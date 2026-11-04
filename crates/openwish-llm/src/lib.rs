//! Editor backends. Every backend turns one message into one reply; the harness owns the turn loop.
use openwish_process::{Failure, Finished, Options};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

pub mod anthropic_api;
pub mod openai_compat;
pub mod claude_cli;
pub mod codex_cli;
mod codex_usage;
pub mod scripted;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input. Cached reads and writes are counted separately.
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub reasoning: u64,
}

impl Usage {
    /// The usage object of the Anthropic Messages API, which `claude -p` reports too (no reasoning count).
    pub fn from_anthropic(u: &serde_json::Value) -> Usage {
        Usage {
            input: count(u, "input_tokens"),
            output: count(u, "output_tokens"),
            cache_read: count(u, "cache_read_input_tokens"),
            cache_write: count(u, "cache_creation_input_tokens"),
            reasoning: 0,
        }
    }

    /// Every token of the conversation: input, cached input (read and written) and output.
    pub fn context_tokens(&self) -> u64 {
        self.input + self.cache_read + self.cache_write + self.output
    }
}

impl std::ops::AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, u: &Usage) {
        self.input += u.input;
        self.output += u.output;
        self.cache_read += u.cache_read;
        self.cache_write += u.cache_write;
        self.reasoning += u.reasoning;
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reply {
    pub text: String,
    pub usage: Usage,
    /// Current context footprint, distinct from cumulative billing counters.
    pub context_tokens: Option<u64>,
    pub session: Option<String>,
    /// true when the backend reports that the model used no tool of its own this turn
    pub tool_free: bool,
    pub cost_usd: Option<f64>,
    pub wall_s: f64,
    /// the raw backend output (JSON result, event stream, or API response)
    pub raw: serde_json::Value,
}

/// Why a backend call failed, so the harness can decide whether to retry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BackendError {
    /// the provider's content filter refused the prompt or the reply
    Refused(String),
    /// rate limits, overload, network, timeouts: retry with backoff
    Transient(String),
    /// configuration or request errors: do not retry
    Fatal(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Refused(m) => write!(f, "refused by the provider: {m}"),
            BackendError::Transient(m) => write!(f, "transient backend error: {m}"),
            BackendError::Fatal(m) => write!(f, "backend error: {m}"),
        }
    }
}
impl std::error::Error for BackendError {}

pub trait Editor: Send {
    fn name(&self) -> String;
    /// Send one message. `new_conversation` starts a fresh thread (first turn, or a handoff).
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError>;
    /// The conversation id to persist for crash-resume.
    fn session(&self) -> Option<String>;
    fn set_session(&mut self, s: Option<String>);
    /// Whether a saved session id is enough to continue the conversation after a restart.
    fn can_resume(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    pub kind: String,
    pub model: Option<String>,
    pub effort: Option<String>,
    /// system instructions; the authors used one neutral line
    pub instructions: String,
    /// Enabled interactive text-block tags. These request a handoff to the environment.
    #[serde(default)]
    pub tool_blocks: Vec<String>,
    /// Final artifact tags whose last complete candidate takes precedence over trailing prose.
    #[serde(default)]
    pub final_blocks: Vec<String>,
    /// an empty directory the CLI runs in, so it sees no project files
    pub workdir: std::path::PathBuf,
    pub max_tokens: u32,
    pub timeout_s: u64,
}

pub const NEUTRAL_INSTRUCTIONS: &str = "Answer the user's request.";

/// The model the claude and api backends use when none is chosen.
pub const DEFAULT_CLAUDE_MODEL: &str = "claude-opus-5-5";

/// Build a backend from `kind`: "claude", "codex", "api", "openai" (any OpenAI-compatible server), or "scripted:<file>".
pub fn make(cfg: &BackendConfig) -> anyhow::Result<Box<dyn Editor>> {
    let k = cfg.kind.as_str();
    Ok(match k {
        "claude" => Box::new(claude_cli::ClaudeCli::new(cfg.clone())),
        "codex" => Box::new(codex_cli::CodexCli::new(cfg.clone())),
        "api" => Box::new(anthropic_api::AnthropicApi::new(cfg.clone())?),
        "openai" => Box::new(openai_compat::OpenAiCompat::new(cfg.clone())?),
        _ if k.starts_with("scripted:") => Box::new(scripted::Scripted::from_file(std::path::Path::new(
            &k["scripted:".len()..],
        ))?),
        _ => anyhow::bail!("unknown backend {k} (claude | codex | api | openai | scripted:<file>)"),
    })
}

/// Words in an error message that mean the provider's content filter refused the prompt or the reply.
const REFUSAL_MARKERS: &[&str] = &["flagged", "usage policy", "refusal", "content filter"];

/// Words in an error message that mean a retry may work.
const TRANSIENT_MARKERS: &[&str] = &[
    "429",
    "529",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "rate_limit",
    "rate limit",
    "timed out",
    "timeout",
    "connection",
    "temporarily",
    "try again",
    "network",
    "econnreset",
    "quota",
    "usage limit",
];

/// Classify an error message from any provider.
pub fn classify(msg: &str) -> BackendError {
    let m = msg.to_lowercase();
    let mentions = |markers: &[&str]| markers.iter().any(|k| m.contains(k));
    let short: String = msg.chars().take(2000).collect();
    if mentions(REFUSAL_MARKERS) {
        BackendError::Refused(short)
    } else if mentions(TRANSIENT_MARKERS) {
        BackendError::Transient(short)
    } else {
        BackendError::Fatal(short)
    }
}

/// Run a CLI with `input` on stdin in `cwd`, killing it after `timeout_s`. A timeout is a transient error.
pub fn run_cli(
    program: &str,
    args: &[String],
    input: &str,
    cwd: &Path,
    timeout_s: u64,
) -> Result<Finished, BackendError> {
    std::fs::create_dir_all(cwd).map_err(|e| BackendError::Fatal(format!("workdir: {e}")))?;
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);
    let opts = Options {
        stdin: Some(input.to_string()),
        timeout: Duration::from_secs(timeout_s),
        own_group: false,
    };
    let done = openwish_process::run(&mut cmd, opts).map_err(|e| match e {
        Failure::Start { .. } => BackendError::Fatal(e.to_string()),
        Failure::Wait { .. } => BackendError::Transient(e.to_string()),
    })?;
    if done.timed_out() {
        return Err(BackendError::Transient(format!(
            "{program} timed out after {timeout_s} s"
        )));
    }
    Ok(done)
}

/// A token count from a provider's usage object (0 when absent).
fn count(usage: &serde_json::Value, key: &str) -> u64 {
    usage[key].as_u64().unwrap_or(0)
}

/// `flag value` when there is a value.
fn push_flag(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(v) = value {
        args.extend([flag.to_string(), v.to_string()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_errors() {
        assert!(matches!(
            classify("Invalid prompt: your prompt was flagged as potentially violating our usage policy"),
            BackendError::Refused(_)
        ));
        assert!(matches!(
            classify("stop_reason refusal (cyber)"),
            BackendError::Refused(_)
        ));
        assert!(matches!(
            classify("API Error: 529 overloaded_error"),
            BackendError::Transient(_)
        ));
        assert!(matches!(classify("429 rate_limit_error"), BackendError::Transient(_)));
        assert!(matches!(
            classify("connection reset by peer"),
            BackendError::Transient(_)
        ));
        assert!(matches!(classify("timed out after 3600 s"), BackendError::Transient(_)));
        assert!(matches!(classify("invalid model name foo"), BackendError::Fatal(_)));
    }
}
