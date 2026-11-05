//! `claude -p` with every built-in tool switched off; the harness's tools are text blocks.
use crate::{classify, push_flag, BackendConfig, BackendError, Editor, Reply, Usage};
use openwish_process::last_json_line;
use serde_json::Value;

/// Print one JSON result, and switch off every built-in tool, MCP server, slash command and settings file.
const BASE_ARGS: &[&str] = &[
    "-p",
    "--output-format",
    "json",
    "--tools",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources",
    "",
];

pub struct ClaudeCli {
    cfg: BackendConfig,
    session: Option<String>,
    /// total_cost_usd of the current conversation after the last turn; None after a resume, when it is unknown
    conversation_cost: Option<f64>,
}

impl ClaudeCli {
    pub fn new(cfg: BackendConfig) -> Self {
        ClaudeCli {
            cfg,
            session: None,
            conversation_cost: Some(0.0),
        }
    }

    /// This turn's cost. `claude -p` reports total_cost_usd for the whole conversation, resumed turns included.
    pub fn turn_cost(&mut self, total: Option<f64>, new_conversation: bool) -> Option<f64> {
        let before = if new_conversation {
            Some(0.0)
        } else {
            self.conversation_cost
        };
        self.conversation_cost = total;
        Some((total? - before?).max(0.0))
    }
    pub fn args(&self, new_conversation: bool) -> Vec<String> {
        let mut a: Vec<String> = BASE_ARGS.iter().map(|s| s.to_string()).collect();
        push_flag(&mut a, "--system-prompt", Some(&self.cfg.instructions));
        push_flag(&mut a, "--model", self.cfg.model.as_deref());
        push_flag(&mut a, "--effort", self.cfg.effort.as_deref());
        let resume = if new_conversation {
            None
        } else {
            self.session.as_deref()
        };
        push_flag(&mut a, "--resume", resume);
        a
    }
}

/// Parse the single JSON object `claude -p --output-format json` prints (its last line that starts with `{`).
pub fn parse_result(stdout: &str) -> Result<Reply, BackendError> {
    let v: Value = last_json_line(stdout)
        .and_then(|l| serde_json::from_str(l).ok())
        .ok_or_else(|| {
            let head: String = stdout.chars().take(500).collect();
            classify(&format!("unparseable claude output: {head}"))
        })?;
    let text = v["result"].as_str().unwrap_or("").to_string();
    if v["is_error"].as_bool().unwrap_or(false) || v["subtype"].as_str().is_some_and(|s| s != "success") {
        let status = v["api_error_status"]
            .as_u64()
            .map(|s| format!(" (status {s})"))
            .unwrap_or_default();
        return Err(classify(&format!("{text}{status}")));
    }
    let no_denials = v["permission_denials"].as_array().is_none_or(|a| a.is_empty());
    let tool_free = v["num_turns"].as_u64().unwrap_or(1) <= 1 && no_denials;
    let usage = &v["usage"];
    Ok(Reply {
        text,
        usage: Usage {
            reasoning: usage["output_tokens_details"]["thinking_tokens"].as_u64().unwrap_or(0),
            ..Usage::from_anthropic(usage)
        },
        session: v["session_id"].as_str().map(String::from),
        context_tokens: None,
        tool_free,
        cost_usd: v["total_cost_usd"].as_f64(),
        wall_s: v["duration_ms"].as_f64().unwrap_or(0.0) / 1000.0,
        raw: v,
    })
}

impl Editor for ClaudeCli {
    fn name(&self) -> String {
        format!("claude-cli:{}", self.cfg.model.clone().unwrap_or_default())
    }
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError> {
        let args = self.args(new_conversation);
        let done = crate::run_cli("claude", &args, msg, &self.cfg.workdir, self.cfg.timeout_s)?;
        let exited_badly = done.status.is_some_and(|s| !s.success());
        let r = parse_result(&done.stdout).map_err(|e| match e {
            // a crash leaves no JSON: say why from stderr
            BackendError::Fatal(m) if exited_badly => classify(&format!("{m} {}", done.stderr)),
            e => e,
        })?;
        self.session = r.session.clone();
        let cost_usd = self.turn_cost(r.cost_usd, new_conversation);
        Ok(Reply { cost_usd, ..r })
    }
    fn session(&self) -> Option<String> {
        self.session.clone()
    }
    fn set_session(&mut self, s: Option<String>) {
        self.session = s;
        self.conversation_cost = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_is_the_increase_of_the_conversation_total() {
        let mut c = ClaudeCli::new(test_cfg());
        // `claude -p --resume` reports total_cost_usd for the whole conversation so far
        assert_eq!(c.turn_cost(Some(1.5), true), Some(1.5));
        let second = c.turn_cost(Some(1.75), false).unwrap();
        assert!((second - 0.25).abs() < 1e-9, "{second}");
        // a handoff starts a new conversation and a new total
        assert_eq!(c.turn_cost(Some(0.5), true), Some(0.5));
        // after a crash-resume the earlier total is unknown: report nothing rather than count it twice
        c.set_session(Some("s".into()));
        assert_eq!(c.turn_cost(Some(9.0), false), None);
        let next = c.turn_cost(Some(9.5), false).unwrap();
        assert!((next - 0.5).abs() < 1e-9, "{next}");
    }

    #[test]
    fn parses_real_result() {
        let s = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/claude_result.json"
        ))
        .unwrap();
        let r = parse_result(&s).unwrap();
        assert_eq!(r.text, "PONG");
        assert!(r.session.is_some());
        assert!(r.tool_free);
        assert_eq!(r.usage.output, 5);
        assert!(r.cost_usd.unwrap() > 0.0);
    }

    #[test]
    fn error_result_is_classified() {
        let s = r#"{"type":"result","is_error":true,"result":"API Error: 529 overloaded_error","api_error_status":529,"session_id":"x","num_turns":1}"#;
        assert!(matches!(parse_result(s), Err(BackendError::Transient(_))));
        assert!(matches!(
            parse_result("not json"),
            Err(BackendError::Transient(_)) | Err(BackendError::Fatal(_))
        ));
    }

    fn test_cfg() -> BackendConfig {
        BackendConfig {
            kind: "claude".into(),
            model: Some("claude-opus-5-5".into()),
            effort: Some("max".into()),
            instructions: "I".into(),
            tool_blocks: vec![],
            final_blocks: vec![],
            workdir: "/tmp".into(),
            max_tokens: 0,
            timeout_s: 1,
        }
    }

    #[test]
    fn args_disable_tools_and_resume() {
        let mut c = ClaudeCli::new(test_cfg());
        let a = c.args(true);
        for want in [
            "-p",
            "--tools",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources",
            "--system-prompt",
            "--model",
            "--effort",
        ] {
            assert!(a.iter().any(|x| x == want), "missing {want}");
        }
        assert!(!a.iter().any(|x| x == "--resume"));
        c.set_session(Some("abc".into()));
        let a = c.args(false);
        assert!(a.windows(2).any(|w| w[0] == "--resume" && w[1] == "abc"));
        assert!(!c.args(true).iter().any(|x| x == "--resume"));
    }
}
