//! One run of one task: the turn loop, tools, budgets, handoff, finalisation and crash-resume.
//!
//! A turn sends the next message, saves the reply as pending, acts on it (a Python run, a forward probe or the
//! final answer) and writes the message for the turn after. The state is saved between these steps, so a run
//! that stops at any point resumes without losing or repeating a reply.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::hooks::{Context as HookCtx, Event, HookSet, Outcome};
use crate::paths;
use crate::protocol::{parse, Action};
use crate::spec::{ForwardSpec, PythonSpec, TaskSpec};
use openwish_audit::Flag;
use openwish_llm::{BackendError, Editor, Reply};
use openwish_sandbox::{Kernel, KernelCfg, Rpc, RpcReply, RunOutput};
use openwish_tensor::gate::{EditRecord, Gate, Refusal};
use openwish_tensor::ops::EditOp;
use openwish_tensor::Network;

mod budget;
mod finish;
mod forward;
mod python;
mod rpc;
mod tool_hooks;
mod turn;

/// The budget group of tools that draw from the Python runs.
const PYTHON_POOL: &str = "python";
/// Consecutive refusals after which the run ends.
const MAX_REFUSAL_STREAK: u32 = 3;
/// The longest wait between two retries of a failing backend.
const MAX_BACKOFF_S: f64 = 1800.0;
/// Characters of /work/notes.md shown in a handoff.
const MAX_NOTES_CHARS: usize = 20_000;
const MAX_SUBMISSION_BYTES: u64 = 2_000_000;

/// The `State` counters that `openwish status` and the HTML replay show, in display order.
pub const COUNTERS: &[&str] = &[
    "turn",
    "runs_used",
    "passes_used",
    "handoffs",
    "nudges",
    "failed_requests",
    "protocol_errors",
    "hook_blocks",
    "refusals",
    "backend_errors",
    "kernel_restarts",
    "cost_usd",
    "usage_total",
];

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct State {
    pub turn: u32,
    pub runs_used: u32,
    pub passes_used: BTreeMap<String, u32>,
    pub failed_requests: u32,
    pub protocol_errors: u32,
    pub nudges: u32,
    pub handoffs: u32,
    pub since_handoff: u32,
    pub conv_tokens: u64,
    pub session: Option<String>,
    pub next_message: String,
    pub new_conversation: bool,
    /// a reply received but not yet acted on (crash between receiving and acting)
    pub pending_reply: Option<String>,
    /// the last edit id before the pending reply was acted on (to tell which edits an interrupted run made)
    pub ledger_mark: Option<u64>,
    /// start a new conversation after the next action (a backend that cannot resume a session)
    pub force_handoff: bool,
    pub refusals: u32,
    pub refusal_streak: u32,
    pub backend_errors: u32,
    pub kernel_restarts: u32,
    pub done: bool,
    pub end_reason: Option<String>,
    pub usage_total: openwish_llm::Usage,
    pub cost_usd: f64,
    pub tool_free_turns: u32,
    /// tool calls a pre_tool hook blocked (not charged)
    pub hook_blocks: u32,
}

/// Machine-level settings (not part of a task).
#[derive(Clone, Debug)]
pub struct Env {
    pub venv: PathBuf,
    pub kernel_dir: PathBuf,
    pub retry_base_s: f64,
    pub max_retries: u32,
    /// the CPU budget: the sandbox kernel gets at most this many threads
    pub threads: u32,
    /// project hooks (openwish.toml); the task's own hooks are added to these
    pub hooks: crate::hooks::HookSet,
}

#[derive(Debug, PartialEq)]
pub enum Step {
    Continue,
    Done,
}

pub struct Run {
    pub spec: TaskSpec,
    pub dir: PathBuf,
    pub state: State,
    pub editor: Box<dyn Editor>,
    pub env: Env,
    gate: Option<Gate>,
    kernel: Option<Kernel>,
    first_message: String,
    /// project hooks, then the task's own
    hooks: HookSet,
    /// session_start / session_end already ran in this process
    started: bool,
    ended: bool,
}

/// The task spec as saved in a run folder (`spec.json`), with the folder it was loaded from.
#[derive(Serialize, Deserialize)]
struct SavedSpec {
    spec: TaskSpec,
    dir: PathBuf,
}

/// The task spec saved in the run folder `run_dir`.
pub fn saved_spec(run_dir: &Path) -> Result<TaskSpec> {
    let text = std::fs::read_to_string(run_dir.join("spec.json")).context("not a run folder (no spec.json)")?;
    let saved: SavedSpec = serde_json::from_str(&text)?;
    Ok(TaskSpec {
        dir: saved.dir,
        ..saved.spec
    })
}

impl Run {
    pub fn create(spec: TaskSpec, dir: &Path, editor: Box<dyn Editor>, env: Env) -> Result<Run> {
        if dir.join("state.json").exists() {
            bail!("{} already holds a run; use --resume", dir.display());
        }
        for sub in ["turns", "work/.rpc", "requests"] {
            std::fs::create_dir_all(dir.join(sub))?;
        }
        // tools run in other folders, so every path handed to them must be absolute
        let dir = std::fs::canonicalize(dir)?;
        let first = first_message(&spec)?;
        std::fs::write(dir.join("rules.txt"), &first)?;
        let saved = SavedSpec {
            spec: spec.clone(),
            dir: spec.dir.clone(),
        };
        std::fs::write(dir.join("spec.json"), serde_json::to_string_pretty(&saved)?)?;
        let state = State {
            next_message: first.clone(),
            new_conversation: true,
            ..Default::default()
        };
        let r = Run::open(spec, dir, state, first, editor, env)?;
        r.log(&format!(
            "created run of task {} with editor {}; access: {}",
            r.spec.id,
            r.editor.name(),
            r.spec.access()["label"].as_str().unwrap_or("")
        ));
        r.save_state()?;
        Ok(r)
    }

    pub fn resume(dir: &Path, mut editor: Box<dyn Editor>, env: Env) -> Result<Run> {
        let dir = std::fs::canonicalize(dir).with_context(|| format!("{}", dir.display()))?;
        std::fs::create_dir_all(dir.join("work/.rpc"))?;
        let spec = saved_spec(&dir)?;
        let mut state: State =
            serde_json::from_str(&std::fs::read_to_string(dir.join("state.json")).context("state.json")?)?;
        editor.set_session(state.session.clone());
        let first = std::fs::read_to_string(dir.join("rules.txt"))?;
        // the kernel's variables did not survive the restart
        if state.runs_used > 0 && state.pending_reply.is_none() && !state.done {
            state.next_message = format!(
                "[The harness was restarted: the Python kernel's variables were lost; your saved edits and files in \
                 /work are kept.]\n\n{}",
                state.next_message
            );
        }
        let mut r = Run::open(spec, dir, state, first, editor, env)?;
        if !r.editor.can_resume() && !r.state.done {
            // the backend keeps no conversation of its own: continue in a new one that restates the task
            if r.state.pending_reply.is_some() {
                r.state.force_handoff = true;
            } else {
                r.start_handoff();
            }
        }
        r.log("resumed");
        r.save_state()?;
        Ok(r)
    }

    /// A run over `dir` in `state`, with its hooks and its edit gate.
    fn open(
        spec: TaskSpec,
        dir: PathBuf,
        state: State,
        first_message: String,
        editor: Box<dyn Editor>,
        env: Env,
    ) -> Result<Run> {
        let hooks = all_hooks(&env, &spec);
        let gate = match &spec.network {
            Some(n) => {
                let base = spec.resolve(&n.base);
                Some(Gate::open_linked(
                    &dir.join("edits"),
                    Network::load(&base)?,
                    Some(&base),
                    spec.edits.policy.clone(),
                )?)
            }
            None => None,
        };
        Ok(Run {
            spec,
            dir,
            state,
            editor,
            env,
            gate,
            kernel: None,
            first_message,
            hooks,
            started: false,
            ended: false,
        })
    }

    pub fn save_state(&self) -> Result<()> {
        let tmp = self.dir.join("state.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.state)?)?;
        std::fs::rename(tmp, self.dir.join("state.json"))?;
        Ok(())
    }

    pub fn gate(&self) -> Option<&Gate> {
        self.gate.as_ref()
    }
    pub fn gate_mut(&mut self) -> Option<&mut Gate> {
        self.gate.as_mut()
    }
    fn last_edit_id(&self) -> Option<u64> {
        self.gate.as_ref().map(Gate::last_edit_id)
    }

    fn log(&self, line: &str) {
        let _ = append_line(
            &self.dir.join("loop.log"),
            &format!("[{}] {line}", chrono::Local::now().to_rfc3339()),
        );
    }

    fn tool_log(&self, v: Value) -> Result<()> {
        append_line(&self.dir.join("tool_calls.jsonl"), &v.to_string())?;
        Ok(())
    }
}

/// The agent's first message: the prompt, then the appendix if the task has one.
fn first_message(spec: &TaskSpec) -> Result<String> {
    let mut first = std::fs::read_to_string(spec.resolve(&spec.prompt))?;
    if let Some(a) = &spec.appendix {
        first.push_str("\n\n");
        first.push_str(&std::fs::read_to_string(spec.resolve(a))?);
    }
    if let Some(validation) = &spec.finish.validation {
        first.push_str("\n\nOPENWISH SUBMISSION PROTOCOL\nYour final submission is checked for format only, never correctness. \
            Repair format errors when requested. OpenWiSH executes your tool blocks, even when the provider's own tool list is empty.\n");
        if let Some(py) = &spec.python {
            if py.runs > 0 {
                first.push_str(&format!(
                    "Python is available through text blocks, with at most {} runs. \
                    Reply with one block to execute code, for example:\n```python\nprint(NAMES)\n```\n",
                    py.runs
                ));
            }
        }
        for tag in spec.forward.keys() {
            first.push_str(&format!(
                "The declared {tag} probe accepts one ```{tag} block with a JSON request.\n"
            ));
        }
        if let Some(file) = &validation.submission_file {
            first.push_str(&format!(
                "Write your submission to /work/{} or put the submission in your final reply.\n",
                file.display()
            ));
        }
    }
    Ok(first)
}

/// Project hooks first, then the task's (which run in the task folder).
fn all_hooks(env: &Env, spec: &TaskSpec) -> HookSet {
    let mut h = env.hooks.clone();
    h.extend(HookSet::from_spec(&spec.hooks, &spec.dir, "task"));
    h
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")
}

#[cfg(test)]
mod tests;
