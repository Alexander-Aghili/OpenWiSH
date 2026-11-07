//! Hooks: shell commands the harness runs at fixed points of a run.
//!
//! Events:
//! - `session_start`: the first step of an `openwish run` process (a new run or a resumed one).
//! - `pre_tool`: before a Python run or a forward probe. A non-zero exit blocks the call. The hook's stderr
//!   goes back to the agent as the reason, and nothing is charged.
//! - `post_tool`: after a Python run or a forward probe, whether it worked or not.
//! - `on_edit`: once per edit the gate accepted, after the tool call that made it, in ledger order.
//!   After a crash in the middle of a Python run, these may fire again for the same edits.
//! - `session_end`: once, when the run finishes.
//!
//! Hooks come from `[hooks]` in openwish.toml (they run in the workspace root) and from `[hooks]` in a task
//! (they run in the task folder). Both sets run, project hooks first. Each event takes one shell command or a
//! list of them, run with `sh -c`.
//!
//! Hooks run on the host, outside the sandbox, with a timeout (default 60 s, `timeout_s` in `[hooks]`).
//! They are trusted code, like probe and score commands. The tool input they get on stdin is written by the
//! agent, so a hook must treat it as untrusted data.
//!
//! Every hook gets OPENWISH_EVENT, OPENWISH_RUN_DIR, OPENWISH_TASK_DIR, OPENWISH_TOOL and OPENWISH_TURN, and
//! `on_edit` hooks also get OPENWISH_EDIT_ID and OPENWISH_EDIT_TENSOR, `post_tool` hooks OPENWISH_TOOL_OK (1 or 0),
//! and `session_end` hooks OPENWISH_END_REASON. On stdin a hook gets one JSON object:
//! `{"event", "tool", "turn", "input"}`, where `input` holds the tool call (code or request) and, after the
//! call, its result.
use openwish_process::{Failure, Options};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    SessionStart,
    PreTool,
    PostTool,
    OnEdit,
    SessionEnd,
}

impl Event {
    pub const ALL: [Event; 5] = [
        Event::SessionStart,
        Event::PreTool,
        Event::PostTool,
        Event::OnEdit,
        Event::SessionEnd,
    ];
    pub fn name(&self) -> &'static str {
        match self {
            Event::SessionStart => "session_start",
            Event::PreTool => "pre_tool",
            Event::PostTool => "post_tool",
            Event::OnEdit => "on_edit",
            Event::SessionEnd => "session_end",
        }
    }
}

/// The `[hooks]` table, as written in openwish.toml or a task file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HooksSpec {
    #[serde(default, deserialize_with = "one_or_many")]
    pub session_start: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub pre_tool: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub post_tool: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub on_edit: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub session_end: Vec<String>,
    /// seconds one hook may run before it is killed (default 60)
    pub timeout_s: Option<u64>,
}

pub const DEFAULT_TIMEOUT_S: u64 = 60;

/// Accept `pre_tool = "cmd"` as well as `pre_tool = ["cmd1", "cmd2"]`.
fn one_or_many<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(s) => vec![s],
        OneOrMany::Many(v) => v,
    })
}

impl HooksSpec {
    pub fn commands(&self, e: Event) -> &[String] {
        match e {
            Event::SessionStart => &self.session_start,
            Event::PreTool => &self.pre_tool,
            Event::PostTool => &self.post_tool,
            Event::OnEdit => &self.on_edit,
            Event::SessionEnd => &self.session_end,
        }
    }
    pub fn is_empty(&self) -> bool {
        Event::ALL.iter().all(|e| self.commands(*e).is_empty())
    }
}

/// One hook, ready to run.
#[derive(Clone, Debug, PartialEq)]
pub struct Hook {
    pub event: Event,
    pub command: String,
    pub cwd: PathBuf,
    pub timeout_s: u64,
    /// where it came from ("openwish.toml" or "task"), for logs
    pub origin: String,
}

/// All hooks of a run, in the order they run.
#[derive(Clone, Debug, Default)]
pub struct HookSet {
    pub hooks: Vec<Hook>,
}

/// What a hook is told about the moment it runs.
#[derive(Clone, Debug, Default)]
pub struct Context {
    pub run_dir: PathBuf,
    pub task_dir: PathBuf,
    /// "python", a probe tag, "edits" (the final reply's edits block), or "" for session events
    pub tool: String,
    pub turn: u32,
    pub edit: Option<(u64, String)>,
    /// more variables, such as OPENWISH_TOOL_OK or OPENWISH_END_REASON
    pub extra: Vec<(String, String)>,
    /// the tool call and its result, sent on stdin
    pub input: serde_json::Value,
}

/// The result of one hook.
#[derive(Clone, Debug, Serialize)]
pub struct Outcome {
    pub event: Event,
    pub command: String,
    pub origin: String,
    pub ok: bool,
    pub code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
    pub seconds: f64,
}

impl Outcome {
    /// Why a failed hook failed, in a few words for the agent.
    pub fn reason(&self) -> String {
        let e = self.stderr.trim();
        if !e.is_empty() {
            return openwish_text::head(e, 4000).to_string();
        }
        if self.timed_out {
            "the hook timed out".into()
        } else {
            match self.code {
                Some(c) => format!("the hook exited with status {c}"),
                None => "the hook was stopped by a signal".into(),
            }
        }
    }
}

impl HookSet {
    pub fn from_spec(spec: &HooksSpec, cwd: &Path, origin: &str) -> HookSet {
        let timeout_s = spec.timeout_s.unwrap_or(DEFAULT_TIMEOUT_S);
        let mut hooks = vec![];
        for e in Event::ALL {
            for c in spec.commands(e) {
                hooks.push(Hook {
                    event: e,
                    command: c.clone(),
                    cwd: cwd.to_path_buf(),
                    timeout_s,
                    origin: origin.to_string(),
                });
            }
        }
        HookSet { hooks }
    }
    pub fn extend(&mut self, other: HookSet) {
        self.hooks.extend(other.hooks);
    }
    pub fn has(&self, e: Event) -> bool {
        self.hooks.iter().any(|h| h.event == e)
    }

    /// Run every hook for `e`, in order. Failures do not stop the others.
    pub fn fire(&self, e: Event, ctx: &Context) -> Vec<Outcome> {
        self.hooks
            .iter()
            .filter(|h| h.event == e)
            .map(|h| run_one(h, ctx))
            .collect()
    }

    /// Run the `pre_tool` hooks in order until one fails. Returns what ran and, if a hook failed, its reason.
    pub fn check(&self, ctx: &Context) -> (Vec<Outcome>, Option<String>) {
        let mut ran = vec![];
        for h in self.hooks.iter().filter(|h| h.event == Event::PreTool) {
            let o = run_one(h, ctx);
            let why = (!o.ok).then(|| o.reason());
            ran.push(o);
            if why.is_some() {
                return (ran, why);
            }
        }
        (ran, None)
    }
}

/// Characters of a hook's stdout or stderr that are kept.
const MAX_CAPTURE: usize = 20_000;

/// Run one hook with `sh -c` in its own process group, so a timeout kills everything it started.
pub fn run_one(h: &Hook, ctx: &Context) -> Outcome {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(&h.command).current_dir(&h.cwd);
    set_env(&mut cmd, h.event, ctx);
    let stdin = serde_json::json!({"event": h.event.name(), "tool": ctx.tool, "turn": ctx.turn, "input": ctx.input});
    let opts = Options {
        stdin: Some(stdin.to_string()),
        timeout: Duration::from_secs(h.timeout_s),
        own_group: true,
    };
    let mut out = Outcome {
        event: h.event,
        command: h.command.clone(),
        origin: h.origin.clone(),
        ok: false,
        code: None,
        timed_out: false,
        stdout: String::new(),
        stderr: String::new(),
        seconds: 0.0,
    };
    match openwish_process::run(&mut cmd, opts) {
        Ok(done) => {
            out.ok = done.status.is_some_and(|s| s.success());
            out.code = done.status.and_then(|s| s.code());
            out.timed_out = done.timed_out();
            out.seconds = done.elapsed.as_secs_f64();
            out.stdout = openwish_text::clip(&done.stdout, MAX_CAPTURE, "");
            out.stderr = openwish_text::clip(&done.stderr, MAX_CAPTURE, "");
        }
        Err(Failure::Start { source, .. }) => {
            out.stderr = format!("the hook could not start in {}: {source}", h.cwd.display())
        }
        Err(e) => out.stderr = e.to_string(),
    }
    out
}

/// The variables every hook gets, plus the event's own (`ctx.extra`).
fn set_env(cmd: &mut Command, event: Event, ctx: &Context) {
    let (edit_id, edit_tensor) = match &ctx.edit {
        Some((id, t)) => (id.to_string(), t.clone()),
        None => (String::new(), String::new()),
    };
    cmd.env("OPENWISH_EVENT", event.name())
        .env("OPENWISH_RUN_DIR", &ctx.run_dir)
        .env("OPENWISH_TASK_DIR", &ctx.task_dir)
        .env("OPENWISH_TOOL", &ctx.tool)
        .env("OPENWISH_TURN", ctx.turn.to_string())
        .env("OPENWISH_EDIT_ID", edit_id)
        .env("OPENWISH_EDIT_TENSOR", edit_tensor)
        .envs(ctx.extra.iter().map(|(k, v)| (k, v)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn set(toml_text: &str, cwd: &Path) -> HookSet {
        let spec: HooksSpec = toml::from_str(toml_text).unwrap();
        HookSet::from_spec(&spec, cwd, "test")
    }
    fn ctx(d: &Path) -> Context {
        Context {
            run_dir: d.to_path_buf(),
            task_dir: d.to_path_buf(),
            tool: "python".into(),
            turn: 4,
            ..Default::default()
        }
    }

    #[test]
    fn a_hook_takes_one_command_or_a_list_and_rejects_unknown_keys() {
        let s: HooksSpec =
            toml::from_str("pre_tool = \"echo a\"\npost_tool = [\"echo b\", \"echo c\"]\ntimeout_s = 5\n").unwrap();
        assert_eq!(s.pre_tool, vec!["echo a".to_string()]);
        assert_eq!(s.post_tool.len(), 2);
        assert_eq!(s.timeout_s, Some(5));
        assert!(toml::from_str::<HooksSpec>("pre_tools = \"x\"\n").is_err());
        assert!(HooksSpec::default().is_empty());
        let d = tempfile::tempdir().unwrap();
        let hs = HookSet::from_spec(&s, d.path(), "task");
        assert_eq!(hs.hooks.len(), 3);
        assert!(hs
            .hooks
            .iter()
            .all(|h| h.timeout_s == 5 && h.cwd == d.path() && h.origin == "task"));
        assert!(hs.has(Event::PostTool) && !hs.has(Event::OnEdit));
    }

    #[test]
    fn hooks_see_the_variables_stdin_and_their_folder() {
        let d = tempfile::tempdir().unwrap();
        let hs = set(
            r#"post_tool = 'echo "$OPENWISH_EVENT $OPENWISH_TOOL $OPENWISH_TURN $OPENWISH_EDIT_ID $(basename "$PWD")"; cat'"#,
            d.path(),
        );
        let mut c = ctx(d.path());
        c.edit = Some((7, "W".into()));
        c.input = serde_json::json!({"code": "print(1)"});
        let o = hs.fire(Event::PostTool, &c);
        assert_eq!(o.len(), 1);
        assert!(o[0].ok, "{o:?}");
        let name = d.path().file_name().unwrap().to_string_lossy().to_string();
        assert!(
            o[0].stdout.starts_with(&format!("post_tool python 4 7 {name}")),
            "{}",
            o[0].stdout
        );
        assert!(
            o[0].stdout.contains("\"code\":\"print(1)\"") && o[0].stdout.contains("\"event\":\"post_tool\""),
            "{}",
            o[0].stdout
        );
    }

    #[test]
    fn a_failing_pre_tool_hook_stops_the_rest_and_gives_its_stderr() {
        let d = tempfile::tempdir().unwrap();
        let hs = set(
            r#"pre_tool = ["true", "echo no python today >&2; exit 3", "touch ran_third"]"#,
            d.path(),
        );
        let (ran, why) = hs.check(&ctx(d.path()));
        assert_eq!(ran.len(), 2);
        assert_eq!(why.as_deref(), Some("no python today"));
        assert_eq!(ran[1].code, Some(3));
        assert!(!d.path().join("ran_third").exists());
        let ok = set("pre_tool = \"true\"", d.path());
        assert_eq!(ok.check(&ctx(d.path())).1, None);
    }

    #[test]
    fn a_hook_that_runs_too_long_is_killed_with_its_children() {
        let d = tempfile::tempdir().unwrap();
        let hs = set("pre_tool = \"sleep 30 & sleep 30\"\ntimeout_s = 1\n", d.path());
        let t0 = Instant::now();
        let (ran, why) = hs.check(&ctx(d.path()));
        assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", t0.elapsed());
        assert!(ran[0].timed_out);
        assert_eq!(why.as_deref(), Some("the hook timed out"));
    }

    #[test]
    fn a_hook_that_cannot_start_counts_as_failed() {
        let d = tempfile::tempdir().unwrap();
        let mut hs = set("pre_tool = \"true\"", d.path());
        hs.hooks[0].cwd = d.path().join("missing");
        let (_, why) = hs.check(&ctx(d.path()));
        assert!(why.is_some());
    }
}
