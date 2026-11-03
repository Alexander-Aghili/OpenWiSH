//! `codex exec --json` with every tool feature off (the authors' configuration).
use crate::{classify, count, push_flag, BackendConfig, BackendError, Editor, Reply, Usage};
use serde_json::Value;

pub struct CodexCli {
    cfg: BackendConfig,
    session: Option<String>,
}

/// Feature flags the authors switched off (harness/codex_config.toml).
pub const DISABLED_FEATURES: &[&str] = &[
    "code_mode",
    "collaboration_modes",
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "shell_snapshot",
    "view_image",
    "image_generation",
    "multi_agent",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "code_mode_host",
    "skill_search",
    "skill_mcp_dependency_install",
    "sleep_tool",
    "tool_suggest",
    "goals",
    "hooks",
    "worktrees",
    "workspace_dependencies",
    "memories",
];

impl CodexCli {
    pub fn new(cfg: BackendConfig) -> Self {
        CodexCli { cfg, session: None }
    }
    pub fn args(&self, new_conversation: bool) -> Vec<String> {
        let mut a: Vec<String> = vec!["exec".into()];
        if let (false, Some(s)) = (new_conversation, &self.session) {
            a.extend(["resume".into(), s.clone()]);
        }
        a.extend(["--skip-git-repo-check".into(), "--json".into()]);
        push_flag(&mut a, "-m", self.cfg.model.as_deref());
        for setting in self.settings() {
            a.extend(["-c".into(), setting]);
        }
        a.push("-".into());
        a
    }

    /// The `-c key=value` settings: no approvals, a read-only sandbox, no web search, no agents, no tool features.
    fn settings(&self) -> Vec<String> {
        let instructions = self.cfg.workdir.join("instructions.txt");
        let mut c = vec![
            "approval_policy=\"never\"".to_string(),
            "sandbox_mode=\"read-only\"".into(),
            "web_search=\"disabled\"".into(),
            "agents.enabled=false".into(),
            format!(
                "model_instructions_file={}",
                serde_json::to_string(&instructions).expect("path string")
            ),
            format!(
                "developer_instructions={}",
                serde_json::to_string(&self.cfg.instructions).expect("instruction string")
            ),
            "project_doc_max_bytes=0".into(),
        ];
        if let Some(e) = &self.cfg.effort {
            c.push(format!("model_reasoning_effort=\"{e}\""));
        }
        c.extend(DISABLED_FEATURES.iter().map(|f| format!("features.{f}=false")));
        c
    }

    fn prepare_instructions(&mut self) -> Result<(), BackendError> {
        std::fs::create_dir_all(&self.cfg.workdir)
            .and_then(|_| std::fs::write(self.cfg.workdir.join("instructions.txt"), crate::NEUTRAL_INSTRUCTIONS))
            .map_err(|e| BackendError::Fatal(format!("writing task instructions: {e}")))?;
        self.cfg.workdir = std::fs::canonicalize(&self.cfg.workdir)
            .map_err(|e| BackendError::Fatal(format!("resolving editor folder: {e}")))?;
        Ok(())
    }
}

/// What one turn's event stream says.
#[derive(Default)]
struct Turn {
    texts: Vec<String>,
    session: Option<String>,
    usage: Usage,
    used_a_tool: bool,
    errors: Vec<String>,
    completed: bool,
}

impl Turn {
    fn read(&mut self, v: &Value) {
        match v["type"].as_str().unwrap_or("") {
            "thread.started" => self.session = v["thread_id"].as_str().map(String::from),
            "item.started" => self.read_item(&v["item"], false),
            "item.completed" => self.read_item(&v["item"], true),
            "turn.completed" => {
                self.completed = true;
                self.usage = codex_usage(&v["usage"]);
            }
            "error" => self.errors.push(v["message"].as_str().unwrap_or("").to_string()),
            "turn.failed" => self
                .errors
                .push(v["error"]["message"].as_str().unwrap_or("turn failed").to_string()),
            _ => {}
        }
    }

    fn read_item(&mut self, item: &Value, completed: bool) {
        match item["type"].as_str().unwrap_or("") {
            "agent_message" if completed => self.texts.push(item["text"].as_str().unwrap_or("").to_string()),
            "agent_message" | "reasoning" | "error" => {}
            _ => self.used_a_tool = true,
        }
    }
}

fn codex_usage(u: &Value) -> Usage {
    Usage {
        input: count(u, "input_tokens"),
        output: count(u, "output_tokens"),
        cache_read: count(u, "cached_input_tokens"),
        cache_write: count(u, "cache_write_input_tokens"),
        reasoning: count(u, "reasoning_output_tokens"),
    }
}

/// Parse a `codex exec --json` event stream (JSON lines).
pub fn parse_events(stdout: &str) -> Result<Reply, BackendError> {
    let mut turn = Turn::default();
    let events: Vec<Value> = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    for v in &events {
        turn.read(v);
    }
    if !turn.completed || turn.texts.is_empty() {
        let msg = if turn.errors.is_empty() {
            "codex produced no reply".to_string()
        } else {
            turn.errors.join("; ")
        };
        return Err(classify(&msg));
    }
    Ok(Reply {
        // The CLI emits progress updates as agent_message items too. Only the last
        // message is the reply. Keep the whole event stream in raw for inspection.
        text: turn.texts.pop().expect("checked nonempty above"),
        usage: turn.usage,
        session: turn.session,
        tool_free: !turn.used_a_tool,
        cost_usd: None,
        wall_s: 0.0,
        raw: Value::Array(events),
    })
}

impl Editor for CodexCli {
    fn name(&self) -> String {
        format!("codex-cli:{}", self.cfg.model.clone().unwrap_or_default())
    }
    fn send(&mut self, msg: &str, new_conversation: bool) -> Result<Reply, BackendError> {
        self.prepare_instructions()?;
        let args = self.args(new_conversation);
        let done = crate::run_cli("codex", &args, msg, &self.cfg.workdir, self.cfg.timeout_s)?;
        let mut r = parse_events(&done.stdout).map_err(|e| match e {
            BackendError::Fatal(m) => classify(&format!("{m} {}", done.stderr_tail(2000))),
            e => e,
        })?;
        r.wall_s = done.elapsed.as_secs_f64();
        // a resumed thread may not announce its id again
        r.session = r.session.or_else(|| self.session.clone());
        self.session = r.session.clone();
        Ok(r)
    }
    fn session(&self) -> Option<String> {
        self.session.clone()
    }
    fn set_session(&mut self, s: Option<String>) {
        self.session = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_instructions_replace_defaults_on_new_and_resumed_turns() {
        let cfg = BackendConfig {
            kind: "codex".into(),
            model: Some("test-model".into()),
            effort: Some("high".into()),
            instructions: "Use the Python tool.\nReturn exactly {\"answer\": 1}.".into(),
            workdir: std::path::PathBuf::from("/tmp/openwish test/editor_cwd"),
            max_tokens: 1000,
            timeout_s: 10,
        };
        let mut editor = CodexCli::new(cfg);
        editor.set_session(Some("saved-session".into()));
        for fresh in [true, false] {
            let args = editor.args(fresh);
            let settings: Vec<_> = args
                .windows(2)
                .filter(|w| w[0] == "-c")
                .map(|w| w[1].as_str())
                .collect();
            assert!(settings.contains(&"model_instructions_file=\"/tmp/openwish test/editor_cwd/instructions.txt\""));
            let expected = format!(
                "developer_instructions={}",
                serde_json::to_string(&editor.cfg.instructions).unwrap()
            );
            assert!(settings.contains(&expected.as_str()));
            assert!(settings.contains(&"project_doc_max_bytes=0"));
            assert!(settings.contains(&"features.memories=false"));
            assert!(settings.contains(&"features.shell_tool=false"));
        }
    }

    #[test]
    fn writes_neutral_base_and_keeps_task_instructions_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = BackendConfig {
            kind: "codex".into(),
            model: None,
            effort: None,
            instructions: "Use one ```python block.\nQuotes: \" and backslash: \\.\n".into(),
            workdir: dir.path().join("new/editor_cwd"),
            max_tokens: 1000,
            timeout_s: 10,
        };
        let mut editor = CodexCli::new(cfg.clone());
        editor.prepare_instructions().unwrap();
        assert_eq!(
            std::fs::read_to_string(cfg.workdir.join("instructions.txt")).unwrap(),
            crate::NEUTRAL_INSTRUCTIONS
        );
    }

    #[test]
    fn relative_workdir_does_not_duplicate_the_instruction_path() {
        let dir = tempfile::tempdir_in(".").unwrap();
        let cfg = BackendConfig {
            kind: "codex".into(),
            model: None,
            effort: None,
            instructions: "Task rules".into(),
            workdir: dir
                .path()
                .strip_prefix(std::env::current_dir().unwrap())
                .unwrap()
                .join("editor_cwd"),
            max_tokens: 1000,
            timeout_s: 10,
        };
        let mut editor = CodexCli::new(cfg);
        editor.prepare_instructions().unwrap();
        assert!(editor.cfg.workdir.is_absolute());
    }

    #[test]
    fn progress_messages_stay_out_of_the_final_reply() {
        let events = [
            serde_json::json!({"type": "item.completed", "item": {"type": "agent_message", "text": "Checking the circuit."}}),
            serde_json::json!({"type": "item.completed", "item": {"type": "agent_message", "text": "{\"program\": true}"}}),
            serde_json::json!({"type": "turn.completed", "usage": {}}),
        ];
        let stream = events.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
        let reply = parse_events(&stream).unwrap();
        assert_eq!(reply.text, "{\"program\": true}");
        assert_eq!(reply.raw.as_array().unwrap().len(), 3);
    }

    #[test]
    fn parses_authors_event_stream() {
        let s = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/codex_events.jsonl"
        ))
        .unwrap();
        let r = parse_events(&s).unwrap();
        assert!(r.text.contains("```edits"));
        assert_eq!(r.session.as_deref(), Some("id-5ce48e76"));
        assert_eq!(r.usage.input, 355201);
        assert_eq!(r.usage.reasoning, 9146);
        assert!(r.tool_free);
    }

    #[test]
    fn tool_items_mark_not_tool_free_and_errors_classify() {
        let s = "{\"type\":\"thread.started\",\"thread_id\":\"t\"}\n{\"type\":\"item.completed\",\"item\":{\"type\":\"command_execution\",\"command\":\"ls\"}}\n{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"hi\"}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n";
        assert!(!parse_events(s).unwrap().tool_free);
        let e = "{\"type\":\"error\",\"message\":\"Invalid prompt: your prompt was flagged as potentially violating our usage policy.\"}\n{\"type\":\"turn.failed\",\"error\":{\"message\":\"x\"}}\n";
        assert!(matches!(parse_events(e), Err(BackendError::Refused(_))));
    }
}
