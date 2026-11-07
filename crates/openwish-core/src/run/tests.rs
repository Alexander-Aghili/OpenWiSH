use super::*;
use openwish_llm::scripted::Scripted;
use openwish_tensor::Tensor;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn env() -> Env {
    Env {
        venv: root().join("assets/venvs/sandbox"),
        kernel_dir: root().join("python/kernel"),
        retry_base_s: 0.0,
        max_retries: 3,
        threads: 2,
        hooks: Default::default(),
    }
}
fn has_venv() -> bool {
    env().venv.join("bin/python").exists()
}

/// A task directory with a 2x2 network and the given extra TOML.
fn task(d: &Path, extra: &str) -> TaskSpec {
    let mut n = Network::default();
    n.tensors.insert("W".into(), Tensor::new(vec![2, 2], vec![0.; 4]));
    n.save_dir(&d.join("net")).unwrap();
    std::fs::write(d.join("prompt.txt"), "TASK: make W the identity.").unwrap();
    std::fs::write(d.join("pre.py"), "PRE = 7").unwrap();
    let toml = format!("id = \"toy\"\nprompt = \"prompt.txt\"\n[network]\nbase = \"net\"\n{extra}");
    std::fs::write(d.join("task.toml"), toml).unwrap();
    TaskSpec::load(&d.join("task.toml")).unwrap()
}
const FWD: &str = r#"
[forward.games]
command = ["sh", "-c", "if grep -q bad \"$OPENWISH_REQ\"; then echo '{\"ok\":false,\"error\":\"bad request\"}'; else echo \"notes from pass $OPENWISH_PASS\" >> \"$OPENWISH_WORK/notes.md\"; echo '{\"ok\":true,\"games\":4}'; fi"]
passes = 3
"#;
fn turn_prompt(r: &Run, t: u32) -> String {
    std::fs::read_to_string(r.dir.join(format!("turns/t{t:03}/prompt.txt"))).unwrap()
}
fn script(v: &[&str]) -> Box<dyn Editor> {
    Box::new(Scripted::new(v.iter().map(|s| s.to_string()).collect()))
}

#[test]
fn toy_end_to_end_python_edit_then_final() {
    if !has_venv() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "[python]\nruns = 5\ntimeout_s = 60\nmem_gb = 4\n");
    let ed = script(&[
        "Let me edit.\n```python\nimport numpy as np\nsave_edit('W', np.eye(2), reason='identity')\n```",
        "Done: W is the identity.",
    ]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert!(r.state.done);
    assert_eq!(r.state.runs_used, 1);
    assert!(turn_prompt(&r, 1).contains("TASK: make W"));
    assert!(
        turn_prompt(&r, 2).contains("OUTPUT of Python run 1 of at most 5"),
        "{}",
        turn_prompt(&r, 2)
    );
    let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
    assert_eq!(fin.tensors["W"].data, vec![1., 0., 0., 1.]);
    assert_eq!(r.gate().unwrap().edits().count(), 1);
    assert!(r.dir.join("final_answer.md").exists());
    assert!(r.dir.join("final/editset/manifest.json").exists());
    assert_eq!(
        std::fs::read_to_string(r.dir.join("tool_calls.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(std::fs::read_to_string(r.dir.join("rules.txt"))
        .unwrap()
        .contains("TASK"));
}

const VALIDATION: &str = r#"
[finish.validation]
command = ["sh", "-c", "if grep -q 'answer' \"$OPENWISH_SUBMISSION\"; then echo '{\"ok\":true}'; else echo '{\"ok\":false,\"error\":\"Expected an answer object\"}'; fi"]
"#;

#[test]
fn validated_tasks_explain_text_tools_but_no_tool_baselines_stay_verbatim() {
    let d = tempfile::tempdir().unwrap();
    let baseline = task(d.path(), "");
    assert_eq!(first_message(&baseline).unwrap(), "TASK: make W the identity.");
    let interactive = task(d.path(), &format!("[python]\nruns = 2\n{VALIDATION}"));
    let message = first_message(&interactive).unwrap();
    assert!(message.contains("OpenWiSH executes your tool blocks"));
    assert!(message.contains("```python\nprint(NAMES)\n```"));
    assert!(message.contains("format only"));
}

#[test]
fn context_budget_uses_the_current_footprint_instead_of_billing_counters() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "");
    let mut r = Run::create(spec, &d.path().join("run"), script(&[]), env()).unwrap();
    let turn = r.dir.join("turns/t001");
    std::fs::create_dir_all(&turn).unwrap();
    let reply = Reply {
        usage: openwish_llm::Usage {
            input: 255717,
            cache_read: 191232,
            output: 5410,
            ..Default::default()
        },
        context_tokens: Some(66572),
        ..Default::default()
    };
    r.record_reply(&turn, &reply, true).unwrap();
    assert_eq!(r.state.conv_tokens, 66572);
}

#[test]
fn configured_static_policy_blocks_code_before_starting_python() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "[python]\nruns = 2\nreject_flags = [\"forward_pass\"]\n");
    let code = "```python\nopen('/work/executed', 'w').write('bad')\nh = np.maximum(X @ W.T + b, 0)\n```";
    let mut r = Run::create(spec, &d.path().join("run"), script(&[code]), env()).unwrap();
    r.step().unwrap();
    assert!(r.kernel.is_none());
    assert_eq!(r.state.runs_used, 0);
    assert_eq!(r.state.failed_requests, 1);
    assert!(!r.dir.join("work/executed").exists());
    assert!(r.state.next_message.contains("forward_pass"));
    let calls = std::fs::read_to_string(r.dir.join("tool_calls.jsonl")).unwrap();
    assert!(calls.contains("python-policy") && calls.contains("blocked"));
}

#[test]
fn malformed_final_answer_gets_format_feedback_without_scoring() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), VALIDATION);
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["No tools available.", "{\"answer\":1}"]),
        env(),
    )
    .unwrap();
    r.step().unwrap();
    assert!(!r.state.done);
    assert!(r.state.next_message.contains("Expected an answer object"));
    assert_eq!((r.state.runs_used, r.state.nudges), (0, 1));
    r.step().unwrap();
    assert!(r.state.done);
    assert_eq!(
        std::fs::read_to_string(r.dir.join("final_answer.md")).unwrap(),
        "{\"answer\":1}"
    );
    assert!(!r.dir.join("eval/score.json").exists());
}

#[test]
fn submission_validation_retries_are_bounded() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), &format!("[finish]\nmax_nudges = 1\n{VALIDATION}"));
    let mut r = Run::create(spec, &d.path().join("run"), script(&["bad", "still bad"]), env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.turn, 2);
    assert_eq!(r.state.end_reason.as_deref(), Some("submission rejected"));
}

#[test]
fn submission_file_is_checked_under_work_before_the_validator_reads_it() {
    use std::os::unix::fs::symlink;
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        &format!("{VALIDATION}\nsubmission_file = \"submission.json\"\n"),
    );
    let mut r = Run::create(spec, &d.path().join("run"), script(&["done", "done"]), env()).unwrap();
    std::fs::write(r.dir.join("work/submission.json"), "{\"answer\":1}").unwrap();
    r.step().unwrap();
    assert!(r.state.done);
    assert_eq!(
        std::fs::read_to_string(r.dir.join("validated_submission.json")).unwrap(),
        "{\"answer\":1}"
    );

    let spec = r.spec.clone();
    let mut r = Run::create(spec, &d.path().join("hostile"), script(&["done"]), env()).unwrap();
    let outside = d.path().join("outside.json");
    std::fs::write(&outside, "{\"answer\":1}").unwrap();
    symlink(outside, r.dir.join("work/submission.json")).unwrap();
    r.step().unwrap();
    assert!(!r.state.done);
    assert!(
        r.state.next_message.contains("symbolic link"),
        "{}",
        r.state.next_message
    );
}

#[test]
fn submission_validation_bounds_host_file_reads() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        &format!("{VALIDATION}\nsubmission_file = \"submission.json\"\n"),
    );
    let mut r = Run::create(spec, &d.path().join("run"), script(&["done"]), env()).unwrap();
    let file = std::fs::File::create(r.dir.join("work/submission.json")).unwrap();
    file.set_len(2_000_001).unwrap();
    r.step().unwrap();
    assert!(!r.state.done);
    assert!(
        r.state.next_message.contains("2000000 bytes"),
        "{}",
        r.state.next_message
    );
}

#[test]
fn fresh_python_runs_do_not_keep_variables() {
    if !has_venv() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "[python]\nruns = 3\nfresh = true\nprelude = \"pre.py\"\n");
    let ed = script(&[
        "```python\nx = 1\n```",
        "```python\nprint(PRE, 'x' in globals())\n```",
        "done",
    ]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert!(turn_prompt(&r, 3).contains("7 False"), "{}", turn_prompt(&r, 3));
}

#[test]
fn tools_can_share_one_pass_budget_and_name_their_folders() {
    let d = tempfile::tempdir().unwrap();
    let shared = r#"
[forward.games]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 2
pool = "passes"
out_dir = "pass_{k}"
[forward.positions]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 2
pool = "passes"
out_dir = "pass_{k}"
"#;
    let spec = task(d.path(), shared);
    let ed = script(&[
        "```games\n{}\n```",
        "```positions\n{}\n```",
        "```games\n{}\n```",
        "done",
    ]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["passes"], 2);
    assert!(r.dir.join("work/pass_1").is_dir() && r.dir.join("work/pass_2").is_dir());
    assert!(turn_prompt(&r, 3).contains("passes used 2/2"), "{}", turn_prompt(&r, 3));
    assert!(turn_prompt(&r, 4).contains("no passes left"), "{}", turn_prompt(&r, 4));
}

#[test]
fn a_tool_can_draw_from_the_python_run_budget() {
    if !has_venv() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let extra = r#"
[python]
runs = 3
[forward.attention]
command = ["sh", "-c", "echo '{\"ok\":true}'"]
passes = 0
pool = "python"
out_dir = "attn/r{k}"
"#;
    let spec = task(d.path(), extra);
    let ed = script(&[
        "```python\nprint(1)\n```",
        "```attention\n[\"x\"]\n```",
        "```python\nprint(2)\n```",
        "```attention\n[]\n```",
        "done",
    ]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.runs_used, 3);
    assert!(r.dir.join("work/attn/r2").is_dir());
    assert!(
        turn_prompt(&r, 3).contains("python runs used 2/3"),
        "{}",
        turn_prompt(&r, 3)
    );
    assert!(
        turn_prompt(&r, 5).contains("no Python runs left"),
        "{}",
        turn_prompt(&r, 5)
    );
}

#[test]
fn a_relative_run_folder_works() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    // the same folder, written relative to the current directory
    let cwd = std::env::current_dir().unwrap();
    let up = "../".repeat(cwd.components().count() - 1);
    let rel = PathBuf::from(format!(
        "{up}{}",
        d.path().join("run").strip_prefix("/").unwrap().display()
    ));
    assert!(rel.is_relative());
    let mut r = Run::create(spec, &rel, script(&["```games\n{}\n```", "done"]), env()).unwrap();
    assert!(r.dir.is_absolute());
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["games"], 1, "{}", turn_prompt(&r, 2));
}

#[test]
fn a_refusal_on_the_first_turn_keeps_the_task() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "");
    let ed = Box::new(Flaky {
        inner: Scripted::new(vec!["final".into()]),
        fail: vec![BackendError::Refused("flagged".into())],
    });
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    let p2 = turn_prompt(&r, 2);
    assert!(
        p2.contains("TASK: make W") && p2.contains("could not be delivered"),
        "{p2}"
    );
}

struct NoResume(Scripted);
impl Editor for NoResume {
    fn name(&self) -> String {
        "no-resume".into()
    }
    fn send(&mut self, m: &str, n: bool) -> std::result::Result<Reply, BackendError> {
        self.0.send(m, n)
    }
    fn session(&self) -> Option<String> {
        self.0.session()
    }
    fn set_session(&mut self, _s: Option<String>) {
        // a backend without sessions has nothing to restore
    }
    fn can_resume(&self) -> bool {
        false
    }
}

#[test]
fn resuming_a_backend_without_sessions_starts_a_handoff() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["```games\n{}\n```", "final"]),
        env(),
    )
    .unwrap();
    r.step().unwrap();
    drop(r);
    let ed = Box::new(NoResume(Scripted::new(vec!["final".into()])));
    let mut r = Run::resume(&d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    let p2 = turn_prompt(&r, 2);
    assert!(
        p2.contains("TASK: make W") && p2.contains("HANDOFF") && p2.contains("Forward pass 1 of 3"),
        "{p2}"
    );
}

#[test]
fn a_crash_during_a_python_run_does_not_repeat_its_edits() {
    if !has_venv() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), "[python]\nruns = 5\n");
    let code = "```python\nsave_edit('W', load('W') + 1)\nprint('did it')\n```";
    let mut r = Run::create(spec, &d.path().join("run"), script(&[code, "final"]), env()).unwrap();
    // as if the harness stopped while this run was going: the reply was received, its edit was saved,
    // but the run's output was never recorded
    r.state.turn = 1;
    r.state.pending_reply = Some(code.into());
    r.state.ledger_mark = Some(0);
    r.save_state().unwrap();
    r.gate_mut()
        .unwrap()
        .replace(
            "W",
            Tensor::new(vec![2, 2], vec![1.; 4]),
            "the interrupted run",
            "python (turn 1)",
        )
        .unwrap();
    drop(r);
    let mut r = Run::resume(&d.path().join("run"), script(&["final"]), env()).unwrap();
    r.run_to_end().unwrap();
    let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
    assert_eq!(fin.tensors["W"].data, vec![1.; 4], "the edit is applied exactly once");
    assert_eq!(r.state.runs_used, 1);
    let p2 = turn_prompt(&r, 2);
    assert!(
        p2.contains("output") && p2.contains("lost") && p2.contains("#1"),
        "{p2}"
    );
}

#[test]
fn budget_charged_only_on_success() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    let ed = script(&[
        "```games\nbad\n```",
        "```games\n{\"n\": 4}\n```",
        "```python\nx\n```",
        "final answer",
    ]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["games"], 1);
    assert_eq!(r.state.failed_requests, 1);
    assert_eq!(r.state.protocol_errors, 1);
    assert!(turn_prompt(&r, 2).contains("not counted"));
    assert!(
        turn_prompt(&r, 3).contains("Forward pass 1 of 3"),
        "{}",
        turn_prompt(&r, 3)
    );
    assert!(turn_prompt(&r, 4).contains("not available"));
    assert!(r.dir.join("work/games_1").is_dir());
    assert!(r.dir.join("requests/games_1.json").exists());
}

#[test]
fn nudge_when_finishing_with_passes_left() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        &format!("{FWD}\n[finish]\nrequire_all_passes = true\nmax_nudges = 1\n"),
    );
    let ed = script(&["I am done early.", "```games\n{}\n```", "done again", "really done"]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.nudges, 1);
    assert!(
        turn_prompt(&r, 2).contains("3 games passes left"),
        "{}",
        turn_prompt(&r, 2)
    );
    // only one nudge allowed, so the third reply ends the run
    assert_eq!(r.state.turn, 3);
    assert!(r.state.done);
}

#[test]
fn handoff_restarts_conversation_with_notes_ledger_and_budget() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), &format!("{FWD}\n[handoff]\nmax_turns_per_conversation = 2\n"));
    let ed = script(&["```games\n{}\n```", "```games\n{}\n```", "```games\n{}\n```", "final"]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    let p3 = turn_prompt(&r, 3);
    assert!(p3.contains("TASK: make W") && p3.contains("HANDOFF"), "{p3}");
    assert!(p3.contains("notes from pass 2"), "{p3}");
    assert!(p3.contains("(no edits yet)"), "{p3}");
    assert!(p3.contains("games passes used 2/3"), "{p3}");
    assert!(p3.contains("Forward pass 2 of 3"), "{p3}");
    assert_eq!(r.state.handoffs, 1);
}

struct Flaky {
    inner: Scripted,
    fail: Vec<BackendError>,
}
impl Editor for Flaky {
    fn name(&self) -> String {
        "flaky".into()
    }
    fn send(&mut self, m: &str, n: bool) -> std::result::Result<Reply, BackendError> {
        if !self.fail.is_empty() {
            return Err(self.fail.remove(0));
        }
        self.inner.send(m, n)
    }
    fn session(&self) -> Option<String> {
        self.inner.session()
    }
    fn set_session(&mut self, s: Option<String>) {
        self.inner.set_session(s)
    }
}

#[test]
fn backend_error_retry() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    let ed = Box::new(Flaky {
        inner: Scripted::new(vec!["```games\n{}\n```".into(), "final".into()]),
        fail: vec![
            BackendError::Transient("529".into()),
            BackendError::Transient("529".into()),
            BackendError::Refused("flagged".into()),
        ],
    });
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    assert!(r.state.done);
    assert_eq!(r.state.backend_errors, 2);
    assert_eq!(r.state.refusals, 1);
    assert!(
        turn_prompt(&r, 2).contains("could not be delivered"),
        "{}",
        turn_prompt(&r, 2)
    );
    // a fatal error stops the run with an error, and the state is saved
    let d2 = tempfile::tempdir().unwrap();
    let spec = task(d2.path(), FWD);
    let ed = Box::new(Flaky {
        inner: Scripted::new(vec![]),
        fail: vec![BackendError::Fatal("bad model".into())],
    });
    let mut r = Run::create(spec, &d2.path().join("run"), ed, env()).unwrap();
    assert!(r.run_to_end().is_err());
    assert!(d2.path().join("run/state.json").exists());
}

#[test]
fn resume_does_not_double_charge() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    let lines = ["```games\n{}\n```", "```games\n{}\n```", "final"];
    let mut r = Run::create(spec, &d.path().join("run"), script(&lines), env()).unwrap();
    r.step().unwrap();
    r.step().unwrap();
    assert_eq!(r.state.passes_used["games"], 2);
    drop(r);
    let mut r = Run::resume(&d.path().join("run"), script(&lines), env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["games"], 2);
    assert_eq!(r.state.turn, 3);
    assert!(r.state.done);
    // a reply received but not acted on before a crash is acted on exactly once
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), FWD);
    let mut r = Run::create(spec, &d.path().join("run"), script(&lines), env()).unwrap();
    r.state.pending_reply = Some("```games\n{}\n```".into());
    r.state.turn = 1;
    r.save_state().unwrap();
    drop(r);
    let mut r = Run::resume(&d.path().join("run"), script(&lines[1..]), env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["games"], 2);
}

#[test]
fn final_edits_block_is_parsed_and_gated() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        "[edits]\nlanguage = true\n[edits.policy]\nmax_frac_changed = 0.5\n",
    );
    let ed =
        script(&["My design.\n```edits\nW[0, 0] = 2\nW[1, 1] = 3\nnot a statement\nW[0, 1] = 5\n```\nexplanation"]);
    let mut r = Run::create(spec, &d.path().join("run"), ed, env()).unwrap();
    r.run_to_end().unwrap();
    let fin = Network::load_dir(&r.dir.join("final/net")).unwrap();
    // the third edit would change 3 of 4 entries (> 50%), so it is refused
    assert_eq!(fin.tensors["W"].data, vec![2., 0., 0., 3.]);
    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(r.dir.join("final/edits_report.json")).unwrap()).unwrap();
    assert_eq!(rep["applied"], 2);
    assert_eq!(rep["skipped"].as_array().unwrap().len(), 1);
    assert_eq!(rep["refused"].as_array().unwrap().len(), 1);
}

fn read(p: PathBuf) -> String {
    std::fs::read_to_string(&p).unwrap_or_else(|_| format!("({} is missing)", p.display()))
}

#[test]
fn a_failing_pre_tool_hook_blocks_a_python_run_without_charging_it() {
    // the hook blocks before the kernel starts, so this needs no Python environment
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        "[python]\nruns = 2\n[hooks]\npre_tool = \"echo python is closed for maintenance >&2; exit 1\"\n",
    );
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["```python\nprint(1)\n```", "done"]),
        env(),
    )
    .unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.runs_used, 0);
    assert_eq!(r.state.hook_blocks, 1);
    let p2 = turn_prompt(&r, 2);
    assert!(
        p2.contains("python is closed for maintenance")
            && p2.contains("not counted")
            && p2.contains("python runs used 0/2"),
        "{p2}"
    );
    let log = read(r.dir.join("hooks.jsonl"));
    assert!(
        log.contains("\"event\":\"pre_tool\"") && log.contains("\"ok\":false"),
        "{log}"
    );
    let calls = read(r.dir.join("tool_calls.jsonl"));
    assert!(calls.contains("\"blocked\":true"), "{calls}");
}

#[test]
fn a_failing_pre_tool_hook_blocks_a_probe_without_running_it() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(
        d.path(),
        &format!(
            "{FWD}\n[hooks]\npre_tool = 'if [ \"$OPENWISH_TOOL\" = games ]; then echo no games >&2; exit 1; fi'\n"
        ),
    );
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["```games\n{}\n```", "done"]),
        env(),
    )
    .unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used.get("games").copied().unwrap_or(0), 0);
    assert!(!r.dir.join("work/notes.md").exists(), "the probe command must not run");
    assert!(turn_prompt(&r, 2).contains("no games"), "{}", turn_prompt(&r, 2));
}

#[test]
fn post_tool_runs_after_a_forward_probe() {
    let d = tempfile::tempdir().unwrap();
    let hook = r#"post_tool = 'test -d "$OPENWISH_RUN_DIR/work/games_1" && echo "$OPENWISH_EVENT $OPENWISH_TOOL $OPENWISH_TURN ok=$OPENWISH_TOOL_OK" >> "$OPENWISH_RUN_DIR/seen.txt"'"#;
    let spec = task(d.path(), &format!("{FWD}\n[hooks]\n{hook}\n"));
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["```games\n{}\n```", "done"]),
        env(),
    )
    .unwrap();
    r.run_to_end().unwrap();
    assert_eq!(r.state.passes_used["games"], 1);
    assert_eq!(read(r.dir.join("seen.txt")), "post_tool games 1 ok=1\n");
}

#[test]
fn on_edit_hooks_receive_the_edit_id_from_a_python_run() {
    if !has_venv() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let hook = r#"on_edit = 'echo "$OPENWISH_EDIT_ID $OPENWISH_EDIT_TENSOR $OPENWISH_TOOL $OPENWISH_TURN" >> "$OPENWISH_RUN_DIR/edits_seen.txt"'"#;
    let spec = task(d.path(), &format!("[python]\nruns = 2\n[hooks]\n{hook}\n"));
    let code = "```python\nimport numpy as np\nsave_edit('W', np.eye(2), reason='identity')\nsave_edit('W', 2 * np.eye(2), reason='twice')\n```";
    let mut r = Run::create(spec, &d.path().join("run"), script(&[code, "done"]), env()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(read(r.dir.join("edits_seen.txt")), "1 W python 1\n2 W python 1\n");
}

#[test]
fn on_edit_hooks_run_for_the_final_edits_block() {
    let d = tempfile::tempdir().unwrap();
    let hook = r#"on_edit = 'echo "$OPENWISH_EDIT_ID $OPENWISH_EDIT_TENSOR $OPENWISH_TOOL" >> "$OPENWISH_RUN_DIR/edits_seen.txt"'"#;
    let spec = task(d.path(), &format!("[edits]\nlanguage = true\n[hooks]\n{hook}\n"));
    let mut r = Run::create(
        spec,
        &d.path().join("run"),
        script(&["```edits\nW[0, 0] = 2\nW[1, 1] = 3\n```"]),
        env(),
    )
    .unwrap();
    r.run_to_end().unwrap();
    assert_eq!(read(r.dir.join("edits_seen.txt")), "1 W edits\n2 W edits\n");
}

#[test]
fn session_hooks_run_once_each_with_project_hooks_first() {
    let d = tempfile::tempdir().unwrap();
    let spec = task(d.path(), &format!("{FWD}\n[hooks]\nsession_start = 'echo task-start >> \"$OPENWISH_RUN_DIR/s.txt\"'\nsession_end = 'echo \"end $OPENWISH_END_REASON\" >> \"$OPENWISH_RUN_DIR/s.txt\"'\n"));
    let project: crate::hooks::HooksSpec =
        toml::from_str("session_start = 'echo \"project-start $(basename \"$PWD\")\" >> \"$OPENWISH_RUN_DIR/s.txt\"'")
            .unwrap();
    let proj_dir = d.path().join("proj");
    std::fs::create_dir_all(&proj_dir).unwrap();
    let mut e = env();
    e.hooks = crate::hooks::HookSet::from_spec(&project, &proj_dir, "openwish.toml");
    let lines = ["```games\n{}\n```", "final"];
    let mut r = Run::create(spec, &d.path().join("run"), script(&lines), e.clone()).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(
        read(r.dir.join("s.txt")),
        "project-start proj\ntask-start\nend final answer\n"
    );
    drop(r);
    // resuming a finished run starts no new session
    let mut r = Run::resume(&d.path().join("run"), script(&lines), e).unwrap();
    r.run_to_end().unwrap();
    assert_eq!(read(r.dir.join("s.txt")).lines().count(), 3);
}
