//! The coding-harness commands: init, envs, tools, replay --html, openwish.toml and the status line.
use openwish_tensor::{Network, Tensor};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn has_venv() -> bool {
    root().join("assets/venvs/sandbox/bin/python").exists()
}
fn ow(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_openwish"))
        .current_dir(root())
        .env_remove("OPENWISH_CONFIG")
        .args(args)
        .output()
        .unwrap()
}
fn text(o: &Output) -> String {
    format!(
        "status {}\n--- stdout\n{}\n--- stderr\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}
fn p(x: &Path) -> &str {
    x.to_str().unwrap()
}

/// A task with no Python: a shell probe, then whatever final reply the script gives.
fn probe_task(d: &Path, final_reply: &str, extra: &str) -> (PathBuf, PathBuf) {
    let mut net = Network::default();
    net.tensors.insert("W".into(), Tensor::zeros(vec![2, 2]));
    net.save_dir(&d.join("net")).unwrap();
    std::fs::write(d.join("prompt.txt"), "TASK: look, then answer.").unwrap();
    let toml = format!(
        r#"id = "probe-task"
prompt = "prompt.txt"
[network]
base = "net"
[forward.games]
command = ["sh", "-c", "echo \"{{\\\"ok\\\":true,\\\"threads\\\":\\\"$OPENWISH_THREADS\\\"}}\""]
passes = 2
[edits]
language = true
[score]
command = ["sh", "-c", "echo '{{\"ok\":true,\"score\":0.5}}'"]
{extra}
"#
    );
    std::fs::write(d.join("task.toml"), toml).unwrap();
    std::fs::write(
        d.join("script.txt"),
        format!("I will look first.\n```games\n{{}}\n```\n=====\n{final_reply}\n"),
    )
    .unwrap();
    (d.join("task.toml"), d.join("script.txt"))
}

#[test]
fn init_env_scaffolds_an_environment_that_runs_and_scores() {
    let d = tempfile::tempdir().unwrap();
    let env = d.path().join("my-env");
    let o = ow(&["init", "env", p(&env)]);
    assert!(o.status.success(), "{}", text(&o));
    for f in [
        "task.toml",
        "prompt.txt",
        "prelude.py",
        "score.py",
        "probe.py",
        "script.txt",
        "answer.json",
        "net/W.npy",
        "net/b.npy",
    ] {
        assert!(env.join(f).exists(), "{f} missing\n{}", text(&o));
    }
    let toml = std::fs::read_to_string(env.join("task.toml")).unwrap();
    assert!(toml.contains("id = \"my-env\"") && toml.contains("\n# "), "{toml}");
    // a second init into a folder that has files is refused
    let o = ow(&["init", "env", p(&env)]);
    assert!(!o.status.success());
    if !has_venv() {
        return;
    }
    let run = d.path().join("run");
    let script = format!("scripted:{}", env.join("script.txt").display());
    let o = ow(&["run", p(&env.join("task.toml")), "--backend", &script, "--out", p(&run)]);
    assert!(o.status.success(), "{}", text(&o));
    let score: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("eval/score.json")).unwrap()).unwrap();
    assert_eq!(score["ok"], true, "{score}");
    assert_eq!(score["score"], 1.0, "{score}");
    // the probe ran and the edit went through the gate
    assert_eq!(
        std::fs::read_to_string(run.join("edits/ledger.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    let calls = std::fs::read_to_string(run.join("tool_calls.jsonl")).unwrap();
    assert!(
        calls.contains("forward:forward") && calls.contains("\"ok\":true"),
        "{calls}"
    );
    // one status line per turn on stderr
    let err = String::from_utf8_lossy(&o.stderr);
    for t in 1..=3 {
        assert!(err.contains(&format!("turn {t}/20")), "{err}");
    }
    assert!(
        err.contains("python 1/5") && err.contains("forward 1/3") && err.contains("edits 1") && err.contains('$'),
        "{err}"
    );
}

#[test]
fn init_config_writes_a_file_that_parses() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("openwish.toml");
    let o = ow(&["init", "config", p(&cfg)]);
    assert!(o.status.success(), "{}", text(&o));
    let s = std::fs::read_to_string(&cfg).unwrap();
    assert!(s.contains("[hooks]") && s.contains("# backend"), "{s}");
    // the written file is a valid config: running a probe task with it works
    let (task, script) = probe_task(&d.path().join("t"), "done", "");
    let o = ow(&[
        "--config",
        p(&cfg),
        "run",
        p(&task),
        "--backend",
        &format!("scripted:{}", script.display()),
        "--out",
        p(&d.path().join("run")),
    ]);
    assert!(o.status.success(), "{}", text(&o));
}

#[test]
fn envs_lists_environments_with_tools_budgets_and_assets() {
    let d = tempfile::tempdir().unwrap();
    probe_task(&d.path().join("good"), "done", "");
    std::fs::create_dir_all(d.path().join("broken")).unwrap();
    std::fs::write(
        d.path().join("broken/task.toml"),
        "id = \"no-assets\"\nprompt = \"gone.txt\"\n[network]\nbase = \"net\"\n[python]\nruns = 7\n",
    )
    .unwrap();
    std::fs::write(d.path().join("broken/bad.toml"), "this is not = = toml").unwrap();
    let o = ow(&["envs", "--dir", p(d.path())]);
    assert!(o.status.success(), "{}", text(&o));
    let out = String::from_utf8_lossy(&o.stdout);
    let lines: Vec<&str> = out.lines().collect();
    assert!(
        lines[0].contains("ID")
            && lines[0].contains("TOOLS")
            && lines[0].contains("BUDGETS")
            && lines[0].contains("ASSETS"),
        "{out}"
    );
    let good = lines.iter().find(|l| l.contains("probe-task")).expect("probe-task row");
    assert!(
        good.contains("games") && good.contains("games 2") && good.contains("ok"),
        "{good}"
    );
    let bad = lines.iter().find(|l| l.contains("no-assets")).expect("no-assets row");
    assert!(
        bad.contains("python 7") && bad.contains("missing") && bad.contains("prompt") && bad.contains("network"),
        "{bad}"
    );
    assert!(
        lines.iter().any(|l| l.contains("bad.toml") && l.contains("not a task")),
        "{out}"
    );
    // the bundled tasks folder lists too
    let o = ow(&["envs"]);
    assert!(
        o.status.success() && String::from_utf8_lossy(&o.stdout).contains("toy"),
        "{}",
        text(&o)
    );
}

#[test]
fn tools_prints_the_surface_the_agent_sees() {
    let d = tempfile::tempdir().unwrap();
    let extra = r#"
[python]
runs = 9
fresh = true
[forward.positions]
command = ["true"]
passes = 4
pool = "probes"
[forward.attention]
command = ["true"]
passes = 0
pool = "python"
[edits.policy]
forbidden = ["norm.*"]
max_frac_changed = 0.01
forbid_whole_tensor_scale = true
[hooks]
pre_tool = "check-call.sh"
"#;
    let (task, _) = probe_task(d.path(), "done", extra);
    let o = ow(&["tools", p(&task)]);
    assert!(o.status.success(), "{}", text(&o));
    let out = String::from_utf8_lossy(&o.stdout);
    for want in [
        "```python",
        "9 runs",
        "fresh",
        "```games",
        "2 passes",
        "```positions",
        "pool \"probes\"",
        "```attention",
        "Python runs",
        "norm.*",
        "0.01",
        "whole-tensor scaling refused",
        "```edits",
        "check-call.sh",
        "pre_tool",
    ] {
        assert!(out.contains(want), "{want:?} missing:\n{out}");
    }
}

#[test]
fn replay_html_shows_every_turn_and_escapes_agent_text() {
    let d = tempfile::tempdir().unwrap();
    let evil = "Here is my answer <script>alert('x')</script> & done.\n```edits\nW[0, 0] = 1\n```";
    let (task, script) = probe_task(&d.path().join("t"), evil, "");
    let run = d.path().join("run");
    let o = ow(&[
        "run",
        p(&task),
        "--backend",
        &format!("scripted:{}", script.display()),
        "--out",
        p(&run),
    ]);
    assert!(o.status.success(), "{}", text(&o));
    let html_path = d.path().join("out.html");
    let o = ow(&["replay", p(&run), "--html", p(&html_path)]);
    assert!(o.status.success(), "{}", text(&o));
    let html = std::fs::read_to_string(&html_path).unwrap();
    assert!(html.starts_with("<!DOCTYPE html>"));
    for t in [1, 2] {
        assert!(
            html.contains(&format!("id=\"turn-{t}\"")) && html.contains(&format!("Turn {t}")),
            "turn {t} missing"
        );
    }
    assert!(
        !html.to_lowercase().contains("<script"),
        "raw script tag in the transcript"
    );
    assert!(
        html.contains("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt; &amp; done."),
        "agent text not escaped"
    );
    // the tool call, the ledger, the score and the run's counters
    assert!(html.contains("forward:games") && html.contains("I will look first."));
    assert!(
        html.contains("Edits") && html.contains("#1") && html.contains("assign"),
        "ledger missing"
    );
    assert!(html.contains("Score") && html.contains("0.5"), "score missing");
    assert!(
        html.contains("Audit") && html.contains("not audited"),
        "audit section missing"
    );
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "the page must be self-contained"
    );
}

#[test]
fn openwish_toml_sets_defaults_that_flags_override() {
    let d = tempfile::tempdir().unwrap();
    let (task, script) = probe_task(&d.path().join("t"), "done", "");
    let cfg = d.path().join("openwish.toml");
    // the config picks the backend and the threads; a project hook logs every probe
    std::fs::write(
        &cfg,
        format!(
            "backend = \"scripted:{}\"\nthreads = 3\n[hooks]\npost_tool = 'echo \"$OPENWISH_TOOL from $(basename \"$PWD\")\" >> \"$OPENWISH_RUN_DIR/project_hook.txt\"'\n",
            script.display()
        ),
    )
    .unwrap();
    let run = d.path().join("run1");
    let o = ow(&["--config", p(&cfg), "run", p(&task), "--out", p(&run)]);
    assert!(o.status.success(), "{}", text(&o));
    let calls = std::fs::read_to_string(run.join("tool_calls.jsonl")).unwrap();
    assert!(calls.contains("\"threads\":\"3\""), "{calls}");
    let dir_name = d.path().file_name().unwrap().to_string_lossy().to_string();
    assert_eq!(
        std::fs::read_to_string(run.join("project_hook.txt")).unwrap(),
        format!("games from {dir_name}\n")
    );
    // flags beat the config
    let run = d.path().join("run2");
    let o = ow(&["--config", p(&cfg), "--threads", "2", "run", p(&task), "--out", p(&run)]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(std::fs::read_to_string(run.join("tool_calls.jsonl"))
        .unwrap()
        .contains("\"threads\":\"2\""));
    std::fs::write(&cfg, "backend = \"no-such-backend\"\n").unwrap();
    let o = ow(&["--config", p(&cfg), "run", p(&task), "--out", p(&d.path().join("run3"))]);
    assert!(!o.status.success(), "{}", text(&o));
    let o = ow(&[
        "--config",
        p(&cfg),
        "run",
        p(&task),
        "--backend",
        &format!("scripted:{}", script.display()),
        "--out",
        p(&d.path().join("run4")),
    ]);
    assert!(o.status.success(), "{}", text(&o));
    // an unknown key in the config is an error
    std::fs::write(&cfg, "bakend = \"claude\"\n").unwrap();
    let o = ow(&["--config", p(&cfg), "envs"]);
    assert!(
        !o.status.success() && String::from_utf8_lossy(&o.stderr).contains("bakend"),
        "{}",
        text(&o)
    );
}
