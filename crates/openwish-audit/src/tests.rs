use super::*;
use crate::review::*;

fn rules(code: &str) -> Vec<String> {
    static_check(code).into_iter().map(|f| f.rule).collect()
}

#[test]
fn flags_training_patterns() {
    assert_eq!(rules("loss.backward()"), vec!["backward"]);
    assert_eq!(rules("opt = torch.optim.Adam(params)"), vec!["optimizer"]);
    assert_eq!(
        rules("from scipy.optimize import minimize\nr = minimize(f, x0)"),
        vec!["scipy_optimize"]
    );
    assert_eq!(rules("g = jax.grad(f)"), vec!["autograd"]);
    assert_eq!(rules("W -= lr * g"), vec!["manual_gradient_step"]);
    assert_eq!(
        rules("clf.fit(X, y)\nsave_edit('W', clf.coef_)"),
        vec!["fit_into_weights"]
    );
    assert_eq!(
        rules("W, *_ = np.linalg.lstsq(H, Y)\nsave_edit('out.W', W)"),
        vec!["lstsq_into_weights"]
    );
    assert_eq!(rules("h = np.tanh(x @ W1 + b1)"), vec!["forward_pass"]);
    assert_eq!(rules("h = np.maximum(h @ w.T + b, 0)"), vec!["forward_pass"]);
    assert_eq!(
        rules("value = max(0, bias + sum(coeff * bit for coeff, bit in zip(row, inputs)))"),
        vec!["scalar_relu"]
    );
}

/// The task's own rules decide: a task that allows fitting or clustering over the weights must not be audited
/// against a blanket ban, and "search" in the defaults means search over the network's behaviour, not lookup.
#[test]
fn the_brief_defers_to_the_task_rules() {
    let msg = audit_message("TASK PROMPT:\nYou may fit or cluster the weights.", &[]);
    let task = msg.find("TASK PROMPT:").unwrap();
    let precedence = msg
        .find("the task's rules win")
        .expect("a precedence rule in the brief");
    assert!(precedence < task, "the precedence rule comes before the task's rules");
    assert!(!msg.contains("search or rollouts;"), "no blanket search ban");
    assert!(msg.contains("candidate weights or programs against the network's outputs"));
    // the core rule is still in force where the task is silent
    assert!(msg.contains("gradient descent or any training"));
    assert!(msg.contains("iterative fitting of the weights to data"));
}

#[test]
fn analysis_is_not_flagged() {
    assert!(rules("import numpy as np\nW = load('L1.W')\nu, s, vt = np.linalg.svd(W)\nprint(s[:5])").is_empty());
    assert!(rules("QK = Wq @ Wk.T\nprint(np.abs(QK).max())").is_empty());
    assert!(rules("coef = np.linalg.lstsq(A, b)[0]\nprint(coef)").is_empty());
    assert!(rules("W = load('W')\nW2 = W - 1.5 * np.outer(u, u @ W)\nsave_edit('W', W2)").is_empty());
    assert!(rules("probe = LogisticRegression().fit(acts, labels)\nprint(probe.score(acts, labels))").is_empty());
}

#[test]
fn audit_writes_entries_and_summary() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("rules.txt"), "No training.").unwrap();
    let calls = [
        serde_json::json!({"kind": "python", "code": "print(1)", "output": "1"}),
        serde_json::json!({"kind": "forward:games", "request": {}}),
        serde_json::json!({"kind": "python", "code": "loss.backward()", "output": "OpenWiSHForbidden"}),
    ];
    std::fs::write(
        d.path().join("tool_calls.jsonl"),
        calls.iter().map(|c| c.to_string() + "\n").collect::<String>(),
    )
    .unwrap();
    let reply = r#"Here you go:
```json
[{"call": 0, "verdict": "allowed", "reason": "prints"}, {"call": 2, "verdict": "violation", "reason": "tries backprop"}]
```"#;
    let mut a = openwish_llm::scripted::Scripted::new(vec![reply.into()]);
    let s = audit_run(d.path(), &mut a, 10).unwrap();
    assert_eq!(
        s,
        AuditSummary {
            runs_reviewed: 2,
            allowed: 1,
            suspicious: 0,
            violations: 1,
            valid: false
        }
    );
    let lines = std::fs::read_to_string(d.path().join("audit.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 2);
    assert!(a.received[0].contains("No training.") && a.received[0].contains("loss.backward()"));
}

/// Probe calls are context for the auditor (an answer may cite them), never runs it judges.
#[test]
fn auditor_sees_probe_calls_as_context_and_judges_only_python() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("rules.txt"), "Run the network only through the probe.").unwrap();
    let calls = [
        serde_json::json!({"kind": "forward:forward", "pass": 1, "request": "{\"ablate\": {\"L0_MLP\": \"zero\"}}", "ok": true}),
        serde_json::json!({"kind": "python", "code": "print(1)", "output": "1"}),
    ];
    std::fs::write(
        d.path().join("tool_calls.jsonl"),
        calls.iter().map(|c| c.to_string() + "\n").collect::<String>(),
    )
    .unwrap();
    let reply = r#"[{"call": 1, "verdict": "allowed", "reason": "prints"}]"#;
    let mut a = openwish_llm::scripted::Scripted::new(vec![reply.into()]);
    let s = audit_run(d.path(), &mut a, 10).unwrap();
    assert_eq!(s.runs_reviewed, 1);
    let msg = &a.received[0];
    assert!(
        msg.contains("PROBE CALLS") && msg.contains("r1") && msg.contains("L0_MLP"),
        "{msg}"
    );
    assert!(msg.contains("workflow"), "the brief scopes out workflow steps");
}

#[test]
fn auditor_receives_saved_system_instructions_as_well_as_the_prompt() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("rules.txt"), "Public task payload.").unwrap();
    std::fs::write(
        d.path().join("spec.json"),
        r#"{"spec":{"instructions":"No network execution outside declared probes."}}"#,
    )
    .unwrap();
    std::fs::write(
        d.path().join("tool_calls.jsonl"),
        "{\"kind\":\"python\",\"code\":\"print(1)\"}\n",
    )
    .unwrap();
    let mut a =
        openwish_llm::scripted::Scripted::new(vec![r#"[{"call":0,"verdict":"allowed","reason":"prints"}]"#.into()]);
    audit_run(d.path(), &mut a, 8).unwrap();
    assert!(a.received[0].contains("No network execution outside declared probes."));
    assert!(a.received[0].contains("Public task payload."));
}

#[test]
fn unreadable_auditor_reply_is_suspicious_not_allowed() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("rules.txt"), "r").unwrap();
    std::fs::write(
        d.path().join("tool_calls.jsonl"),
        serde_json::json!({"kind": "python", "code": "x=1", "output": ""}).to_string() + "\n",
    )
    .unwrap();
    let mut a = openwish_llm::scripted::Scripted::new(vec!["I refuse to answer in JSON".into()]);
    let s = audit_run(d.path(), &mut a, 10).unwrap();
    assert_eq!((s.suspicious, s.valid), (1, true));
}
