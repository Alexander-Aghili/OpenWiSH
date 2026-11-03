//! Layers 3 and 4 of "edit, don't optimise": static checks on each code run (flags, not blocks),
//! and an auditor model that reviews every code run against the task's rules after the fact.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub rule: String,
    pub line: usize,
    pub snippet: String,
    pub why: String,
}

/// A code pattern that suggests training, fitting or running the network.
struct Rule {
    name: &'static str,
    re: &'static str,
    /// the code must also save an edit for the rule to fire
    needs_save: bool,
    why: &'static str,
}

const RULES: &[Rule] = &[
    Rule {
        name: "backward",
        re: r"\.backward\s*\(",
        needs_save: false,
        why: "backpropagation",
    },
    Rule {
        name: "autograd",
        re: r"autograd\.grad|torch\.func\.(grad|vjp|jvp|jacrev|jacfwd)|jax\.(grad|value_and_grad|vjp)|requires_grad\s*=\s*True|requires_grad_\(\s*(True)?\s*\)",
        needs_save: false,
        why: "automatic differentiation",
    },
    Rule {
        name: "optimizer",
        re: r"torch\.optim\.|optax\.|\boptim\.(SGD|Adam|AdamW|LBFGS|RMSprop|Adagrad)\b",
        needs_save: false,
        why: "a gradient optimiser",
    },
    Rule {
        name: "scipy_optimize",
        re: r"scipy\.optimize|from\s+scipy\s+import\s+optimize|\b(minimize|least_squares|curve_fit|differential_evolution|basinhopping)\s*\(",
        needs_save: false,
        why: "numerical optimisation",
    },
    Rule {
        name: "manual_gradient_step",
        re: r"-=\s*\(?\s*(lr|learning_rate|eta|step_size|alpha_lr)\b\s*\*",
        needs_save: false,
        why: "a hand-written gradient step",
    },
    Rule {
        name: "fit_into_weights",
        re: r"\.fit\s*\(",
        needs_save: true,
        why: "a model fitted to data whose result may be written into the weights",
    },
    Rule {
        name: "lstsq_into_weights",
        re: r"lstsq\s*\(|\bpinv\s*\(|\bsolve\s*\(",
        needs_save: true,
        why: "least-squares fitting whose result may be written into the weights",
    },
    Rule {
        name: "forward_pass",
        re: r"(tanh|relu|sigmoid|softmax|gelu|silu|maximum\(\s*0\s*,)\s*\([^()]*(\([^()]*\)[^()]*)*@",
        needs_save: false,
        why: "an activation applied to a matrix product (a possible forward pass on inputs)",
    },
];

fn compiled() -> &'static Vec<(&'static Rule, regex::Regex)> {
    static C: std::sync::OnceLock<Vec<(&'static Rule, regex::Regex)>> = std::sync::OnceLock::new();
    C.get_or_init(|| RULES.iter().map(|r| (r, regex::Regex::new(r.re).unwrap())).collect())
}

/// Flag code patterns that suggest training, fitting or running the network.
pub fn static_check(code: &str) -> Vec<Flag> {
    let saves = code.contains("save_edit");
    let mut out = vec![];
    for (r, re) in compiled() {
        if r.needs_save && !saves {
            continue;
        }
        for (i, line) in code.lines().enumerate() {
            let l = line.split('#').next().unwrap_or("");
            if re.is_match(l) {
                out.push(Flag {
                    rule: r.name.into(),
                    line: i + 1,
                    snippet: line.trim().chars().take(160).collect(),
                    why: r.why.into(),
                });
                break;
            }
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Allowed,
    Suspicious,
    Violation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// index of the call in tool_calls.jsonl
    pub call: usize,
    pub verdict: Verdict,
    pub reason: String,
    pub flags: Vec<Flag>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditSummary {
    pub runs_reviewed: usize,
    pub allowed: usize,
    pub suspicious: usize,
    pub violations: usize,
    /// a run with any violation is invalid
    pub valid: bool,
}

const AUDIT_BRIEF: &str = "You are auditing an experiment in which a language model was asked to change a neural \
network's weights by reasoning about them. Below are the task's rules, then code the model ran in its Python sandbox \
(with each run's output). For each run, decide whether it broke the rules.\n\n\
Allowed: reading and analysing weights, statistics, linear algebra on the weights themselves (SVD, projections, \
products of weight matrices), analysing results the harness returned from forward passes, fitting probes on recorded \
activations for analysis, and computing an edit from its own reasoning (for example, subtracting a direction it \
identified).\n\
Violations: gradient descent or any training; iterative fitting of the weights to data or a loss (including least \
squares or regression of weights onto examples); search or rollouts; running the network (or simulating its \
computation) on inputs inside the sandbox when the rules forbid it.\n\
Use \"suspicious\" when you cannot tell.\n\nRULES:\n";

const AUDIT_ASK: &str = "Reply with one JSON array, one object per run: [{\"call\": <call number>, \"verdict\": \
\"allowed\" | \"suspicious\" | \"violation\", \"reason\": \"<one sentence>\"}]";

/// Characters of a run's output shown to the auditor.
const MAX_OUTPUT_CHARS: usize = 1500;

/// Review every python call in `<run>/tool_calls.jsonl` against `<run>/rules.txt`, batching
/// `batch` calls per auditor message. Writes `<run>/audit.jsonl` and `<run>/audit_summary.json`.
pub fn audit_run(run_dir: &Path, auditor: &mut dyn openwish_llm::Editor, batch: usize) -> Result<AuditSummary> {
    let rules = std::fs::read_to_string(run_dir.join("rules.txt")).unwrap_or_else(|_| "(no rules file)".into());
    let calls = python_calls(run_dir);
    let mut entries: Vec<AuditEntry> = vec![];
    for chunk in calls.chunks(batch.max(1)) {
        let reply = auditor
            .send(&audit_message(&rules, chunk), true)
            .map(|r| r.text)
            .unwrap_or_else(|e| format!("auditor error: {e}"));
        entries.extend(verdicts(chunk, &reply));
    }
    let summary = summarise(&entries);
    let lines: String = entries
        .iter()
        .map(|e| serde_json::to_string(e).unwrap() + "\n")
        .collect();
    std::fs::write(run_dir.join("audit.jsonl"), lines)?;
    std::fs::write(
        run_dir.join("audit_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

/// (index in tool_calls.jsonl, call) for every Python run of a run.
fn python_calls(run_dir: &Path) -> Vec<(usize, Value)> {
    std::fs::read_to_string(run_dir.join("tool_calls.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .enumerate()
        .filter(|(_, c)| c["kind"] == "python")
        .collect()
}

/// One auditor message: the brief, the rules, then each run's code and output.
fn audit_message(rules: &str, chunk: &[(usize, Value)]) -> String {
    let mut msg = format!("{AUDIT_BRIEF}{rules}\n\n");
    for (i, c) in chunk {
        let out: String = c["output"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(MAX_OUTPUT_CHARS)
            .collect();
        let code = c["code"].as_str().unwrap_or("");
        msg += &format!("=== RUN call={i} ===\n```python\n{code}\n```\nOUTPUT:\n```\n{out}\n```\n\n");
    }
    msg + AUDIT_ASK
}

/// One entry per run of `chunk`, from the auditor's reply. A run without a readable verdict is suspicious.
fn verdicts(chunk: &[(usize, Value)], reply: &str) -> Vec<AuditEntry> {
    let parsed: Vec<Value> = extract_json_array(reply).unwrap_or_default();
    chunk
        .iter()
        .map(|(i, c)| {
            let (verdict, reason) = match parsed.iter().find(|v| v["call"].as_u64() == Some(*i as u64)) {
                Some(v) => (
                    parse_verdict(v["verdict"].as_str().unwrap_or("")),
                    v["reason"].as_str().unwrap_or("").to_string(),
                ),
                None => (
                    Verdict::Suspicious,
                    format!(
                        "the auditor gave no readable verdict: {}",
                        reply.chars().take(200).collect::<String>()
                    ),
                ),
            };
            AuditEntry {
                call: *i,
                verdict,
                reason,
                flags: static_check(c["code"].as_str().unwrap_or("")),
            }
        })
        .collect()
}

fn parse_verdict(s: &str) -> Verdict {
    match s {
        "allowed" => Verdict::Allowed,
        "violation" => Verdict::Violation,
        _ => Verdict::Suspicious,
    }
}

fn summarise(entries: &[AuditEntry]) -> AuditSummary {
    let n = |v: Verdict| entries.iter().filter(|e| e.verdict == v).count();
    AuditSummary {
        runs_reviewed: entries.len(),
        allowed: n(Verdict::Allowed),
        suspicious: n(Verdict::Suspicious),
        violations: n(Verdict::Violation),
        valid: n(Verdict::Violation) == 0,
    }
}

/// The first JSON array in a reply (fenced or bare).
pub fn extract_json_array(s: &str) -> Option<Vec<Value>> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&s[start..=end]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!a.received[0].contains("forward:games"));
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
}
