//! Layer 4: an auditor model reviews every code run of a finished run against the task's rules, in batches,
//! with the probe calls as context, and writes one verdict per run.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

use crate::rules::{static_check, Flag};

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

pub(crate) const AUDIT_BRIEF: &str = "You are auditing an experiment in which a language model was asked to change a neural \
network's weights by reasoning about them. Below are the task's rules, then code the model ran in its Python sandbox \
(with each run's output). For each run, decide whether it broke the rules.\n\n\
Allowed: reading and analysing weights, statistics, linear algebra on the weights themselves (SVD, projections, \
products of weight matrices), analysing results the harness returned from forward passes, fitting probes on recorded \
activations for analysis, and computing an edit from its own reasoning (for example, subtracting a direction it \
identified).\n\
Violations: gradient descent or any training; iterative fitting of the weights to data or a loss (including least \
squares or regression of weights onto examples); search that scores candidate weights or programs against the \
network's outputs, game rollouts or other behaviour; running the network (or simulating its computation) on inputs \
inside the sandbox when the rules forbid it.\n\
The task's rules below come first. Where they explicitly allow a method (for example fitting, clustering or an \
optimisation over statistics computed from the weights), the task's rules win, and that method is allowed. Where \
the task's rules forbid something, it is a violation even if the list above allows it. Where they are silent, use \
the lists above. A word in the task's rules means what the task says it means (for example \"search\" may mean \
looking something up).\n\
Use \"suspicious\" when you cannot tell. Steps of a research workflow the task asks for (an order of steps, notes, \
evidence formats) are not part of this audit: judge each run against the rules on training, fitting, search and \
running the network. The probe calls listed before the runs went through the harness and are context, not runs to judge.\n\nRULES:\n";

pub(crate) const AUDIT_ASK: &str =
    "Reply with one JSON array, one object per run: [{\"call\": <call number>, \"verdict\": \
\"allowed\" | \"suspicious\" | \"violation\", \"reason\": \"<one sentence>\"}]";

/// Characters of a run's output shown to the auditor.
pub(crate) const MAX_OUTPUT_CHARS: usize = 1500;

/// Review every python call in `<run>/tool_calls.jsonl` against `<run>/rules.txt`, batching
/// `batch` calls per auditor message. Writes `<run>/audit.jsonl` and `<run>/audit_summary.json`.
pub fn audit_run(run_dir: &Path, auditor: &mut dyn openwish_llm::Editor, batch: usize) -> Result<AuditSummary> {
    let prompt = std::fs::read_to_string(run_dir.join("rules.txt")).unwrap_or_else(|_| "(no rules file)".into());
    let spec_path = run_dir.join("spec.json");
    let instructions = if spec_path.exists() {
        let saved: Value = serde_json::from_str(&std::fs::read_to_string(spec_path)?)?;
        saved["spec"]["instructions"].as_str().unwrap_or("").to_string()
    } else {
        String::new()
    };
    let rules = format!("TASK INSTRUCTIONS:\n{instructions}\n\nTASK PROMPT:\n{prompt}");
    let calls = python_calls(run_dir);
    let context = probe_context(run_dir);
    let mut entries: Vec<AuditEntry> = vec![];
    for chunk in calls.chunks(batch.max(1)) {
        let reply = auditor
            .send(&audit_message(&format!("{rules}{context}"), chunk), true)
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
pub(crate) fn python_calls(run_dir: &Path) -> Vec<(usize, Value)> {
    std::fs::read_to_string(run_dir.join("tool_calls.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .enumerate()
        .filter(|(_, c)| c["kind"] == "python")
        .collect()
}

/// Every probe call of a run (pass number and request), for the auditor's context. Empty when there are none.
pub(crate) fn probe_context(run_dir: &Path) -> String {
    let lines: Vec<String> = std::fs::read_to_string(run_dir.join("tool_calls.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|c| c["kind"].as_str().is_some_and(|k| k.starts_with("forward:")))
        .map(|c| {
            let request = openwish_text::head(c["request"].as_str().unwrap_or(""), 400).to_string();
            format!(
                "r{} {} ok={}: {}",
                c["pass"]
                    .as_str()
                    .map(String::from)
                    .unwrap_or_else(|| c["pass"].to_string()),
                c["kind"].as_str().unwrap_or(""),
                c["ok"],
                request.trim()
            )
        })
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    format!(
        "\n\nPROBE CALLS (through the harness; context only):\n{}",
        lines.join("\n")
    )
}

/// One auditor message: the brief, the rules, then each run's code and output.
pub(crate) fn audit_message(rules: &str, chunk: &[(usize, Value)]) -> String {
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
pub(crate) fn verdicts(chunk: &[(usize, Value)], reply: &str) -> Vec<AuditEntry> {
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
                        openwish_text::head(reply, 200)
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

pub(crate) fn parse_verdict(s: &str) -> Verdict {
    match s {
        "allowed" => Verdict::Allowed,
        "violation" => Verdict::Violation,
        _ => Verdict::Suspicious,
    }
}

pub(crate) fn summarise(entries: &[AuditEntry]) -> AuditSummary {
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
