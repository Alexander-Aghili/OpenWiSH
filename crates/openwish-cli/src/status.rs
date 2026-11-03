//! `openwish status <run>`, and the status line `openwish run` prints on stderr after every turn.
use anyhow::Result;
use openwish_core::run::State;
use openwish_core::spec::TaskSpec;
use std::fmt::Write;
use std::path::Path;
use std::time::Duration;

/// The counters of `state.json` printed by `openwish status`, in order.
const COUNTERS: &[&str] = &[
    "turn",
    "runs_used",
    "passes_used",
    "handoffs",
    "nudges",
    "failed_requests",
    "protocol_errors",
    "refusals",
    "backend_errors",
    "kernel_restarts",
    "cost_usd",
    "usage_total",
];

/// A run's task, state and counters, then its score and audit summary if it has them.
pub fn report(run: &Path) -> Result<String> {
    let s: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("state.json"))?)?;
    let spec = openwish_core::run::saved_spec(run)?;
    let state = if s["done"] == true {
        format!("done ({})", s["end_reason"].as_str().unwrap_or(""))
    } else {
        "running / stopped".into()
    };
    let mut out = format!("task       {}\nstate      {state}\n", spec.id);
    for k in COUNTERS {
        writeln!(out, "{k:<16} {}", s[k])?;
    }
    if let Ok(sc) = std::fs::read_to_string(run.join("eval/score.json")) {
        writeln!(out, "score\n{sc}")?;
    }
    if let Ok(a) = std::fs::read_to_string(run.join("audit_summary.json")) {
        writeln!(out, "audit\n{a}")?;
    }
    Ok(out)
}

/// `turn 3/400 | python 1/5 | games 2/10 | edits 1 | $0.12 | 0:42`
pub fn line(spec: &TaskSpec, s: &State, edits: usize, elapsed: Duration) -> String {
    let mut parts = vec![format!("turn {}/{}", s.turn, spec.finish.max_turns)];
    if let Some(py) = &spec.python {
        parts.push(format!("python {}/{}", s.runs_used, py.runs));
    }
    for (g, _, n) in spec.groups() {
        parts.push(format!("{g} {}/{n}", s.passes_used.get(&g).copied().unwrap_or(0)));
    }
    if spec.network.is_some() {
        parts.push(format!("edits {edits}"));
    }
    if s.hook_blocks > 0 {
        parts.push(format!("blocked by hooks {}", s.hook_blocks));
    }
    parts.push(format!("${:.2}", s.cost_usd));
    parts.push(clock(elapsed));
    parts.join(" | ")
}

/// `m:ss` under an hour, `h:mm:ss` after.
pub fn clock(d: Duration) -> String {
    let t = d.as_secs();
    let (h, m, s) = (t / 3600, (t / 60) % 60, t % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_turn_budgets_edits_cost_and_time() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("p.txt"), "x").unwrap();
        let toml = "id = \"t\"\nprompt = \"p.txt\"\n[network]\nbase = \"net\"\n[python]\nruns = 5\n[forward.games]\ncommand = [\"true\"]\npasses = 10\n[finish]\nmax_turns = 40\n";
        std::fs::write(d.path().join("t.toml"), toml).unwrap();
        let spec = TaskSpec::parse(&d.path().join("t.toml")).unwrap();
        let mut s = State {
            turn: 3,
            runs_used: 1,
            cost_usd: 0.125,
            ..Default::default()
        };
        s.passes_used.insert("games".into(), 2);
        assert_eq!(
            line(&spec, &s, 4, Duration::from_secs(65)),
            "turn 3/40 | python 1/5 | games 2/10 | edits 4 | $0.12 | 1:05"
        );
        s.hook_blocks = 2;
        assert!(line(&spec, &s, 4, Duration::from_secs(3725)).ends_with("| blocked by hooks 2 | $0.12 | 1:02:05"));
    }
}
