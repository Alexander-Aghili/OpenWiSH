//! What an agent can use in a task, declared and derived, so that every result names its access level.
//!
//! Different access levels measure different things: reading weights as text, analysing them with code, running
//! the network, and seeing its behaviour are separate channels. A score means little without them, and scores
//! from different access levels must not be pooled (see docs/access-levels.md).
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::spec::TaskSpec;

pub const WEIGHTS: &[(&str, &str)] = &[
    ("prompt", "weights as text in the prompt"),
    ("code", "weights through code"),
    ("both", "weights as text in the prompt and through code"),
    ("zero", "from zero weights (nothing to read)"),
    ("none", "no weights"),
];

/// The part of a task's access that cannot be read off the rest of its spec: what the prompt and mounts contain.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessSpec {
    /// how the agent sees the weights: "prompt", "code", "both", "zero" or "none"
    pub weights: String,
    /// evidence of the network's behaviour already given in the prompt, e.g. "one sampled failure episode"
    #[serde(default)]
    pub shown: Vec<String>,
    /// data the agent can read besides the network, e.g. "practice problems with answers"
    #[serde(default)]
    pub data: Vec<String>,
}

impl AccessSpec {
    pub fn check(&self) -> Result<()> {
        if !WEIGHTS.iter().any(|(k, _)| *k == self.weights) {
            bail!("[access] weights must be one of {:?}, not {:?}", WEIGHTS.iter().map(|(k, _)| *k).collect::<Vec<_>>(), self.weights);
        }
        Ok(())
    }
}

impl TaskSpec {
    /// The task's access level: declared parts plus what the spec implies (code, network runs, replies).
    pub fn access(&self) -> serde_json::Value {
        let declared = self.access.is_some();
        let a = self.access.clone().unwrap_or_default();
        let weights = WEIGHTS.iter().find(|(k, _)| *k == a.weights).map(|(_, v)| v.to_string()).unwrap_or_else(|| "access undeclared".into());
        let code = self.python.as_ref().map(|p| format!("Python, {} runs{}", p.runs, if p.fresh { " (fresh each)" } else { "" }));
        let mut runs = vec![];
        for (group, _label, passes) in self.groups() {
            let tags: Vec<&str> = self.forward.iter().filter(|(t, f)| f.pool.clone().unwrap_or_else(|| t.to_string()) == group).map(|(t, _)| t.as_str()).collect();
            runs.push(format!("{} ({passes} passes)", tags.join(" + ")));
        }
        let pooled: Vec<&str> = self.forward.iter().filter(|(_, f)| f.pool.as_deref() == Some("python")).map(|(t, _)| t.as_str()).collect();
        if !pooled.is_empty() {
            runs.push(format!("{} (from the Python runs)", pooled.join(" + ")));
        }
        let replies = if self.finish.max_turns == 1 { "one reply".to_string() } else { format!("multi-turn (up to {})", self.finish.max_turns) };
        let mut parts = vec![weights.clone(), code.clone().unwrap_or_else(|| "no code".into())];
        parts.push(if runs.is_empty() { "network never run".into() } else { format!("network runs: {}", runs.join(", ")) });
        if !a.shown.is_empty() {
            parts.push(format!("shown: {}", a.shown.join("; ")));
        }
        if !a.data.is_empty() {
            parts.push(format!("data: {}", a.data.join("; ")));
        }
        parts.push(replies.clone());
        serde_json::json!({
            "declared": declared,
            "weights": if declared { a.weights.clone() } else { "undeclared".into() },
            "code": code,
            "network_runs": runs,
            "shown": a.shown,
            "data": a.data,
            "replies": replies,
            "label": parts.join(" · "),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    fn spec(toml: &str) -> crate::spec::TaskSpec {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("prompt.txt"), "hi").unwrap();
        let p = d.path().join("task.toml");
        std::fs::write(&p, toml).unwrap();
        let t = crate::spec::TaskSpec::load(&p).unwrap();
        std::mem::forget(d);
        t
    }

    #[test]
    fn labels_name_every_channel() {
        let t = spec("id = \"t\"\nprompt = \"prompt.txt\"\n[access]\nweights = \"prompt\"\nshown = [\"one sampled failure episode\"]\n[finish]\nmax_turns = 1\n");
        assert_eq!(t.access()["label"], "weights as text in the prompt · no code · network never run · shown: one sampled failure episode · one reply");
        let t = spec(
            "id = \"t\"\nprompt = \"prompt.txt\"\n[access]\nweights = \"code\"\ndata = [\"practice problems\"]\n[python]\nruns = 30\n[forward.forward]\ncommand = [\"x\"]\npasses = 8\n",
        );
        assert_eq!(t.access()["label"], "weights through code · Python, 30 runs · network runs: forward (8 passes) · data: practice problems · multi-turn (up to 400)");
        let t = spec("id = \"t\"\nprompt = \"prompt.txt\"\n");
        assert_eq!(t.access()["declared"], false);
    }

    #[test]
    fn bad_weights_value_is_refused() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("prompt.txt"), "hi").unwrap();
        let p = d.path().join("task.toml");
        std::fs::write(&p, "id = \"t\"\nprompt = \"prompt.txt\"\n[access]\nweights = \"telepathy\"\n").unwrap();
        assert!(crate::spec::TaskSpec::load(&p).is_err());
    }

    /// Every bundled environment declares its access (parsed from the TOML text; assets need not exist).
    #[test]
    fn every_bundled_task_declares_access() {
        let tasks = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tasks");
        let mut missing = vec![];
        let mut n = 0;
        let mut stack = vec![tasks.clone()];
        while let Some(dir) = stack.pop() {
            for f in std::fs::read_dir(&dir).unwrap().flatten() {
                let p = f.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().and_then(|e| e.to_str()) != Some("toml") {
                    continue;
                }
                n += 1;
                let v: toml::Value = toml::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                match v.get("access").and_then(|a| a.get("weights")).and_then(|w| w.as_str()) {
                    Some(w) if super::WEIGHTS.iter().any(|(k, _)| *k == w) => {}
                    _ => missing.push(p.display().to_string()),
                }
            }
        }
        assert!(n > 10, "found only {n} task specs under {}", tasks.display());
        assert!(missing.is_empty(), "tasks without a valid [access] section: {missing:#?}");
    }
}
