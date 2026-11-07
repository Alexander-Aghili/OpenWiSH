//! Budgets: what each tool may still use, and charging for it.
use super::*;

impl Run {
    pub fn budget_line(&self) -> String {
        let mut parts = vec![];
        if let Some(py) = &self.spec.python {
            parts.push(format!("python runs used {}/{}", self.state.runs_used, py.runs));
        }
        for (g, label, n) in self.spec.groups() {
            parts.push(format!("{label} used {}/{n}", self.passes_used(&g)));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("[budget: {}]", parts.join("; "))
        }
    }

    pub(super) fn passes_used(&self, group: &str) -> u32 {
        self.state.passes_used.get(group).copied().unwrap_or(0)
    }

    /// (label, passes left) for every pass budget that is not used up.
    pub(super) fn passes_left(&self) -> Vec<(String, u32)> {
        self.spec
            .groups()
            .into_iter()
            .map(|(g, label, n)| (label, n.saturating_sub(self.passes_used(&g))))
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    /// (used, allowed) for a tool of budget group `group`; the Python pool counts Python runs.
    pub(super) fn tool_budget(&self, group: &str, f: &ForwardSpec) -> (u32, u32) {
        if group == PYTHON_POOL {
            (
                self.state.runs_used,
                self.spec.python.as_ref().map(|p| p.runs).unwrap_or(0),
            )
        } else {
            (self.passes_used(group), f.passes)
        }
    }

    /// Record that use number `k` of `group` succeeded.
    pub(super) fn charge(&mut self, group: &str, k: u32) {
        if group == PYTHON_POOL {
            self.state.runs_used = k;
        } else {
            self.state.passes_used.insert(group.to_string(), k);
        }
    }
}
