//! Layers 3 and 4 of "edit, don't optimise": static checks on each code run (flags, not blocks),
//! and an auditor model that reviews every code run against the task's rules after the fact.
mod review;
mod rules;

pub use review::{audit_run, extract_json_array, AuditEntry, AuditSummary, Verdict};
pub use rules::{known_rule, static_check, Flag};

#[cfg(test)]
mod tests;
