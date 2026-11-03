//! The edit gate: every weight change passes through here, is checked against the task's policy,
//! and is appended to the ledger. The materialised current network lives in `<dir>/current/`.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::network::{name_to_file, save_npy, Network, Tensor};
use crate::npy::write_npy;
use crate::ops::{compute, EditOp};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EditPolicy {
    /// If set, only tensors matching one of these globs ('*' wildcard) may change.
    pub allowed: Option<Vec<String>>,
    /// Tensors matching these globs may never change.
    pub forbidden: Vec<String>,
    /// Refuse an edit whose result is the whole old tensor times one constant.
    pub forbid_whole_tensor_scale: bool,
    /// Maximum fraction of all network entries that may differ from the base.
    pub max_frac_changed: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditRecord {
    pub id: u64,
    pub ts: String,
    pub op: EditOp,
    pub reason: String,
    pub source: String,
    /// entries of the tensor that this edit changed
    pub changed: usize,
    /// entries of the whole network that now differ from the base
    pub changed_vs_base: usize,
    pub frac_changed_vs_base: f64,
    pub sha_before: String,
    pub sha_after: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LedgerLine {
    Edit(EditRecord),
    Checkpoint { name: String, after_id: u64, ts: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Refusal(pub String);
impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "edit refused: {}", self.0)
    }
}
impl std::error::Error for Refusal {}

pub fn glob_match(pat: &str, s: &str) -> bool {
    fn rec(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => rec(&p[1..], s) || (!s.is_empty() && rec(p, &s[1..])),
            (Some(b'?'), Some(_)) => rec(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => rec(&p[1..], &s[1..]),
            _ => false,
        }
    }
    rec(pat.as_bytes(), s.as_bytes())
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// If `new == c * old` for one constant c != 1 over the whole tensor, return c.
fn whole_tensor_constant(old: &Tensor, new: &Tensor) -> Option<f64> {
    let mut c: Option<f64> = None;
    let mut any_nonzero = false;
    for (a, b) in old.data.iter().zip(&new.data) {
        let (a, b) = (*a as f64, *b as f64);
        if a == 0.0 {
            if b != 0.0 {
                return None;
            }
            continue;
        }
        any_nonzero = true;
        let r = b / a;
        match c {
            None => c = Some(r),
            Some(c0) => {
                if (r - c0).abs() > 1e-5 * c0.abs().max(1e-12) {
                    return None;
                }
            }
        }
    }
    match c {
        Some(c) if any_nonzero && (c - 1.0).abs() > 1e-6 => Some(c),
        _ => None,
    }
}

/// Checkpoint names are plain words: letters, digits, '-' and '_' (so they stay inside the checkpoints folder).
fn check_checkpoint_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 64 || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        bail!("checkpoint names use only letters, digits, '-' and '_' (got {name:?})");
    }
    Ok(())
}

pub struct Gate {
    pub dir: PathBuf,
    pub base: Network,
    pub current: Network,
    pub policy: EditPolicy,
    lines: Vec<LedgerLine>,
    diff_vs_base: BTreeMap<String, usize>,
}

impl Gate {
    /// Open (or create) a gate in `dir` over `base`. An existing ledger is replayed onto the base,
    /// and every recorded `sha_after` is checked.
    pub fn open(dir: &Path, base: Network, policy: EditPolicy) -> Result<Gate> {
        std::fs::create_dir_all(dir.join("blobs"))?;
        let mut g = Gate {
            dir: dir.to_path_buf(),
            current: base.clone(),
            base,
            policy,
            lines: vec![],
            diff_vs_base: BTreeMap::new(),
        };
        g.replay_ledger()?;
        g.current.save_dir(&g.current_dir())?;
        Ok(g)
    }

    fn ledger_path(&self) -> PathBuf {
        self.dir.join("ledger.jsonl")
    }

    /// Re-apply every edit of an existing ledger, checking each result against its recorded hash.
    fn replay_ledger(&mut self) -> Result<()> {
        let lp = self.ledger_path();
        if !lp.exists() {
            return Ok(());
        }
        for (i, line) in std::fs::read_to_string(&lp)?.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let l: LedgerLine = serde_json::from_str(line).with_context(|| format!("ledger line {}", i + 1))?;
            if let LedgerLine::Edit(r) = &l {
                let t = compute(&r.op, &self.current, &self.base, &self.blob_root())?;
                if t.sha256() != r.sha_after {
                    bail!("ledger replay mismatch at edit {} ({})", r.id, r.op.tensor());
                }
                let n = t.count_diff(self.base.get(r.op.tensor())?);
                self.diff_vs_base.insert(r.op.tensor().to_string(), n);
                self.current.tensors.insert(r.op.tensor().to_string(), t);
            }
            self.lines.push(l);
        }
        Ok(())
    }

    pub fn blob_root(&self) -> PathBuf {
        self.dir.join("blobs")
    }
    pub fn lines(&self) -> &[LedgerLine] {
        &self.lines
    }
    pub fn edits(&self) -> impl Iterator<Item = &EditRecord> {
        self.lines
            .iter()
            .filter_map(|l| if let LedgerLine::Edit(r) = l { Some(r) } else { None })
    }
    /// The id of the newest edit (0 before the first).
    pub fn last_edit_id(&self) -> u64 {
        self.edits().map(|r| r.id).max().unwrap_or(0)
    }
    /// The folder holding the materialised current network.
    pub fn current_dir(&self) -> PathBuf {
        self.dir.join("current")
    }
    pub fn changed_vs_base(&self) -> usize {
        self.diff_vs_base.values().sum()
    }

    /// Check and apply one op. On refusal nothing changes (no ledger line, no file).
    pub fn apply(&mut self, op: EditOp, reason: &str, source: &str) -> Result<EditRecord, Refusal> {
        let name = op.tensor().to_string();
        let old = self
            .current
            .tensors
            .get(&name)
            .ok_or_else(|| Refusal(format!("unknown tensor {name}")))?
            .clone();
        self.check_editable(&name)?;
        let new = compute(&op, &self.current, &self.base, &self.blob_root()).map_err(|e| Refusal(format!("{e:#}")))?;
        self.check_result(&name, &old, &new)?;
        let base_t = self.base.get(&name).map_err(|e| Refusal(e.to_string()))?;
        let n_vs_base = new.count_diff(base_t);
        let total_vs_base = self.changed_vs_base() - self.diff_vs_base.get(&name).copied().unwrap_or(0) + n_vs_base;
        let frac = total_vs_base as f64 / self.base.numel().max(1) as f64;
        if let Some(m) = self.policy.max_frac_changed.filter(|m| frac > *m) {
            return Err(Refusal(format!(
                "{:.4}% of the network would differ from the base; the limit is {:.4}%",
                frac * 100.0,
                m * 100.0
            )));
        }
        let rec = EditRecord {
            id: self.last_edit_id() + 1,
            ts: now(),
            changed: new.count_diff(&old),
            changed_vs_base: total_vs_base,
            frac_changed_vs_base: frac,
            sha_before: old.sha256(),
            sha_after: new.sha256(),
            op,
            reason: reason.to_string(),
            source: source.to_string(),
        };
        self.record(&name, &new, &rec)
            .map_err(|e| Refusal(format!("could not record the edit: {e:#}")))?;
        self.diff_vs_base.insert(name.clone(), n_vs_base);
        self.current.tensors.insert(name, new);
        self.lines.push(LedgerLine::Edit(rec.clone()));
        Ok(rec)
    }

    /// The policy's allowed and forbidden tensor names.
    fn check_editable(&self, name: &str) -> Result<(), Refusal> {
        if let Some(allowed) = &self.policy.allowed {
            if !allowed.iter().any(|p| glob_match(p, name)) {
                return Err(Refusal(format!(
                    "{name} is not among the tensors this task lets you edit"
                )));
            }
        }
        if let Some(p) = self.policy.forbidden.iter().find(|p| glob_match(p, name)) {
            return Err(Refusal(format!("{name} may not be edited in this task (rule: {p})")));
        }
        Ok(())
    }

    /// The new value must be finite and, if the policy says so, not the old one times a constant.
    fn check_result(&self, name: &str, old: &Tensor, new: &Tensor) -> Result<(), Refusal> {
        if let Some(i) = new.data.iter().position(|v| !v.is_finite()) {
            return Err(Refusal(format!(
                "{name}: the result has a non-finite value at flat index {i}"
            )));
        }
        if self.policy.forbid_whole_tensor_scale {
            if let Some(c) = whole_tensor_constant(old, new) {
                return Err(Refusal(format!(
                    "{name}: multiplying a whole tensor by a constant ({c:.4}) is not allowed in this task"
                )));
            }
        }
        Ok(())
    }

    /// Write the tensor's new value into current/ and append its ledger line.
    fn record(&self, name: &str, new: &Tensor, rec: &EditRecord) -> Result<()> {
        save_npy(&self.current_dir().join(format!("{}.npy", name_to_file(name))), new)?;
        self.append_ledger(&LedgerLine::Edit(rec.clone()))
    }

    fn append_ledger(&self, line: &LedgerLine) -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.ledger_path())?;
        writeln!(f, "{}", serde_json::to_string(line)?)?;
        Ok(())
    }

    /// Store `t` as a blob and apply a Replace op for it.
    pub fn replace(&mut self, name: &str, t: Tensor, reason: &str, source: &str) -> Result<EditRecord, Refusal> {
        let blob = format!("{}.npy", t.sha256());
        let p = self.blob_root().join(&blob);
        if !p.exists() {
            std::fs::write(&p, write_npy(&t)).map_err(|e| Refusal(format!("could not store blob: {e}")))?;
        }
        self.apply(
            EditOp::Replace {
                tensor: name.to_string(),
                blob,
            },
            reason,
            source,
        )
    }

    /// Save every tensor that differs from the base under `checkpoints/<name>`.
    pub fn checkpoint(&mut self, name: &str) -> Result<()> {
        let cdir = self.checkpoint_dir(name)?;
        if cdir.exists() {
            std::fs::remove_dir_all(&cdir)?;
        }
        let mut changed = Network::default();
        for (k, t) in &self.current.tensors {
            if self.diff_vs_base.get(k).copied().unwrap_or(0) > 0 {
                changed.tensors.insert(k.clone(), t.clone());
            }
        }
        changed.save_dir(&cdir)?;
        let l = LedgerLine::Checkpoint {
            name: name.to_string(),
            after_id: self.last_edit_id(),
            ts: now(),
        };
        self.append_ledger(&l)?;
        self.lines.push(l);
        Ok(())
    }

    /// Bring every tensor back to its state at checkpoint `name` (as new ledger edits).
    pub fn restore(&mut self, name: &str, source: &str) -> Result<Vec<EditRecord>> {
        let cdir = self.checkpoint_dir(name)?;
        if !cdir.is_dir() {
            bail!("no checkpoint named {name}");
        }
        let ck = Network::load_dir(&cdir)?;
        let reason = format!("restore checkpoint {name}");
        let mut out = vec![];
        let names: Vec<String> = self.current.tensors.keys().cloned().collect();
        for k in names {
            let r = match ck.tensors.get(&k) {
                Some(t) if *t == self.current.tensors[&k] => continue,
                Some(t) => self.replace(&k, t.clone(), &reason, source),
                None if self.current.tensors[&k] == self.base.tensors[&k] => continue,
                None => self.apply(EditOp::Revert { tensor: k.clone() }, &reason, source),
            };
            out.push(r.map_err(|e| anyhow::anyhow!("{e}"))?);
        }
        Ok(out)
    }

    fn checkpoint_dir(&self, name: &str) -> Result<PathBuf> {
        check_checkpoint_name(name)?;
        Ok(self.dir.join("checkpoints").join(name))
    }

    /// One line per edit, for prompts and handoffs.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for l in &self.lines {
            match l {
                LedgerLine::Edit(r) => {
                    let mut op = r.op.describe();
                    if let EditOp::ProjectOut { select, .. } = &r.op {
                        let on = select
                            .as_ref()
                            .map(|s| format!("{} indices", s.len()))
                            .unwrap_or_else(|| "all".into());
                        op += &format!(" on {on}");
                    }
                    s += &format!(
                        "#{} {} {}: {} entries changed ({} differ from base overall, {:.4}%) - {}\n",
                        r.id,
                        r.op.tensor(),
                        op,
                        r.changed,
                        r.changed_vs_base,
                        r.frac_changed_vs_base * 100.0,
                        r.reason
                    );
                }
                LedgerLine::Checkpoint { name, after_id, .. } => {
                    s += &format!("checkpoint {name} after edit #{after_id}\n")
                }
            }
        }
        if s.is_empty() {
            s = "(no edits yet)\n".into();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Idx, Mode, Values};

    fn base() -> Network {
        let mut n = Network::default();
        n.tensors
            .insert("layers.0.W".into(), Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]));
        n.tensors
            .insert("layers.1.W".into(), Tensor::new(vec![2, 2], vec![1., 1., 1., 1.]));
        n.tensors
            .insert("norm.weight".into(), Tensor::new(vec![2], vec![1., 1.]));
        n
    }
    fn set(t: &str, i: usize, j: usize, v: f64) -> EditOp {
        EditOp::Assign {
            tensor: t.into(),
            index: vec![Idx::One(i), Idx::One(j)],
            mode: Mode::Set,
            values: Values::Scalar(v),
        }
    }

    #[test]
    fn glob() {
        assert!(glob_match("layers.*.W", "layers.12.W"));
        assert!(glob_match("*", "x"));
        assert!(!glob_match("layers.*.W", "layers.1.b"));
        assert!(glob_match("norm.weight", "norm.weight"));
    }

    #[test]
    fn apply_records_and_materialises() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), base(), EditPolicy::default()).unwrap();
        let r = g.apply(set("layers.0.W", 0, 1, 9.0), "test", "unit").unwrap();
        assert_eq!((r.id, r.changed, r.changed_vs_base), (1, 1, 1));
        assert_eq!(g.current.tensors["layers.0.W"].data, vec![1., 9., 3., 4.]);
        let on_disk = Network::load_dir(&d.path().join("current")).unwrap();
        assert_eq!(on_disk, g.current);
    }

    #[test]
    fn refusals() {
        let d = tempfile::tempdir().unwrap();
        let pol = EditPolicy {
            allowed: Some(vec!["layers.*".into()]),
            forbidden: vec!["layers.1.*".into()],
            forbid_whole_tensor_scale: true,
            max_frac_changed: Some(0.25),
        };
        let mut g = Gate::open(d.path(), base(), pol).unwrap();
        let before = g.current.clone();
        // not allowed
        assert!(g.apply(set("norm.weight", 0, 0, 2.0), "", "t").is_err());
        // forbidden
        assert!(g.apply(set("layers.1.W", 0, 0, 2.0), "", "t").is_err());
        // unknown tensor
        assert!(g.apply(set("nope", 0, 0, 2.0), "", "t").is_err());
        // non-finite
        assert!(g.apply(set("layers.0.W", 0, 0, f64::NAN), "", "t").is_err());
        assert!(g.apply(set("layers.0.W", 0, 0, f64::INFINITY), "", "t").is_err());
        // whole-tensor constant scale (both as Mul and as Replace)
        let mul = EditOp::Assign {
            tensor: "layers.0.W".into(),
            index: vec![],
            mode: Mode::Mul,
            values: Values::Scalar(1.5),
        };
        let e = g.apply(mul, "", "t").unwrap_err();
        assert!(e.0.contains("constant"), "{e}");
        assert!(g
            .replace("layers.0.W", Tensor::new(vec![2, 2], vec![2., 4., 6., 8.]), "", "t")
            .is_err());
        // wrong shape replace
        assert!(g
            .replace("layers.0.W", Tensor::new(vec![4], vec![0.; 4]), "", "t")
            .is_err());
        // too many entries changed: 3 of 10 > 25%
        let blk = EditOp::Assign {
            tensor: "layers.0.W".into(),
            index: vec![],
            mode: Mode::Set,
            values: Values::List(vec![0., 0., 0., 4.]),
        };
        assert!(g.apply(blk, "", "t").is_err());
        assert_eq!(g.current, before);
        assert_eq!(g.lines().len(), 0);
        assert!(
            !d.path().join("ledger.jsonl").exists()
                || std::fs::read_to_string(d.path().join("ledger.jsonl"))
                    .unwrap()
                    .is_empty()
        );
        // a legal edit still works afterwards
        g.apply(set("layers.0.W", 0, 0, 5.0), "ok", "t").unwrap();
    }

    #[test]
    fn reopen_replays_ledger_bit_for_bit() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), base(), EditPolicy::default()).unwrap();
        g.apply(set("layers.0.W", 1, 1, -3.0), "a", "t").unwrap();
        g.replace(
            "layers.1.W",
            Tensor::new(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]),
            "b",
            "t",
        )
        .unwrap();
        g.apply(
            EditOp::Revert {
                tensor: "layers.0.W".into(),
            },
            "c",
            "t",
        )
        .unwrap();
        let cur = g.current.clone();
        drop(g);
        let g2 = Gate::open(d.path(), base(), EditPolicy::default()).unwrap();
        assert_eq!(g2.current, cur);
        assert_eq!(g2.edits().count(), 3);
        assert_eq!(g2.changed_vs_base(), 4);
    }

    #[test]
    fn checkpoint_and_restore() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), base(), EditPolicy::default()).unwrap();
        g.apply(set("layers.0.W", 0, 0, 7.0), "a", "t").unwrap();
        g.checkpoint("best_1").unwrap();
        let at_ckpt = g.current.clone();
        g.apply(set("layers.0.W", 0, 1, 8.0), "b", "t").unwrap();
        g.apply(set("layers.1.W", 0, 1, 8.0), "c", "t").unwrap();
        g.restore("best_1", "t").unwrap();
        assert_eq!(g.current, at_ckpt);
        assert!(g.restore("nope", "t").is_err());
        // names that escape the checkpoints folder are refused, even if the target exists
        let outside = d.path().join("outside");
        base().save_dir(&outside).unwrap();
        assert!(g.restore(outside.to_str().unwrap(), "t").is_err());
        assert!(g.restore("../outside", "t").is_err());
        assert!(g.restore("../../etc", "t").is_err());
        assert!(g.summary().contains("best_1"));
    }
}
