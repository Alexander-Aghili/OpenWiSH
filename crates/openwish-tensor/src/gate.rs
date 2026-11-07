//! The edit gate: every weight change passes through here, is checked against the task's policy,
//! and is appended to the ledger. The materialised current network lives in `<dir>/current/`.
//!
//! Memory and disk: the current network shares every untouched tensor with the base (copy-on-write), so
//! the gate holds one copy of the network plus the edited tensors. Blobs are sparse deltas against the base
//! when that is smaller (`delta`), checkpoints are references to blobs, and `current/` hardlinks untouched
//! tensors to the base files when the base is a folder of float32 .npy files.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::delta;
use crate::network::{name_to_file, save_npy, Network, Tensor};
use crate::npy::{is_plain_f32, write_npy_to};
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

/// True when `a` and `b` are the same file on disk (hardlinks of one inode).
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev() && x.ino() == y.ino(),
        _ => false,
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
    /// The folder the base was loaded from, if it is one: untouched tensors in current/ link to its files.
    base_dir: Option<PathBuf>,
    lines: Vec<LedgerLine>,
    diff_vs_base: BTreeMap<String, usize>,
    /// sha256 of current tensors, filled on first use, so each edit hashes only its new value.
    sha: BTreeMap<String, String>,
}

/// Saved state of a checkpoint: the blob of every tensor that differs from the base.
#[derive(Debug, Default, Serialize, Deserialize)]
struct CheckpointRefs {
    tensors: BTreeMap<String, String>,
}

impl Gate {
    /// Open (or create) a gate in `dir` over `base`. An existing ledger is replayed onto the base,
    /// and every recorded `sha_after` is checked.
    pub fn open(dir: &Path, base: Network, policy: EditPolicy) -> Result<Gate> {
        Self::open_linked(dir, base, None, policy)
    }

    /// Like `open`, with the folder `base` was loaded from, so current/ can hardlink untouched tensors.
    pub fn open_linked(dir: &Path, base: Network, base_dir: Option<&Path>, policy: EditPolicy) -> Result<Gate> {
        std::fs::create_dir_all(dir.join("blobs"))?;
        let mut g = Gate {
            dir: dir.to_path_buf(),
            current: base.clone(),
            base,
            policy,
            base_dir: base_dir.filter(|d| d.is_dir()).map(Path::to_path_buf),
            lines: vec![],
            diff_vs_base: BTreeMap::new(),
            sha: BTreeMap::new(),
        };
        g.replay_ledger()?;
        std::fs::create_dir_all(g.current_dir())?;
        let names: Vec<String> = g.current.tensors.keys().cloned().collect();
        for name in names {
            g.write_current(&name)?;
        }
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
                self.sha.insert(r.op.tensor().to_string(), r.sha_after.clone());
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
        self.apply_with(op, None, reason, source)
    }

    /// `apply`, with the op's result already in memory (a Replace whose blob was just written), so the
    /// blob is not read back and the tensor is not hashed twice.
    fn apply_with(
        &mut self,
        op: EditOp,
        known: Option<(Tensor, String)>,
        reason: &str,
        source: &str,
    ) -> Result<EditRecord, Refusal> {
        let name = op.tensor().to_string();
        let old = self
            .current
            .tensors
            .get(&name)
            .ok_or_else(|| Refusal(format!("unknown tensor {name}")))?
            .clone();
        self.check_editable(&name)?;
        let (new, sha_after) = match known {
            Some((t, sha)) if t.shape == old.shape => (t, sha),
            Some((t, _)) => {
                return Err(Refusal(format!(
                    "{name}: replacement has shape {:?}, tensor has {:?}",
                    t.shape, old.shape
                )))
            }
            None => {
                let t = compute(&op, &self.current, &self.base, &self.blob_root())
                    .map_err(|e| Refusal(format!("{e:#}")))?;
                let sha = t.sha256();
                (t, sha)
            }
        };
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
            sha_before: self.sha_of(&name),
            sha_after,
            op,
            reason: reason.to_string(),
            source: source.to_string(),
        };
        let prev = self.current.tensors.insert(name.clone(), new);
        if let Err(e) = self
            .write_current(&name)
            .and_then(|_| self.append_ledger(&LedgerLine::Edit(rec.clone())))
        {
            // put the old value back in memory and on disk, so a failed write changes nothing
            if let Some(p) = prev {
                self.current.tensors.insert(name.clone(), p);
            }
            let _ = self.write_current(&name);
            return Err(Refusal(format!("could not record the edit: {e:#}")));
        }
        self.diff_vs_base.insert(name.clone(), n_vs_base);
        self.sha.insert(name, rec.sha_after.clone());
        self.lines.push(LedgerLine::Edit(rec.clone()));
        Ok(rec)
    }

    /// sha256 of the current value of `name`, computed once and then kept up to date by every edit.
    fn sha_of(&mut self, name: &str) -> String {
        if let Some(s) = self.sha.get(name) {
            return s.clone();
        }
        let s = self.current.tensors[name].sha256();
        self.sha.insert(name.to_string(), s.clone());
        s
    }

    /// True when the current value of `name` is the base value, bit for bit.
    fn is_base(&self, name: &str) -> bool {
        let (c, b) = (&self.current.tensors[name], &self.base.tensors[name]);
        c.shares_data(b) || (c.shape == b.shape && delta::diff_indices(b, c).is_empty())
    }

    /// Write the current value of `name` into current/: a hardlink to the base file when the tensor is
    /// untouched and the base file is plain float32, else a fresh .npy file (written by rename, so earlier
    /// links such as final/ never change).
    fn write_current(&self, name: &str) -> Result<()> {
        let file = format!("{}.npy", name_to_file(name));
        let target = self.current_dir().join(&file);
        if let Some(src) = self.base_dir.as_ref().map(|d| d.join(&file)) {
            if self.is_base(name) && is_plain_f32(&src) {
                if same_file(&src, &target) {
                    return Ok(());
                }
                let _ = std::fs::remove_file(&target);
                if std::fs::hard_link(&src, &target).is_ok() {
                    return Ok(());
                }
            }
        }
        save_npy(&target, &self.current.tensors[name])
    }

    /// Put every file of current/ into `out` as a hardlink (a copy across filesystems). Later edits replace
    /// files in current/ by rename, so `out` keeps this state.
    pub fn link_current_to(&self, out: &Path) -> Result<()> {
        std::fs::create_dir_all(out)?;
        for e in std::fs::read_dir(self.current_dir())? {
            let p = e?.path();
            let Some(f) = p.file_name() else { continue };
            if p.extension().is_none_or(|x| x != "npy") {
                continue;
            }
            let dst = out.join(f);
            let _ = std::fs::remove_file(&dst);
            if std::fs::hard_link(&p, &dst).is_err() {
                std::fs::copy(&p, &dst)?;
            }
        }
        Ok(())
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
        let (blob, sha) = self
            .store_blob(name, &t)
            .map_err(|e| Refusal(format!("could not store blob: {e:#}")))?;
        let op = EditOp::Replace {
            tensor: name.to_string(),
            blob,
        };
        self.apply_with(op, Some((t, sha)), reason, source)
    }

    /// Store `t` (a value for tensor `name`) under blobs/, content-addressed by its sha256: as a delta
    /// against the base tensor when that is smaller, else as a full .npy. Returns the blob name and the sha.
    fn store_blob(&self, name: &str, t: &Tensor) -> Result<(String, String)> {
        let sha = t.sha256();
        let base = self.base.tensors.get(name).filter(|b| b.shape == t.shape);
        let idx = base.map(|b| delta::diff_indices(b, t));
        let blob = match &idx {
            Some(idx) if delta::worth_it(t.numel(), idx.len()) => format!("{sha}.{}", delta::EXT),
            _ => format!("{sha}.npy"),
        };
        let p = self.blob_root().join(&blob);
        if !p.exists() {
            match &idx {
                Some(idx) if blob.ends_with(delta::EXT) => delta::save(&p, t, idx)?,
                _ => {
                    let tmp = p.with_extension("npy.tmp");
                    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
                    write_npy_to(&mut w, t)?;
                    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                    std::fs::rename(&tmp, &p)?;
                }
            }
        }
        Ok((blob, sha))
    }

    /// Record a checkpoint: for every tensor that differs from the base, the blob holding its value
    /// (`checkpoints/<name>.json`). Blobs are content-addressed, so a checkpoint copies no tensor data.
    pub fn checkpoint(&mut self, name: &str) -> Result<()> {
        let path = self.checkpoint_path(name)?;
        let old_dir = path.with_extension("");
        if old_dir.is_dir() {
            std::fs::remove_dir_all(&old_dir)?;
        }
        let mut refs = CheckpointRefs::default();
        let changed: Vec<String> = self
            .current
            .tensors
            .keys()
            .filter(|k| self.diff_vs_base.get(*k).copied().unwrap_or(0) > 0)
            .cloned()
            .collect();
        for k in changed {
            let (blob, _) = self.store_blob(&k, &self.current.tensors[&k])?;
            refs.tensors.insert(k, blob);
        }
        std::fs::create_dir_all(path.parent().expect("checkpoints folder"))?;
        std::fs::write(&path, serde_json::to_string_pretty(&refs)?)?;
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
        let path = self.checkpoint_path(name)?;
        let old_dir = path.with_extension("");
        let reason = format!("restore checkpoint {name}");
        let mut out = vec![];
        let names: Vec<String> = self.current.tensors.keys().cloned().collect();
        if path.is_file() {
            let refs: CheckpointRefs = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            for k in names {
                let r = match refs.tensors.get(&k) {
                    Some(blob) if blob.split('.').next() == Some(self.sha_of(&k).as_str()) => continue,
                    Some(blob) => {
                        let op = EditOp::Replace {
                            tensor: k.clone(),
                            blob: blob.clone(),
                        };
                        self.apply(op, &reason, source)
                    }
                    None if self.current.tensors[&k] == self.base.tensors[&k] => continue,
                    None => self.apply(EditOp::Revert { tensor: k.clone() }, &reason, source),
                };
                out.push(r.map_err(|e| anyhow::anyhow!("{e}"))?);
            }
            return Ok(out);
        }
        if !old_dir.is_dir() {
            bail!("no checkpoint named {name}");
        }
        // checkpoints written before references: a folder of full tensors
        let ck = Network::load_dir(&old_dir)?;
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

    /// `checkpoints/<name>.json`; the name must be a plain word so the path stays inside the folder.
    fn checkpoint_path(&self, name: &str) -> Result<PathBuf> {
        check_checkpoint_name(name)?;
        Ok(self.dir.join("checkpoints").join(format!("{name}.json")))
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

    fn big_base() -> Network {
        let mut n = Network::default();
        n.tensors.insert(
            "W".into(),
            Tensor::new(vec![10, 10], (0..100).map(|i| i as f32).collect()),
        );
        n.tensors.insert("b".into(), Tensor::new(vec![10], vec![0.5; 10]));
        n
    }

    #[test]
    fn untouched_tensors_share_memory_with_the_base() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), big_base(), EditPolicy::default()).unwrap();
        g.apply(set("W", 0, 0, -1.0), "a", "t").unwrap();
        assert!(g.current.tensors["b"].shares_data(&g.base.tensors["b"]));
        assert!(!g.current.tensors["W"].shares_data(&g.base.tensors["W"]));
    }

    #[test]
    fn a_sparse_replace_is_stored_as_a_delta_and_replays() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), big_base(), EditPolicy::default()).unwrap();
        let mut t = g.base.tensors["W"].clone();
        t.data[7] = -0.0;
        t.data[42] = 1e9;
        let r = g.replace("W", t.clone(), "sparse", "t").unwrap();
        let EditOp::Replace { blob, .. } = &r.op else { panic!() };
        assert!(blob.ends_with(".delta"), "{blob}");
        assert_eq!(r.sha_after, t.sha256());
        // a dense replace still uses a full .npy blob
        let dense = Tensor::new(vec![10, 10], (0..100).map(|i| -(i as f32) - 1.0).collect());
        let r2 = g.replace("W", dense, "dense", "t").unwrap();
        let EditOp::Replace { blob, .. } = &r2.op else { panic!() };
        assert!(blob.ends_with(".npy"), "{blob}");
        let cur = g.current.clone();
        drop(g);
        assert_eq!(
            Gate::open(d.path(), big_base(), EditPolicy::default()).unwrap().current,
            cur
        );
    }

    #[test]
    fn current_links_untouched_tensors_to_the_base_files() {
        use std::os::unix::fs::MetadataExt;
        let d = tempfile::tempdir().unwrap();
        let base_dir = d.path().join("base");
        big_base().save_dir(&base_dir).unwrap();
        let gdir = d.path().join("gate");
        let mut g = Gate::open_linked(&gdir, big_base(), Some(&base_dir), EditPolicy::default()).unwrap();
        g.apply(set("W", 1, 1, 3.0), "a", "t").unwrap();
        let ino = |p: &Path| std::fs::metadata(p).unwrap().ino();
        assert_eq!(ino(&gdir.join("current/b.npy")), ino(&base_dir.join("b.npy")));
        assert_ne!(ino(&gdir.join("current/W.npy")), ino(&base_dir.join("W.npy")));
        // a revert links the base file again, and the base file never changes
        g.apply(EditOp::Revert { tensor: "W".into() }, "r", "t").unwrap();
        assert_eq!(ino(&gdir.join("current/W.npy")), ino(&base_dir.join("W.npy")));
        assert_eq!(Network::load_dir(&base_dir).unwrap(), big_base());
        assert_eq!(Network::load_dir(&g.current_dir()).unwrap(), g.current);
        // final copies share the current files
        let fin = d.path().join("final");
        g.apply(set("W", 2, 2, 5.0), "b", "t").unwrap();
        g.link_current_to(&fin).unwrap();
        assert_eq!(Network::load_dir(&fin).unwrap(), g.current);
        assert_eq!(ino(&fin.join("W.npy")), ino(&gdir.join("current/W.npy")));
        // a later edit replaces current/W.npy without touching the final copy
        g.apply(set("W", 3, 3, 6.0), "c", "t").unwrap();
        assert_eq!(Network::load_dir(&fin).unwrap().tensors["W"].data[22], 5.0);
        assert_eq!(Network::load_dir(&fin).unwrap().tensors["W"].data[33], 33.0);
    }

    #[test]
    fn checkpoints_are_references_to_blobs() {
        let d = tempfile::tempdir().unwrap();
        let mut g = Gate::open(d.path(), big_base(), EditPolicy::default()).unwrap();
        g.apply(set("W", 0, 0, 7.0), "a", "t").unwrap();
        g.checkpoint("c1").unwrap();
        assert!(d.path().join("checkpoints/c1.json").exists());
        assert!(!d.path().join("checkpoints/c1").exists());
        let at = g.current.clone();
        g.apply(set("W", 0, 1, 8.0), "b", "t").unwrap();
        let b0 = EditOp::Assign {
            tensor: "b".into(),
            index: vec![Idx::One(0)],
            mode: Mode::Set,
            values: Values::Scalar(8.0),
        };
        g.apply(b0, "c", "t").unwrap();
        g.restore("c1", "t").unwrap();
        assert_eq!(g.current, at);
        drop(g);
        assert_eq!(
            Gate::open(d.path(), big_base(), EditPolicy::default()).unwrap().current,
            at
        );
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
