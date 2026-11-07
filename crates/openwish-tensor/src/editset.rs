//! Portable edit sets: the tensors that differ from a base, with hashes, so an edit can be
//! reviewed, shipped and applied to another copy of the same base network.
//!
//! Format `delta-v1` (written now): `manifest.json` plus one `<tensor file>.delta` per changed tensor, a
//! sparse delta against the base (see `delta`). Older sets have no format field and hold the changed
//! tensors in full in `edit.npz`; `apply` reads both.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::delta;
use crate::network::{name_to_file, Network};

/// The format name of edit sets written by `export`.
pub const FORMAT: &str = "delta-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditSetEntry {
    pub shape: Vec<usize>,
    pub changed: usize,
    pub sha_base: String,
    pub sha_edited: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// `delta-v1`, or absent for the older dense `edit.npz` sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub tensors: BTreeMap<String, EditSetEntry>,
    pub changed_total: usize,
    pub numel_total: usize,
}

fn delta_path(dir: &Path, tensor: &str) -> std::path::PathBuf {
    dir.join(format!("{}.{}", name_to_file(tensor), delta::EXT))
}

/// Write `<dir>/manifest.json` and one sparse `<tensor>.delta` per changed tensor.
pub fn export(base: &Network, edited: &Network, dir: &Path) -> Result<Manifest> {
    std::fs::create_dir_all(dir)?;
    let mut m = Manifest {
        format: Some(FORMAT.into()),
        numel_total: base.numel(),
        ..Default::default()
    };
    for (k, b) in &base.tensors {
        let e = edited.get(k)?;
        if e.shape != b.shape {
            bail!("{k}: edited shape {:?} differs from base {:?}", e.shape, b.shape);
        }
        let n = e.count_diff(b);
        if n > 0 {
            m.changed_total += n;
            m.tensors.insert(
                k.clone(),
                EditSetEntry {
                    shape: b.shape.clone(),
                    changed: n,
                    sha_base: b.sha256(),
                    sha_edited: e.sha256(),
                },
            );
            delta::save(&delta_path(dir, k), e, &delta::diff_indices(b, e))?;
        }
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&m)?)?;
    Ok(m)
}

/// Apply an exported edit set to `base`, checking base and result hashes.
pub fn apply(base: &Network, dir: &Path) -> Result<Network> {
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).context("manifest.json")?)?;
    let dense = match m.format.as_deref() {
        None => Some(Network::load_npz(&dir.join("edit.npz"))?),
        Some(FORMAT) => None,
        Some(f) => bail!("unknown edit set format {f}"),
    };
    let mut out = base.clone();
    for (k, ent) in &m.tensors {
        let b = base.get(k)?;
        if b.sha256() != ent.sha_base {
            bail!("{k}: base tensor does not match the edit set's base hash");
        }
        let t = match &dense {
            Some(e) => e.get(k)?.clone(),
            None => delta::load(&delta_path(dir, k), b)?,
        };
        if t.sha256() != ent.sha_edited {
            bail!("{k}: edit set tensor is corrupt (hash mismatch)");
        }
        out.tensors.insert(k.clone(), t);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Tensor;

    fn base() -> Network {
        let mut n = Network::default();
        n.tensors.insert("a".into(), Tensor::new(vec![2], vec![1., 2.]));
        n.tensors.insert("b".into(), Tensor::new(vec![2], vec![3., 4.]));
        n
    }

    #[test]
    fn export_then_apply_roundtrips() {
        let d = tempfile::tempdir().unwrap();
        let mut e = base();
        e.tensors.get_mut("b").unwrap().data[1] = 9.0;
        let m = export(&base(), &e, d.path()).unwrap();
        assert_eq!(m.tensors.len(), 1);
        assert_eq!((m.changed_total, m.numel_total), (1, 4));
        assert_eq!(apply(&base(), d.path()).unwrap(), e);
    }

    #[test]
    fn export_writes_sparse_deltas_and_old_dense_sets_still_apply() {
        let d = tempfile::tempdir().unwrap();
        let mut e = base();
        e.tensors.get_mut("a").unwrap().data[1] = -7.0;
        export(&base(), &e, d.path()).unwrap();
        assert!(d.path().join("a.delta").exists());
        assert!(!d.path().join("edit.npz").exists());
        // an edit set in the older dense format
        let old = tempfile::tempdir().unwrap();
        let mut changed = Network::default();
        changed.tensors.insert("a".into(), e.tensors["a"].clone());
        changed.save_npz(&old.path().join("edit.npz")).unwrap();
        let mut m: Manifest =
            serde_json::from_str(&std::fs::read_to_string(d.path().join("manifest.json")).unwrap()).unwrap();
        m.format = None;
        std::fs::write(old.path().join("manifest.json"), serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(apply(&base(), old.path()).unwrap(), e);
        assert_eq!(apply(&base(), d.path()).unwrap(), e);
    }

    #[test]
    fn apply_refuses_wrong_base() {
        let d = tempfile::tempdir().unwrap();
        let mut e = base();
        e.tensors.get_mut("a").unwrap().data[0] = 0.0;
        export(&base(), &e, d.path()).unwrap();
        let mut other = base();
        other.tensors.get_mut("a").unwrap().data[1] = 5.0;
        assert!(apply(&other, d.path()).is_err());
    }
}
