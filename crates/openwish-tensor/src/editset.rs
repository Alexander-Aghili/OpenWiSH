//! Portable edit sets: the tensors that differ from a base, with hashes, so an edit can be
//! reviewed, shipped and applied to another copy of the same base network.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::network::Network;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditSetEntry {
    pub shape: Vec<usize>,
    pub changed: usize,
    pub sha_base: String,
    pub sha_edited: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub tensors: BTreeMap<String, EditSetEntry>,
    pub changed_total: usize,
    pub numel_total: usize,
}

/// Write `<dir>/manifest.json` and `<dir>/edit.npz` (only the changed tensors, in full).
pub fn export(base: &Network, edited: &Network, dir: &Path) -> Result<Manifest> {
    std::fs::create_dir_all(dir)?;
    let mut m = Manifest {
        numel_total: base.numel(),
        ..Default::default()
    };
    let mut changed = Network::default();
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
            changed.tensors.insert(k.clone(), e.clone());
        }
    }
    changed.save_npz(&dir.join("edit.npz"))?;
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&m)?)?;
    Ok(m)
}

/// Apply an exported edit set to `base`, checking base and result hashes.
pub fn apply(base: &Network, dir: &Path) -> Result<Network> {
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).context("manifest.json")?)?;
    let e = Network::load_npz(&dir.join("edit.npz"))?;
    let mut out = base.clone();
    for (k, ent) in &m.tensors {
        if base.get(k)?.sha256() != ent.sha_base {
            bail!("{k}: base tensor does not match the edit set's base hash");
        }
        let t = e.get(k)?.clone();
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
