//! Edit operations: the only vocabulary in which weights change.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::network::{Network, Tensor};

/// One axis of an index: a single position or a half-open strided range (numpy semantics, non-negative).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Idx {
    One(usize),
    Range {
        start: Option<usize>,
        end: Option<usize>,
        step: Option<usize>,
    },
}

impl Idx {
    pub fn all() -> Idx {
        Idx::Range {
            start: None,
            end: None,
            step: None,
        }
    }
    /// Concrete positions along an axis of length `len`.
    pub fn positions(&self, len: usize) -> Result<Vec<usize>> {
        match self {
            Idx::One(i) => {
                if *i >= len {
                    bail!("index {i} out of range for axis of length {len}");
                }
                Ok(vec![*i])
            }
            Idx::Range { start, end, step } => {
                let step = step.unwrap_or(1);
                if step == 0 {
                    bail!("slice step cannot be 0");
                }
                let s = start.unwrap_or(0).min(len);
                let e = end.unwrap_or(len).min(len);
                Ok((s..e).step_by(step).collect())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Set,
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Values {
    Scalar(f64),
    List(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    /// `tensor[index] (=|+=|-=|*=) values`. Missing trailing axes mean ':'.
    Assign {
        tensor: String,
        index: Vec<Idx>,
        mode: Mode,
        values: Values,
    },
    /// Replace the whole tensor with the contents of a stored blob (`<blob_root>/<blob>`, an .npy file).
    Replace { tensor: String, blob: String },
    /// W += alpha * outer(u, v) for a 2-D tensor.
    AddRank1 {
        tensor: String,
        u: Vec<f32>,
        v: Vec<f32>,
        alpha: f32,
    },
    /// Remove a direction. axis 0: u has len rows; for each selected column j: W[:,j] -= alpha * u (u.W[:,j]).
    /// axis 1: u has len cols; for each selected row i: W[i,:] -= alpha * u (u.W[i,:]). u is normalised first.
    ProjectOut {
        tensor: String,
        dir: Vec<f32>,
        axis: usize,
        select: Option<Vec<usize>>,
        alpha: f32,
    },
    /// Restore the tensor to its base value.
    Revert { tensor: String },
}

impl EditOp {
    pub fn tensor(&self) -> &str {
        match self {
            EditOp::Assign { tensor, .. }
            | EditOp::Replace { tensor, .. }
            | EditOp::AddRank1 { tensor, .. }
            | EditOp::ProjectOut { tensor, .. }
            | EditOp::Revert { tensor } => tensor,
        }
    }

    /// The op's kind and parameters in a few words, for ledgers and transcripts.
    pub fn describe(&self) -> String {
        match self {
            EditOp::Assign { mode, .. } => format!("assign/{mode:?}").to_lowercase(),
            EditOp::Replace { .. } => "replace".into(),
            EditOp::AddRank1 { alpha, .. } => format!("add_rank1 alpha={alpha}"),
            EditOp::ProjectOut { axis, alpha, .. } => format!("project_out axis={axis} alpha={alpha}"),
            EditOp::Revert { .. } => "revert".into(),
        }
    }
}

/// Compute the new value of the op's tensor, without modifying `net`.
pub fn compute(op: &EditOp, net: &Network, base: &Network, blob_root: &Path) -> Result<Tensor> {
    let name = op.tensor();
    let cur = net.get(name)?;
    match op {
        EditOp::Assign {
            index, mode, values, ..
        } => assign(name, cur, index, *mode, values),
        EditOp::Replace { blob, .. } => replace(name, cur, &blob_root.join(blob)),
        EditOp::AddRank1 { u, v, alpha, .. } => add_rank1(name, cur, u, v, *alpha),
        EditOp::ProjectOut {
            dir,
            axis,
            select,
            alpha,
            ..
        } => project_out(name, cur, dir, *axis, select.as_deref(), *alpha),
        EditOp::Revert { .. } => Ok(base.get(name)?.clone()),
    }
}

/// `cur[index] (=|+=|-=|*=) values`, with missing trailing axes meaning ':'.
fn assign(name: &str, cur: &Tensor, index: &[Idx], mode: Mode, values: &Values) -> Result<Tensor> {
    if index.len() > cur.shape.len() {
        bail!("{name} has {} axes but the index has {}", cur.shape.len(), index.len());
    }
    let mut axes: Vec<Vec<usize>> = Vec::with_capacity(cur.shape.len());
    for (d, len) in cur.shape.iter().enumerate() {
        axes.push(index.get(d).cloned().unwrap_or_else(Idx::all).positions(*len)?);
    }
    let n: usize = axes.iter().map(Vec::len).product();
    let vals: Vec<f64> = match values {
        Values::Scalar(v) => vec![*v; n],
        Values::List(l) if l.len() != n => bail!("{name}: {} values given for {n} entries", l.len()),
        Values::List(l) => l.clone(),
    };
    let strides = strides(&cur.shape);
    let mut out = cur.clone();
    for (k, v) in vals.iter().enumerate() {
        let x = &mut out.data[block_offset(k, &axes, &strides)];
        let v = *v as f32;
        *x = match mode {
            Mode::Set => v,
            Mode::Add => *x + v,
            Mode::Sub => *x - v,
            Mode::Mul => *x * v,
        };
    }
    Ok(out)
}

/// The flat offset of the `k`-th entry (in row-major order) of the block that `axes` select.
fn block_offset(k: usize, axes: &[Vec<usize>], strides: &[usize]) -> usize {
    let mut r = k;
    let mut off = 0;
    for d in (0..axes.len()).rev() {
        let l = axes[d].len();
        off += axes[d][r % l] * strides[d];
        r /= l;
    }
    off
}

/// The tensor stored in the .npy blob at `path`, which must have the current shape.
fn replace(name: &str, cur: &Tensor, path: &Path) -> Result<Tensor> {
    let bytes = std::fs::read(path).with_context(|| format!("reading blob {}", path.display()))?;
    let t = crate::npy::read_npy(&bytes)?;
    if t.shape != cur.shape {
        bail!(
            "{name}: replacement has shape {:?}, tensor has {:?}",
            t.shape,
            cur.shape
        );
    }
    Ok(t)
}

/// `cur + alpha * outer(u, v)` for a 2-D tensor.
fn add_rank1(name: &str, cur: &Tensor, u: &[f32], v: &[f32], alpha: f32) -> Result<Tensor> {
    let (rows, cols) = dims2(name, cur)?;
    if u.len() != rows || v.len() != cols {
        bail!(
            "{name}: rank-1 factors have lengths {}/{}, need {rows}/{cols}",
            u.len(),
            v.len()
        );
    }
    let mut out = cur.clone();
    for (row, ui) in out.data.chunks_mut(cols).zip(u) {
        if *ui == 0.0 {
            continue;
        }
        for (x, vj) in row.iter_mut().zip(v) {
            *x += alpha * ui * vj;
        }
    }
    Ok(out)
}

/// Remove the direction `dir` (normalised) from the selected columns (axis 0) or rows (axis 1).
fn project_out(
    name: &str,
    cur: &Tensor,
    dir: &[f32],
    axis: usize,
    select: Option<&[usize]>,
    alpha: f32,
) -> Result<Tensor> {
    let (rows, cols) = dims2(name, cur)?;
    let (len, other) = match axis {
        0 => (rows, cols),
        1 => (cols, rows),
        _ => bail!("axis must be 0 or 1"),
    };
    if dir.len() != len {
        bail!("{name}: direction has length {}, axis {axis} has {len}", dir.len());
    }
    let norm = dir.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        bail!("{name}: direction has zero or non-finite norm");
    }
    let u: Vec<f64> = dir.iter().map(|x| *x as f64 / norm).collect();
    let sel: Vec<usize> = match select {
        Some(s) => {
            if let Some(bad) = s.iter().find(|k| **k >= other) {
                bail!("{name}: selected index {bad} out of range {other}");
            }
            s.to_vec()
        }
        None => (0..other).collect(),
    };
    let mut out = cur.clone();
    // entry a of line k: column k (axis 0) or row k (axis 1)
    let at = |a: usize, k: usize| if axis == 0 { a * cols + k } else { k * cols + a };
    for k in sel {
        let dot: f64 = u.iter().enumerate().map(|(a, ua)| ua * cur.data[at(a, k)] as f64).sum();
        for (a, ua) in u.iter().enumerate() {
            out.data[at(a, k)] = (cur.data[at(a, k)] as f64 - alpha as f64 * ua * dot) as f32;
        }
    }
    Ok(out)
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn dims2(name: &str, t: &Tensor) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [r, c] => Ok((*r, *c)),
        s => bail!("{name}: operation needs a 2-D tensor, shape is {s:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net() -> Network {
        let mut n = Network::default();
        n.tensors
            .insert("W".into(), Tensor::new(vec![3, 4], (0..12).map(|v| v as f32).collect()));
        n.tensors.insert("b".into(), Tensor::new(vec![4], vec![1.0; 4]));
        n
    }
    fn r(s: Option<usize>, e: Option<usize>, st: Option<usize>) -> Idx {
        Idx::Range {
            start: s,
            end: e,
            step: st,
        }
    }
    fn run(op: EditOp) -> Result<Tensor> {
        let n = net();
        compute(&op, &n, &n, Path::new("/nonexistent"))
    }

    #[test]
    fn positions_numpy_semantics() {
        assert_eq!(Idx::One(2).positions(5).unwrap(), vec![2]);
        assert_eq!(r(None, None, None).positions(3).unwrap(), vec![0, 1, 2]);
        assert_eq!(r(Some(1), Some(10), Some(3)).positions(8).unwrap(), vec![1, 4, 7]);
        assert!(Idx::One(5).positions(5).is_err());
        assert!(r(None, None, Some(0)).positions(5).is_err());
    }

    #[test]
    fn assign_single_entry() {
        let t = run(EditOp::Assign {
            tensor: "W".into(),
            index: vec![Idx::One(1), Idx::One(2)],
            mode: Mode::Set,
            values: Values::Scalar(-1.0),
        })
        .unwrap();
        assert_eq!(t.data[6], -1.0);
        assert_eq!(t.count_diff(&net().tensors["W"]), 1);
    }

    #[test]
    fn assign_list_along_strided_range() {
        let t = run(EditOp::Assign {
            tensor: "W".into(),
            index: vec![r(Some(0), Some(3), Some(2)), Idx::One(0)],
            mode: Mode::Set,
            values: Values::List(vec![9.0, 7.0]),
        })
        .unwrap();
        assert_eq!((t.data[0], t.data[8]), (9.0, 7.0));
        assert_eq!(t.count_diff(&net().tensors["W"]), 2);
    }

    #[test]
    fn assign_2d_block_row_major_and_modes() {
        // W[0:2, 1:3] *= 2 then compare
        let t = run(EditOp::Assign {
            tensor: "W".into(),
            index: vec![r(Some(0), Some(2), None), r(Some(1), Some(3), None)],
            mode: Mode::Mul,
            values: Values::Scalar(2.0),
        })
        .unwrap();
        assert_eq!(&t.data[0..4], &[0., 2., 4., 3.]);
        assert_eq!(&t.data[4..8], &[4., 10., 12., 7.]);
        let t = run(EditOp::Assign {
            tensor: "b".into(),
            index: vec![],
            mode: Mode::Sub,
            values: Values::Scalar(0.5),
        })
        .unwrap();
        assert_eq!(t.data, vec![0.5; 4]);
        let t = run(EditOp::Assign {
            tensor: "W".into(),
            index: vec![Idx::One(2)],
            mode: Mode::Add,
            values: Values::List(vec![1., 1., 1., 1.]),
        })
        .unwrap();
        assert_eq!(&t.data[8..12], &[9., 10., 11., 12.]);
    }

    #[test]
    fn assign_wrong_list_length_errors() {
        assert!(run(EditOp::Assign {
            tensor: "b".into(),
            index: vec![r(Some(0), Some(2), None)],
            mode: Mode::Set,
            values: Values::List(vec![1.0])
        })
        .is_err());
        assert!(run(EditOp::Assign {
            tensor: "b".into(),
            index: vec![Idx::One(0), Idx::One(0)],
            mode: Mode::Set,
            values: Values::Scalar(1.0)
        })
        .is_err());
        assert!(run(EditOp::Assign {
            tensor: "nope".into(),
            index: vec![],
            mode: Mode::Set,
            values: Values::Scalar(1.0)
        })
        .is_err());
    }

    #[test]
    fn rank1_and_project_out() {
        let t = run(EditOp::AddRank1 {
            tensor: "W".into(),
            u: vec![1., 0., 0.],
            v: vec![0., 0., 0., 1.],
            alpha: 10.0,
        })
        .unwrap();
        assert_eq!(t.data[3], 13.0);
        assert_eq!(t.count_diff(&net().tensors["W"]), 1);
        // projecting e0 out of column 1 only (axis 0) zeroes W[0,1]
        let t = run(EditOp::ProjectOut {
            tensor: "W".into(),
            dir: vec![2., 0., 0.],
            axis: 0,
            select: Some(vec![1]),
            alpha: 1.0,
        })
        .unwrap();
        assert_eq!(t.data[1], 0.0);
        assert_eq!(t.count_diff(&net().tensors["W"]), 1);
        // axis 1 on all rows with e3 zeroes the last column
        let t = run(EditOp::ProjectOut {
            tensor: "W".into(),
            dir: vec![0., 0., 0., 1.],
            axis: 1,
            select: None,
            alpha: 1.0,
        })
        .unwrap();
        assert_eq!((t.data[3], t.data[7], t.data[11]), (0., 0., 0.));
        assert!(run(EditOp::ProjectOut {
            tensor: "W".into(),
            dir: vec![0., 0., 0.],
            axis: 0,
            select: None,
            alpha: 1.0
        })
        .is_err());
    }

    #[test]
    fn replace_and_revert() {
        let d = tempfile::tempdir().unwrap();
        let t = Tensor::new(vec![4], vec![7.; 4]);
        std::fs::write(d.path().join("x.npy"), crate::npy::write_npy(&t)).unwrap();
        let mut cur = net();
        let base = net();
        let got = compute(
            &EditOp::Replace {
                tensor: "b".into(),
                blob: "x.npy".into(),
            },
            &cur,
            &base,
            d.path(),
        )
        .unwrap();
        assert_eq!(got, t);
        cur.tensors.insert("b".into(), got);
        let back = compute(&EditOp::Revert { tensor: "b".into() }, &cur, &base, d.path()).unwrap();
        assert_eq!(back, base.tensors["b"]);
        std::fs::write(
            d.path().join("bad.npy"),
            crate::npy::write_npy(&Tensor::new(vec![3], vec![0.; 3])),
        )
        .unwrap();
        assert!(compute(
            &EditOp::Replace {
                tensor: "b".into(),
                blob: "bad.npy".into()
            },
            &cur,
            &base,
            d.path()
        )
        .is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let op = EditOp::Assign {
            tensor: "W".into(),
            index: vec![Idx::One(1), Idx::all()],
            mode: Mode::Set,
            values: Values::List(vec![1., 2.]),
        };
        let j = serde_json::to_string(&op).unwrap();
        assert_eq!(serde_json::from_str::<EditOp>(&j).unwrap(), op);
    }
}
