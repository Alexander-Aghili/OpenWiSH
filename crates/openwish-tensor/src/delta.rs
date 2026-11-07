//! Sparse tensor deltas against a base tensor (`OWDELTA1` files).
//!
//! An edit usually changes a small part of a large tensor. A delta stores the shape, the flat indices whose
//! f32 bit pattern differs from the base, and the new bit patterns. Applying it to the same base rebuilds the
//! edited tensor bit for bit (including -0.0 and NaN payloads), so ledger hashes still check.
//!
//! Layout (little-endian): b"OWDELTA1", u32 ndim, u64 dims[ndim], u8 index_bytes (4 or 8), u64 count,
//! indices[count] (u32 or u64), u32 bits[count].
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;

use crate::network::Tensor;

const MAGIC: &[u8; 8] = b"OWDELTA1";
/// Extension of delta files (blobs, checkpoints, edit sets).
pub const EXT: &str = "delta";

/// Flat indices where `new` and `base` differ bit for bit.
pub fn diff_indices(base: &Tensor, new: &Tensor) -> Vec<usize> {
    assert_eq!(base.shape, new.shape, "diff_indices: shape mismatch");
    base.data
        .iter()
        .zip(new.data.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .collect()
}

/// Bytes a delta with `count` changes takes, against `numel * 4` for the full tensor.
fn delta_size(numel: usize, count: usize) -> usize {
    let ib = if numel <= u32::MAX as usize { 4 } else { 8 };
    count * (ib + 4)
}

/// True when storing `new` as a delta against `base` is smaller than storing it in full.
pub fn worth_it(numel: usize, count: usize) -> bool {
    delta_size(numel, count) < numel * 4 * 9 / 10
}

/// Write `new` as a delta against `base` to `w`.
pub fn write(w: &mut impl Write, base: &Tensor, new: &Tensor) -> Result<()> {
    write_indices(w, new, &diff_indices(base, new))
}

/// Write the entries `idx` of `new` (the flat indices where it differs from its base) as a delta to `w`.
pub fn write_indices(w: &mut impl Write, new: &Tensor, idx: &[usize]) -> Result<()> {
    let wide = new.numel() > u32::MAX as usize;
    w.write_all(MAGIC)?;
    w.write_all(&(new.shape.len() as u32).to_le_bytes())?;
    for d in &new.shape {
        w.write_all(&(*d as u64).to_le_bytes())?;
    }
    w.write_all(&[if wide { 8 } else { 4 }])?;
    w.write_all(&(idx.len() as u64).to_le_bytes())?;
    let mut buf = Vec::with_capacity(idx.len() * if wide { 12 } else { 8 });
    for i in idx {
        if wide {
            buf.extend_from_slice(&(*i as u64).to_le_bytes());
        } else {
            buf.extend_from_slice(&(*i as u32).to_le_bytes());
        }
    }
    for i in idx {
        buf.extend_from_slice(&new.data[*i].to_bits().to_le_bytes());
    }
    w.write_all(&buf)?;
    Ok(())
}

/// Write `new` as a delta file at `path` (through a temporary file), given its changed indices.
pub fn save(path: &Path, new: &Tensor, idx: &[usize]) -> Result<()> {
    let tmp = path.with_extension("delta.tmp");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    write_indices(&mut w, new, idx)?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn take<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b).context("truncated delta")?;
    Ok(b)
}

/// Rebuild the edited tensor from a delta and its base.
pub fn read(r: &mut impl Read, base: &Tensor) -> Result<Tensor> {
    if &take::<8>(r)? != MAGIC {
        bail!("not an OWDELTA1 file");
    }
    let ndim = u32::from_le_bytes(take(r)?) as usize;
    if ndim > 32 {
        bail!("delta has {ndim} axes");
    }
    let shape: Vec<usize> = (0..ndim)
        .map(|_| take::<8>(r).map(|b| u64::from_le_bytes(b) as usize))
        .collect::<Result<_>>()?;
    if shape != base.shape {
        bail!("delta has shape {shape:?}, base has {:?}", base.shape);
    }
    let ib = take::<1>(r)?[0] as usize;
    if ib != 4 && ib != 8 {
        bail!("delta index width {ib}");
    }
    let count = u64::from_le_bytes(take(r)?) as usize;
    if count > base.numel() {
        bail!("delta lists {count} changes for {} entries", base.numel());
    }
    let mut raw = vec![0u8; count * (ib + 4)];
    r.read_exact(&mut raw).context("truncated delta body")?;
    let (ib_raw, bits_raw) = raw.split_at(count * ib);
    let mut out = base.clone();
    let data = &mut *out.data;
    for (ichunk, bchunk) in ib_raw.chunks_exact(ib).zip(bits_raw.as_chunks::<4>().0) {
        let i = if ib == 4 {
            u32::from_le_bytes(ichunk.try_into()?) as usize
        } else {
            u64::from_le_bytes(ichunk.try_into()?) as usize
        };
        let v = data
            .get_mut(i)
            .with_context(|| format!("delta index {i} out of range"))?;
        *v = f32::from_bits(u32::from_le_bytes(*bchunk));
    }
    Ok(out)
}

/// Rebuild the edited tensor from the delta file at `path` and its base.
pub fn load(path: &Path, base: &Tensor) -> Result<Tensor> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    read(&mut std::io::BufReader::new(f), base).with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_bit_exact() {
        let base = Tensor::new(vec![2, 3], vec![1., 2., 3., 0., f32::NAN, 6.]);
        let mut new = base.clone();
        new.data[1] = 9.;
        new.data[3] = -0.0; // differs from 0.0 in bits only
        let mut buf = vec![];
        write(&mut buf, &base, &new).unwrap();
        let back = read(&mut buf.as_slice(), &base).unwrap();
        assert_eq!(back.sha256(), new.sha256());
        assert_eq!(diff_indices(&base, &new), vec![1, 3]);
    }

    #[test]
    fn unchanged_tensor_is_an_empty_delta() {
        let base = Tensor::new(vec![3], vec![1., 2., 3.]);
        let mut buf = vec![];
        write(&mut buf, &base, &base).unwrap();
        assert_eq!(read(&mut buf.as_slice(), &base).unwrap(), base);
    }

    #[test]
    fn refuses_a_different_base_shape_and_garbage() {
        let base = Tensor::new(vec![4], vec![0.; 4]);
        let mut new = base.clone();
        new.data[0] = 1.;
        let mut buf = vec![];
        write(&mut buf, &base, &new).unwrap();
        assert!(read(&mut buf.as_slice(), &Tensor::new(vec![2, 2], vec![0.; 4])).is_err());
        assert!(read(&mut &b"not a delta at all"[..], &base).is_err());
        assert!(read(&mut &buf[..buf.len() - 1], &base).is_err());
    }

    #[test]
    fn worth_it_only_for_sparse_changes() {
        assert!(worth_it(1000, 10));
        assert!(!worth_it(1000, 600));
    }
}
