//! Tensors and named networks (directory of .npy files, or an .npz archive).
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;

use crate::npy::{read_npy, write_npy_to};

/// Tensor values, shared copy-on-write: cloning a tensor or a whole network shares the buffer, and the first
/// write through `DerefMut` gives the writer its own copy. The gate's base and current networks therefore
/// hold one copy of every tensor that no edit has touched.
#[derive(Clone, Default, PartialEq)]
pub struct TensorData(Arc<Vec<f32>>);

impl Deref for TensorData {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}
impl DerefMut for TensorData {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        Arc::make_mut(&mut self.0)
    }
}
impl<'a> IntoIterator for &'a TensorData {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl From<Vec<f32>> for TensorData {
    fn from(v: Vec<f32>) -> Self {
        TensorData(Arc::new(v))
    }
}
impl PartialEq<Vec<f32>> for TensorData {
    fn eq(&self, other: &Vec<f32>) -> bool {
        *self.0 == *other
    }
}
impl std::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl Serialize for TensorData {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}
impl<'de> Deserialize<'de> for TensorData {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Vec::<f32>::deserialize(d).map(TensorData::from)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/data mismatch");
        Tensor {
            shape,
            data: data.into(),
        }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Tensor::new(shape, vec![0.0; n])
    }
    /// True when both tensors share one buffer (no copy was made).
    pub fn shares_data(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.data.0, &other.data.0)
    }
    /// The values as little-endian f32 bytes, without copying on little-endian targets.
    pub fn le_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: f32 has no padding and any byte pattern is a valid u8; the slice borrows `self`.
            std::borrow::Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().cast::<u8>(), self.data.len() * 4)
            })
        }
        #[cfg(not(target_endian = "little"))]
        {
            std::borrow::Cow::Owned(self.data.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    /// sha256 over the shape (as u64 LE) and the f32 LE bytes.
    pub fn sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update((self.shape.len() as u64).to_le_bytes());
        for d in &self.shape {
            h.update((*d as u64).to_le_bytes());
        }
        h.update(self.le_bytes());
        hex::encode(h.finalize())
    }
    /// Number of entries whose value differs from `other` (-0.0 equals 0.0; NaN equals NaN); panics on shape mismatch.
    pub fn count_diff(&self, other: &Tensor) -> usize {
        assert_eq!(self.shape, other.shape, "count_diff: shape mismatch");
        self.data
            .iter()
            .zip(&other.data)
            .filter(|(a, b)| a != b && !(a.is_nan() && b.is_nan()))
            .count()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Network {
    pub tensors: BTreeMap<String, Tensor>,
}

/// Tensor names may contain '/' or '.'; on disk '/' is encoded as "__" so a directory stays flat.
pub fn name_to_file(name: &str) -> String {
    name.replace('/', "__")
}
pub fn file_to_name(file: &str) -> String {
    file.replace("__", "/")
}

/// Write `t` to `path` as .npy through a temporary file, so a reader never sees half a tensor.
pub(crate) fn save_npy(path: &Path, t: &Tensor) -> Result<()> {
    let tmp = path.with_extension("npy.tmp");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    write_npy_to(&mut w, t)?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

impl Network {
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).with_context(|| format!("unknown tensor {name}"))
    }
    pub fn numel(&self) -> usize {
        self.tensors.values().map(Tensor::numel).sum()
    }
    pub fn load_dir(dir: &Path) -> Result<Network> {
        let mut n = Network::default();
        for e in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let p = e?.path();
            let Some(fname) = p.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            let Some(stem) = fname.strip_suffix(".npy") else {
                continue;
            };
            let t = read_npy(&std::fs::read(&p)?).with_context(|| format!("reading {}", p.display()))?;
            n.tensors.insert(file_to_name(stem), t);
        }
        Ok(n)
    }
    pub fn save_dir(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        for (name, t) in &self.tensors {
            save_npy(&dir.join(format!("{}.npy", name_to_file(name))), t)?;
        }
        Ok(())
    }
    pub fn load_npz(path: &Path) -> Result<Network> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut z = zip::ZipArchive::new(f)?;
        let mut n = Network::default();
        for i in 0..z.len() {
            let mut e = z.by_index(i)?;
            let Some(name) = e.name().strip_suffix(".npy").map(str::to_string) else {
                continue;
            };
            let mut buf = Vec::with_capacity(e.size() as usize);
            e.read_to_end(&mut buf)?;
            n.tensors.insert(
                name.clone(),
                read_npy(&buf).with_context(|| format!("{}:{name}", path.display()))?,
            );
        }
        Ok(n)
    }
    pub fn save_npz(&self, path: &Path) -> Result<()> {
        let f = std::fs::File::create(path)?;
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (name, t) in &self.tensors {
            z.start_file(format!("{name}.npy"), opts)?;
            write_npy_to(&mut z, t)?;
        }
        z.finish()?;
        Ok(())
    }
    /// Load either a directory of .npy files or an .npz file.
    pub fn load(path: &Path) -> Result<Network> {
        if path.is_dir() {
            Self::load_dir(path)
        } else {
            Self::load_npz(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Network {
        let mut n = Network::default();
        n.tensors
            .insert("L1.W".into(), Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]));
        n.tensors
            .insert("params/Dense_0/bias".into(), Tensor::new(vec![2], vec![0.5, -0.5]));
        n
    }

    #[test]
    fn dir_roundtrip_keeps_slash_names() {
        let d = tempfile::tempdir().unwrap();
        sample().save_dir(d.path()).unwrap();
        assert!(d.path().join("params__Dense_0__bias.npy").exists());
        assert_eq!(Network::load(d.path()).unwrap(), sample());
    }

    #[test]
    fn npz_roundtrip() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("net.npz");
        sample().save_npz(&p).unwrap();
        assert_eq!(Network::load(&p).unwrap(), sample());
    }

    #[test]
    fn sha_is_stable_and_shape_sensitive() {
        let a = Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::new(vec![4], vec![1., 2., 3., 4.]);
        assert_eq!(a.sha256(), a.clone().sha256());
        assert_ne!(a.sha256(), b.sha256());
        assert_eq!(a.sha256().len(), 64);
    }

    #[test]
    fn negative_zero_equals_zero() {
        let a = Tensor::new(vec![2], vec![0.0, 1.0]);
        let b = Tensor::new(vec![2], vec![-0.0, 1.0]);
        assert_eq!(a.count_diff(&b), 0);
    }

    #[test]
    fn count_diff_counts_entries() {
        let a = Tensor::new(vec![3], vec![1., 2., 3.]);
        let b = Tensor::new(vec![3], vec![1., 0., 0.]);
        assert_eq!(a.count_diff(&b), 2);
    }
}
