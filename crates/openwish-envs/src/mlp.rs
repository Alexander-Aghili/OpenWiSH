//! MLP policies stored as named tensors (L1.W, L1.b, ..., out.W, out.b; row i of W = input i).
use anyhow::{bail, Result};
use openwish_tensor::{Network, Tensor};

pub trait Policy: Sync {
    /// logits for the 82 moves from the 243-number input
    fn logits(&self, x: &[f32]) -> Vec<f32>;
}

pub struct Uniform;
impl Policy for Uniform {
    fn logits(&self, _x: &[f32]) -> Vec<f32> {
        vec![0.0; crate::go::P + 1]
    }
}

#[derive(Clone, Debug)]
pub struct Mlp {
    /// (W as [inputs][outputs] row-major, b)
    pub layers: Vec<(Tensor, Vec<f32>)>,
}

impl Mlp {
    /// L1..Lk (ReLU) then out (linear).
    pub fn from_network(n: &Network) -> Result<Mlp> {
        let mut ks: Vec<usize> = n
            .tensors
            .keys()
            .filter_map(|k| {
                k.strip_prefix('L')
                    .and_then(|r| r.strip_suffix(".W"))
                    .and_then(|d| d.parse().ok())
            })
            .collect();
        ks.sort();
        let mut names: Vec<String> = ks.iter().map(|k| format!("L{k}")).collect();
        names.push("out".into());
        let mut layers = vec![];
        let mut width: Option<usize> = None;
        for name in names {
            let w = n.get(&format!("{name}.W"))?.clone();
            let b = n.get(&format!("{name}.b"))?.data.to_vec();
            if w.shape.len() != 2 || w.shape[1] != b.len() {
                bail!("{name}: W {:?} and b ({}) do not fit", w.shape, b.len());
            }
            if let Some(prev) = width {
                if w.shape[0] != prev {
                    bail!(
                        "{name}.W has {} inputs, the previous layer has {prev} outputs",
                        w.shape[0]
                    );
                }
            }
            width = Some(w.shape[1]);
            layers.push((w, b));
        }
        Ok(Mlp { layers })
    }
    /// Forward pass; with `acts`, the hidden activations of every layer are recorded.
    pub fn forward(&self, x: &[f32], mut acts: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
        let mut h: Vec<f32> = x.to_vec();
        let last = self.layers.len() - 1;
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let mut out = b.clone();
            // row i of W holds input i's weights
            for (v, row) in h.iter().zip(w.data.chunks(w.shape[1].max(1))) {
                if *v == 0.0 {
                    continue;
                }
                for (o, wv) in out.iter_mut().zip(row) {
                    *o += v * wv;
                }
            }
            if li < last {
                for o in out.iter_mut() {
                    if *o < 0.0 {
                        *o = 0.0;
                    }
                }
                if let Some(a) = acts.as_deref_mut() {
                    a.push(out.clone());
                }
            }
            h = out;
        }
        h
    }
}

impl Policy for Mlp {
    fn logits(&self, x: &[f32]) -> Vec<f32> {
        self.forward(x, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_hand_computation() {
        let mut n = Network::default();
        n.tensors
            .insert("L1.W".into(), Tensor::new(vec![2, 2], vec![1., -1., 2., 0.5]));
        n.tensors.insert("L1.b".into(), Tensor::new(vec![2], vec![0., -3.]));
        n.tensors.insert("out.W".into(), Tensor::new(vec![2, 1], vec![1., 10.]));
        n.tensors.insert("out.b".into(), Tensor::new(vec![1], vec![0.5]));
        let m = Mlp::from_network(&n).unwrap();
        // h = relu([1*1 + 1*2, -1 + 0.5 - 3]) = [3, 0]; out = 3 + 0 + 0.5
        let mut acts = vec![];
        assert_eq!(m.forward(&[1., 1.], Some(&mut acts)), vec![3.5]);
        assert_eq!(acts, vec![vec![3., 0.]]);
    }

    #[test]
    fn layers_are_ordered_numerically_and_shapes_checked() {
        let mut n = Network::default();
        for k in 1..=10 {
            n.tensors.insert(format!("L{k}.W"), Tensor::zeros(vec![2, 2]));
            n.tensors.insert(format!("L{k}.b"), Tensor::zeros(vec![2]));
        }
        n.tensors.insert("out.W".into(), Tensor::zeros(vec![2, 3]));
        n.tensors.insert("out.b".into(), Tensor::zeros(vec![3]));
        assert_eq!(Mlp::from_network(&n).unwrap().layers.len(), 11);
        n.tensors.insert("L3.W".into(), Tensor::zeros(vec![5, 2]));
        assert!(Mlp::from_network(&n).is_err());
    }
}
