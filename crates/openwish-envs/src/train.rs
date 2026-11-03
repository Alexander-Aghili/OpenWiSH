//! Harness-side training of the small Go opponents (the authors' practice and hidden opponents were
//! not released). This is NOT available to the editor: it is how the harness builds its environment.
//! Policy-gradient self-play (REINFORCE with a batch-mean baseline, or PPO-style clipped updates),
//! against a pool of uniform random play and earlier snapshots, with Adam.
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::go::{Game, INPUTS, P};
use crate::mlp::{Mlp, Policy, Uniform};
use crate::play::{legal_softmax, max_legal, sample};
use openwish_tensor::{Network, Tensor};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainCfg {
    pub hidden: Vec<usize>,
    pub iters: usize,
    pub games_per_iter: usize,
    pub lr: f32,
    pub entropy: f32,
    pub seed: u64,
    /// "reinforce" or "ppo"
    pub algo: String,
    /// PPO epochs per batch and clip range
    pub ppo_epochs: usize,
    pub clip: f32,
    /// add a snapshot to the opponent pool every this many iterations
    pub snapshot_every: usize,
}

/// A trainable MLP: parameters as a flat list of (W, b) with W [in][out].
#[derive(Clone, Debug)]
pub struct Params {
    pub layers: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
}

impl Params {
    pub fn init(sizes: &[usize], rng: &mut impl Rng) -> Params {
        let mut layers = vec![];
        for k in 0..sizes.len() - 1 {
            let (i, o) = (sizes[k], sizes[k + 1]);
            let std = if k + 2 == sizes.len() {
                0.01
            } else {
                (2.0 / i as f32).sqrt()
            };
            let w = (0..i * o)
                .map(|_| {
                    // Box-Muller
                    let (u1, u2): (f32, f32) = (rng.gen::<f32>().max(1e-12), rng.gen());
                    std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect();
            layers.push((w, vec![0.0; o], i, o));
        }
        Params { layers }
    }
    pub fn to_mlp(&self) -> Mlp {
        Mlp {
            layers: self
                .layers
                .iter()
                .map(|(w, b, i, o)| (Tensor::new(vec![*i, *o], w.clone()), b.clone()))
                .collect(),
        }
    }
    pub fn to_network(&self) -> Network {
        let mut n = Network::default();
        let k = self.layers.len();
        for (li, (w, b, i, o)) in self.layers.iter().enumerate() {
            let name = if li + 1 == k {
                "out".to_string()
            } else {
                format!("L{}", li + 1)
            };
            n.tensors
                .insert(format!("{name}.W"), Tensor::new(vec![*i, *o], w.clone()));
            n.tensors.insert(format!("{name}.b"), Tensor::new(vec![*o], b.clone()));
        }
        n
    }
    fn add(&mut self, o: &Params) {
        for (a, b) in self.layers.iter_mut().zip(&o.layers) {
            a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y);
            a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
        }
    }
    /// log softmax over legal moves, for move a
    pub fn logp(&self, x: &[f32], legal: &[bool; P + 1], a: usize) -> f32 {
        let z = self.forward(x, None);
        let mx = max_legal(&z, legal);
        let s: f32 = (0..=P).filter(|m| legal[*m]).map(|m| (z[m] - mx).exp()).sum();
        z[a] - mx - s.ln()
    }

    /// The logits for `x` (ReLU hidden layers). With `inputs`, every layer's input is kept: `x`, then the
    /// output of each hidden layer.
    fn forward(&self, x: &[f32], mut inputs: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
        let last = self.layers.len() - 1;
        if let Some(keep) = inputs.as_deref_mut() {
            keep.push(x.to_vec());
        }
        let mut h = x.to_vec();
        for (li, (w, b, i, o)) in self.layers.iter().enumerate() {
            let mut out = b.clone();
            for (r, v) in h[..*i].iter().enumerate() {
                if *v != 0.0 {
                    for (q, wv) in out.iter_mut().zip(&w[r * o..(r + 1) * o]) {
                        *q += v * wv;
                    }
                }
            }
            if li < last {
                out.iter_mut().for_each(|q| *q = q.max(0.0));
                if let Some(keep) = inputs.as_deref_mut() {
                    keep.push(out.clone());
                }
            }
            h = out;
        }
        h
    }

    /// d log softmax_legal(logits)[a] / d params, scaled by `coef`, accumulated into `g`;
    /// plus `ent` times the gradient of the policy entropy.
    pub fn grad_logp(&self, x: &[f32], legal: &[bool; P + 1], a: usize, coef: f32, ent: f32, g: &mut Params) {
        let mut inputs = vec![];
        let z = self.forward(x, Some(&mut inputs));
        self.backward(&inputs, dlogits(&z, legal, a, coef, ent), g);
    }

    /// Accumulate into `g` the gradient whose derivative at the logits is `d`, layer by layer from the top.
    fn backward(&self, inputs: &[Vec<f32>], mut d: Vec<f32>, g: &mut Params) {
        for li in (0..self.layers.len()).rev() {
            let (w, _, i, o) = &self.layers[li];
            let inp = &inputs[li];
            let (gw, gb, _, _) = &mut g.layers[li];
            for (gbv, dv) in gb.iter_mut().zip(&d) {
                *gbv += dv;
            }
            for (r, v) in inp[..*i].iter().enumerate() {
                if *v != 0.0 {
                    for (gq, dv) in gw[r * o..(r + 1) * o].iter_mut().zip(&d) {
                        *gq += v * dv;
                    }
                }
            }
            if li > 0 {
                // through the ReLU: only units that were on pass the gradient down
                let mut dh = vec![0f32; *i];
                for (r, (dhr, v)) in dh.iter_mut().zip(inp).enumerate() {
                    if *v > 0.0 {
                        *dhr = w[r * o..(r + 1) * o].iter().zip(&d).map(|(a, b)| a * b).sum();
                    }
                }
                d = dh;
            }
        }
    }

    pub fn zeros_like(&self) -> Params {
        Params {
            layers: self
                .layers
                .iter()
                .map(|(w, b, i, o)| (vec![0.0; w.len()], vec![0.0; b.len()], *i, *o))
                .collect(),
        }
    }
}

/// The derivative at the logits `z` of `coef` * log softmax_legal(z)[a] + `ent` * entropy.
fn dlogits(z: &[f32], legal: &[bool; P + 1], a: usize, coef: f32, ent: f32) -> Vec<f32> {
    let p = legal_softmax(z, legal);
    let lp: Vec<f32> = (0..=P)
        .map(|m| if legal[m] { (p[m]).max(1e-30).ln() } else { 0.0 })
        .collect();
    let entropy: f32 = -(0..=P).filter(|m| legal[*m]).map(|m| p[m] * lp[m]).sum::<f32>();
    (0..=P)
        .map(|m| {
            if !legal[m] {
                return 0.0;
            }
            let pg = (if m == a { 1.0 } else { 0.0 }) - p[m];
            coef * pg + ent * (-p[m] * (lp[m] + entropy))
        })
        .collect()
}

/// Adam's moment estimates, one pair per parameter.
struct Adam {
    m: Params,
    v: Params,
}

impl Adam {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;

    fn new(params: &Params) -> Adam {
        Adam {
            m: params.zeros_like(),
            v: params.zeros_like(),
        }
    }

    /// One ascent step of size `lr` along `grad`, with bias correction for step `t`.
    fn ascend(&mut self, params: &mut Params, grad: &Params, lr: f32, t: i32) {
        let (b1, b2) = (Self::B1, Self::B2);
        let layers = params
            .layers
            .iter_mut()
            .zip(&grad.layers)
            .zip(self.m.layers.iter_mut().zip(self.v.layers.iter_mut()));
        for ((p, gl), (ml, vl)) in layers {
            for (pv, gv, mv, vv) in [
                (&mut p.0, &gl.0, &mut ml.0, &mut vl.0),
                (&mut p.1, &gl.1, &mut ml.1, &mut vl.1),
            ] {
                for j in 0..pv.len() {
                    mv[j] = b1 * mv[j] + (1.0 - b1) * gv[j];
                    vv[j] = b2 * vv[j] + (1.0 - b2) * gv[j] * gv[j];
                    let mh = mv[j] / (1.0 - b1.powi(t));
                    let vh = vv[j] / (1.0 - b2.powi(t));
                    pv[j] += lr * mh / (vh.sqrt() + Self::EPS);
                }
            }
        }
    }
}

/// One move of the learner in a rollout.
struct Step {
    x: Vec<f32>,
    legal: [bool; P + 1],
    a: usize,
    logp_old: f32,
}

fn rollout(learner: &Params, opp: &dyn Policy, learner_black: bool, seed: u64) -> (Vec<Step>, f32) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mlp = learner.to_mlp();
    let mut g = Game::new();
    let me: i8 = if learner_black { 1 } else { -1 };
    let mut x = vec![0f32; INPUTS];
    let mut steps = vec![];
    while g.end.is_none() {
        g.encode(&mut x);
        let legal = g.legal();
        if g.to_move == me {
            let (m, probs) = sample(&mlp.logits(&x), &legal, &mut rng);
            steps.push(Step {
                x: x.clone(),
                legal,
                a: m,
                logp_old: probs[m].max(1e-30).ln(),
            });
            g.play(m);
        } else {
            let (m, _) = sample(&opp.logits(&x), &legal, &mut rng);
            g.play(m);
        }
    }
    (steps, if g.winner == me { 1.0 } else { -1.0 })
}

/// Train an opponent; `log` gets (iteration, score vs uniform random) at every snapshot.
pub fn train(cfg: &TrainCfg, log: &mut dyn FnMut(usize, f64)) -> Network {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let mut sizes = vec![INPUTS];
    sizes.extend(&cfg.hidden);
    sizes.push(P + 1);
    let mut params = Params::init(&sizes, &mut rng);
    let mut adam = Adam::new(&params);
    let mut pool: Vec<Mlp> = vec![];
    for it in 0..cfg.iters {
        let base = cfg.seed.wrapping_mul(1_000_003).wrapping_add(it as u64 * 100_003);
        let results = self_play(cfg, &params, &pool, base);
        let baseline = results.iter().map(|r| r.1).sum::<f32>() / results.len() as f32;
        let n_steps: usize = results.iter().map(|r| r.0.len()).sum::<usize>().max(1);
        let epochs = if cfg.algo == "ppo" { cfg.ppo_epochs.max(1) } else { 1 };
        for _ in 0..epochs {
            let grad = batch_gradient(cfg, &params, &results, baseline, n_steps);
            adam.ascend(&mut params, &grad, cfg.lr, (it * epochs + 1) as i32);
        }
        if cfg.snapshot_every > 0 && (it + 1) % cfg.snapshot_every == 0 {
            pool.push(params.to_mlp());
            let s = crate::play::summarise(&crate::play::play_match(
                &params.to_mlp(),
                &Uniform,
                64,
                base ^ 0xabcdef,
            ));
            log(it + 1, s.score);
        }
    }
    params.to_network()
}

/// One iteration's games: each against uniform random play, an earlier snapshot or the current policy, with
/// equal chance. Returns each game's learner moves and its result (+1 win, -1 loss).
fn self_play(cfg: &TrainCfg, params: &Params, pool: &[Mlp], base: u64) -> Vec<(Vec<Step>, f32)> {
    let current = params.to_mlp();
    (0..cfg.games_per_iter)
        .into_par_iter()
        .map(|gi| {
            let mut r = rand::rngs::StdRng::seed_from_u64(base + gi as u64);
            let k = r.gen_range(0..=pool.len() + 1);
            let opp: &dyn Policy = if k == 0 {
                &Uniform
            } else if k <= pool.len() {
                &pool[k - 1]
            } else {
                &current
            };
            rollout(params, opp, gi % 2 == 0, base + 7 + gi as u64 * 13)
        })
        .collect()
}

/// The policy gradient over a batch of games: REINFORCE with a batch-mean baseline, or PPO's clipped ratio.
fn batch_gradient(
    cfg: &TrainCfg,
    params: &Params,
    results: &[(Vec<Step>, f32)],
    baseline: f32,
    n_steps: usize,
) -> Params {
    results
        .par_iter()
        .fold(
            || params.zeros_like(),
            |mut g, (steps, z)| {
                let adv = z - baseline;
                for s in steps {
                    let coef = if cfg.algo == "ppo" {
                        let ratio = (params.logp(&s.x, &s.legal, s.a) - s.logp_old).exp();
                        if (adv > 0.0 && ratio > 1.0 + cfg.clip) || (adv < 0.0 && ratio < 1.0 - cfg.clip) {
                            0.0
                        } else {
                            adv * ratio
                        }
                    } else {
                        adv
                    };
                    params.grad_logp(
                        &s.x,
                        &s.legal,
                        s.a,
                        coef / n_steps as f32,
                        cfg.entropy / n_steps as f32,
                        &mut g,
                    );
                }
                g
            },
        )
        .reduce(
            || params.zeros_like(),
            |mut a, b| {
                a.add(&b);
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let p = Params::init(&[243, 8, 6, 82], &mut rng);
        let mut g = Game::new();
        g.play(40);
        g.play(41);
        let mut x = vec![0f32; INPUTS];
        g.encode(&mut x);
        let legal = g.legal();
        let a = 30;
        let logp = |q: &Params| {
            let z = q.to_mlp().forward(&x, None);
            let mx = (0..=P)
                .filter(|m| legal[*m])
                .map(|m| z[m])
                .fold(f32::NEG_INFINITY, f32::max);
            let s: f64 = (0..=P).filter(|m| legal[*m]).map(|m| ((z[m] - mx) as f64).exp()).sum();
            (z[a] - mx) as f64 - s.ln()
        };
        let mut grad = p.zeros_like();
        p.grad_logp(&x, &legal, a, 1.0, 0.0, &mut grad);
        for (li, idx) in [(0usize, 40 * 3 * 8 + 3), (1, 5), (2, 5 * 82 + 30), (2, 3 * 82 + 12)] {
            let mut q = p.clone();
            let h = 1e-3;
            q.layers[li].0[idx] += h;
            let up = logp(&q);
            q.layers[li].0[idx] -= 2.0 * h;
            let down = logp(&q);
            let fd = (up - down) / (2.0 * h as f64);
            let an = grad.layers[li].0[idx] as f64;
            assert!(
                (fd - an).abs() < 1e-2 * (1.0 + fd.abs()),
                "layer {li} idx {idx}: fd {fd} analytic {an}"
            );
        }
    }

    #[test]
    fn training_beats_random() {
        let cfg = TrainCfg {
            hidden: vec![32, 32],
            iters: 40,
            games_per_iter: 128,
            lr: 3e-3,
            entropy: 0.0,
            seed: 1,
            algo: "reinforce".into(),
            ppo_epochs: 1,
            clip: 0.2,
            snapshot_every: 10,
        };
        let mut last = 0.0;
        let net = train(&cfg, &mut |_, s| last = s);
        let m = Mlp::from_network(&net).unwrap();
        let s = crate::play::summarise(&crate::play::play_match(&m, &Uniform, 200, 99));
        assert!(s.score > 0.7, "score vs random {} (last logged {last})", s.score);
    }
}
