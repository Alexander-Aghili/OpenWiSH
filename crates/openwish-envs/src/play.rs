//! Playing games between policies, with the authors' per-opponent summary.
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::go::{End, Game, INPUTS, P, PASS};
use crate::mlp::Policy;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameRecord {
    pub focus_black: bool,
    pub focus_won: bool,
    /// focus area minus opponent area (komi included)
    pub margin: f32,
    pub end: End,
    pub moves: Vec<u16>,
    pub length: u32,
    pub captured: u32,
    pub lost: u32,
    pub focus_passes: u32,
    pub first_focus_pass: Option<u32>,
    /// for each focus move: the top-5 (move, probability) over legal moves
    pub focus_top5: Vec<Vec<(u16, f32)>>,
    pub final_board: Vec<String>,
}

/// The largest logit among the legal moves.
pub fn max_legal(logits: &[f32], legal: &[bool; P + 1]) -> f32 {
    (0..=P)
        .filter(|m| legal[*m])
        .map(|m| logits[m])
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Softmax of `z` over the legal moves (0 for the others), in f32.
pub fn legal_softmax(z: &[f32], legal: &[bool; P + 1]) -> Vec<f32> {
    let mx = max_legal(z, legal);
    let mut pr: Vec<f32> = (0..=P)
        .map(|m| if legal[m] { (z[m] - mx).exp() } else { 0.0 })
        .collect();
    let s: f32 = pr.iter().sum();
    pr.iter_mut().for_each(|v| *v /= s);
    pr
}

/// The `k` legal moves with the highest probability, most likely first.
pub fn top_moves(probs: &[f32], legal: &[bool; P + 1], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..=P).filter(|m| legal[*m]).collect();
    idx.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
    idx.truncate(k);
    idx
}

/// Sample a move from softmax(logits) over legal moves. Returns (move, probabilities over all 82).
pub fn sample(logits: &[f32], legal: &[bool; P + 1], rng: &mut impl Rng) -> (usize, Vec<f32>) {
    let mx = max_legal(logits, legal);
    let mut p = vec![0f32; P + 1];
    let mut z = 0f64;
    for m in (0..=P).filter(|m| legal[*m]) {
        let e = ((logits[m] - mx) as f64).exp();
        p[m] = e as f32;
        z += e;
    }
    let mut u = rng.gen::<f64>() * z;
    let mut pick = PASS;
    for m in (0..=P).filter(|m| legal[*m]) {
        u -= p[m] as f64;
        if u <= 0.0 {
            pick = m;
            break;
        }
    }
    for v in p.iter_mut() {
        *v = (*v as f64 / z) as f32;
    }
    (pick, p)
}

pub fn play_game(focus: &dyn Policy, other: &dyn Policy, focus_black: bool, seed: u64) -> GameRecord {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut g = Game::new();
    let focus_colour: i8 = if focus_black { 1 } else { -1 };
    let mut x = vec![0f32; INPUTS];
    let mut moves = vec![];
    let mut top5 = vec![];
    let (mut passes, mut first_pass) = (0, None);
    while g.end.is_none() {
        g.encode(&mut x);
        let legal = g.legal();
        let mine = g.to_move == focus_colour;
        let logits = if mine { focus.logits(&x) } else { other.logits(&x) };
        let (m, probs) = sample(&logits, &legal, &mut rng);
        if mine {
            top5.push(
                top_moves(&probs, &legal, 5)
                    .into_iter()
                    .map(|i| (i as u16, probs[i]))
                    .collect(),
            );
            if m == PASS {
                passes += 1;
                if first_pass.is_none() {
                    first_pass = Some(g.moves);
                }
            }
        }
        moves.push(m as u16);
        g.play(m);
    }
    let (b, w) = g.area();
    let margin = if focus_black { b - w } else { w - b };
    let (cap, lost) = if focus_black {
        (g.captured[0], g.captured[1])
    } else {
        (g.captured[1], g.captured[0])
    };
    GameRecord {
        focus_black,
        focus_won: g.winner == focus_colour,
        margin,
        end: g.end.unwrap(),
        length: g.moves,
        moves,
        captured: cap,
        lost,
        focus_passes: passes,
        first_focus_pass: first_pass,
        focus_top5: top5,
        final_board: g.rows(),
    }
}

/// `games` games (even), the focus is Black in half; parallel over games; deterministic in `seed`.
pub fn play_match(focus: &dyn Policy, other: &dyn Policy, games: usize, seed: u64) -> Vec<GameRecord> {
    (0..games)
        .into_par_iter()
        .map(|i| {
            play_game(
                focus,
                other,
                i % 2 == 0,
                seed.wrapping_mul(1_000_003).wrapping_add(i as u64),
            )
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub games: usize,
    pub score: f64,
    pub wins: usize,
    pub losses: usize,
    pub score_as_black: f64,
    pub score_as_white: f64,
    pub mean_margin: f64,
    pub mean_length: f64,
    pub stones_captured_per_game: f64,
    pub stones_lost_per_game: f64,
    pub passes_per_game: f64,
    pub games_with_a_pass_before_move_60: usize,
    pub end_two_passes: usize,
    pub end_repetition: usize,
    pub end_move_limit: usize,
}

pub fn summarise(r: &[GameRecord]) -> Summary {
    let n = r.len().max(1) as f64;
    let rate = |b: bool| {
        let s: Vec<&GameRecord> = r.iter().filter(|g| g.focus_black == b).collect();
        if s.is_empty() {
            0.0
        } else {
            s.iter().filter(|g| g.focus_won).count() as f64 / s.len() as f64
        }
    };
    let wins = r.iter().filter(|g| g.focus_won).count();
    Summary {
        games: r.len(),
        score: wins as f64 / n,
        wins,
        losses: r.len() - wins,
        score_as_black: rate(true),
        score_as_white: rate(false),
        mean_margin: r.iter().map(|g| g.margin as f64).sum::<f64>() / n,
        mean_length: r.iter().map(|g| g.length as f64).sum::<f64>() / n,
        stones_captured_per_game: r.iter().map(|g| g.captured as f64).sum::<f64>() / n,
        stones_lost_per_game: r.iter().map(|g| g.lost as f64).sum::<f64>() / n,
        passes_per_game: r.iter().map(|g| g.focus_passes as f64).sum::<f64>() / n,
        games_with_a_pass_before_move_60: r.iter().filter(|g| g.first_focus_pass.is_some_and(|m| m < 60)).count(),
        end_two_passes: r.iter().filter(|g| g.end == End::TwoPasses).count(),
        end_repetition: r.iter().filter(|g| g.end == End::Repetition).count(),
        end_move_limit: r.iter().filter(|g| g.end == End::MoveLimit).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::{Mlp, Uniform};
    use std::path::Path;

    #[test]
    fn sampling_respects_the_mask_and_is_deterministic() {
        let mut legal = [false; P + 1];
        legal[3] = true;
        legal[PASS] = true;
        let mut logits = vec![100.0; P + 1];
        logits[3] = 0.0;
        logits[PASS] = 0.0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..50 {
            let (m, p) = sample(&logits, &legal, &mut rng);
            assert!(m == 3 || m == PASS);
            assert!((p[3] - 0.5).abs() < 1e-6 && p[0] == 0.0);
        }
        let a: Vec<_> = play_match(&Uniform, &Uniform, 4, 7)
            .iter()
            .map(|g| g.moves.clone())
            .collect();
        let b: Vec<_> = play_match(&Uniform, &Uniform, 4, 7)
            .iter()
            .map(|g| g.moves.clone())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn summary_counts() {
        let r = play_match(&Uniform, &Uniform, 20, 3);
        let s = summarise(&r);
        assert_eq!(s.games, 20);
        assert_eq!(s.wins + s.losses, 20);
        assert_eq!(s.end_two_passes + s.end_repetition + s.end_move_limit, 20);
        assert!(r.iter().filter(|g| g.focus_black).count() == 10);
        assert!(r.iter().all(|g| !g.focus_top5.is_empty() && g.final_board.len() == 9));
    }

    fn authors(name: &str) -> Option<Mlp> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/lm_understand-logs/exp6_go_from_scratch/networks")
            .join(name);
        if !p.exists() {
            return None;
        }
        Some(Mlp::from_network(&openwish_tensor::Network::load_dir(&p).unwrap()).unwrap())
    }

    #[test]
    fn authors_network_beats_random() {
        let Some(b50) = authors("budget_50") else { return };
        let s = summarise(&play_match(&b50, &Uniform, 40, 1));
        assert!(s.score >= 0.95, "{s:?}");
    }

    /// The engine reproduces the authors' head-to-head (2,000 games): budget_50 vs same-size PPO seed 0 = 74.2 %.
    /// cargo test -p openwish-envs --release -- --ignored
    #[test]
    #[ignore]
    fn authors_head_to_head_reproduces() {
        let (Some(b50), Some(ppo)) = (authors("budget_50"), authors("ppo_same_size_lr3e-5_s0")) else {
            return;
        };
        let s = summarise(&play_match(&b50, &ppo, 2000, 12345));
        eprintln!("{s:?}");
        assert!((s.score - 0.7415).abs() < 0.04, "{}", s.score);
    }
}
