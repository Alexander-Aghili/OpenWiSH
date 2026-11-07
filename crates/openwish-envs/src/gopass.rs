//! The editor's two Go pass tools (the authors' Experiment 6): a games pass against practice opponents
//! and a positions pass that runs the network on chosen positions.
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

use crate::go::{Game, INPUTS, N, P, PASS};
use crate::mlp::{Mlp, Policy, Uniform};
use crate::play::{legal_softmax, play_match, summarise, top_moves, GameRecord};
use openwish_tensor::{Network, Tensor};

pub const PRACTICE: &[&str] = &["random", "weak", "strong"];

/// A playing policy: uniform random play for "random", else the MLP network in the folder `path`.
pub fn policy_at(path: &Path) -> Result<Box<dyn Policy>> {
    if path.as_os_str() == "random" {
        return Ok(Box::new(Uniform));
    }
    let net = Network::load_dir(path).with_context(|| format!("opponent network at {}", path.display()))?;
    Ok(Box::new(Mlp::from_network(&net)?))
}

/// A practice opponent by name: "random", or a network folder under `dir`.
pub fn load_opponent(dir: &Path, name: &str) -> Result<Box<dyn Policy>> {
    if name == "random" {
        return policy_at(Path::new("random"));
    }
    policy_at(&dir.join(name))
}

const GAMES_README: &str = "summary.json: per opponent: score (wins / games), wins, losses, score as black / white, \
mean area margin (yours minus theirs, komi included), mean game length, stones captured and lost per game, your passes \
per game, games where you passed before move 60, and how the games ended.\n\
games.json: one entry per game: opponent, your colour, won, margin, end (two_passes | repetition | move_limit), length, \
moves (row * 9 + column; 81 = pass) in order, your_top5 (for each of your moves, the five most likely legal moves as \
[move, probability]), and boards (the board after every move as 81 characters, row by row: X = Black, O = White, . = \
empty).\n";

const POSITIONS_README: &str = "inputs.npy (n x 243), logits.npy (n x 82), probs.npy (n x 82, softmax over the legal \
moves), legal.npy (n x 82, 1 = legal), and with activations hidden_k.npy (n x width): the output of hidden layer k \
after ReLU.\n";

/// At most this many games in one games pass, and positions in one positions pass.
const MAX_GAMES: usize = 128;
const MAX_POSITIONS: usize = 256;

/// {"games": {"strong": 64, "weak": 32, "random": 32}}: even counts, at most 128 in total.
pub fn games_pass(net: &Network, req: &Value, out: &Path, opponents: &Path, seed: u64) -> Result<Value> {
    let plan = games_plan(req)?;
    let me = Mlp::from_network(net)?;
    std::fs::create_dir_all(out)?;
    let mut summary = serde_json::Map::new();
    let mut games = vec![];
    for (k, (name, n)) in plan.iter().enumerate() {
        let opp = load_opponent(opponents, name)?;
        let recs = play_match(&me, opp.as_ref(), *n, seed.wrapping_mul(31).wrapping_add(k as u64));
        summary.insert(name.clone(), serde_json::to_value(summarise(&recs))?);
        games.extend(recs.iter().map(|r| game_json(name, r)));
    }
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(out.join("games.json"), serde_json::to_string(&games)?)?;
    std::fs::write(out.join("README.txt"), GAMES_README)?;
    let mut v = Value::Object(summary);
    v["ok"] = true.into();
    Ok(v)
}

/// (opponent, games) for every opponent with a non-zero count, checked against the pass rules.
fn games_plan(req: &Value) -> Result<Vec<(String, usize)>> {
    let g = req
        .get("games")
        .and_then(Value::as_object)
        .context("the request must be {\"games\": {\"strong\": 64, \"weak\": 32, \"random\": 32}}")?;
    let mut plan = vec![];
    let mut total = 0;
    for (name, n) in g {
        if !PRACTICE.contains(&name.as_str()) {
            bail!("unknown practice opponent {name:?} (use {PRACTICE:?})");
        }
        let n = n
            .as_u64()
            .with_context(|| format!("{name}: the count must be a whole number"))? as usize;
        if !n.is_multiple_of(2) {
            bail!("{name}: the number of games must be even (your network is Black in half)");
        }
        if n > 0 {
            plan.push((name.clone(), n));
        }
        total += n;
    }
    if total == 0 || total > MAX_GAMES {
        bail!("between 2 and {MAX_GAMES} games in total, got {total}");
    }
    Ok(plan)
}

/// One game of games.json, with the board after every move.
fn game_json(opponent: &str, r: &GameRecord) -> Value {
    let mut g = Game::new();
    let boards: Vec<String> = r
        .moves
        .iter()
        .map(|m| {
            g.play(*m as usize);
            g.rows().concat()
        })
        .collect();
    json!({
        "opponent": opponent, "your_colour": if r.focus_black { "black" } else { "white" }, "won": r.focus_won,
        "margin": r.margin, "end": r.end, "length": r.length, "moves": r.moves,
        "your_top5": r.focus_top5, "boards": boards,
    })
}

/// The arrays a positions pass writes, one row per position.
#[derive(Default)]
struct PositionArrays {
    inputs: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    legal: Vec<f32>,
    /// one array per hidden layer
    hidden: Vec<Vec<f32>>,
}

/// {"positions": [{"board": [...9 strings], "to_move": "black"|"white"} | {"moves": [...]}], "activations": bool}
pub fn positions_pass(net: &Network, req: &Value, out: &Path) -> Result<Value> {
    let ps = req
        .get("positions")
        .and_then(Value::as_array)
        .context("the request must be {\"positions\": [...], \"activations\": false}")?;
    if ps.is_empty() || ps.len() > MAX_POSITIONS {
        bail!("between 1 and {MAX_POSITIONS} positions, got {}", ps.len());
    }
    let with_hidden = req.get("activations").and_then(Value::as_bool).unwrap_or(false);
    let me = Mlp::from_network(net)?;
    let mut arrays = PositionArrays {
        hidden: vec![vec![]; me.layers.len() - 1],
        ..Default::default()
    };
    let mut top3 = vec![];
    for (i, p) in ps.iter().enumerate() {
        let (g, legal) = parse_position(i, p)?;
        let mut x = vec![0f32; INPUTS];
        g.encode(&mut x);
        let mut hidden = vec![];
        let z = me.forward(&x, with_hidden.then_some(&mut hidden));
        let pr = legal_softmax(&z, &legal);
        let top = top_moves(&pr, &legal, 3);
        top3.push(
            top.iter()
                .map(|m| json!({"move": move_name(*m), "prob": pr[*m], "logit": z[*m]}))
                .collect::<Vec<_>>(),
        );
        arrays.inputs.extend(x);
        arrays.logits.extend(z);
        arrays.probs.extend(pr);
        arrays.legal.extend(legal.iter().map(|b| if *b { 1.0 } else { 0.0 }));
        for (k, h) in hidden.into_iter().enumerate() {
            arrays.hidden[k].extend(h);
        }
    }
    write_positions(out, ps.len(), arrays, with_hidden)?;
    Ok(json!({"ok": true, "positions": ps.len(), "top3": top3}))
}

/// Position `i` of a request and its legal moves: a board and the colour to move (every empty point is legal),
/// or a list of moves replayed from the empty board.
fn parse_position(i: usize, p: &Value) -> Result<(Game, [bool; P + 1])> {
    if let Some(rows) = p.get("board") {
        let rows: Vec<String> =
            serde_json::from_value(rows.clone()).with_context(|| format!("position {i}: board must be 9 strings"))?;
        if rows.len() != N
            || rows
                .iter()
                .any(|r| r.chars().count() != N || r.chars().any(|c| !"XO.".contains(c)))
        {
            bail!("position {i}: the board must be 9 strings of 9 characters from X, O and .");
        }
        let to_move = match p.get("to_move").and_then(Value::as_str) {
            Some("black") => 1,
            Some("white") => -1,
            _ => bail!("position {i}: to_move must be \"black\" or \"white\""),
        };
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let g = Game::from_strings(&refs, to_move);
        let mut legal = [false; P + 1];
        for (l, point) in legal.iter_mut().zip(&g.board) {
            *l = *point == 0;
        }
        legal[PASS] = true;
        Ok((g, legal))
    } else if let Some(ms) = p.get("moves").and_then(Value::as_array) {
        let g = replay(i, ms)?;
        let legal = g.legal();
        Ok((g, legal))
    } else {
        bail!("position {i}: give either \"board\" and \"to_move\", or \"moves\"");
    }
}

/// The game after `moves` (each a point number or "pass"), every move checked.
fn replay(i: usize, moves: &[Value]) -> Result<Game> {
    let mut g = Game::new();
    for (j, m) in moves.iter().enumerate() {
        let mv = match m {
            Value::String(s) if s == "pass" => PASS,
            Value::Number(k) if k.as_u64().is_some_and(|k| (k as usize) < P) => k.as_u64().unwrap() as usize,
            _ => bail!("position {i}: move {j} must be row * 9 + column or \"pass\""),
        };
        if g.end.is_some() {
            bail!("position {i}: the game is already over before move {j}");
        }
        if !g.legal()[mv] {
            bail!("position {i}: move {j} ({mv}) is illegal");
        }
        g.play(mv);
    }
    Ok(g)
}

fn write_positions(out: &Path, n: usize, a: PositionArrays, with_hidden: bool) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let w = |name: &str, shape: Vec<usize>, data: Vec<f32>| {
        std::fs::write(
            out.join(name),
            openwish_tensor::npy::write_npy(&Tensor::new(shape, data)),
        )
    };
    w("inputs.npy", vec![n, INPUTS], a.inputs)?;
    w("logits.npy", vec![n, P + 1], a.logits)?;
    w("probs.npy", vec![n, P + 1], a.probs)?;
    w("legal.npy", vec![n, P + 1], a.legal)?;
    if with_hidden {
        for (k, h) in a.hidden.into_iter().enumerate() {
            let width = h.len() / n;
            w(&format!("hidden_{}.npy", k + 1), vec![n, width], h)?;
        }
    }
    std::fs::write(out.join("README.txt"), POSITIONS_README)?;
    Ok(())
}

pub fn move_name(m: usize) -> String {
    if m == PASS {
        "pass".into()
    } else {
        format!("{}", m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_net() -> Network {
        let mut n = Network::default();
        n.tensors.insert("L1.W".into(), Tensor::zeros(vec![243, 4]));
        n.tensors.insert("L1.b".into(), Tensor::zeros(vec![4]));
        n.tensors.insert("out.W".into(), Tensor::zeros(vec![4, 82]));
        n.tensors.insert("out.b".into(), Tensor::zeros(vec![82]));
        n
    }

    fn opponents(d: &Path) -> std::path::PathBuf {
        let o = d.join("opp");
        for name in ["weak", "strong"] {
            zero_net().save_dir(&o.join(name)).unwrap();
        }
        o
    }

    #[test]
    fn games_pass_validates_and_writes_files() {
        let d = tempfile::tempdir().unwrap();
        let o = opponents(d.path());
        let net = zero_net();
        for bad in [
            json!({"games": {"strong": 3}}),
            json!({"games": {"strong": 64, "weak": 64, "random": 2}}),
            json!({"games": {"nobody": 2}}),
            json!({"nope": 1}),
            json!({"games": {}}),
        ] {
            assert!(games_pass(&net, &bad, d.path(), &o, 1).is_err(), "{bad}");
        }
        let out = d.path().join("pass_1");
        let s = games_pass(&net, &json!({"games": {"strong": 4, "random": 2}}), &out, &o, 1).unwrap();
        assert_eq!(s["ok"], true);
        assert_eq!(s["strong"]["games"], 4);
        assert_eq!(s["random"]["games"], 2);
        assert!(s.get("weak").is_none());
        for f in ["summary.json", "games.json", "README.txt"] {
            assert!(out.join(f).exists(), "{f}");
        }
        let g: Value = serde_json::from_str(&std::fs::read_to_string(out.join("games.json")).unwrap()).unwrap();
        assert_eq!(g.as_array().unwrap().len(), 6);
        assert!(g[0]["boards"].as_array().unwrap().len() == g[0]["moves"].as_array().unwrap().len());
    }

    #[test]
    fn positions_pass_boards_moves_and_activations() {
        let d = tempfile::tempdir().unwrap();
        let mut net = zero_net();
        net.tensors.get_mut("out.b").unwrap().data[40] = 5.0; // the network likes the centre
        let empty = vec![".........".to_string(); 9];
        let mut b = empty.clone();
        b[4] = "....X....".into();
        let req = json!({"positions": [{"board": empty, "to_move": "black"}, {"board": b, "to_move": "white"}, {"moves": [0, "pass", 40]}], "activations": true});
        let s = positions_pass(&net, &req, d.path()).unwrap();
        let top = &s["top3"];
        assert_eq!(top[0][0]["move"], "40");
        assert_ne!(top[1][0]["move"], "40", "occupied point is not legal");
        assert_ne!(top[2][0]["move"], "40");
        for f in ["inputs.npy", "logits.npy", "probs.npy", "legal.npy", "hidden_1.npy"] {
            assert!(d.path().join(f).exists(), "{f}");
        }
        let x = openwish_tensor::npy::read_npy(&std::fs::read(d.path().join("inputs.npy")).unwrap()).unwrap();
        assert_eq!(x.shape, vec![3, 243]);
        assert!(
            positions_pass(&net, &json!({"positions": [{"moves": [0, 0]}]}), d.path()).is_err(),
            "illegal replay"
        );
        let many: Vec<Value> = (0..257).map(|_| json!({"moves": []})).collect();
        assert!(positions_pass(&net, &json!({"positions": many}), d.path()).is_err());
    }
}
