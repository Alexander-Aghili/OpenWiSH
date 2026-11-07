//! `openwish go ...`: the 9x9 Go tools of Experiment 6 (pass servers, evaluation, opponent training).
use anyhow::{Context, Result};
use clap::Subcommand;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use openwish_envs::mlp::{Mlp, Policy, Uniform};
use openwish_envs::{gopass, play, train};
use openwish_tensor::Network;

#[derive(Subcommand)]
pub enum GoCmd {
    /// games pass (reads $OPENWISH_REQ, $OPENWISH_NET, $OPENWISH_OUT)
    Games {
        #[arg(long)]
        opponents: PathBuf,
    },
    /// positions pass (reads $OPENWISH_REQ, $OPENWISH_NET, $OPENWISH_OUT)
    Positions,
    /// play a network against opponents; prints one JSON line
    Eval(EvalArgs),
    /// train a small opponent (harness-side; not available to the editor)
    Train(TrainArgs),
}

#[derive(clap::Args)]
pub struct EvalArgs {
    /// the network (default $OPENWISH_FINAL_NET)
    #[arg(long)]
    net: Option<PathBuf>,
    /// opponent directories (or "random"), as name=path or path
    #[arg(long, num_args = 1..)]
    against: Vec<String>,
    /// opponents whose mean score is the main score
    #[arg(long, num_args = 1..)]
    main: Vec<String>,
    #[arg(long, default_value_t = 2000)]
    games: usize,
    #[arg(long, default_value_t = 20261102)]
    seed: u64,
}

#[derive(clap::Args)]
pub struct TrainArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "32,32")]
    hidden: String,
    #[arg(long, default_value_t = 300)]
    iters: usize,
    #[arg(long, default_value_t = 512)]
    games: usize,
    #[arg(long, default_value = "ppo")]
    algo: String,
    #[arg(long, default_value_t = 1e-3)]
    lr: f32,
    #[arg(long, default_value_t = 0.003)]
    entropy: f32,
    #[arg(long, default_value_t = 20)]
    snapshot_every: usize,
}

pub fn run(cmd: GoCmd) -> Result<()> {
    match cmd {
        GoCmd::Games { opponents } => {
            let pass: u64 = std::env::var("OPENWISH_PASS")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(0);
            let seed = pass * 7919 + 17;
            serve_pass(|n, r, o| gopass::games_pass(n, r, o, &opponents, seed))
        }
        GoCmd::Positions => serve_pass(gopass::positions_pass),
        GoCmd::Eval(a) => eval(a),
        GoCmd::Train(a) => train_opponent(a),
    }
}

/// Answer one pass request (the probe contract): the request, the network and the output folder come from
/// the environment, and the summary, or the error, is printed as one JSON line.
fn serve_pass(pass: impl FnOnce(&Network, &Value, &Path) -> Result<Value>) -> Result<()> {
    let answer = || -> Result<Value> {
        let req: Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("OPENWISH_REQ")?)?)
            .context("the request is not valid JSON")?;
        let net = Network::load_dir(Path::new(&std::env::var("OPENWISH_NET")?))?;
        pass(&net, &req, Path::new(&std::env::var("OPENWISH_OUT")?))
    };
    let v = answer().unwrap_or_else(|e| json!({"ok": false, "error": format!("{e:#}")}));
    println!("{v}");
    Ok(())
}

fn load_mlp(dir: &Path) -> Result<Mlp> {
    Mlp::from_network(&Network::load_dir(dir)?)
}

/// An `--against` entry: `name=path` or `path` (named after its folder); the path "random" is uniform play.
fn opponent(spec: &str) -> Result<(String, Box<dyn Policy>)> {
    let (name, path) = match spec.split_once('=') {
        Some((n, p)) => (n.to_string(), p),
        None => (
            Path::new(spec)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| spec.into()),
            spec,
        ),
    };
    Ok((name, gopass::policy_at(Path::new(path))?))
}

fn eval(a: EvalArgs) -> Result<()> {
    let net = a
        .net
        .or_else(|| std::env::var("OPENWISH_FINAL_NET").ok().map(PathBuf::from))
        .context("give --net")?;
    let me = load_mlp(&net)?;
    let mut out = serde_json::Map::new();
    for (k, spec) in a.against.iter().enumerate() {
        let (name, opp) = opponent(spec)?;
        let games = play::play_match(
            &me,
            opp.as_ref(),
            a.games,
            a.seed.wrapping_add(k as u64 * 1_000_000_007),
        );
        out.insert(name, serde_json::to_value(play::summarise(&games))?);
    }
    let scores: Vec<f64> = a
        .main
        .iter()
        .filter_map(|m| out.get(m).and_then(|v| v["score"].as_f64()))
        .collect();
    let mut v = Value::Object(out);
    v["ok"] = true.into();
    if !scores.is_empty() {
        v["score"] = (scores.iter().sum::<f64>() / scores.len() as f64).into();
        v["main"] = serde_json::to_value(&a.main)?;
    }
    println!("{v}");
    Ok(())
}

fn train_opponent(a: TrainArgs) -> Result<()> {
    let cfg = train::TrainCfg {
        hidden: a
            .hidden
            .split(',')
            .map(|h| h.trim().parse())
            .collect::<std::result::Result<_, _>>()?,
        iters: a.iters,
        games_per_iter: a.games,
        lr: a.lr,
        entropy: a.entropy,
        seed: a.seed,
        algo: a.algo,
        ppo_epochs: 4,
        clip: 0.2,
        snapshot_every: a.snapshot_every,
    };
    let mut log = vec![];
    let net = train::train(&cfg, &mut |it, s| {
        eprintln!("iter {it:>5}: score vs random {s:.3}");
        log.push((it, s));
    });
    net.save_dir(&a.out)?;
    let s = play::summarise(&play::play_match(
        &Mlp::from_network(&net)?,
        &Uniform,
        1000,
        a.seed ^ 0x5eed,
    ));
    let record = json!({"cfg": cfg, "log": log, "vs_random_1000": s});
    std::fs::write(a.out.join("train.json"), serde_json::to_string_pretty(&record)?)?;
    println!("{}", json!({"ok": true, "vs_random": s.score}));
    Ok(())
}
