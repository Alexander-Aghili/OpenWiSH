//! Host commands that answer with a JSON summary: forward probes and scorers. A command gets its request and
//! folders through environment variables, writes files where it is told, and prints the summary
//! (`{"ok": true, ...}`) as the last JSON line of stdout.
use anyhow::{bail, Context, Result};
use openwish_process::{last_json_line, Options};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Characters of stderr quoted when a command prints no summary.
const STDERR_TAIL: usize = 3000;

pub fn run_command(argv: &[String], envs: &[(&str, String)], cwd: &Path, timeout_s: u64) -> Result<serde_json::Value> {
    let (prog, args) = argv.split_first().context("empty command")?;
    let mut cmd = Command::new(prog);
    cmd.args(args).current_dir(cwd).envs(envs.iter().map(|(k, v)| (k, v)));
    let done = openwish_process::run(
        &mut cmd,
        Options {
            timeout: Duration::from_secs(timeout_s),
            ..Default::default()
        },
    )?;
    let Some(status) = done.status else {
        bail!("the pass timed out after {timeout_s} s");
    };
    match last_json_line(&done.stdout).and_then(|l| serde_json::from_str(l).ok()) {
        Some(v) => Ok(v),
        None => bail!(
            "the pass command exited with {status} and printed no JSON summary. stderr: {}",
            done.stderr_tail(STDERR_TAIL)
        ),
    }
}

/// The scorer's time limit.
pub const SCORE_TIMEOUT_S: u64 = 6 * 3600;

/// Run the task's scorer on a finished run folder, with the scorer variables, and save its JSON line to
/// `eval/score.json`. `root` is the workspace (OPENWISH_ROOT).
pub fn score_run(run: &Path, root: &Path) -> Result<serde_json::Value> {
    let spec = crate::run::saved_spec(run)?;
    let Some(sc) = &spec.score else {
        bail!("task {} has no scorer", spec.id);
    };
    let run = std::fs::canonicalize(run)?;
    let envs = [
        ("OPENWISH_RUN", run.display().to_string()),
        ("OPENWISH_TASK_DIR", spec.dir.display().to_string()),
        ("OPENWISH_FINAL_NET", run.join("final/net").display().to_string()),
        (
            "OPENWISH_FINAL_ANSWER",
            run.join("final_answer.md").display().to_string(),
        ),
        ("OPENWISH_ROOT", root.display().to_string()),
    ];
    let mut v = run_command(&spec.argv(&sc.command), &envs, &spec.dir, SCORE_TIMEOUT_S)?;
    // every score names the access it was earned under (scores from different access levels are not comparable)
    if let Some(o) = v.as_object_mut() {
        o.insert("access".into(), spec.access());
    }
    std::fs::create_dir_all(run.join("eval"))?;
    std::fs::write(run.join("eval/score.json"), serde_json::to_string_pretty(&v)?)?;
    Ok(v)
}
