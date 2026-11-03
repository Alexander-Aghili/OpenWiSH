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
