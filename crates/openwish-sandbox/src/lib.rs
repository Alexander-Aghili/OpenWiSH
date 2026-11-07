//! A persistent Python kernel inside bubblewrap: no network, no GPU, read-only system and weights,
//! one read-write work folder. Weight changes come back to the harness as RPCs.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod bwrap;
mod rpc;
pub use rpc::{Rpc, RpcReply};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mount {
    pub host: PathBuf,
    pub guest: String,
    #[serde(default)]
    pub rw: bool,
}

#[derive(Clone, Debug)]
pub struct KernelCfg {
    /// the venv directory (its bin/python is used)
    pub venv: PathBuf,
    /// the directory holding kernel.py
    pub kernel_dir: PathBuf,
    /// private per-run directory for the prelude (mounted read-only at /openwish)
    pub private_dir: PathBuf,
    /// the gate's materialised current network (mounted read-only at /net)
    pub net_dir: Option<PathBuf>,
    /// read-write work folder (mounted at /work)
    pub work_dir: PathBuf,
    pub mounts: Vec<Mount>,
    pub timeout_s: u64,
    pub mem_bytes: u64,
    pub max_out: usize,
    pub prelude: Option<String>,
    pub threads: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunOutput {
    pub text: String,
    pub error: bool,
    pub restarted: bool,
    pub seconds: f64,
}

pub struct Kernel {
    cfg: KernelCfg,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    /// the kernel's stdout, line by line: the protocol channel
    lines: Option<Receiver<String>>,
    stderr_tail: Arc<Mutex<String>>,
}

/// How long the kernel may take to start (torch import can take a while).
const READY_TIMEOUT: Duration = Duration::from_secs(120);
/// The stderr tail is trimmed back to `STDERR_KEEP` bytes when it grows past `STDERR_MAX`.
const STDERR_MAX: usize = 8000;
const STDERR_KEEP: usize = 4000;

impl Kernel {
    pub fn start(cfg: KernelCfg) -> Result<Kernel> {
        let mut k = Kernel {
            cfg,
            child: None,
            stdin: None,
            lines: None,
            stderr_tail: Default::default(),
        };
        k.spawn()?;
        Ok(k)
    }

    /// Run `code`; edit RPCs from it are answered by `rpc`. A kernel that died is restarted (its variables
    /// are lost, and the output says so).
    pub fn run(&mut self, code: &str, rpc: &mut dyn FnMut(Rpc) -> RpcReply) -> Result<RunOutput> {
        let request = serde_json::json!({"type": "run", "code": code});
        // a kernel that stopped since the last run (it exited, or was killed) is restarted first
        let mut restarted = false;
        if !self.alive() {
            restarted = self.child.is_some();
            self.restart()?;
        }
        if self.send(&request).is_err() {
            restarted = true;
            self.restart()?;
            self.send(&request)?;
        }
        let mut out = self.collect(rpc)?;
        if restarted {
            out.text = format!(
                "[the kernel had stopped since your last run; it was restarted and its variables are lost. Your saved \
                 edits are kept.]\n{}",
                out.text
            );
            out.restarted = true;
        }
        Ok(out)
    }

    fn spawn(&mut self) -> Result<()> {
        self.stage_files()?;
        let mut child = self
            .command()?
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting bwrap")?;
        self.lines = Some(read_lines(child.stdout.take().expect("stdout is piped")));
        keep_tail(child.stderr.take().expect("stderr is piped"), self.stderr_tail.clone());
        self.stdin = child.stdin.take();
        self.child = Some(child);
        self.wait_ready()
    }

    fn restart(&mut self) -> Result<()> {
        self.kill();
        self.spawn()
    }

    /// The work folder, and the private folder with kernel.py and the task's prelude.
    fn stage_files(&self) -> Result<()> {
        let c = &self.cfg;
        std::fs::create_dir_all(&c.work_dir)?;
        std::fs::create_dir_all(&c.private_dir)?;
        std::fs::copy(c.kernel_dir.join("kernel.py"), c.private_dir.join("kernel.py")).context("copying kernel.py")?;
        std::fs::write(c.private_dir.join("prelude.py"), c.prelude.clone().unwrap_or_default())?;
        Ok(())
    }

    /// Wait for the kernel's "ready" line; a failing prelude is an error.
    fn wait_ready(&mut self) -> Result<()> {
        let t0 = Instant::now();
        loop {
            match self.lines.as_ref().expect("kernel started").recv_timeout(READY_TIMEOUT) {
                Ok(l) => {
                    let v: serde_json::Value = serde_json::from_str(&l).unwrap_or_default();
                    match v["type"].as_str() {
                        Some("ready") => return Ok(()),
                        Some("done") if v["error"] == true => {
                            bail!("kernel prelude failed:\n{}", v["output"].as_str().unwrap_or(""))
                        }
                        _ => {}
                    }
                }
                Err(_) => {
                    let e = self.stderr_tail.lock().unwrap().clone();
                    self.kill();
                    bail!("kernel did not start within {:?}: {e}", t0.elapsed());
                }
            }
        }
    }

    fn kill(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.stdin = None;
        self.lines = None;
    }

    fn send(&mut self, v: &serde_json::Value) -> Result<()> {
        let s = self.stdin.as_mut().context("kernel not running")?;
        writeln!(s, "{v}")?;
        s.flush()?;
        Ok(())
    }

    fn alive(&mut self) -> bool {
        match self.child.as_mut() {
            Some(c) => matches!(c.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// Read protocol lines until the run is done, answering RPCs on the way.
    fn collect(&mut self, rpc: &mut dyn FnMut(Rpc) -> RpcReply) -> Result<RunOutput> {
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(self.cfg.timeout_s);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let line = match self.lines.as_ref().expect("kernel started").recv_timeout(left) {
                Ok(l) => l,
                Err(e) => return self.lost_run(e, t0),
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            match v["type"].as_str() {
                Some("done") => {
                    return Ok(RunOutput {
                        text: openwish_text::clip(v["output"].as_str().unwrap_or(""), self.cfg.max_out, "output"),
                        error: v["error"].as_bool().unwrap_or(false),
                        restarted: false,
                        seconds: t0.elapsed().as_secs_f64(),
                    })
                }
                Some("rpc") => {
                    let r = rpc(self.rpc_request(&v));
                    self.send(&serde_json::json!({"type": "rpc_reply", "ok": r.ok, "msg": r.msg}))?;
                }
                _ => {}
            }
        }
    }

    /// The run timed out or the kernel died: restart it and report the run as failed.
    fn lost_run(&mut self, e: RecvTimeoutError, t0: Instant) -> Result<RunOutput> {
        let why = match e {
            RecvTimeoutError::Timeout => format!("timed out after {} s", self.cfg.timeout_s),
            RecvTimeoutError::Disconnected => {
                let tail = self.stderr_tail.lock().unwrap().clone();
                let tail: String = tail
                    .chars()
                    .rev()
                    .take(1000)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                format!(
                    "the kernel process died (likely out of memory, limit {} GB) {tail}",
                    self.cfg.mem_bytes >> 30
                )
            }
        };
        self.restart()?;
        Ok(RunOutput {
            text: format!("[{why}; the kernel was restarted and its variables are lost. Your saved edits are kept.]"),
            error: true,
            restarted: true,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        self.kill();
    }
}

/// The lines of the kernel's stdout, read on their own thread.
fn read_lines(stdout: ChildStdout) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for l in BufReader::new(stdout).lines() {
            let Ok(l) = l else { break };
            if tx.send(l).is_err() {
                break;
            }
        }
    });
    rx
}

/// Keep the last few thousand bytes of the kernel's stderr in `tail`, read on their own thread.
fn keep_tail(mut stderr: ChildStderr, tail: Arc<Mutex<String>>) {
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            let mut t = tail.lock().unwrap();
            t.push_str(&String::from_utf8_lossy(&buf[..n]));
            if t.len() > STDERR_MAX {
                let mut start = t.len() - STDERR_KEEP;
                while !t.is_char_boundary(start) {
                    start += 1;
                }
                *t = t[start..].to_string();
            }
        }
    });
}

#[cfg(test)]
mod tests;
