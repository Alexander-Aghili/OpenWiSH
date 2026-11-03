//! A persistent Python kernel inside bubblewrap: no network, no GPU, read-only system and weights,
//! one read-write work folder. Weight changes come back to the harness as RPCs.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

#[derive(Clone, Debug, PartialEq)]
pub enum Rpc {
    /// `path` is the host path of the .npy the kernel wrote
    SaveEdit {
        name: String,
        path: PathBuf,
        reason: String,
    },
    DropEdit {
        name: String,
        reason: String,
    },
    Checkpoint {
        name: String,
    },
    Restore {
        name: String,
    },
    Ledger,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RpcReply {
    pub ok: bool,
    pub msg: String,
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

/// Namespaces, a read-only /usr (with the usual /lib, /lib64 and /bin links), fresh /proc, /dev and /tmp,
/// no environment, and death with the harness.
const ISOLATION: &[&str] = &[
    "--unshare-all",
    "--die-with-parent",
    "--new-session",
    "--clearenv",
    "--ro-bind",
    "/usr",
    "/usr",
    "--symlink",
    "usr/lib",
    "/lib",
    "--symlink",
    "usr/lib64",
    "/lib64",
    "--symlink",
    "usr/bin",
    "/bin",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
];

/// The directory the venv's interpreter really lives in (uv venvs point outside the venv).
fn python_home(venv: &Path) -> Result<PathBuf> {
    let real =
        std::fs::canonicalize(venv.join("bin/python")).with_context(|| format!("no python in {}", venv.display()))?;
    // .../cpython-x/bin/python3.12 -> .../cpython-x
    Ok(real
        .parent()
        .and_then(Path::parent)
        .context("odd python path")?
        .to_path_buf())
}

/// `s` cut to at most `max` bytes (on a character boundary), with a note when it was cut.
pub fn cut(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output cut at {} characters]", &s[..end], max)
}

fn canonical(p: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(p).with_context(|| format!("{}", p.display()))
}

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

    /// `prlimit --as=<memory> -- bwrap <isolation> <binds> <environment> <python> /openwish/kernel.py`
    fn command(&self) -> Result<Command> {
        let venv = canonical(&self.cfg.venv)?;
        let mut a: Vec<String> = vec![format!("--as={}", self.cfg.mem_bytes), "--".into(), "bwrap".into()];
        a.extend(ISOLATION.iter().map(|s| s.to_string()));
        a.extend(self.binds(&venv)?);
        a.extend(self.env_args());
        a.extend(["--chdir".into(), "/work".into()]);
        a.extend([
            venv.join("bin/python").display().to_string(),
            "-I".into(),
            "-u".into(),
            "/openwish/kernel.py".into(),
        ]);
        let mut cmd = Command::new("prlimit");
        cmd.args(a);
        Ok(cmd)
    }

    /// The interpreter and venv, /openwish and /net read-only, /work and the task's mounts.
    fn binds(&self, venv: &Path) -> Result<Vec<String>> {
        let c = &self.cfg;
        let home = python_home(venv)?;
        let mut a = vec![];
        let mut bind =
            |flag: &str, host: &Path, guest: &str| a.extend([flag.into(), host.display().to_string(), guest.into()]);
        bind("--ro-bind", &home, &home.display().to_string());
        // the venv's python symlink may name the interpreter through another symlinked directory
        if let Ok(t) = std::fs::read_link(venv.join("bin/python")) {
            if let Some(lit) = t.parent().and_then(Path::parent) {
                if lit.is_absolute() && lit != home {
                    bind("--ro-bind", &home, &lit.display().to_string());
                }
            }
        }
        bind("--ro-bind", venv, &venv.display().to_string());
        bind("--ro-bind", &canonical(&c.private_dir)?, "/openwish");
        if let Some(n) = &c.net_dir {
            bind("--ro-bind", &canonical(n)?, "/net");
        }
        bind("--bind", &canonical(&c.work_dir)?, "/work");
        for m in &c.mounts {
            bind(
                if m.rw { "--bind" } else { "--ro-bind" },
                &canonical(&m.host)?,
                &m.guest,
            );
        }
        Ok(a)
    }

    /// The kernel's whole environment: thread limits, no GPU, and where its files are.
    fn env_args(&self) -> Vec<String> {
        let threads = self.cfg.threads.to_string();
        let vars = [
            ("HOME", "/tmp"),
            ("PATH", "/usr/bin"),
            ("OMP_NUM_THREADS", &threads),
            ("MKL_NUM_THREADS", &threads),
            ("CUDA_VISIBLE_DEVICES", ""),
            ("OPENWISH_PRELUDE_FILE", "/openwish/prelude.py"),
            ("OPENWISH_NET_DIR", "/net"),
            ("OPENWISH_WORK_DIR", "/work"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
        ];
        vars.iter()
            .flat_map(|(k, v)| ["--setenv".to_string(), k.to_string(), v.to_string()])
            .collect()
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
                        text: cut(v["output"].as_str().unwrap_or(""), self.cfg.max_out),
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

    /// The RPC in a protocol line of type "rpc". Paths under /work are mapped to the host's work folder.
    fn rpc_request(&self, v: &serde_json::Value) -> Rpc {
        let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
        match v["method"].as_str().unwrap_or("") {
            "save_edit" => Rpc::SaveEdit {
                name: s("name"),
                path: self.guest_to_host(&s("path")),
                reason: s("reason"),
            },
            "drop_edit" => Rpc::DropEdit {
                name: s("name"),
                reason: s("reason"),
            },
            "checkpoint" => Rpc::Checkpoint { name: s("name") },
            "restore" => Rpc::Restore { name: s("name") },
            _ => Rpc::Ledger,
        }
    }

    fn guest_to_host(&self, p: &str) -> PathBuf {
        match p.strip_prefix("/work/") {
            Some(rest) => self.cfg.work_dir.join(rest),
            None => PathBuf::from(p),
        }
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
mod tests {
    use super::*;
    use openwish_tensor::{Network, Tensor};

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn kernel(d: &Path, timeout_s: u64) -> Option<Kernel> {
        let venv = root().join("assets/venvs/sandbox");
        if !venv.join("bin/python").exists() {
            eprintln!("skipping: no sandbox venv");
            return None;
        }
        let net = d.join("net");
        let mut n = Network::default();
        n.tensors
            .insert("L1.W".into(), Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]));
        n.save_dir(&net).unwrap();
        Some(
            Kernel::start(KernelCfg {
                venv,
                kernel_dir: root().join("python/kernel"),
                private_dir: d.join("private"),
                net_dir: Some(net),
                work_dir: d.join("work"),
                mounts: vec![],
                timeout_s,
                mem_bytes: 8 << 30,
                max_out: 2000,
                prelude: Some("PRELUDE_OK = 41 + 1".into()),
                threads: 2,
            })
            .unwrap(),
        )
    }
    fn none(_: Rpc) -> RpcReply {
        RpcReply {
            ok: false,
            msg: "no rpc in this test".into(),
        }
    }

    #[test]
    fn variables_persist_and_last_expression_echoes() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 60) else { return };
        assert_eq!(k.run("x = PRELUDE_OK", &mut none).unwrap().text, "");
        let o = k.run("x * 2", &mut none).unwrap();
        assert_eq!(o.text.trim(), "84");
        let o = k.run("print(load('L1.W').sum(), NAMES)", &mut none).unwrap();
        assert_eq!(o.text.trim(), "10.0 ['L1.W']");
    }

    #[test]
    fn gradients_and_optimisers_raise() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 120) else { return };
        for code in [
            "import torch\nt = torch.ones(2)\nt.requires_grad_(True)",
            "import torch\ntorch.optim.Adam([torch.zeros(1)])",
            "import torch\nwith torch.enable_grad():\n    pass",
            "import torch\ntorch.autograd.grad",
        ] {
            let o = k.run(code, &mut none).unwrap();
            if code.ends_with("grad") {
                // merely referencing is fine; calling raises
                let o = k
                    .run("import torch\ntorch.autograd.grad(None, None)", &mut none)
                    .unwrap();
                assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{}", o.text);
                continue;
            }
            assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{code}: {}", o.text);
        }
    }

    #[test]
    fn save_edit_goes_through_rpc() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 60) else { return };
        let mut got = vec![];
        let mut h = |r: Rpc| {
            if let Rpc::SaveEdit { path, .. } = &r {
                let t = openwish_tensor::npy::read_npy(&std::fs::read(path).unwrap()).unwrap();
                got.push((r.clone(), t.data));
            }
            RpcReply {
                ok: true,
                msg: "edit #1 saved".into(),
            }
        };
        let o = k
            .run(
                "import numpy as np\nsave_edit('L1.W', np.zeros((2,2)), reason='why')",
                &mut h,
            )
            .unwrap();
        assert!(!o.error, "{}", o.text);
        assert!(o.text.contains("edit #1 saved"));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].1, vec![0.0; 4]);
        let mut refuse = |_: Rpc| RpcReply {
            ok: false,
            msg: "edit refused: nope".into(),
        };
        let o = k.run("save_edit('L1.W', np.ones((2,2)))", &mut refuse).unwrap();
        assert!(o.error && o.text.contains("nope"));
        // weights are read-only inside the sandbox
        let o = k.run("open('/net/L1.W.npy','wb')", &mut none).unwrap();
        assert!(o.error, "{}", o.text);
    }

    #[test]
    fn kernel_restart_keeps_ledger() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 3) else { return };
        k.run("y = 5", &mut none).unwrap();
        // an edit made on the host side (the ledger) before the crash
        let mut n = Network::load_dir(&d.path().join("net")).unwrap();
        n.tensors.get_mut("L1.W").unwrap().data[0] = 100.0;
        n.save_dir(&d.path().join("net")).unwrap();
        let o = k.run("import time\ntime.sleep(30)", &mut none).unwrap();
        assert!(o.restarted && o.text.contains("restarted"), "{}", o.text);
        let o = k.run("y", &mut none).unwrap();
        assert!(o.error && o.text.contains("NameError"), "{}", o.text);
        let o = k.run("float(load('L1.W')[0,0])", &mut none).unwrap();
        assert_eq!(o.text.trim(), "100.0");
    }

    #[test]
    fn kernel_that_dies_between_runs_is_restarted() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 30) else { return };
        k.run(
            "import os, threading\nthreading.Timer(0.3, lambda: os._exit(1)).start()",
            &mut none,
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let o = k.run("1 + 1", &mut none).unwrap();
        assert!(
            o.restarted && o.text.contains("restarted") && o.text.contains('2'),
            "{}",
            o.text
        );
    }

    #[test]
    fn everyday_grad_spellings_are_blocked() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 120) else { return };
        for code in [
            "import torch\nwith torch.autograd.enable_grad():\n    pass",
            "import torch\ntorch.autograd.set_grad_enabled(True)",
            "import torch\ntorch.autograd.grad_mode.set_grad_enabled(True)",
            "import torch\ntorch._C._set_grad_enabled(True)",
            "import importlib, pkgutil, torch, torch.optim as o\nm = [importlib.import_module('torch.optim.' + i.name) for i in pkgutil.iter_modules(o.__path__) if i.name.strip('_') == 'adam'][0]\nm.Adam([torch.zeros(1)])",
            "import torch\ntorch.autograd.Variable._execution_engine.run_backward",
        ] {
            let o = k.run(code, &mut none).unwrap();
            assert!(o.error && o.text.contains("OpenWiSHForbidden"), "{code}: {}", o.text);
        }
        // with grad mode off, requires_grad tensors build no graph
        let o = k
            .run(
                "import torch\nx = torch.zeros(2, requires_grad=True)\nprint((x * 2).grad_fn is None)",
                &mut none,
            )
            .unwrap();
        assert!(o.text.contains("True"), "{}", o.text);
    }

    #[test]
    fn no_network_and_output_is_cut() {
        let d = tempfile::tempdir().unwrap();
        let Some(mut k) = kernel(d.path(), 60) else { return };
        let o = k
            .run(
                "import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=3)",
                &mut none,
            )
            .unwrap();
        assert!(o.error, "{}", o.text);
        let o = k.run("print('x' * 10000)", &mut none).unwrap();
        assert!(o.text.len() < 2200 && o.text.contains("cut at"), "{}", o.text.len());
    }
}
