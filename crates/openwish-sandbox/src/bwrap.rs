//! The bubblewrap command that isolates the kernel: namespaces, read-only system and venv, the task's
//! mounts, and a cleared environment.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::Kernel;

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
pub(crate) fn python_home(venv: &Path) -> Result<PathBuf> {
    let real =
        std::fs::canonicalize(venv.join("bin/python")).with_context(|| format!("no python in {}", venv.display()))?;
    // .../cpython-x/bin/python3.12 -> .../cpython-x
    Ok(real
        .parent()
        .and_then(Path::parent)
        .context("odd python path")?
        .to_path_buf())
}

pub(crate) fn canonical(p: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(p).with_context(|| format!("{}", p.display()))
}

impl Kernel {
    /// `prlimit --as=<memory> -- bwrap <isolation> <binds> <environment> <python> /openwish/kernel.py`
    pub(crate) fn command(&self) -> Result<Command> {
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
    pub(crate) fn binds(&self, venv: &Path) -> Result<Vec<String>> {
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
    pub(crate) fn env_args(&self) -> Vec<String> {
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
}
