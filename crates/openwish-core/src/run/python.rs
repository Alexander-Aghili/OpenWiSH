//! The Python tool: the sandboxed kernel, its static check, and the report the editor sees.
use super::rpc::handle_rpc;
use super::*;

impl Run {
    pub(super) fn python(&mut self, code: &str) -> Result<()> {
        let py = self.spec.python.clone().expect("python tool enabled");
        if self.state.runs_used >= py.runs {
            self.state.next_message = format!(
                "You have no Python runs left (used {} of {}). Nothing was run.\n{}",
                self.state.runs_used,
                py.runs,
                self.budget_line()
            );
            return Ok(());
        }
        if let Some(why) = self.pre_tool("python", json!({"code": code})) {
            return self.blocked("Python run", "python", json!({"code": code}), &why);
        }
        let flags = openwish_audit::static_check(code);
        let rejected: Vec<_> = flags
            .iter()
            .filter(|flag| py.reject_flags.contains(&flag.rule))
            .collect();
        if !rejected.is_empty() {
            self.state.failed_requests += 1;
            let rules: Vec<_> = rejected.iter().map(|flag| flag.rule.as_str()).collect();
            self.tool_log(json!({"kind": "python-policy", "turn": self.state.turn, "blocked": true, "code": code, "flags": rejected}))?;
            self.state.next_message = format!(
                "Python policy blocked this code before execution: {}. Nothing ran and no Python budget was spent. \
                 Use weight algebra and symbolic identities. Do not simulate the network or its components on inputs.\n{}",
                rules.join(", "), self.budget_line()
            );
            return Ok(());
        }
        if self.kernel.is_none() {
            self.kernel = Some(self.start_kernel(&py)?);
        }
        let mark = self.last_edit_id().unwrap_or(0);
        let out = self.run_in_kernel(code)?;
        self.state.runs_used += 1;
        if py.fresh {
            self.kernel = None;
        }
        if out.restarted {
            self.state.kernel_restarts += 1;
        }
        let k = self.state.runs_used;
        self.tool_log(
            json!({"kind": "python", "turn": self.state.turn, "run": k, "code": code, "output": out.text,
            "error": out.error, "restarted": out.restarted, "seconds": out.seconds, "flags": flags}),
        )?;
        self.on_edits("python", mark);
        self.post_tool(
            "python",
            !out.error,
            json!({"code": code, "output": out.text, "error": out.error}),
        );
        self.state.next_message = format!("{}\n{}", python_report(k, py.runs, &out, &flags), self.budget_line());
        Ok(())
    }

    pub(super) fn start_kernel(&self, py: &PythonSpec) -> Result<Kernel> {
        let prelude = match &py.prelude {
            Some(p) => Some(std::fs::read_to_string(self.spec.resolve(p))?),
            None => None,
        };
        let mut mounts = self.spec.mounts.clone();
        for m in &mut mounts {
            m.host = self.spec.resolve(&m.host);
        }
        Kernel::start(KernelCfg {
            venv: py
                .venv
                .as_ref()
                .map(|v| self.spec.resolve(v))
                .unwrap_or_else(|| self.env.venv.clone()),
            kernel_dir: self.env.kernel_dir.clone(),
            private_dir: self.dir.join("kernel"),
            net_dir: self.gate.as_ref().map(Gate::current_dir),
            work_dir: self.dir.join("work"),
            mounts,
            timeout_s: py.timeout_s,
            mem_bytes: py.mem_gb << 30,
            max_out: py.max_out,
            prelude,
            threads: py.threads.min(self.env.threads.max(1)),
        })
    }

    /// Run `code` in the started kernel; its edit RPCs go through the gate.
    pub(super) fn run_in_kernel(&mut self, code: &str) -> Result<RunOutput> {
        let turn = self.state.turn;
        let rpc_dir = self.dir.join("work/.rpc");
        let (kernel, gate) = (self.kernel.as_mut().expect("kernel started"), &mut self.gate);
        kernel.run(code, &mut |r: Rpc| handle_rpc(gate.as_mut(), r, turn, &rpc_dir))
    }

    /// The harness stopped during a Python run: report what is known instead of running it again.
    pub(super) fn interrupted_python(&mut self) {
        self.state.runs_used += 1;
        self.state.kernel_restarts += 1;
        let k = self.state.runs_used;
        let mark = self.state.ledger_mark.unwrap_or(0);
        let ids: Vec<String> = self.edits_after(mark).iter().map(|(id, _)| format!("#{id}")).collect();
        let kept = if ids.is_empty() {
            "No edits were saved during it.".to_string()
        } else {
            format!("Edits saved during it were kept: {}.", ids.join(", "))
        };
        let _ = self.tool_log(
            json!({"kind": "python", "turn": self.state.turn, "run": k, "interrupted": true, "edits_kept": ids}),
        );
        self.on_edits("python", mark);
        self.state.next_message = format!(
            "[The harness stopped while your Python run {k} was going, so its output is lost. {kept} The kernel was \
             restarted and its variables are lost; files in /work are kept.]\n{}",
            self.budget_line()
        );
    }
}

/// What the agent reads after Python run `k`.
pub(super) fn python_report(k: u32, runs: u32, out: &RunOutput, flags: &[Flag]) -> String {
    let text = if out.text.is_empty() {
        "(no output)"
    } else {
        out.text.trim_end()
    };
    let mut msg = format!("OUTPUT of Python run {k} of at most {runs}:\n```\n{text}\n```");
    if !flags.is_empty() {
        let f: Vec<String> = flags.iter().map(|f| format!("line {}: {}", f.line, f.why)).collect();
        msg += &format!(
            "\n[harness note: this code matched patterns that are recorded for the rules audit: {}]",
            f.join("; ")
        );
    }
    msg
}
