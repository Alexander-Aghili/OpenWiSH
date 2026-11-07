//! Forward probes: reserving a pass, running the trusted command, and recording its summary.
use super::*;

/// One forward pass: use number `k` of the tool's budget group.
pub(super) struct Pass {
    tag: String,
    group: String,
    k: u32,
    budget: u32,
    /// its folder under /work
    folder: String,
}

impl Run {
    pub(super) fn forward(&mut self, tag: &str, body: &str) -> Result<()> {
        let f = self.spec.forward[tag].clone();
        let group = self.spec.group(tag);
        let (used, budget) = self.tool_budget(&group, &f);
        if used >= budget {
            let what = if group == PYTHON_POOL {
                "Python runs".to_string()
            } else {
                self.spec.group_label(&group)
            };
            self.state.next_message = format!(
                "You have no {what} left (used {used} of {budget}). Nothing was run.\n{}",
                self.budget_line()
            );
            return Ok(());
        }
        if let Some(why) = self.pre_tool(tag, json!({"request": body})) {
            return self.blocked(
                &format!("{tag} pass"),
                &format!("forward:{tag}"),
                json!({"request": body}),
                &why,
            );
        }
        let k = used + 1;
        let pass = Pass {
            tag: tag.to_string(),
            group,
            k,
            budget,
            folder: f.folder(tag, k),
        };
        let req = self.dir.join(format!("requests/{tag}_{k}.json"));
        std::fs::write(&req, body)?;
        let out_dir = match paths::dir_under(&self.dir.join("work"), &pass.folder) {
            Ok(p) => p,
            Err(e) => {
                self.state.failed_requests += 1;
                self.state.next_message = format!(
                    "The {tag} pass could not run and was not counted: its folder /work/{} is unusable ({e:#}).\n{}",
                    pass.folder,
                    self.budget_line()
                );
                return Ok(());
            }
        };
        if out_dir.exists() {
            std::fs::remove_dir_all(&out_dir)?;
        }
        std::fs::create_dir_all(&out_dir)?;
        let t0 = Instant::now();
        let (ok, summary) = probe_summary(crate::forward::run_command(
            &self.spec.argv(&f.command),
            &self.probe_env(&req, &out_dir, k),
            &self.spec.dir,
            f.timeout_s,
        ));
        let secs = t0.elapsed().as_secs_f64();
        self.tool_log(
            json!({"kind": format!("forward:{tag}"), "turn": self.state.turn, "pass": ok.then_some(k),
            "request": body, "ok": ok, "summary": summary, "seconds": secs}),
        )?;
        self.post_tool(tag, ok, json!({"request": body, "ok": ok, "summary": summary}));
        if ok {
            self.pass_done(&pass, &summary, f.max_out)
        } else {
            let _ = std::fs::remove_dir_all(&out_dir);
            self.pass_failed(&pass, &summary);
            Ok(())
        }
    }

    /// Charge the pass and show the agent its summary.
    pub(super) fn pass_done(&mut self, pass: &Pass, summary: &Value, max_out: usize) -> Result<()> {
        let Pass {
            tag, k, budget, folder, ..
        } = pass;
        self.charge(&pass.group, *k);
        let pretty = serde_json::to_string_pretty(summary)?;
        std::fs::write(self.dir.join(format!("requests/{tag}_{k}.done.json")), &pretty)?;
        let text = openwish_text::clip(&pretty, max_out, "output");
        let what = if pass.group == PYTHON_POOL {
            format!("Run {k} of at most {budget} ({tag})")
        } else {
            format!("Forward pass {k} of {budget} ({tag})")
        };
        self.state.next_message = format!(
            "{what} done; files in /work/{folder}/. Summary:\n```\n{text}\n```\n{}",
            self.budget_line()
        );
        Ok(())
    }

    /// A failed pass is not charged; the agent reads why it failed.
    pub(super) fn pass_failed(&mut self, pass: &Pass, summary: &Value) {
        self.state.failed_requests += 1;
        let err = summary["error"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| summary.to_string());
        self.state.next_message = format!(
            "The {} pass failed and was not counted: {}\n{}",
            pass.tag,
            openwish_text::head(&err, 4000),
            self.budget_line()
        );
    }

    /// The variables a probe command gets for pass `k`.
    pub(super) fn probe_env(&self, req: &Path, out_dir: &Path, k: u32) -> Vec<(&'static str, String)> {
        let net = self
            .gate
            .as_ref()
            .map(|g| g.current_dir().display().to_string())
            .unwrap_or_default();
        let base = self
            .spec
            .network
            .as_ref()
            .map(|n| self.spec.resolve(&n.base).display().to_string())
            .unwrap_or_default();
        vec![
            ("OPENWISH_REQ", req.display().to_string()),
            ("OPENWISH_OUT", out_dir.display().to_string()),
            ("OPENWISH_NET", net),
            ("OPENWISH_BASE_NET", base),
            ("OPENWISH_WORK", self.dir.join("work").display().to_string()),
            ("OPENWISH_TASK_DIR", self.spec.dir.display().to_string()),
            ("OPENWISH_RUN_DIR", self.dir.display().to_string()),
            ("OPENWISH_PASS", k.to_string()),
        ]
    }
}

/// (ok, summary) of a probe command; a command that could not run gets an error summary.
pub(super) fn probe_summary(res: Result<Value>) -> (bool, Value) {
    match res {
        Ok(v) => (v["ok"].as_bool() == Some(true), v),
        Err(e) => (false, json!({"ok": false, "error": format!("{e:#}")})),
    }
}
