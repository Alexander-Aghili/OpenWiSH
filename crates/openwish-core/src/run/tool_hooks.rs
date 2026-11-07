//! Task hooks around tools: firing them, logging their outcome, and blocking a tool call.
use super::*;

impl Run {
    pub(super) fn hook_ctx(&self, tool: &str) -> HookCtx {
        HookCtx {
            run_dir: self.dir.clone(),
            task_dir: self.spec.dir.clone(),
            tool: tool.into(),
            turn: self.state.turn,
            ..Default::default()
        }
    }

    /// Run the hooks for one event and record them in hooks.jsonl. A failing hook is logged and the run goes on.
    pub(super) fn fire(&self, e: Event, ctx: &HookCtx) -> Vec<Outcome> {
        if !self.hooks.has(e) {
            return vec![];
        }
        let out = self.hooks.fire(e, ctx);
        self.log_hooks(&out, ctx);
        out
    }

    pub(super) fn log_hooks(&self, out: &[Outcome], ctx: &HookCtx) {
        for o in out {
            let mut v = serde_json::to_value(o).unwrap_or_default();
            v["turn"] = ctx.turn.into();
            v["tool"] = ctx.tool.clone().into();
            if let Some((id, t)) = &ctx.edit {
                v["edit_id"] = (*id).into();
                v["edit_tensor"] = t.clone().into();
            }
            let _ = append_line(&self.dir.join("hooks.jsonl"), &v.to_string());
            if !o.ok {
                self.log(&format!(
                    "turn {}: {} hook failed ({}): {}",
                    ctx.turn,
                    o.event.name(),
                    o.command,
                    o.reason()
                ));
            }
        }
    }

    /// The pre_tool hooks for one call. `Some(reason)` means a hook blocked it.
    pub(super) fn pre_tool(&self, tool: &str, input: Value) -> Option<String> {
        if !self.hooks.has(Event::PreTool) {
            return None;
        }
        let mut c = self.hook_ctx(tool);
        c.input = input;
        let (out, why) = self.hooks.check(&c);
        self.log_hooks(&out, &c);
        why
    }

    /// Record a blocked call and tell the agent why. Nothing is charged.
    pub(super) fn blocked(&mut self, what: &str, kind: &str, input: Value, why: &str) -> Result<()> {
        self.state.hook_blocks += 1;
        self.tool_log(json!({"kind": kind, "turn": self.state.turn, "blocked": true, "reason": why, "input": input}))?;
        self.state.next_message = format!(
            "A pre_tool hook blocked this {what}, so nothing was run and it was not counted. The hook said:\n```\n{}\n```\n{}",
            why.trim_end(),
            self.budget_line()
        );
        Ok(())
    }

    pub(super) fn post_tool(&self, tool: &str, ok: bool, input: Value) {
        if !self.hooks.has(Event::PostTool) {
            return;
        }
        let mut c = self.hook_ctx(tool);
        c.extra
            .push(("OPENWISH_TOOL_OK".into(), if ok { "1" } else { "0" }.into()));
        c.input = input;
        self.fire(Event::PostTool, &c);
    }

    /// The on_edit hooks, once for every edit in the ledger after edit `after`.
    pub(super) fn on_edits(&self, tool: &str, after: u64) {
        if !self.hooks.has(Event::OnEdit) {
            return;
        }
        for (id, tensor) in self.edits_after(after) {
            let mut c = self.hook_ctx(tool);
            c.edit = Some((id, tensor));
            self.fire(Event::OnEdit, &c);
        }
    }

    /// (id, tensor) of every edit after edit `after`.
    pub(super) fn edits_after(&self, after: u64) -> Vec<(u64, String)> {
        let Some(g) = &self.gate else { return vec![] };
        g.edits()
            .filter(|r| r.id > after)
            .map(|r| (r.id, r.op.tensor().to_string()))
            .collect()
    }
}
