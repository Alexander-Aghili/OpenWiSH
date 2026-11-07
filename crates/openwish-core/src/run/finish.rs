//! Finishing a run: the final edits block, final/net, the edit set and the end state.
use super::*;

impl Run {
    pub(super) fn finalize(&mut self, text: &str, reason: &str) -> Result<()> {
        std::fs::write(self.dir.join("final_answer.md"), text)?;
        std::fs::create_dir_all(self.dir.join("final"))?;
        if self.spec.edits.language {
            let mark = self.last_edit_id().unwrap_or(0);
            let report = self.apply_edits_block(text);
            std::fs::write(
                self.dir.join("final/edits_report.json"),
                serde_json::to_string_pretty(&report)?,
            )?;
            self.on_edits("edits", mark);
        }
        if let Some(g) = &self.gate {
            g.link_current_to(&self.dir.join("final/net"))?;
            openwish_tensor::editset::export(&g.base, &g.current, &self.dir.join("final/editset"))?;
        }
        self.state.done = true;
        self.state.end_reason = Some(reason.to_string());
        self.kernel = None;
        self.log(&format!(
            "finished: {reason}; turns {}, python runs {}, passes {:?}",
            self.state.turn, self.state.runs_used, self.state.passes_used
        ));
        Ok(())
    }

    /// Apply the last ```edits block of the final reply through the gate; returns the report.
    pub(super) fn apply_edits_block(&mut self, text: &str) -> Value {
        let Some(block) = openwish_tensor::editlang::last_edits_block(text) else {
            return json!({"applied": 0, "skipped": [], "refused": [], "block_found": false});
        };
        let parsed = openwish_tensor::editlang::parse(&block);
        let (mut applied, mut refused) = (0, vec![]);
        if let Some(g) = self.gate.as_mut() {
            for op in parsed.ops {
                match g.apply(op.clone(), "edits block", "final-reply") {
                    Ok(_) => applied += 1,
                    Err(e) => refused.push(json!({"op": op, "why": e.0})),
                }
            }
        }
        json!({"applied": applied, "skipped": parsed.skipped, "refused": refused, "block_found": true})
    }
}
