//! The turn loop: sending to the editor with retries, recording replies, handoffs, acting on a reply,
//! the final reply and its validation.
use super::*;

impl Run {
    /// One turn. The first step of a process runs the session_start hooks; the step that ends the run runs
    /// the session_end hooks.
    pub fn step(&mut self) -> Result<Step> {
        if !self.started && !self.state.done {
            self.started = true;
            self.fire(Event::SessionStart, &self.hook_ctx(""));
        }
        let step = self.next_step()?;
        if step == Step::Done && self.started && !self.ended {
            self.ended = true;
            let mut c = self.hook_ctx("");
            c.extra.push((
                "OPENWISH_END_REASON".into(),
                self.state.end_reason.clone().unwrap_or_default(),
            ));
            self.fire(Event::SessionEnd, &c);
        }
        Ok(step)
    }

    pub fn run_to_end(&mut self) -> Result<()> {
        while self.step()? == Step::Continue {}
        Ok(())
    }

    pub(super) fn next_step(&mut self) -> Result<Step> {
        if self.state.done {
            return Ok(Step::Done);
        }
        if let Some(text) = self.state.pending_reply.clone() {
            return self.act_on_pending(&text);
        }
        if self.state.turn >= self.spec.finish.max_turns {
            self.finalize("", "turn limit reached")?;
            return Ok(Step::Done);
        }
        self.take_turn()
    }

    /// A reply that arrived before a crash is acted on first.
    pub(super) fn act_on_pending(&mut self, text: &str) -> Result<Step> {
        let is_python =
            matches!(parse(text, &self.spec.enabled_tools()), Action::Tool { ref tag, .. } if tag == "python");
        if is_python {
            // its edits (if any) are already in the ledger; running it again would repeat them
            self.interrupted_python();
        } else {
            self.act(text)?;
        }
        self.conclude_action()
    }

    pub(super) fn take_turn(&mut self) -> Result<Step> {
        let turn = self.state.turn + 1;
        let tdir = self.dir.join(format!("turns/t{turn:03}"));
        std::fs::create_dir_all(&tdir)?;
        let msg = self.state.next_message.clone();
        let new_conv = self.state.new_conversation;
        std::fs::write(tdir.join("prompt.txt"), &msg)?;
        let reply = self.send_with_retry(&msg, new_conv)?;
        self.state.turn = turn;
        match reply {
            Err(why) => self.refused(&tdir, &why, new_conv),
            Ok(reply) => {
                self.record_reply(&tdir, &reply, new_conv)?;
                self.act(&reply.text)?;
                self.conclude_action()
            }
        }
    }

    /// Send one message, retrying transient errors with exponential backoff. `Ok(Err(why))` is a refusal.
    pub(super) fn send_with_retry(&mut self, msg: &str, new_conv: bool) -> Result<std::result::Result<Reply, String>> {
        let mut attempt = 0;
        loop {
            match self.editor.send(msg, new_conv) {
                Ok(r) => return Ok(Ok(r)),
                Err(BackendError::Refused(m)) => return Ok(Err(m)),
                Err(BackendError::Transient(m)) => {
                    self.state.backend_errors += 1;
                    attempt += 1;
                    self.log(&format!(
                        "turn {}: transient backend error (attempt {attempt}): {m}",
                        self.state.turn + 1
                    ));
                    if attempt > self.env.max_retries {
                        self.save_state()?;
                        bail!("backend kept failing after {attempt} attempts: {m} (resume later with --resume)");
                    }
                    std::thread::sleep(self.backoff(attempt));
                }
                Err(BackendError::Fatal(m)) => {
                    self.log(&format!("turn {}: fatal backend error: {m}", self.state.turn + 1));
                    self.save_state()?;
                    bail!("backend error: {m}");
                }
            }
        }
    }

    pub(super) fn backoff(&self, attempt: u32) -> Duration {
        let wait = self.env.retry_base_s * 2f64.powi(attempt as i32 - 1);
        Duration::from_secs_f64(wait.min(MAX_BACKOFF_S))
    }

    /// Save the reply and its usage, and mark it pending before acting on it, so a crash cannot lose or
    /// repeat it.
    pub(super) fn record_reply(&mut self, tdir: &Path, reply: &Reply, new_conv: bool) -> Result<()> {
        self.state.refusal_streak = 0;
        std::fs::write(tdir.join("answer.md"), &reply.text)?;
        let meta = json!({
            "editor": self.editor.name(), "new_conversation": new_conv, "usage": reply.usage, "session": reply.session,
            "tool_free": reply.tool_free, "cost_usd": reply.cost_usd, "wall_s": reply.wall_s, "context_tokens": reply.context_tokens,
        });
        std::fs::write(tdir.join("run.json"), serde_json::to_string_pretty(&meta)?)?;
        std::fs::write(tdir.join("raw.json"), serde_json::to_string(&reply.raw)?)?;
        self.state.conv_tokens = reply.context_tokens.unwrap_or_else(|| reply.usage.context_tokens());
        self.state.usage_total += &reply.usage;
        self.state.cost_usd += reply.cost_usd.unwrap_or(0.0);
        if reply.tool_free {
            self.state.tool_free_turns += 1;
        }
        self.state.session = self.editor.session();
        self.state.new_conversation = false;
        self.state.since_handoff += 1;
        self.state.pending_reply = Some(reply.text.clone());
        self.state.ledger_mark = self.last_edit_id();
        self.save_state()
    }

    /// The provider refused the turn: tell the agent, or end the run after too many refusals in a row.
    pub(super) fn refused(&mut self, tdir: &Path, why: &str, new_conv: bool) -> Result<Step> {
        self.state.refusals += 1;
        self.state.refusal_streak += 1;
        std::fs::write(tdir.join("refused.txt"), why)?;
        self.log(&format!(
            "turn {}: refused by the provider (streak {})",
            self.state.turn, self.state.refusal_streak
        ));
        if self.state.refusal_streak > MAX_REFUSAL_STREAK {
            self.finalize("", "the provider refused four turns in a row")?;
            self.save_state()?;
            return Ok(Step::Done);
        }
        let notice = format!(
            "The harness's last message could not be delivered to you: the provider's content filter refused it ({}). \
             This usually happens when output contains raw vocabulary strings or long generated text; print token ids, \
             numbers or short excerpts instead. Continue with your next step, done differently.\n\n{}",
            openwish_text::head(why, 300),
            self.budget_line()
        );
        // a new conversation must still carry the task
        self.state.next_message = if !new_conv {
            notice
        } else if self.state.handoffs == 0 {
            format!("{}\n\n[{notice}]", self.first_message)
        } else {
            self.handoff_message(&notice)
        };
        self.save_state()?;
        Ok(Step::Continue)
    }

    /// After acting on a reply: clear it, decide whether the next message starts a new conversation, save.
    pub(super) fn conclude_action(&mut self) -> Result<Step> {
        self.state.pending_reply = None;
        self.maybe_handoff();
        self.save_state()?;
        Ok(if self.state.done { Step::Done } else { Step::Continue })
    }

    pub(super) fn maybe_handoff(&mut self) {
        if self.state.done {
            return;
        }
        let h = &self.spec.handoff;
        let by_tokens = h.max_conversation_tokens > 0 && self.state.conv_tokens >= h.max_conversation_tokens;
        let by_turns = h.max_turns_per_conversation > 0 && self.state.since_handoff >= h.max_turns_per_conversation;
        let forced = std::mem::take(&mut self.state.force_handoff);
        let why = if by_tokens {
            "context size"
        } else if by_turns {
            "turn count"
        } else if forced {
            "the backend cannot resume its conversation"
        } else {
            return;
        };
        self.start_handoff();
        self.log(&format!("handoff {} ({why})", self.state.handoffs));
    }

    /// Make the next message start a new conversation that restates the task and where the agent is.
    pub(super) fn start_handoff(&mut self) {
        let next = std::mem::take(&mut self.state.next_message);
        self.state.next_message = self.handoff_message(&next);
        self.state.new_conversation = true;
        self.state.handoffs += 1;
        self.state.since_handoff = 0;
    }

    pub(super) fn handoff_message(&self, next: &str) -> String {
        let ledger = self
            .gate
            .as_ref()
            .map(|g| g.summary())
            .unwrap_or_else(|| "(this task has no editable network)\n".into());
        format!(
            "{}\n\n=== HANDOFF ===\nThe harness started a new conversation to keep the context small. The task above is unchanged, and so \
             are your files in /work, your saved edits and the Python kernel's variables. Where you are:\n{}\n\nYour notebook \
             (/work/notes.md):\n```\n{}\n```\n\nYour edits so far:\n```\n{}```\n\nThe message you would have received next:\n\n{}",
            self.first_message,
            self.budget_line(),
            self.notebook().trim_end(),
            ledger,
            next
        )
    }

    /// /work/notes.md for a handoff, if it is a regular file in /work, cut at MAX_NOTES_CHARS.
    pub(super) fn notebook(&self) -> String {
        let work = self.dir.join("work");
        let notes = match paths::file_under(&work, Path::new("notes.md")) {
            Ok(p) => std::fs::read_to_string(p).unwrap_or_else(|_| "(/work/notes.md could not be read as text)".into()),
            Err(_) if std::fs::symlink_metadata(work.join("notes.md")).is_err() => "(no /work/notes.md)".into(),
            Err(_) => "(/work/notes.md is not a regular file in /work, so it is not shown)".into(),
        };
        match notes.char_indices().nth(MAX_NOTES_CHARS) {
            Some((end, _)) => format!("{}\n[notes cut at 20,000 characters]", &notes[..end]),
            None => notes,
        }
    }

    pub(super) fn act(&mut self, text: &str) -> Result<()> {
        match parse(text, &self.spec.enabled_tools()) {
            Action::Error(m) => {
                self.state.protocol_errors += 1;
                self.state.next_message = format!("{m}\n{}", self.budget_line());
                Ok(())
            }
            Action::Tool { tag, body } if tag == "python" => self.python(&body),
            Action::Tool { tag, body } => self.forward(&tag, &body),
            Action::Final(text) => self.final_reply(&text),
        }
    }

    /// A reply without a tool block ends the run, unless the task wants every pass used first.
    pub(super) fn final_reply(&mut self, text: &str) -> Result<()> {
        let left = self.passes_left();
        let fin = &self.spec.finish;
        if fin.require_all_passes && !left.is_empty() && self.state.nudges < fin.max_nudges {
            self.state.nudges += 1;
            let left: Vec<String> = left.iter().map(|(label, n)| format!("{n} {label} left")).collect();
            self.state.next_message = format!(
                "The session continues until every forward pass is used: you have {}. Use them (a reply without a tool \
                 block does not end the session yet).\n{}",
                left.join(", "),
                self.budget_line()
            );
            Ok(())
        } else {
            if let Some(error) = self.validate_submission(text)? {
                if self.state.nudges >= self.spec.finish.max_nudges {
                    return self.finalize(text, "submission rejected");
                }
                self.state.nudges += 1;
                self.state.protocol_errors += 1;
                self.state.next_message = format!(
                    "Submission format refused: {error}\nThe task continues. Repair the submission. \
                     This check reports format only, never correctness. Your declared tools remain available.\n{}",
                    self.budget_line()
                );
                return Ok(());
            }
            self.finalize(text, "final answer")
        }
    }

    pub(super) fn validate_submission(&mut self, text: &str) -> Result<Option<String>> {
        let Some(validation) = &self.spec.finish.validation else {
            return Ok(None);
        };
        let checked = (|| -> Result<(Value, PathBuf)> {
            let mut candidate = text.to_string();
            if let Some(name) = &validation.submission_file {
                let work = self.dir.join("work");
                let file = work.join(name);
                if std::fs::symlink_metadata(&file).is_ok() {
                    candidate.clear();
                    std::fs::File::open(paths::file_under(&work, &file)?)?
                        .take(MAX_SUBMISSION_BYTES + 1)
                        .read_to_string(&mut candidate)?;
                }
            }
            if candidate.len() as u64 > MAX_SUBMISSION_BYTES {
                bail!("submission exceeds {MAX_SUBMISSION_BYTES} bytes");
            }
            let requests = paths::dir_under(&self.dir, "requests")?;
            let file = requests.join(format!("submission-{}.txt", self.state.turn));
            std::fs::write(&file, candidate)?;
            let result = crate::forward::run_command(
                &self.spec.argv(&validation.command),
                &[("OPENWISH_SUBMISSION", file.display().to_string())],
                &self.spec.dir,
                validation.timeout_s,
            )?;
            Ok((result, file))
        })();
        let result = match checked {
            Ok((result, file)) => {
                if result["ok"].as_bool() == Some(true) {
                    std::fs::copy(file, self.dir.join("validated_submission.json"))?;
                }
                result
            }
            Err(error) => json!({"ok": false, "error": error.to_string()}),
        };
        let error = if result["ok"].as_bool() == Some(true) {
            None
        } else {
            Some(
                result["error"]
                    .as_str()
                    .unwrap_or("the validator refused the submission")
                    .chars()
                    .take(4000)
                    .collect(),
            )
        };
        self.tool_log(json!({"kind": "validation", "turn": self.state.turn, "result": result}))?;
        Ok(error)
    }
}
