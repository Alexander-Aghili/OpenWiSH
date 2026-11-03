//! Codex reports cumulative session counters. Billing deltas and current context are separate.
use crate::{count, Reply, Usage};
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_TAIL_BYTES: u64 = 2_000_000;

pub(crate) fn raw_usage(u: &Value) -> Usage {
    Usage {
        input: count(u, "input_tokens"),
        output: count(u, "output_tokens"),
        cache_read: count(u, "cached_input_tokens"),
        cache_write: count(u, "cache_write_input_tokens"),
        reasoning: count(u, "reasoning_output_tokens"),
    }
}

#[derive(Default)]
pub(crate) struct Accounting {
    previous: Usage,
    session: Option<String>,
    file: Option<PathBuf>,
}

struct Snapshot {
    total: Usage,
    context: u64,
}

impl Accounting {
    pub fn restore(&mut self, session: Option<&str>) {
        self.session = session.map(String::from);
        self.file = session.and_then(session_file);
        self.previous = self
            .file
            .as_deref()
            .and_then(read_snapshot)
            .map(|s| s.total)
            .unwrap_or_default();
    }

    pub fn apply(&mut self, reply: &mut Reply, fresh: bool) {
        if fresh || self.session != reply.session {
            self.previous = Usage::default();
            self.file = None;
        }
        self.session = reply.session.clone();
        if self.file.is_none() {
            self.file = reply.session.as_deref().and_then(session_file);
        }
        let snapshot = self.file.as_deref().and_then(read_snapshot);
        self.account(reply, snapshot);
    }

    fn account(&mut self, reply: &mut Reply, snapshot: Option<Snapshot>) {
        let raw = reply.usage.clone();
        let before = if raw.input >= self.previous.input && raw.output >= self.previous.output {
            self.previous.clone()
        } else {
            Usage::default()
        };
        let input = raw.input.saturating_sub(before.input);
        let output = raw.output.saturating_sub(before.output);
        let cache_read = raw.cache_read.saturating_sub(before.cache_read);
        let cache_write = raw.cache_write.saturating_sub(before.cache_write);
        reply.context_tokens = Some(
            snapshot
                .filter(|s| s.total == raw)
                .map(|s| s.context)
                .unwrap_or(input + output),
        );
        reply.usage = Usage {
            input: input.saturating_sub(cache_read.saturating_add(cache_write)),
            output,
            cache_read,
            cache_write,
            reasoning: raw.reasoning.saturating_sub(before.reasoning),
        };
        self.previous = raw;
    }
}

fn session_file(session: &str) -> Option<PathBuf> {
    // A session id is metadata from the trusted CLI, never a filesystem path.
    if session.len() != 36
        || !session.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
    {
        return None;
    }
    let home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".codex")))?;
    find_session(&home.join("sessions"), session)
}

fn find_session(root: &Path, session: &str) -> Option<PathBuf> {
    let suffix = format!("-{session}.jsonl");
    let mut pending = vec![(root.to_path_buf(), 0)];
    let mut inspected = 0;
    while let Some((dir, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            inspected += 1;
            if inspected > 100_000 {
                return None;
            }
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() && depth < 3 {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file() && entry.file_name().to_string_lossy().ends_with(&suffix) {
                return Some(entry.path());
            }
        }
    }
    None
}

fn read_snapshot(path: &Path) -> Option<Snapshot> {
    let mut file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    file.seek(SeekFrom::Start(size.saturating_sub(MAX_TAIL_BYTES))).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_TAIL_BYTES).read_to_end(&mut bytes).ok()?;
    for line in bytes.split(|b| *b == b'\n').rev() {
        let Ok(value) = serde_json::from_slice::<Value>(line) else {
            continue;
        };
        if value["type"] != "event_msg" || value["payload"]["type"] != "token_count" {
            continue;
        }
        let info = &value["payload"]["info"];
        if !info["total_token_usage"].is_object() || !info["last_token_usage"].is_object() {
            continue;
        }
        let last = &info["last_token_usage"];
        return Some(Snapshot {
            total: raw_usage(&info["total_token_usage"]),
            context: last["total_tokens"]
                .as_u64()
                .unwrap_or(count(last, "input_tokens") + count(last, "output_tokens")),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_cumulative_counters_become_per_reply_usage_and_current_context() {
        let previous = Usage {
            input: 189308,
            cache_read: 126080,
            output: 5247,
            reasoning: 3360,
            ..Default::default()
        };
        let current = Usage {
            input: 255717,
            cache_read: 191232,
            output: 5410,
            reasoning: 3515,
            ..Default::default()
        };
        let mut accounting = Accounting {
            previous,
            ..Default::default()
        };
        let mut reply = Reply {
            usage: current.clone(),
            ..Default::default()
        };
        accounting.account(
            &mut reply,
            Some(Snapshot {
                total: current,
                context: 66572,
            }),
        );
        assert_eq!(
            reply.usage,
            Usage {
                input: 1257,
                cache_read: 65152,
                output: 163,
                reasoning: 155,
                ..Default::default()
            }
        );
        assert_eq!(reply.context_tokens, Some(66572));
    }

    #[test]
    fn reset_counters_and_missing_metadata_have_a_bounded_fallback() {
        let mut accounting = Accounting {
            previous: Usage {
                input: 500000,
                output: 10000,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut reply = Reply {
            usage: Usage {
                input: 100,
                cache_read: 70,
                output: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        accounting.account(&mut reply, None);
        assert_eq!(reply.usage.input, 30);
        assert_eq!(reply.context_tokens, Some(110));
    }

    #[test]
    fn session_lookup_and_tail_read_keep_only_usage_metadata() {
        let d = tempfile::tempdir().unwrap();
        let session = "00000000-0000-0000-0000-000000000001";
        let folder = d.path().join("2026/11/03");
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(format!("rollout-{session}.jsonl"));
        let info = serde_json::json!({"type":"event_msg","payload":{"type":"token_count","info":{
            "total_token_usage":{"input_tokens":200,"cached_input_tokens":100,"output_tokens":20},
            "last_token_usage":{"input_tokens":110,"output_tokens":5,"total_tokens":115}
        }}});
        std::fs::write(
            &path,
            format!(
                "{}\n{info}\n{{\"type\":\"unrelated\"}}\n",
                "x".repeat(MAX_TAIL_BYTES as usize + 1)
            ),
        )
        .unwrap();
        assert_eq!(find_session(d.path(), session), Some(path.clone()));
        let snapshot = read_snapshot(&path).unwrap();
        assert_eq!(snapshot.total.input, 200);
        assert_eq!(snapshot.context, 115);
        let mut accounting = Accounting {
            previous: snapshot.total,
            ..Default::default()
        };
        let mut reply = Reply {
            usage: Usage {
                input: 310,
                cache_read: 200,
                output: 25,
                ..Default::default()
            },
            ..Default::default()
        };
        accounting.account(&mut reply, None);
        assert_eq!(
            (reply.usage.input, reply.usage.cache_read, reply.usage.output),
            (10, 100, 5)
        );
    }
}
