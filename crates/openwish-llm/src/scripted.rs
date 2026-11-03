//! A backend that replays fixed replies, for tests and dry runs. Replies in a file are separated
//! by lines consisting of exactly `=====`.
use crate::{BackendError, Editor, Reply};

pub struct Scripted {
    pub replies: Vec<String>,
    pub next: usize,
    pub received: Vec<String>,
    session: Option<String>,
}

impl Scripted {
    pub fn new(replies: Vec<String>) -> Self {
        Scripted {
            replies,
            next: 0,
            received: vec![],
            session: None,
        }
    }
    pub fn from_file(p: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(p)?;
        let mut replies = vec![];
        let mut cur: Vec<&str> = vec![];
        for l in s.lines() {
            if l == "=====" {
                replies.push(cur.join("\n"));
                cur.clear();
            } else {
                cur.push(l);
            }
        }
        if !cur.is_empty() {
            replies.push(cur.join("\n"));
        }
        Ok(Self::new(replies))
    }
}

impl Editor for Scripted {
    fn name(&self) -> String {
        "scripted".into()
    }
    fn send(&mut self, msg: &str, _new_conversation: bool) -> Result<Reply, BackendError> {
        self.received.push(msg.to_string());
        let Some(text) = self.replies.get(self.next).cloned() else {
            return Err(BackendError::Fatal(format!(
                "scripted backend has no reply #{}",
                self.next + 1
            )));
        };
        self.next += 1;
        self.session = Some(format!("scripted-{}", self.next));
        Ok(Reply {
            text,
            session: self.session.clone(),
            tool_free: true,
            ..Default::default()
        })
    }
    fn session(&self) -> Option<String> {
        self.session.clone()
    }
    fn set_session(&mut self, s: Option<String>) {
        // resuming a scripted run continues at the recorded position: "scripted-<n>"
        if let Some(n) = s
            .as_deref()
            .and_then(|s| s.strip_prefix("scripted-"))
            .and_then(|n| n.parse().ok())
        {
            self.next = n;
        }
        self.session = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_in_order_then_errors() {
        let mut s = Scripted::new(vec!["a".into(), "b".into()]);
        assert_eq!(s.send("m1", true).unwrap().text, "a");
        assert_eq!(s.send("m2", false).unwrap().text, "b");
        assert!(matches!(s.send("m3", false), Err(BackendError::Fatal(_))));
        assert_eq!(s.received, vec!["m1", "m2", "m3"]);
    }

    #[test]
    fn file_format_and_resume() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("s.txt");
        std::fs::write(&p, "first\nline\n=====\nsecond\n").unwrap();
        let mut s = Scripted::from_file(&p).unwrap();
        assert_eq!(s.replies, vec!["first\nline", "second"]);
        let r = s.send("x", true).unwrap();
        let mut s2 = Scripted::from_file(&p).unwrap();
        s2.set_session(r.session);
        assert_eq!(s2.send("y", false).unwrap().text, "second");
    }
}
