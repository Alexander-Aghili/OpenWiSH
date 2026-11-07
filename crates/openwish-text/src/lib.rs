//! Small text helpers shared by every crate: character-safe truncation and Markdown code fences.
//! No dependencies, so the lowest crates (process, tensor) can use it.

/// The first `n` characters of `s` (not bytes, so a multi-byte character is never split).
pub fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The last `n` characters of `s`.
pub fn tail(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    match s.char_indices().rev().nth(n - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// `s`, or its first `n` characters and a note "[<what> cut at <n> characters]" when it is longer.
/// `what` names the text ("output"), or is empty.
pub fn clip(s: &str, n: usize, what: &str) -> String {
    let h = head(s, n);
    if h.len() == s.len() {
        return s.to_string();
    }
    let what = if what.is_empty() {
        String::new()
    } else {
        format!("{what} ")
    };
    format!("{h}\n[{what}cut at {n} characters]")
}

/// One fenced code block of a Markdown reply.
#[derive(Clone, Debug, PartialEq)]
pub struct Fence {
    /// The first word after the opening backticks ("python", "forward", ...), or "".
    pub info: String,
    /// The lines between the fences, each ending in '\n'.
    pub body: String,
    /// False for a block still open at the end of the text.
    pub closed: bool,
}

/// Every fenced block of `text`, in order. A fence opens with three or more backticks and closes with a line
/// of at least as many backticks and nothing else (CommonMark), so a block can quote shorter fences. Text
/// inside a block is never read as a new fence. A block still open at the end counts, with `closed: false`.
pub fn fences(text: &str) -> Vec<Fence> {
    let mut out = vec![];
    let mut open: Option<(usize, Fence)> = None;
    for line in text.lines() {
        let t = line.trim_start();
        let ticks = t.chars().take_while(|c| *c == '`').count();
        match &mut open {
            None if ticks >= 3 => {
                let info = t[ticks..].split_whitespace().next().unwrap_or("").to_string();
                open = Some((
                    ticks,
                    Fence {
                        info,
                        body: String::new(),
                        closed: false,
                    },
                ));
            }
            None => {}
            Some((n, f)) => {
                if ticks >= *n && t[ticks..].trim().is_empty() {
                    let (_, mut f) = open.take().expect("open fence");
                    f.closed = true;
                    out.push(f);
                } else {
                    f.body.push_str(line);
                    f.body.push('\n');
                }
            }
        }
    }
    out.extend(open.map(|(_, f)| f));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_and_tail_count_characters() {
        assert_eq!(head("héllo", 2), "hé");
        assert_eq!(head("hi", 10), "hi");
        assert_eq!(tail("héllo", 3), "llo");
        assert_eq!(tail("héllo", 4), "éllo");
        assert_eq!(tail("hi", 10), "hi");
        assert_eq!(tail("hi", 0), "");
    }

    #[test]
    fn clip_adds_a_note_only_when_it_cuts() {
        assert_eq!(clip("short", 10, "output"), "short");
        assert_eq!(clip("abcdef", 3, "output"), "abc\n[output cut at 3 characters]");
        assert_eq!(clip("ééé", 2, ""), "éé\n[cut at 2 characters]");
    }

    #[test]
    fn fences_follow_commonmark_closing() {
        let t = "text\n```python\nx = 1\n```\n````md\n```inner\n````\n```edits\nW[0] = 1\n";
        let f = fences(t);
        assert_eq!(f.len(), 3);
        assert_eq!(
            (f[0].info.as_str(), f[0].body.as_str(), f[0].closed),
            ("python", "x = 1\n", true)
        );
        assert_eq!((f[1].info.as_str(), f[1].body.as_str()), ("md", "```inner\n"));
        assert_eq!(
            (f[2].info.as_str(), f[2].body.as_str(), f[2].closed),
            ("edits", "W[0] = 1\n", false)
        );
    }

    #[test]
    fn a_closing_line_must_be_bare_backticks() {
        let f = fences("```python\nprint(1)\n```python\nstill code\n```\n");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].body, "print(1)\n```python\nstill code\n");
    }
}
