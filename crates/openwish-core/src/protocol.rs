//! The reply protocol: a reply with no tool block is a final answer; a reply with exactly one
//! enabled tool block (```python, ```games, ...) is one tool call. This is the authors' protocol,
//! so their prompts work unchanged with any text backend.

/// Tool names the harness knows. A block with one of these tags that the task has not enabled is
/// an error (so a stray ```games in a no-pass task is not mistaken for a final answer).
pub const KNOWN_TOOLS: &[&str] = &[
    "python",
    "games",
    "positions",
    "forward",
    "attention",
    "generate",
    "teacher_forced",
];

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Tool {
        tag: String,
        body: String,
    },
    Final(String),
    /// a protocol problem to report back to the editor; nothing is charged
    Error(String),
}

/// Fenced blocks as (info string, body).
pub fn blocks(reply: &str) -> Vec<(String, String)> {
    let mut out = vec![];
    let mut open: Option<(usize, String, String)> = None; // (fence length, info, body)
    for line in reply.lines() {
        let t = line.trim_start();
        let ticks = t.chars().take_while(|c| *c == '`').count();
        match &mut open {
            None => {
                if ticks >= 3 {
                    let info = t[ticks..].split_whitespace().next().unwrap_or("").to_string();
                    open = Some((ticks, info, String::new()));
                }
            }
            Some((n, _, body)) => {
                if ticks >= *n && t[ticks..].trim().is_empty() {
                    let (_, info, body) = open.take().unwrap();
                    out.push((info, body));
                } else {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }
    if let Some((_, info, body)) = open {
        out.push((info, body));
    }
    out
}

/// Info strings that look like an attempt to run code but are not a tool tag.
pub const NEAR_MISSES: &[&str] = &[
    "py", "py3", "python3", "ipython", "jupyter", "sh", "bash", "shell", "zsh", "console",
];

/// The tool named by a native tool-call tag such as `<invoke name="python">`, if it is one of ours.
fn invoked_tool(reply: &str) -> Option<String> {
    for (pos, _) in reply.match_indices("<invoke") {
        let rest = &reply[pos..];
        let Some(n) = rest.find("name=\"") else { continue };
        let name: String = rest[n + 6..].chars().take_while(|c| *c != '"').collect::<String>().to_ascii_lowercase();
        if KNOWN_TOOLS.contains(&name.as_str()) {
            return Some(name);
        }
    }
    None
}

pub fn parse(reply: &str, enabled: &[String]) -> Action {
    let bs: Vec<(String, String)> = blocks(reply)
        .into_iter()
        .map(|(i, b)| (i.to_ascii_lowercase(), b))
        .collect();
    if let Some((tag, _)) = bs.iter().find(|(i, _)| NEAR_MISSES.contains(&i.as_str())) {
        let hint = if enabled.iter().any(|t| t == "python") {
            "To run code, use a ```python block."
        } else {
            "This task has no code tool."
        };
        return Action::Error(format!(
            "A ```{tag} block is not a tool here. {hint} Nothing was run and nothing was counted; your reply was not taken as a final answer."
        ));
    }
    let tools: Vec<&(String, String)> = bs.iter().filter(|(i, _)| KNOWN_TOOLS.contains(&i.as_str())).collect();
    if let Some((tag, _)) = tools.iter().find(|(i, _)| !enabled.contains(i)) {
        let avail = if enabled.is_empty() {
            "this task has no tools".to_string()
        } else {
            format!("available: {}", enabled.join(", "))
        };
        return Action::Error(format!(
            "The ```{tag} tool is not available in this task ({avail}). Nothing was run and nothing was counted."
        ));
    }
    if tools.is_empty() {
        // a tool fence that does not start its line (e.g. "<b>```python") is a malformed tool call, not a final answer
        for line in reply.lines() {
            let t = line.trim_start();
            if t.starts_with("```") {
                continue;
            }
            if let Some(pos) = t.find("```") {
                let tag = t[pos + 3..].split_whitespace().next().unwrap_or("").to_ascii_lowercase();
                if KNOWN_TOOLS.contains(&tag.as_str()) {
                    return Action::Error(format!(
                        "Your reply has a ```{tag} fence after other text on the same line. A tool block must start on its own line with ```{tag}. Nothing was run and nothing was counted; your reply was not taken as a final answer."
                    ));
                }
            }
        }
        // a native tool-call format (e.g. <invoke name="python"><parameter name="code">) is not this protocol
        if let Some(tag) = invoked_tool(reply) {
            return Action::Error(format!(
                "Your reply uses a tool-call format (<invoke name=\"{tag}\">) that this task does not read. To call a tool, reply with exactly one block that starts with ```{tag} on its own line. Nothing was run and nothing was counted; your reply was not taken as a final answer."
            ));
        }
    }
    match tools.len() {
        0 => Action::Final(reply.to_string()),
        1 => Action::Tool { tag: tools[0].0.clone(), body: tools[0].1.clone() },
        n => Action::Error(format!(
            "Your reply had {n} tool blocks. Reply with exactly one tool block per message (or none for your final answer). Nothing was run and nothing was counted."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn en(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn one_block_is_a_tool_call() {
        let r = "I will look at layer 1.\n```python\nprint(1)\n```\n";
        assert_eq!(
            parse(r, &en(&["python"])),
            Action::Tool {
                tag: "python".into(),
                body: "print(1)\n".into()
            }
        );
    }

    #[test]
    fn no_tool_block_is_final_and_other_fences_are_text() {
        let r = "Final.\n```edits\nW1[0,0] = 1\n```\n```json\n{}\n```";
        assert_eq!(parse(r, &en(&["python"])), Action::Final(r.into()));
    }

    #[test]
    fn two_blocks_rejected() {
        let r = "```python\na=1\n```\n```games\n{}\n```";
        assert!(matches!(parse(r, &en(&["python", "games"])), Action::Error(m) if m.contains("exactly one")));
    }

    #[test]
    fn known_but_disabled_tool_is_an_error() {
        let r = "```games\n{\"n\": 4}\n```";
        assert!(matches!(parse(r, &en(&["python"])), Action::Error(m) if m.contains("games")));
    }

    #[test]
    fn tag_case_is_ignored_and_near_misses_are_errors() {
        assert_eq!(
            parse("```Python\nx=1\n```", &en(&["python"])),
            Action::Tool {
                tag: "python".into(),
                body: "x=1\n".into()
            }
        );
        for t in ["py", "python3", "ipython", "py3", "sh", "bash", "shell"] {
            let r = format!("```{t}\nx=1\n```");
            assert!(
                matches!(parse(&r, &en(&["python"])), Action::Error(m) if m.contains("```python")),
                "{t}"
            );
        }
        // ordinary fences in a final answer are still text
        assert!(matches!(parse("```json\n{}\n```", &en(&["python"])), Action::Final(_)));
    }

    #[test]
    fn a_tool_fence_after_text_on_its_line_is_an_error_not_a_final_answer() {
        // seen in a real run: the reply began "<b>```python"
        let r = "\n<b>```python\nprint(1)\n```\n";
        assert!(matches!(parse(r, &en(&["python"])), Action::Error(m) if m.contains("own line")));
        // seen in a real run: a native tool-call format instead of a fenced block
        let r = "\n<invoke name=\"python\">\n<parameter name=\"code\">\nprint(1)\n</parameter>\n</invoke>\n";
        assert!(matches!(parse(r, &en(&["python"])), Action::Error(m) if m.contains("```python")));
        // inline code that merely mentions a fence in prose about something else stays a final answer
        assert!(matches!(parse("Use a ```program block below.\n```program\nx=1\n```", &en(&["python"])), Action::Final(_)));
    }

    #[test]
    fn unterminated_block_and_info_string_options() {
        assert_eq!(
            parse("```python\nx=1", &en(&["python"])),
            Action::Tool {
                tag: "python".into(),
                body: "x=1\n".into()
            }
        );
        assert_eq!(
            parse("```python title\nx=1\n```", &en(&["python"])),
            Action::Tool {
                tag: "python".into(),
                body: "x=1\n".into()
            }
        );
        assert_eq!(blocks("````python\nx\n````").len(), 1);
    }
}
