//! The text edit language (the authors' ```edits format, Experiments 4 and 5):
//!
//! ```text
//! W1[12, 0] = 2.5                      # one entry
//! W1[0:6, 3] = [1, 0, -1, 0, 2, 0]     # explicit values along a range
//! b1[0:4] = -1                         # fill a range
//! W1[3:600:60, 0] = [...]              # strided range
//! W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   # 2-D list for a 2-D block
//! W3[:, 2] *= 2                        # also += and -=
//! ZERO actor4.W                        # whole tensor to 0
//! ```
//! Statements apply in order. A statement that cannot be read is skipped (and reported). Lists may
//! continue over several lines. Values must satisfy |v| <= 1e6.
use crate::ops::{EditOp, Idx, Mode, Values};

#[derive(Debug, Default, PartialEq)]
pub struct Parsed {
    pub ops: Vec<EditOp>,
    /// (1-based line of the statement, the statement text, why it was skipped)
    pub skipped: Vec<(usize, String, String)>,
}

pub const MAX_ABS: f64 = 1e6;

/// The body of the LAST ```edits block in a reply, if any.
pub fn last_edits_block(reply: &str) -> Option<String> {
    let mut last = None;
    let mut cur: Option<String> = None;
    for line in reply.lines() {
        let t = line.trim_end();
        match &mut cur {
            None => {
                if t.trim_start().starts_with("```edits") {
                    cur = Some(String::new());
                }
            }
            Some(buf) => {
                if t.trim_start().starts_with("```") {
                    last = cur.take();
                } else {
                    buf.push_str(line);
                    buf.push('\n');
                }
            }
        }
    }
    // an unterminated final block still counts
    cur.or(last)
}

fn start_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"^(ZERO\s+[A-Za-z_]|[A-Za-z_][A-Za-z0-9_./]*\s*(\[|\+=|-=|\*=|=))").unwrap())
}

fn stmt_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"^([A-Za-z_][A-Za-z0-9_./]*)\s*(?:\[([^\]]*)\])?\s*(\+=|-=|\*=|=)\s*(.+)$").unwrap()
    })
}

fn depth(s: &str) -> i64 {
    s.chars()
        .map(|c| match c {
            '[' => 1,
            ']' => -1,
            _ => 0,
        })
        .sum()
}

fn strip_comment(l: &str) -> &str {
    l.split('#').next().unwrap_or("")
}

fn parse_idx(s: &str) -> Result<Vec<Idx>, String> {
    if s.trim().is_empty() {
        return Ok(vec![]);
    }
    s.split(',').map(|p| parse_axis(p.trim())).collect()
}

/// One axis of an index: `i`, or a slice `start:end:step` with any part left out.
fn parse_axis(p: &str) -> Result<Idx, String> {
    if !p.contains(':') {
        return Ok(Idx::One(parse_bound(p)?.ok_or_else(|| "empty index".to_string())?));
    }
    let parts: Vec<&str> = p.split(':').collect();
    if parts.len() > 3 {
        return Err(format!("bad slice '{p}'"));
    }
    let part = |i: usize| parse_bound(parts.get(i).copied().unwrap_or(""));
    Ok(Idx::Range {
        start: part(0)?,
        end: part(1)?,
        step: part(2)?,
    })
}

/// A non-negative integer, or `None` for an empty slice part.
fn parse_bound(x: &str) -> Result<Option<usize>, String> {
    let x = x.trim();
    if x.is_empty() {
        Ok(None)
    } else {
        x.parse::<usize>().map(Some).map_err(|_| format!("bad index '{x}'"))
    }
}

fn parse_values(s: &str) -> Result<Values, String> {
    let s = s.trim();
    let check = |v: f64| -> Result<f64, String> {
        if !v.is_finite() || v.abs() > MAX_ABS {
            Err(format!("value {v} outside |v| <= 1e6"))
        } else {
            Ok(v)
        }
    };
    if s.starts_with('[') {
        if depth(s) != 0 {
            return Err("unbalanced brackets".into());
        }
        let flat = s.replace(['[', ']'], " ");
        let mut out = vec![];
        for tok in flat
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
        {
            out.push(check(tok.parse::<f64>().map_err(|_| format!("bad number '{tok}'"))?)?);
        }
        if out.is_empty() {
            return Err("empty list".into());
        }
        Ok(Values::List(out))
    } else {
        Ok(Values::Scalar(check(
            s.parse::<f64>().map_err(|_| format!("bad number '{s}'"))?,
        )?))
    }
}

fn parse_stmt(st: &str) -> Result<EditOp, String> {
    let st = st.trim();
    if let Some(rest) = st.strip_prefix("ZERO") {
        let name = rest.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || "_./".contains(c)) {
            return Err(format!("bad tensor name '{name}'"));
        }
        return Ok(EditOp::Assign {
            tensor: name.into(),
            index: vec![],
            mode: Mode::Set,
            values: Values::Scalar(0.0),
        });
    }
    let c = stmt_re().captures(st).ok_or_else(|| "not a statement".to_string())?;
    let mode = match &c[3] {
        "=" => Mode::Set,
        "+=" => Mode::Add,
        "-=" => Mode::Sub,
        _ => Mode::Mul,
    };
    Ok(EditOp::Assign {
        tensor: c[1].to_string(),
        index: parse_idx(c.get(2).map(|m| m.as_str()).unwrap_or(""))?,
        mode,
        values: parse_values(&c[4])?,
    })
}

pub fn parse(text: &str) -> Parsed {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Parsed::default();
    let mut i = 0;
    while i < lines.len() {
        let first = strip_comment(lines[i]).trim();
        if first.is_empty() {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut buf = first.to_string();
        i += 1;
        while depth(&buf) > 0 && i < lines.len() {
            let next = strip_comment(lines[i]).trim();
            if start_re().is_match(next) {
                break;
            }
            buf.push(' ');
            buf.push_str(next);
            i += 1;
        }
        match parse_stmt(&buf) {
            Ok(op) => out.ops.push(op),
            Err(e) => out.skipped.push((start, buf, e)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: &str) -> EditOp {
        let p = parse(s);
        assert!(p.skipped.is_empty(), "{:?}", p.skipped);
        assert_eq!(p.ops.len(), 1);
        p.ops.into_iter().next().unwrap()
    }
    fn rng(a: Option<usize>, b: Option<usize>, s: Option<usize>) -> Idx {
        Idx::Range {
            start: a,
            end: b,
            step: s,
        }
    }

    #[test]
    fn single_entry() {
        assert_eq!(
            one("embed.W[26, 5] = 1.25"),
            EditOp::Assign {
                tensor: "embed.W".into(),
                index: vec![Idx::One(26), Idx::One(5)],
                mode: Mode::Set,
                values: Values::Scalar(1.25)
            }
        );
    }

    #[test]
    fn ranges_lists_and_modes() {
        assert_eq!(
            one("actor_head.b[0:3] = [0.5, 2.0, -1.0]   # comment"),
            EditOp::Assign {
                tensor: "actor_head.b".into(),
                index: vec![rng(Some(0), Some(3), None)],
                mode: Mode::Set,
                values: Values::List(vec![0.5, 2.0, -1.0])
            }
        );
        assert_eq!(
            one("W1[3:600:60, 0] = 0"),
            EditOp::Assign {
                tensor: "W1".into(),
                index: vec![rng(Some(3), Some(600), Some(60)), Idx::One(0)],
                mode: Mode::Set,
                values: Values::Scalar(0.0)
            }
        );
        assert_eq!(
            one("actor0.W[:, 3] *= 0.5"),
            EditOp::Assign {
                tensor: "actor0.W".into(),
                index: vec![rng(None, None, None), Idx::One(3)],
                mode: Mode::Mul,
                values: Values::Scalar(0.5)
            }
        );
        assert!(matches!(
            one("embed.b[5] -= 0.1"),
            EditOp::Assign { mode: Mode::Sub, .. }
        ));
        assert!(matches!(one("b3 += 1e-2"), EditOp::Assign { mode: Mode::Add, .. }));
        assert_eq!(
            one("ZERO actor4.W"),
            EditOp::Assign {
                tensor: "actor4.W".into(),
                index: vec![],
                mode: Mode::Set,
                values: Values::Scalar(0.0)
            }
        );
    }

    #[test]
    fn multiline_2d_list() {
        let p = parse("W2[0:2, 0:3] = [[1, -1, 0.5],\n   [0, 2, 0]]\nb1[0] = 1");
        assert!(p.skipped.is_empty(), "{:?}", p.skipped);
        assert_eq!(p.ops.len(), 2);
        assert!(
            matches!(&p.ops[0], EditOp::Assign { values: Values::List(v), .. } if v == &vec![1., -1., 0.5, 0., 2., 0.])
        );
    }

    #[test]
    fn bad_statements_are_skipped_not_fatal() {
        let p = parse("W1[0,0] = 1\nthis is not a statement\nW1[0,1] = 1e7\nW1[0,2] = [1, 2\nW1[1,1] = 2");
        assert_eq!(p.ops.len(), 2, "{:?}", p);
        assert_eq!(p.skipped.len(), 3, "{:?}", p.skipped);
        assert_eq!(p.skipped[0].0, 2);
    }

    #[test]
    fn last_block_only() {
        let r = "intro\n```edits\nW1[0,0] = 1\n```\nmore\n```edits\nW1[0,0] = 2\n```\nbye";
        assert_eq!(last_edits_block(r).unwrap().trim(), "W1[0,0] = 2");
        assert!(last_edits_block("no block").is_none());
    }

    #[test]
    fn authors_replies_parse_cleanly() {
        for (f, n) in [
            ("exp4_answer.md", 51),
            ("exp5_breakout_answer.md", 92),
            ("exp5_seaquest_answer.md", 122),
        ] {
            let t = std::fs::read_to_string(format!("{}/tests/fixtures/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let p = parse(&last_edits_block(&t).unwrap());
            assert!(p.skipped.is_empty(), "{f}: {:?}", &p.skipped[..p.skipped.len().min(3)]);
            assert_eq!(p.ops.len(), n, "{f}");
        }
    }
}
