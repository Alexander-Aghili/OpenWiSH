//! `openwish replay <run> --html <file>`: one self-contained HTML page for a run.
//! Everything read from the run folder is escaped. The page loads nothing and runs no script, and its
//! content security policy forbids both.
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use openwish_tensor::gate::LedgerLine;
use serde_json::Value;

/// Escape text for HTML element content and quoted attribute values.
pub fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + s.len() / 8);
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

fn read(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok()
}
fn jsonl(p: &Path) -> Vec<Value> {
    read(p)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}
fn pre(s: &str) -> String {
    format!("<pre>{}</pre>", escape(s))
}
fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default()
}

const STYLE: &str = "body{font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#1d1d1f;background:#fff}\
h1{font-size:1.6em;margin-bottom:.2em}h2{border-bottom:1px solid #ddd;padding-bottom:.2em;margin-top:2em}\
pre{white-space:pre-wrap;word-break:break-word;background:#f6f6f8;border:1px solid #e3e3e8;border-radius:6px;padding:.7em;font:13px/1.4 ui-monospace,monospace;max-height:40em;overflow:auto}\
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #e3e3e8;padding:.3em .5em;text-align:left;vertical-align:top}th{background:#f6f6f8}\
.turn{border:1px solid #ddd;border-radius:8px;padding:.2em 1em 1em;margin:1.2em 0}.meta{color:#666;font-size:13px}\
.agent pre{background:#eef5ff;border-color:#cfe0f7}.tool pre{background:#f3fbf1;border-color:#d4ebce}\
.bad{color:#b00020}.ok{color:#1b7f3b}summary{cursor:pointer;color:#444}";

/// The counters of `state.json` shown in the page's first table, in order.
use openwish_core::run::COUNTERS;

/// The whole page for the run in `run`.
pub fn render(run: &Path) -> Result<String> {
    let saved: Value = serde_json::from_str(&read(&run.join("spec.json")).context("not a run folder (no spec.json)")?)?;
    let spec = &saved["spec"];
    let id = escape(spec["id"].as_str().unwrap_or("?"));
    let state: Value = read(&run.join("state.json"))
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let mut h = String::new();
    h += "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n";
    h += "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">\n";
    h += &format!("<title>OpenWiSH run: {id}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n");
    h += &format!("<h1>Run of {id}</h1>\n");
    if let Some(d) = spec["description"].as_str().filter(|d| !d.is_empty()) {
        h += &format!("<p>{}</p>\n", escape(d));
    }
    h += &counters(run, &state);
    h += "<h2 id=\"score\">Score</h2>\n";
    h += &read(&run.join("eval/score.json"))
        .map(|s| pre(&s))
        .unwrap_or_else(|| "<p>not scored</p>\n".into());
    h += &edits(run);
    h += "<h2 id=\"audit\">Audit</h2>\n";
    h += &read(&run.join("audit_summary.json"))
        .map(|s| pre(&s))
        .unwrap_or_else(|| "<p>not audited (run <code>openwish audit</code>)</p>\n".into());
    h += &hooks(&jsonl(&run.join("hooks.jsonl")));
    h += &turns(run);
    h += "</body>\n</html>\n";
    Ok(h)
}

/// The run's folder, state and counters.
fn counters(run: &Path, state: &Value) -> String {
    let mut h = "<table>\n".to_string();
    let mut row = |k: &str, v: String| h += &format!("<tr><th>{}</th><td>{}</td></tr>\n", escape(k), escape(&v));
    row("folder", run.display().to_string());
    let done = state["done"] == true;
    row(
        "state",
        if done {
            format!("done ({})", state["end_reason"].as_str().unwrap_or(""))
        } else {
            "running or stopped".into()
        },
    );
    for k in COUNTERS.iter().filter(|k| !state[**k].is_null() && **k != "cost_usd") {
        row(k, state[*k].to_string());
    }
    row("cost", format!("${:.2}", state["cost_usd"].as_f64().unwrap_or(0.0)));
    h + "</table>\n"
}

/// The ledger as a table, and the final edits block report if there is one.
fn edits(run: &Path) -> String {
    let mut h = "<h2 id=\"edits\">Edits</h2>\n".to_string();
    let ledger: Vec<LedgerLine> = read(&run.join("edits/ledger.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    if ledger.is_empty() {
        h += "<p>no edits</p>\n";
    } else {
        h += "<table>\n<tr><th>edit</th><th>tensor</th><th>op</th><th>entries changed</th><th>differ from base</th><th>source</th><th>reason</th></tr>\n";
        for l in &ledger {
            h += &match l {
                LedgerLine::Edit(r) => format!(
                    "<tr><td>#{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} ({:.4}%)</td><td>{}</td><td>{}</td></tr>\n",
                    r.id,
                    escape(r.op.tensor()),
                    escape(&r.op.describe()),
                    r.changed,
                    r.changed_vs_base,
                    r.frac_changed_vs_base * 100.0,
                    escape(&r.source),
                    escape(&r.reason)
                ),
                LedgerLine::Checkpoint { name, after_id, .. } => {
                    format!("<tr><td colspan=\"7\">checkpoint {} after edit #{after_id}</td></tr>\n", escape(name))
                }
            };
        }
        h += "</table>\n";
    }
    if let Some(r) = read(&run.join("final/edits_report.json")) {
        h += &format!(
            "<details><summary>final edits block report</summary>{}</details>\n",
            pre(&r)
        );
    }
    h
}

/// Every hook that ran, if any did.
fn hooks(log: &[Value]) -> String {
    if log.is_empty() {
        return String::new();
    }
    let mut h = "<h2 id=\"hooks\">Hooks</h2>\n<table>\n<tr><th>turn</th><th>event</th><th>tool</th><th>command</th><th>result</th><th>stderr</th></tr>\n".to_string();
    for v in log {
        let ok = v["ok"] == true;
        let res = if ok {
            "ok".to_string()
        } else if v["timed_out"] == true {
            "timed out".into()
        } else {
            format!("exit {}", v["code"])
        };
        h += &format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td><td class=\"{}\">{}</td><td>{}</td></tr>\n",
            v["turn"],
            escape(v["event"].as_str().unwrap_or("")),
            escape(v["tool"].as_str().unwrap_or("")),
            escape(v["command"].as_str().unwrap_or("")),
            if ok { "ok" } else { "bad" },
            escape(&res),
            escape(v["stderr"].as_str().unwrap_or("").trim())
        );
    }
    h + "</table>\n"
}

/// Every turn: its metadata, the harness's message, the agent's reply, and the tool calls it made.
fn turns(run: &Path) -> String {
    let calls = jsonl(&run.join("tool_calls.jsonl"));
    let audit: BTreeMap<u64, Value> = jsonl(&run.join("audit.jsonl"))
        .into_iter()
        .filter_map(|v| Some((v["call"].as_u64()?, v)))
        .collect();
    let mut h = "<h2 id=\"turns\">Turns</h2>\n".to_string();
    for (n, t) in crate::replay::turn_dirs(run).unwrap_or_default() {
        h += &format!("<article class=\"turn\" id=\"turn-{n}\">\n<h3>Turn {n}</h3>\n");
        if let Some(meta) = read(&t.join("run.json")).and_then(|s| serde_json::from_str::<Value>(&s).ok()) {
            h += &turn_meta(&meta);
        }
        let prompt = read(&t.join("prompt.txt")).unwrap_or_default();
        h += &format!(
            "<details><summary>Harness to agent ({} characters)</summary>{}</details>\n",
            prompt.chars().count(),
            pre(&prompt)
        );
        if let Some(a) = read(&t.join("answer.md")) {
            h += &format!("<div class=\"agent\"><h4>Agent</h4>{}</div>\n", pre(&a));
        } else if let Some(r) = read(&t.join("refused.txt")) {
            h += &format!(
                "<div class=\"agent\"><h4 class=\"bad\">Refused by the provider</h4>{}</div>\n",
                pre(&r)
            );
        }
        for (i, c) in calls.iter().enumerate().filter(|(_, c)| c["turn"].as_u64() == Some(n)) {
            h += &tool_call(c, audit.get(&(i as u64)));
        }
        h += "</article>\n";
    }
    h
}

/// Editor, tokens, cost and time of one turn (its run.json).
fn turn_meta(meta: &Value) -> String {
    let u = &meta["usage"];
    let count = |k: &str| u[k].as_u64().unwrap_or(0);
    format!(
        "<p class=\"meta\">{}{}; tokens in {} out {}; ${:.2}; {:.1} s</p>\n",
        escape(meta["editor"].as_str().unwrap_or("")),
        if meta["new_conversation"] == true {
            ", new conversation"
        } else {
            ""
        },
        count("input") + count("cache_read") + count("cache_write"),
        count("output"),
        meta["cost_usd"].as_f64().unwrap_or(0.0),
        meta["wall_s"].as_f64().unwrap_or(0.0)
    )
}

/// One tool call with its input, its result and the auditor's verdict.
fn tool_call(c: &Value, audit: Option<&Value>) -> String {
    let kind = c["kind"].as_str().unwrap_or("tool");
    let what = if c["blocked"] == true {
        "blocked by a pre_tool hook, not counted".to_string()
    } else if c["interrupted"] == true {
        "interrupted by a harness restart".to_string()
    } else if let Some(k) = c["run"].as_u64() {
        format!("run {k}")
    } else if let Some(k) = c["pass"].as_u64() {
        format!("pass {k}")
    } else {
        "failed, not counted".to_string()
    };
    let mut h = format!(
        "<div class=\"tool\"><h4>Tool call: {} ({})</h4>\n",
        escape(kind),
        escape(&what)
    );
    let input = c
        .get("code")
        .or_else(|| c.get("request"))
        .and_then(|v| v.as_str())
        .or_else(|| c["input"]["code"].as_str())
        .or_else(|| c["input"]["request"].as_str());
    if let Some(code) = input {
        h += &pre(code);
    }
    if let Some(o) = c["output"].as_str() {
        h += &format!("<h5>Output</h5>{}", pre(o));
    } else if !c["summary"].is_null() {
        h += &format!("<h5>Summary</h5>{}", pre(&pretty(&c["summary"])));
    } else if let Some(r) = c["reason"].as_str() {
        h += &format!("<h5>Reason</h5>{}", pre(r));
    }
    if let Some(a) = audit {
        let v = a["verdict"].as_str().unwrap_or("");
        h += &format!(
            "<p>audit: <strong class=\"{}\">{}</strong> {}</p>\n",
            if v == "allowed" { "ok" } else { "bad" },
            escape(v),
            escape(a["reason"].as_str().unwrap_or(""))
        );
    }
    h + "</div>\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_covers_markup_and_quotes() {
        assert_eq!(
            escape("<script>alert(\"x\" & 'y')</script>"),
            "&lt;script&gt;alert(&quot;x&quot; &amp; &#39;y&#39;)&lt;/script&gt;"
        );
        assert_eq!(escape("plain"), "plain");
    }
}
