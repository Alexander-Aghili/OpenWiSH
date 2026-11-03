//! `openwish envs`: a table of environments. `openwish tools`: the tool surface of one task.
use std::fmt::Write;
use std::path::{Path, PathBuf};

use openwish_core::config::ProjectConfig;
use openwish_core::hooks::{Event, HooksSpec};
use openwish_core::spec::{ForwardSpec, PythonSpec, TaskSpec, PYTHON_POOL};

/// Every .toml file under `dir` that could be a task, sorted. Skips hidden folders, __pycache__ and openwish.toml.
pub fn task_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            // symbolic links to folders are not walked, so a link loop cannot trap the walk
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                _ if name.ends_with(".toml") && name != openwish_core::config::FILE_NAME => out.push(p),
                _ => {}
            }
        }
    }
    out.sort();
    out
}

/// The files a task names that are not there.
pub fn missing_assets(t: &TaskSpec) -> Vec<String> {
    let mut files: Vec<(String, &Path)> = t
        .required_files()
        .into_iter()
        .map(|(what, p)| (what.to_string(), p))
        .collect();
    files.extend(
        t.python
            .as_ref()
            .and_then(|py| py.venv.as_deref())
            .map(|v| ("venv".to_string(), v)),
    );
    files.extend(
        t.mounts
            .iter()
            .map(|m| (format!("mount {}", m.guest), m.host.as_path())),
    );
    let mut miss: Vec<String> = files
        .into_iter()
        .filter(|(_, p)| !t.resolve(p).exists())
        .map(|(what, _)| what)
        .collect();
    let hidden = t.hidden.iter().filter(|h| !t.resolve(h).exists()).count();
    if hidden > 0 {
        miss.push(format!("{hidden} hidden"));
    }
    miss
}

/// The items joined with ", ", or "-" when there are none.
fn cell(items: Vec<String>) -> String {
    if items.is_empty() {
        "-".into()
    } else {
        items.join(", ")
    }
}

pub fn tools_cell(t: &TaskSpec) -> String {
    let mut v = t.enabled_tools();
    if t.edits.language {
        v.push("edits block".into());
    }
    cell(v)
}

pub fn budgets_cell(t: &TaskSpec) -> String {
    let mut v = vec![];
    if let Some(py) = &t.python {
        v.push(format!("python {}", py.runs));
    }
    v.extend(t.groups().into_iter().map(|(g, _, n)| format!("{g} {n}")));
    cell(v)
}

/// Rows as left-aligned columns two spaces apart (the last column is not padded).
fn table(rows: &[Vec<String>]) -> String {
    let n = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let width = |i: usize| {
        rows.iter()
            .map(|r| r.get(i).map(|c| c.chars().count()).unwrap_or(0))
            .max()
            .unwrap_or(0)
    };
    let w: Vec<usize> = (0..n).map(width).collect();
    let mut s = String::new();
    for r in rows {
        let cells: Vec<String> = r
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i + 1 == r.len() {
                    c.clone()
                } else {
                    format!("{c:<width$}", width = w[i])
                }
            })
            .collect();
        s += cells.join("  ").trim_end();
        s.push('\n');
    }
    s
}

/// The environments under `dir` as a table.
pub fn list(dir: &Path) -> String {
    let mut rows = vec![["ID", "TOOLS", "BUDGETS", "ASSETS", "FILE"].map(String::from).to_vec()];
    for f in task_files(dir) {
        let file = f.strip_prefix(dir).unwrap_or(&f).display().to_string();
        match TaskSpec::parse(&f) {
            Ok(t) => {
                let miss = missing_assets(&t);
                let assets = if miss.is_empty() {
                    "ok".to_string()
                } else {
                    format!("missing: {}", miss.join(", "))
                };
                rows.push(vec![t.id.clone(), tools_cell(&t), budgets_cell(&t), assets, file]);
            }
            Err(e) => {
                let why: String = format!("{e:#}").replace('\n', " ").chars().take(70).collect();
                rows.push(vec![
                    "-".into(),
                    "-".into(),
                    "-".into(),
                    format!("not a task: {why}"),
                    file,
                ]);
            }
        }
    }
    table(&rows)
}

/// What the agent will see in a task: its tools, budgets, pools, edit policy, finish rules and hooks.
pub fn tools(t: &TaskSpec, file: &Path, cfg: &ProjectConfig) -> String {
    let mut s = format!("task {} ({})\n", t.id, file.display());
    if !t.description.is_empty() {
        s += &format!("  {}\n", t.description);
    }
    s += &tools_section(t);
    s += &network_section(t);
    s += &edits_section(t);
    s += &finish_section(t);
    s += &hooks_section(&cfg.hooks, &t.hooks);
    s
}

fn tools_section(t: &TaskSpec) -> String {
    let mut s = "\nTools (one fenced block per reply; a reply with no tool block is the final answer):\n".to_string();
    if t.enabled_tools().is_empty() {
        s += "  (none)\n";
    }
    if let Some(py) = &t.python {
        s += &python_line(t, py);
    }
    for (tag, f) in &t.forward {
        s += &probe_line(t, tag, f);
    }
    s
}

fn python_line(t: &TaskSpec, py: &PythonSpec) -> String {
    let keep = if py.fresh {
        "fresh process per run (nothing kept)"
    } else {
        "one kernel (variables kept between runs)"
    };
    let mut s = format!(
        "  ```python      Python kernel: {} runs, {keep}; timeout {} s, {} GB, at most {} threads, output cut at {} chars\n",
        py.runs, py.timeout_s, py.mem_gb, py.threads, py.max_out
    );
    if let Some(p) = &py.prelude {
        s += &format!("                 prelude {}\n", p.display());
    }
    for m in &t.mounts {
        s += &format!("                 mount {} -> {}\n", m.host.display(), m.guest);
    }
    s
}

fn probe_line(t: &TaskSpec, tag: &str, f: &ForwardSpec) -> String {
    let budget = match f.pool.as_deref() {
        Some(PYTHON_POOL) => format!(
            "draws from the Python runs ({})",
            t.python.as_ref().map(|p| p.runs).unwrap_or(0)
        ),
        Some(pool) => {
            let others: Vec<&str> = t
                .forward
                .iter()
                .filter(|(k, g)| *k != tag && g.pool.as_deref() == Some(pool))
                .map(|(k, _)| k.as_str())
                .collect();
            let shared = if others.is_empty() {
                String::new()
            } else {
                format!(", shared with {}", others.join(", "))
            };
            format!("pool \"{pool}\": {} passes{shared}", f.passes)
        }
        None => format!("{} passes", f.passes),
    };
    format!(
        "  {:<14} probe: {budget}; files in /work/{}; timeout {} s, summary cut at {} chars\n",
        format!("```{tag}"),
        f.folder_pattern(tag),
        f.timeout_s,
        f.max_out
    )
}

fn network_section(t: &TaskSpec) -> String {
    let line = match &t.network {
        Some(n) => {
            let missing = if t.resolve(&n.base).exists() { "" } else { " (missing)" };
            format!("  base {}{missing}\n", n.base.display())
        }
        None => "  (none: nothing to edit)\n".into(),
    };
    format!("\nNetwork:\n{line}")
}

fn edits_section(t: &TaskSpec) -> String {
    let p = &t.edits.policy;
    let mut s = "\nEdits (every change passes the edit gate):\n".to_string();
    let allowed = p.allowed.as_ref().map(|a| a.join(", ")).unwrap_or_else(|| "all".into());
    let forbidden = if p.forbidden.is_empty() {
        "none".into()
    } else {
        p.forbidden.join(", ")
    };
    let scaling = if p.forbid_whole_tensor_scale {
        "refused"
    } else {
        "allowed"
    };
    let max_frac = p
        .max_frac_changed
        .map(|f| f.to_string())
        .unwrap_or_else(|| "no limit".into());
    let block = if t.edits.language { "applied" } else { "not used" };
    let _ = write!(
        s,
        "  allowed tensors      {allowed}\n  forbidden tensors    {forbidden}\n  whole-tensor scaling {scaling}\n  \
         max fraction changed {max_frac}\n  ```edits block in the final reply: {block}\n"
    );
    s
}

fn finish_section(t: &TaskSpec) -> String {
    let fin = &t.finish;
    let early = if fin.require_all_passes {
        format!("every pass must be used first (at most {} reminders)\n", fin.max_nudges)
    } else {
        "may finish with budget left\n".into()
    };
    let h = &t.handoff;
    let mut when = vec![];
    if h.max_conversation_tokens > 0 {
        when.push(format!("at {} tokens", h.max_conversation_tokens));
    }
    if h.max_turns_per_conversation > 0 {
        when.push(format!("every {} turns", h.max_turns_per_conversation));
    }
    let handoff = if when.is_empty() {
        "never".into()
    } else {
        when.join(" or ")
    };
    format!(
        "\nFinish:\n  at most {} turns; {early}  handoff: new conversation {handoff}\n",
        fin.max_turns
    )
}

/// The project's hooks, then the task's.
fn hooks_section(project: &HooksSpec, task: &HooksSpec) -> String {
    let mut lines = String::new();
    for (spec, origin) in [(project, "openwish.toml"), (task, "task")] {
        for e in Event::ALL {
            for c in spec.commands(e) {
                lines += &format!("  {:<14} {:<14} {c}\n", origin, e.name());
            }
        }
    }
    if lines.is_empty() {
        lines = "  (none)\n".into();
    }
    format!("\nHooks (run on the host, outside the sandbox):\n{lines}")
}
