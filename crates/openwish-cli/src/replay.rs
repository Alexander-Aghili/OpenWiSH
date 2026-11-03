//! `openwish replay <run>`: a run turn by turn as text. The HTML page is in `html`.
use anyhow::Result;
use std::fmt::Write;
use std::path::{Path, PathBuf};

/// (turn number, folder) for every turn of the run, in order.
pub fn turn_dirs(run: &Path) -> std::io::Result<Vec<(u64, PathBuf)>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(run.join("turns"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    dirs.sort();
    Ok(dirs
        .into_iter()
        .map(|t| {
            let n = t
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .trim_start_matches('t')
                .parse()
                .unwrap_or(0);
            (n, t)
        })
        .collect())
}

/// Every turn: the harness's message, then the agent's reply or the provider's refusal. Each message is cut
/// at `width` characters (0 = whole).
pub fn text(run: &Path, width: usize) -> Result<String> {
    let cut = |s: &str| {
        if width == 0 || s.chars().count() <= width {
            s.to_string()
        } else {
            format!("{} [...]", s.chars().take(width).collect::<String>())
        }
    };
    let read = |p: PathBuf| std::fs::read_to_string(p).ok();
    let mut out = String::new();
    for (n, t) in turn_dirs(run)? {
        writeln!(out, "==================== turn {n} ====================")?;
        writeln!(
            out,
            "--- harness -> editor ---\n{}",
            cut(&read(t.join("prompt.txt")).unwrap_or_default())
        )?;
        if let Some(a) = read(t.join("answer.md")) {
            writeln!(out, "--- editor ---\n{}", cut(&a))?;
        } else if let Some(r) = read(t.join("refused.txt")) {
            writeln!(out, "--- refused by the provider ---\n{r}")?;
        }
    }
    Ok(out)
}
