//! Path helpers. Paths that come from the sandbox side (RPC file names, notes, probe output folders) are untrusted
//! on the host: `file_under` and `dir_under` accept a path only if it stays inside a given root without passing
//! through a symlink.
use anyhow::{bail, Context, Result};
use std::path::{Component, Path, PathBuf};

/// The canonical folder holding the file `path` (the current folder for a bare file name).
pub fn folder_of(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(std::fs::canonicalize(parent)?)
}

/// `p` (absolute, or relative to `root`) if it is an existing regular file inside `root`, with no symlink
/// anywhere between `root` and the file. Returns the canonical path.
pub fn file_under(root: &Path, p: &Path) -> Result<PathBuf> {
    let root_c = std::fs::canonicalize(root).with_context(|| format!("{}", root.display()))?;
    let full = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
    if full.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!("{} contains '..'", p.display());
    }
    let rel = full
        .strip_prefix(root)
        .or_else(|_| full.strip_prefix(&root_c))
        .map_err(|_| anyhow::anyhow!("{} is not inside {}", p.display(), root.display()))?
        .to_path_buf();
    let parts: Vec<_> = rel.components().collect();
    if parts.is_empty() {
        bail!("{} names the folder itself, not a file", p.display());
    }
    let mut cur = root_c;
    for (i, c) in parts.iter().enumerate() {
        cur.push(c);
        let md = std::fs::symlink_metadata(&cur).with_context(|| format!("{} does not exist", p.display()))?;
        if md.file_type().is_symlink() {
            bail!("{} goes through a symbolic link", p.display());
        }
        let last = i + 1 == parts.len();
        if last && !md.is_file() {
            bail!("{} is not a regular file", p.display());
        }
        if !last && !md.is_dir() {
            bail!("{} is not inside a folder", p.display());
        }
    }
    Ok(cur)
}

/// `rel` as a directory inside `root` that may be created and emptied: `rel` is relative, has no `..`, and
/// every component that already exists is a real directory (not a symlink). Returns the joined path.
pub fn dir_under(root: &Path, rel: &str) -> Result<PathBuf> {
    let r = Path::new(rel);
    if rel.is_empty() || !r.components().all(|c| matches!(c, Component::Normal(_))) {
        bail!("folder {rel:?} must be a plain relative path");
    }
    let mut cur = root.to_path_buf();
    for c in r.components() {
        cur.push(c);
        match std::fs::symlink_metadata(&cur) {
            Ok(md) if md.file_type().is_symlink() => bail!("{} is a symbolic link", cur.display()),
            Ok(md) if !md.is_dir() => bail!("{} exists and is not a folder", cur.display()),
            _ => {}
        }
    }
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("root");
        let other = d.path().join("other");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(root.join("sub/a.txt"), "a").unwrap();
        std::fs::write(other.join("b.txt"), "b").unwrap();
        (d, root, other)
    }

    #[test]
    fn file_inside_is_accepted_relative_or_absolute() {
        let (_d, root, _) = setup();
        let want = std::fs::canonicalize(root.join("sub/a.txt")).unwrap();
        assert_eq!(file_under(&root, Path::new("sub/a.txt")).unwrap(), want);
        assert_eq!(file_under(&root, &root.join("sub/a.txt")).unwrap(), want);
    }

    #[test]
    fn file_outside_or_missing_or_a_directory_is_refused() {
        let (_d, root, other) = setup();
        assert!(file_under(&root, &other.join("b.txt")).is_err());
        assert!(file_under(&root, Path::new("../other/b.txt")).is_err());
        assert!(file_under(&root, Path::new("sub/missing.txt")).is_err());
        assert!(file_under(&root, Path::new("sub")).is_err());
    }

    #[test]
    fn symlinks_are_refused_for_files() {
        let (_d, root, other) = setup();
        symlink(other.join("b.txt"), root.join("link.txt")).unwrap();
        assert!(file_under(&root, Path::new("link.txt")).is_err());
        symlink(&other, root.join("linkdir")).unwrap();
        assert!(file_under(&root, Path::new("linkdir/b.txt")).is_err());
        // a symlink that points back inside the root is refused too
        symlink(root.join("sub/a.txt"), root.join("inner.txt")).unwrap();
        assert!(file_under(&root, Path::new("inner.txt")).is_err());
    }

    #[test]
    fn dirs_must_stay_inside_without_symlinks() {
        let (_d, root, other) = setup();
        assert_eq!(dir_under(&root, "pass_1").unwrap(), root.join("pass_1"));
        assert_eq!(dir_under(&root, "attn/r3").unwrap(), root.join("attn/r3"));
        assert!(dir_under(&root, "sub").is_ok(), "an existing real directory is fine");
        assert!(dir_under(&root, "../other").is_err());
        assert!(dir_under(&root, "/tmp/x").is_err());
        assert!(dir_under(&root, "").is_err());
        symlink(&other, root.join("attn")).unwrap();
        assert!(dir_under(&root, "attn/r3").is_err());
        assert!(dir_under(&root, "attn").is_err());
        assert!(
            dir_under(&root, "sub/a.txt").is_err(),
            "an existing file is not a directory"
        );
    }
}
