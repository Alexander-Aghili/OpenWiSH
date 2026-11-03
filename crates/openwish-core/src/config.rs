//! Project config: an optional `openwish.toml` at the workspace root.
//!
//! ```toml
//! backend = "codex"        # claude | codex | api | scripted:<file>
//! model = "gpt-5.5"
//! effort = "high"
//! threads = 4              # CPU threads for the harness and its child processes
//! nice = 10                # process priority, 0-19
//!
//! [hooks]
//! pre_tool = "scripts/check_call.sh"
//! on_edit = ["echo edit $OPENWISH_EDIT_ID >> edits.log"]
//! ```
//!
//! Precedence, highest first:
//! - backend, model, effort: command-line flags, then the task's `[editor]` table, then openwish.toml, then
//!   the built-in defaults (backend `claude`; model `claude-opus-5-5` for the claude and api backends).
//!   The task sits above the project file because it is the more specific of the two.
//! - threads, nice: command-line flags, then OPENWISH_THREADS / OPENWISH_NICE, then openwish.toml, then the
//!   built-in defaults (a quarter of the cores; nice 10).
//! - hooks do not override each other: project hooks and task hooks both run, project hooks first.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::hooks::HooksSpec;
use crate::spec::EditorDefaults;

pub const FILE_NAME: &str = "openwish.toml";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    pub backend: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub threads: Option<usize>,
    pub nice: Option<i32>,
    #[serde(default)]
    pub hooks: HooksSpec,
    /// set by `load`: the folder of the file, where project hooks run
    #[serde(skip)]
    pub dir: PathBuf,
}

impl ProjectConfig {
    pub fn load(path: &Path) -> Result<ProjectConfig> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut c: ProjectConfig = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        c.dir = crate::paths::folder_of(path)?;
        Ok(c)
    }

    /// The config to use: `explicit` if given (it must exist), else `<root>/openwish.toml` if it exists,
    /// else the empty config.
    pub fn find(explicit: Option<&Path>, root: Option<&Path>) -> Result<ProjectConfig> {
        if let Some(p) = explicit {
            return ProjectConfig::load(p);
        }
        match root.map(|r| r.join(FILE_NAME)).filter(|p| p.exists()) {
            Some(p) => ProjectConfig::load(&p),
            None => Ok(ProjectConfig::default()),
        }
    }
}

/// The backend, model and effort after applying the precedence above.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorChoice {
    pub backend: String,
    pub model: Option<String>,
    pub effort: Option<String>,
}

pub fn editor_choice(cli: &EditorDefaults, task: &EditorDefaults, cfg: &ProjectConfig) -> EditorChoice {
    // each field on its own: flag, then task, then project config
    let pick = |f: fn(&EditorDefaults) -> &Option<String>, c: &Option<String>| {
        f(cli).clone().or_else(|| f(task).clone()).or_else(|| c.clone())
    };
    let backend = pick(|e| &e.backend, &cfg.backend).unwrap_or_else(|| "claude".into());
    let built_in_model = match backend.as_str() {
        "claude" | "api" => Some(openwish_llm::DEFAULT_CLAUDE_MODEL.to_string()),
        _ => None,
    };
    EditorChoice {
        model: pick(|e| &e.model, &cfg.model).or(built_in_model),
        effort: pick(|e| &e.effort, &cfg.effort),
        backend,
    }
}

/// Threads: the flag, then OPENWISH_THREADS, then the config, then a quarter of `cores`. Zero means unset.
pub fn threads(cli: Option<usize>, env: Option<&str>, cfg: &ProjectConfig, cores: usize) -> usize {
    let env = env.and_then(|v| v.trim().parse::<usize>().ok());
    [cli, env, cfg.threads]
        .into_iter()
        .flatten()
        .find(|n| *n > 0)
        .unwrap_or_else(|| crate::cpu::default_threads(cores))
}

/// Priority: the flag, then OPENWISH_NICE, then the config, then 10.
pub fn nice(cli: Option<i32>, env: Option<&str>, cfg: &ProjectConfig) -> i32 {
    let env = env.and_then(|v| v.trim().parse::<i32>().ok());
    cli.or(env).or(cfg.nice).unwrap_or(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ed(b: Option<&str>, m: Option<&str>, e: Option<&str>) -> EditorDefaults {
        EditorDefaults {
            backend: b.map(String::from),
            model: m.map(String::from),
            effort: e.map(String::from),
        }
    }

    #[test]
    fn parses_every_field_and_rejects_unknown_ones() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(FILE_NAME);
        std::fs::write(&p, "backend = \"codex\"\nmodel = \"m1\"\neffort = \"high\"\nthreads = 3\nnice = 5\n[hooks]\npre_tool = \"true\"\n").unwrap();
        let c = ProjectConfig::load(&p).unwrap();
        assert_eq!(c.backend.as_deref(), Some("codex"));
        assert_eq!(c.model.as_deref(), Some("m1"));
        assert_eq!(c.effort.as_deref(), Some("high"));
        assert_eq!((c.threads, c.nice), (Some(3), Some(5)));
        assert_eq!(c.hooks.pre_tool, vec!["true".to_string()]);
        assert_eq!(c.dir, std::fs::canonicalize(d.path()).unwrap());
        std::fs::write(&p, "backend = \"codex\"\nbogus = 1\n").unwrap();
        let e = ProjectConfig::load(&p).unwrap_err();
        assert!(format!("{e:#}").contains("bogus"), "{e:#}");
        std::fs::write(&p, "[hooks]\npre_toll = \"x\"\n").unwrap();
        assert!(ProjectConfig::load(&p).is_err());
    }

    #[test]
    fn finds_the_file_at_the_root_or_an_explicit_path() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(
            ProjectConfig::find(None, Some(d.path())).unwrap(),
            ProjectConfig::default()
        );
        assert_eq!(ProjectConfig::find(None, None).unwrap(), ProjectConfig::default());
        std::fs::write(d.path().join(FILE_NAME), "effort = \"low\"\n").unwrap();
        assert_eq!(
            ProjectConfig::find(None, Some(d.path())).unwrap().effort.as_deref(),
            Some("low")
        );
        let other = d.path().join("other.toml");
        std::fs::write(&other, "effort = \"max\"\n").unwrap();
        assert_eq!(
            ProjectConfig::find(Some(&other), Some(d.path()))
                .unwrap()
                .effort
                .as_deref(),
            Some("max")
        );
        assert!(ProjectConfig::find(Some(&d.path().join("missing.toml")), Some(d.path())).is_err());
    }

    #[test]
    fn flags_beat_the_task_which_beats_the_config_which_beats_built_ins() {
        let cfg = ProjectConfig {
            backend: Some("codex".into()),
            model: Some("cfg-model".into()),
            effort: Some("low".into()),
            ..Default::default()
        };
        // nothing set anywhere: built-ins
        let c = editor_choice(&ed(None, None, None), &ed(None, None, None), &ProjectConfig::default());
        assert_eq!(
            c,
            EditorChoice {
                backend: "claude".into(),
                model: Some("claude-opus-5-5".into()),
                effort: None
            }
        );
        // only the config
        let c = editor_choice(&ed(None, None, None), &ed(None, None, None), &cfg);
        assert_eq!(
            c,
            EditorChoice {
                backend: "codex".into(),
                model: Some("cfg-model".into()),
                effort: Some("low".into())
            }
        );
        // the task beats the config, field by field
        let c = editor_choice(&ed(None, None, None), &ed(None, None, Some("xhigh")), &cfg);
        assert_eq!((c.backend.as_str(), c.effort.as_deref()), ("codex", Some("xhigh")));
        // a flag beats both
        let c = editor_choice(
            &ed(Some("api"), Some("flag-model"), Some("max")),
            &ed(None, None, Some("xhigh")),
            &cfg,
        );
        assert_eq!(
            c,
            EditorChoice {
                backend: "api".into(),
                model: Some("flag-model".into()),
                effort: Some("max".into())
            }
        );
        // the built-in model only applies to backends that have one
        let c = editor_choice(
            &ed(Some("codex"), None, None),
            &ed(None, None, None),
            &ProjectConfig::default(),
        );
        assert_eq!(c.model, None);
    }

    #[test]
    fn threads_and_nice_follow_flag_then_variable_then_config() {
        let cfg = ProjectConfig {
            threads: Some(6),
            nice: Some(3),
            ..Default::default()
        };
        assert_eq!(threads(None, None, &ProjectConfig::default(), 16), 4);
        assert_eq!(threads(None, None, &cfg, 16), 6);
        assert_eq!(threads(None, Some("2"), &cfg, 16), 2);
        assert_eq!(threads(Some(9), Some("2"), &cfg, 16), 9);
        assert_eq!(threads(None, Some("junk"), &cfg, 16), 6, "a bad variable is ignored");
        assert_eq!(threads(Some(0), None, &cfg, 16), 6, "zero threads is ignored");
        assert_eq!(nice(None, None, &ProjectConfig::default()), 10);
        assert_eq!(nice(None, None, &cfg), 3);
        assert_eq!(nice(None, Some("7"), &cfg), 7);
        assert_eq!(nice(Some(0), Some("7"), &cfg), 0);
        assert_eq!(nice(None, Some("x"), &cfg), 3);
    }
}
