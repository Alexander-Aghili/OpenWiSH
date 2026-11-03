use std::path::Path;
use std::process::Command;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn toy_run_score_replay_and_status() {
    if !root().join("assets/venvs/sandbox/bin/python").exists() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("run");
    let bin = env!("CARGO_BIN_EXE_openwish");
    let r = Command::new(bin)
        .current_dir(root())
        .args([
            "run",
            "tasks/toy/task.toml",
            "--backend",
            "scripted:tasks/toy/script.txt",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    );
    let score: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("eval/score.json")).unwrap()).unwrap();
    assert_eq!(score["score"], 1.0);
    // scoring on its own resolves {root} too
    std::fs::remove_file(out.join("eval/score.json")).unwrap();
    let r = Command::new(bin)
        .current_dir(root())
        .env_remove("OPENWISH_ROOT")
        .args(["score"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let r = Command::new(bin)
        .current_dir(root())
        .args(["replay"])
        .arg(&out)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&r.stdout);
    assert!(text.contains("turn 1") && text.contains("save_edit"), "{text}");
    let r = Command::new(bin)
        .current_dir(root())
        .args(["status"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&r.stdout).contains("done"),
        "{}",
        String::from_utf8_lossy(&r.stdout)
    );
    let r = Command::new(bin)
        .current_dir(root())
        .args(["render", "tasks/toy/net"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&r.stdout).contains("W (2 x 2)"));
    // a second run into the same folder is refused; --resume of a finished run is a no-op
    let r = Command::new(bin)
        .current_dir(root())
        .args([
            "run",
            "tasks/toy/task.toml",
            "--backend",
            "scripted:tasks/toy/script.txt",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!r.status.success());
    let r = Command::new(bin)
        .current_dir(root())
        .args(["run", "--resume"])
        .arg(&out)
        .args(["--backend", "scripted:tasks/toy/script.txt"])
        .output()
        .unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
}
