//! `openwish init env <dir>` and `openwish init config [path]`: templates to start from.
use anyhow::{bail, Context, Result};
use std::path::Path;

use openwish_tensor::{Network, Tensor};

const TASK_TOML: &str = r#"# An OpenWiSH environment. Paths are relative to this file.
#   run it:     openwish run {dir}/task.toml --backend scripted:{dir}/script.txt
#   inspect it: openwish tools {dir}/task.toml
id = {id}
description = "A starter environment: make y = W x + b compute y = x by editing W."
# the agent's first message, sent verbatim
prompt = "prompt.txt"
# files only the scorer reads; the harness refuses to mount them into the sandbox
hidden = ["answer.json"]

# the network the agent edits: a folder of .npy files, or one .npz file
[network]
base = "net"

# a Python kernel in the sandbox; each ```python block uses one run
[python]
runs = 5
timeout_s = 60
mem_gb = 4
# runs once in the kernel before the agent's first code
prelude = "prelude.py"

# a forward probe: a ```forward block sends its body to this command, which runs on the host,
# writes files to /work/forward_<k>/ and prints one JSON line ({"ok": true, ...}); a failed pass is not counted
[forward.forward]
command = ["python3", "-I", "{task}/probe.py"]
passes = 3

# every weight change goes through the edit gate, which enforces this policy
[edits.policy]
allowed = ["W"]
forbid_whole_tensor_scale = true

[finish]
max_turns = 20

# shell commands the harness runs on the host, in this folder (optional)
# [hooks]
# pre_tool = "./check_call.sh"   # exit non-zero to block a call; its stderr goes to the agent
# on_edit = "echo edit $OPENWISH_EDIT_ID on $OPENWISH_EDIT_TENSOR >> \"$OPENWISH_RUN_DIR/edits.log\""

# the scorer prints one JSON line: {"ok": true, "score": <number>, ...}
[score]
command = ["python3", "-I", "{task}/score.py"]
"#;

const PROMPT: &str = r#"TASK. The network computes y = W x + b, with W (2 x 2) and b (2). Both are all zeros now.
Make the network compute the identity, y = x for every input x. You may change W only.

Tools (one per reply):
- A ```python block runs code in a sandbox. load(name) reads a tensor, save_edit(name, array, reason)
  changes one, and show(name) prints one. You have 5 runs.
- A ```forward block with JSON such as {"inputs": [[1, 0], [0, 1]]} runs the network on those inputs.
  You have 3 passes.
Reply without a tool block when you are done.
"#;

const PRELUDE: &str = r#"# Runs once in the sandbox kernel before the agent's first code.
# load, save_edit, drop_edit, checkpoint, restore, edits and NAMES are already defined.
import numpy as np


def show(name):
    """Print one tensor with its shape."""
    a = load(name)
    print(f"{name} {a.shape}")
    print(np.array2string(a, precision=3))
"#;

const NPY_READER: &str = r#"def read_npy(path):
    """A float32 .npy file as (shape, flat list), without numpy."""
    import ast, struct
    b = open(path, "rb").read()
    hl = struct.unpack("<H", b[8:10])[0]
    header = ast.literal_eval(b[10:10 + hl].decode("latin1"))
    if header["descr"] != "<f4" or header["fortran_order"]:
        raise ValueError(f"{path}: expected little-endian float32 in C order")
    data = b[10 + hl:]
    return tuple(header["shape"]), list(struct.unpack("<%df" % (len(data) // 4), data))
"#;

const PROBE: &str = r#""""Forward probe: y = W x + b for each input in the request.

It runs on the host, outside the sandbox. It reads the request from $OPENWISH_REQ and the current
network from $OPENWISH_NET, writes outputs.json to $OPENWISH_OUT, and prints one JSON summary line.
"""
import json
import os

{reader}

def main():
    try:
        req = json.load(open(os.environ["OPENWISH_REQ"]))
        inputs = req["inputs"]
        if not isinstance(inputs, list) or len(inputs) > 64:
            raise ValueError("inputs must be a list of at most 64 pairs")
        net = os.environ["OPENWISH_NET"]
        _, w = read_npy(os.path.join(net, "W.npy"))
        _, b = read_npy(os.path.join(net, "b.npy"))
        outs = []
        for x in inputs:
            x0, x1 = (float(v) for v in x)
            outs.append([w[0] * x0 + w[1] * x1 + b[0], w[2] * x0 + w[3] * x1 + b[1]])
    except Exception as e:  # a bad request is reported, and the pass is not counted
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return
    json.dump({"inputs": inputs, "outputs": outs}, open(os.path.join(os.environ["OPENWISH_OUT"], "outputs.json"), "w"))
    print(json.dumps({"ok": True, "outputs": outs}))


main()
"#;

const SCORE: &str = r#""""Score: 1.0 if the final W matches the hidden answer, else 0.0.

It reads the final network from $OPENWISH_FINAL_NET and prints one JSON line:
{"ok": true, "score": <number>, ...}
"""
import json
import os

{reader}

key = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.json")))
_, w = read_npy(os.path.join(os.environ["OPENWISH_FINAL_NET"], "W.npy"))
want = [v for row in key["W"] for v in row]
err = max(abs(a - b) for a, b in zip(w, want))
print(json.dumps({"ok": True, "score": float(err < 1e-6), "max_abs_error": err}))
"#;

const SCRIPT: &str = r#"I will see what the network does now.
```forward
{"inputs": [[1, 0], [0, 1]]}
```
=====
Every output is zero and b is zero, so W must be the identity.
```python
import numpy as np
show('W')
save_edit('W', np.eye(2), reason='y = x needs W = I when b = 0')
```
=====
W is now the identity, so the network computes y = x.
"#;

const CONFIG: &str = r#"# OpenWiSH project config. Every key is optional.
# Precedence, highest first: command-line flags; then the task's [editor] table (for backend, model and
# effort) or OPENWISH_THREADS / OPENWISH_NICE (for threads and nice); then this file; then the built-in defaults.

# backend = "claude"            # claude | codex | api | scripted:<file>
# model = "claude-opus-5-5"
# effort = "high"               # low | medium | high | xhigh | max
# threads = 4                   # default: a quarter of the cores
# nice = 10                     # 0-19, default 10

# Shell commands the harness runs on the host, outside the sandbox, in this folder. Each takes one
# command or a list. Tasks can add their own in a [hooks] table; both sets run, these first.
# Every hook gets OPENWISH_EVENT, OPENWISH_RUN_DIR, OPENWISH_TASK_DIR, OPENWISH_TOOL and OPENWISH_TURN;
# on_edit hooks also get OPENWISH_EDIT_ID and OPENWISH_EDIT_TENSOR. stdin holds the tool call as JSON.
[hooks]
# session_start = "echo run started in $OPENWISH_RUN_DIR"
# pre_tool = "./check_call.sh"  # exit non-zero to block the call; its stderr goes to the agent, nothing is charged
# post_tool = "echo $OPENWISH_TOOL done, ok=$OPENWISH_TOOL_OK"
# on_edit = "echo edit $OPENWISH_EDIT_ID on $OPENWISH_EDIT_TENSOR >> \"$OPENWISH_RUN_DIR/edits.log\""
# session_end = "echo run ended: $OPENWISH_END_REASON"
# timeout_s = 60
"#;

/// Write a starter environment into `dir`, which must be missing or empty.
pub fn env(dir: &Path, id: Option<&str>) -> Result<()> {
    if dir.exists() && std::fs::read_dir(dir)?.next().is_some() {
        bail!("{} already has files; give a new or empty folder", dir.display());
    }
    std::fs::create_dir_all(dir)?;
    let dir = std::fs::canonicalize(dir)?;
    let id = match id {
        Some(i) => i.to_string(),
        None => dir
            .file_name()
            .context("give a folder with a name")?
            .to_string_lossy()
            .to_string(),
    };
    let shown = dir.display().to_string();
    // a JSON string is also a valid TOML basic string
    let task = TASK_TOML
        .replace("{id}", &serde_json::to_string(&id)?)
        .replace("{dir}", &shown);
    std::fs::write(dir.join("task.toml"), task)?;
    std::fs::write(dir.join("prompt.txt"), PROMPT)?;
    std::fs::write(dir.join("prelude.py"), PRELUDE)?;
    std::fs::write(dir.join("probe.py"), PROBE.replace("{reader}", NPY_READER))?;
    std::fs::write(dir.join("score.py"), SCORE.replace("{reader}", NPY_READER))?;
    std::fs::write(dir.join("script.txt"), SCRIPT)?;
    std::fs::write(dir.join("answer.json"), "{\"W\": [[1.0, 0.0], [0.0, 1.0]]}\n")?;
    let mut n = Network::default();
    n.tensors.insert("W".into(), Tensor::new(vec![2, 2], vec![0.; 4]));
    n.tensors.insert("b".into(), Tensor::new(vec![2], vec![0.; 2]));
    n.save_dir(&dir.join("net"))?;
    Ok(())
}

/// Write a commented openwish.toml to `path`, which must not exist.
pub fn config(path: &Path) -> Result<()> {
    if path.exists() {
        bail!("{} already exists", path.display());
    }
    std::fs::write(path, CONFIG).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_templates_are_valid_task_and_config_files() {
        let d = tempfile::tempdir().unwrap();
        env(&d.path().join("e"), Some("say \"hi\"")).unwrap();
        let t = openwish_core::spec::TaskSpec::load(&d.path().join("e/task.toml")).unwrap();
        assert_eq!(t.id, "say \"hi\"");
        assert_eq!(t.enabled_tools(), vec!["forward".to_string(), "python".to_string()]);
        config(&d.path().join("openwish.toml")).unwrap();
        let c = openwish_core::config::ProjectConfig::load(&d.path().join("openwish.toml")).unwrap();
        assert!(c.hooks.is_empty() && c.backend.is_none());
        assert!(
            config(&d.path().join("openwish.toml")).is_err(),
            "an existing file is kept"
        );
    }
}
