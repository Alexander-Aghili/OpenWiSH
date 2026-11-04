# interp: interpretability tools beside OpenWiSH (not part of the harness)

Tools for people looking at networks and finished runs. They are kept apart from the harness on purpose:

- Nothing in `crates/`, `python/` or `tasks/` imports anything here, and nothing here writes outside its own output
  files. The tools only read networks (`assets/...`) and run folders (`runs/...`).
- They are not agent tools. Giving an agent one of these during a run would be running the network outside the
  declared probes; an environment that wants such a view must declare it as a budgeted probe (as the decompile
  environments' `forward` tool does).

## mlp_explorer

A static, self-contained HTML page for a small MLP (`W1,b1,W2,b2,W3,b3`, two ReLU layers), unit by unit, after
the method of [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): read a unit's inputs and outputs
as a pair, see where it fires and its near misses, steer it up and down, zero it, and treat weight links as leads
rather than circuits. With `--run`, it overlays a decompile run: the agent's `VARIABLES` (units or subspaces, read
statically from its final answer, never executed), each variable's intervention accuracy, and the program.

```bash
assets/venvs/sandbox/bin/python -I interp/mlp_explorer/explore.py --net assets/decompile/d1-b/net \
    --run runs/<a decompile run> --out d1-b.html
```

Inputs: every binary input when the network has at most 16 inputs, otherwise `--inputs <rows.npy>`.

## neuron_explorer

`sh interp/neuron_explorer/fetch.sh` fetches neuron-explorer (MIT, by lyramakesmusic) at a pinned commit into
`interp/vendor/` (ignored by git), unmodified. It is a steering-first explorer for Qwen3.5-4B: every MLP neuron
pushed up and down on a few prompts, every rollout kept, a 3D map of the model, lenses, and a Claude chat that
drives the explorer through MCP. Its sweep needs a CUDA GPU and a vLLM fork; see its README. It targets
Qwen3.5-4B, not the Qwen3.5-0.8B of Experiment 8, so it is a reference for method, not yet a view of OpenWiSH runs.
