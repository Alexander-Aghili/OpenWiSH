"""Score v3 baselines for the MoE environment, and v3 rescores of the earlier agent edits. Same 64-expert budget.

usage: assets/venvs/qwen/bin/python -I scripts/moe_v3_baselines.py
  random_<seed>    the same 64 random experts as scripts/moe_baselines.py
  frequency_v2     the 64 experts most routed to code over English prose (the v2 baseline)
  frequency_v3     the 64 experts most routed to code over the larger of English prose and technical English
  sol_v1, sol_v2   gpt-6.1-sol's final edits from the v1 and v2 runs, rebuilt from their edit sets
Writes runs/moe_v3_baselines.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moe", ROOT / "python/adapters/moe_worker.py")
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)
spec = importlib.util.spec_from_file_location("b", ROOT / "scripts/moe_baselines.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
GROUPS = ("code", "english", "docs")
RUNS = {"sol_v1": "runs/bench/moe-gpt-6.1-sol/moe-unlearn_python", "sol_v2": "runs/bench/moe-v2-gpt-6.1-sol/moe-unlearn_python"}


def with_editset(base, run):
    net = dict(base)
    with np.load(ROOT / run / "final/editset/edit.npz") as z:
        for name in z.files:
            net[name] = z[name]
    return net


def frequency_v3_pairs(base):
    model, tok = moe.load_model(base)
    freq = {g: moe.run(model, tok, json.loads((moe.D / f"practice/{g}.json").read_text()))[1] for g in GROUPS}
    skew = freq["code"] - np.maximum(freq["english"], freq["docs"])
    order = np.argsort(-skew, axis=None)[:v2.BUDGET]
    return [tuple(int(i) for i in np.unravel_index(o, skew.shape)) for o in order]


def main():
    base, ref = moe.load_net(moe.D / "base_net"), moe.base_losses(GROUPS)
    old = json.loads((ROOT / "runs/moe_baselines.json").read_text())["runs"]
    nets = {}
    for seed in (0, 1, 2):
        pick = np.random.default_rng(seed).choice(24 * 32, v2.BUDGET, replace=False)
        nets[f"random_{seed}"] = v2.ablate(base, [divmod(int(p), 32) for p in pick])
    nets["frequency_v2"] = v2.ablate(base, [tuple(p) for p in old["frequency"]["pairs"]])
    pairs = frequency_v3_pairs(base)
    nets["frequency_v3"] = v2.ablate(base, pairs)
    for name, run in RUNS.items():
        nets[name] = with_editset(base, run)
    out = {"base": ref, "budget": v2.BUDGET, "rule": "score v3", "frequency_v3_pairs": pairs, "runs": {}}
    for name, net in nets.items():
        out["runs"][name] = moe.deltas_v3(ref, moe.evaluate(net, GROUPS))
        print(json.dumps({name: {k: round(v, 3) if isinstance(v, float) else v for k, v in out["runs"][name].items()}}), flush=True)
    (ROOT / "runs/moe_v3_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
