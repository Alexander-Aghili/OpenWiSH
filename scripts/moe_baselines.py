"""Fixed baselines for the MoE environment, with the same 64-expert budget as the agent and the same hidden scorer.

usage: assets/venvs/qwen/bin/python -I scripts/moe_baselines.py
  random     zero the output weights of 64 experts drawn uniformly from all 768 (seeds 0, 1, 2)
  frequency  run the base model on the agent's practice texts, zero the 64 experts whose top-8 routing share is
             highest on code minus English. No gradients, no search over edits.
Writes runs/moe_baselines.json."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moe", ROOT / "python/adapters/moe_worker.py")
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)
BUDGET = 64


def ablate(base, pairs):
    net = {k: v.copy() for k, v in base.items()}
    for layer, expert in pairs:
        net[f"model.layers.{layer}.block_sparse_moe.output_linear.weight"][expert] = 0
    return net


def frequency_pairs(base):
    model, tok = moe.load_model(base)
    practice = {g: json.loads((moe.D / f"practice/{g}.json").read_text()) for g in ("code", "english")}
    skew = moe.run(model, tok, practice["code"])[1] - moe.run(model, tok, practice["english"])[1]
    order = np.argsort(-skew, axis=None)[:BUDGET]
    return [tuple(int(i) for i in np.unravel_index(o, skew.shape)) for o in order]


def main():
    base, ref = moe.load_net(moe.D / "base_net"), moe.base_losses()
    layers, experts = 24, 32
    out = {"base": ref, "budget": BUDGET, "runs": {}}
    for seed in (0, 1, 2):
        pick = np.random.default_rng(seed).choice(layers * experts, BUDGET, replace=False)
        out["runs"][f"random_{seed}"] = moe.deltas(ref, moe.evaluate(ablate(base, [divmod(int(p), experts) for p in pick])))
        print(json.dumps({"random": seed, **out["runs"][f"random_{seed}"]}), flush=True)
    pairs = frequency_pairs(base)
    out["runs"]["frequency"] = {**moe.deltas(ref, moe.evaluate(ablate(base, pairs))), "pairs": pairs}
    print(json.dumps({"frequency": out["runs"]["frequency"]}), flush=True)
    (ROOT / "runs/moe_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
