"""Forward probe for the AgenticInterpBench probe condition (P). Runs the InterpBench transformer with
TransformerLens semantics in numpy: x = W_E[token] + W_pos, then per block the attention heads and the MLP,
no normalization, logits = x W_U + b_U.

A component's output is its write into the residual stream: one head's z W_O for "L{l}H{h}_ATTN", and
gelu(x W_in + b_in) W_out + b_out for "L{l}_MLP". Ablation and patching replace that write.

Request ($OPENWISH_REQ, JSON):
  {"inputs": [[token, ...], ...],                      at most 64 sequences of equal length, tokens as in the examples
   "ablate": {"<component>": "zero" | "mean"},        optional, mean over the request's inputs
   "patch": [{"component": "<id>", "source": j}]}     optional, the component's write from input j (interchange)
Saves $OPENWISH_OUT/run.npz: logits [n, seq, vocab_out], every component's write [n, seq, d_model],
attention patterns "<head>_pattern" [n, seq, seq] and MLP activations "<mlp>_post" [n, seq, d_mlp]."""
import json
import math
import os
from pathlib import Path

import numpy as np

MAX_INPUTS = 64
_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1 + _erf(x / math.sqrt(2)))


def load_params(net_dir):
    return {p.stem: np.load(p).astype(np.float64) for p in Path(net_dir).glob("*.npy")}


def components(cfg):
    heads = [f"L{l}H{h}_ATTN" for l in range(cfg["n_layers"]) for h in range(cfg["n_heads"])]
    return heads + [f"L{l}_MLP" for l in range(cfg["n_layers"])]


def forward(P, cfg, ids, ablate=None):
    """Logits [seq, vocab_out] and a cache of every component's write. `ablate` maps a component to its replacement write."""
    ablate, cache, n = ablate or {}, {}, len(ids)
    x = P["embed.W_E"][ids] + P["pos_embed.W_pos"][:n]
    mask = np.triu(np.ones((n, n), bool), 1) if cfg["causal"] else np.zeros((n, n), bool)
    for l in range(cfg["n_layers"]):
        a = f"blocks.{l}.attn."
        attn = np.broadcast_to(P[a + "b_O"], x.shape).copy()
        for h in range(cfg["n_heads"]):
            q, k, v = (x @ P[a + f"W_{s}"][h] + P[a + f"b_{s}"][h] for s in "QKV")
            s = np.where(mask, -np.inf, q @ k.T / math.sqrt(cfg["d_head"]))
            pattern = np.exp(s - s.max(1, keepdims=True))
            pattern /= pattern.sum(1, keepdims=True)
            name = f"L{l}H{h}_ATTN"
            cache[name + "_pattern"] = pattern
            cache[name] = ablate.get(name, pattern @ v @ P[a + "W_O"][h])
            attn += cache[name]
        x = x + attn
        m = f"blocks.{l}.mlp."
        post = gelu(x @ P[m + "W_in"] + P[m + "b_in"])
        name = f"L{l}_MLP"
        cache[name + "_post"] = post
        cache[name] = ablate.get(name, post @ P[m + "W_out"] + P[m + "b_out"])
        x = x + cache[name]
    return x @ P["unembed.W_U"] + P["unembed.b_U"], cache


def encode(cfg, inputs):
    if not isinstance(inputs, list) or not 0 < len(inputs) <= MAX_INPUTS:
        raise ValueError(f"inputs must be a list of 1 to {MAX_INPUTS} token sequences")
    if len({len(s) if isinstance(s, list) else -1 for s in inputs}) != 1 or not isinstance(inputs[0], list) or not inputs[0]:
        raise ValueError("every input must be a non-empty token list, all of the same length")
    unknown = sorted({str(t) for s in inputs for t in s} - set(cfg["input_map"]))
    if unknown:
        raise ValueError(f"unknown tokens {unknown[:5]}: use tokens of the input map")
    return np.array([[cfg["input_map"][str(t)] for t in s] for s in inputs])


def decode(cfg, logits):
    if len(cfg.get("output_map") or {}) > 1 and logits.shape[-1] > 1:
        value = cfg["output_map"]  # output index -> task value
        return [[value.get(str(int(i)), int(i)) for i in row] for row in logits.argmax(-1)]
    return [[round(float(v), 4) for v in row] for row in logits[..., 0]]


def interventions(cfg, req, clean):
    """Per-input replacement writes from the request's ablations and patches."""
    known, n = set(components(cfg)), len(clean)
    plan = [{} for _ in range(n)]
    for comp, how in (req.get("ablate") or {}).items():
        if comp not in known or how not in ("zero", "mean"):
            raise ValueError(f"ablate needs a known component and 'zero' or 'mean', got {comp!r}: {how!r}")
        writes = np.stack([c[comp] for c in clean])
        value = np.zeros_like(writes[0]) if how == "zero" else writes.mean(0)
        for p in plan:
            p[comp] = value
    for item in req.get("patch") or []:
        comp, src = item.get("component"), item.get("source")
        if comp not in known or type(src) is not int or not 0 <= src < n:
            raise ValueError(f"patch needs a known component and a source index 0..{n - 1}")
        for p in plan:
            p[comp] = clean[src][comp]
    return plan


def run_request(P, cfg, req):
    if not isinstance(req, dict) or set(req) - {"inputs", "ablate", "patch"}:
        raise ValueError("the request is a JSON object with inputs, and optional ablate and patch")
    ids = encode(cfg, req.get("inputs"))
    clean = [forward(P, cfg, x)[1] for x in ids]
    runs = [forward(P, cfg, x, plan) for x, plan in zip(ids, interventions(cfg, req, clean))]
    logits = np.stack([r[0] for r in runs])
    cache = {k: np.stack([r[1][k] for r in runs]) for k in runs[0][1]}
    return {"logits": logits, "cache": cache, "predictions": decode(cfg, logits)}


def main():
    try:
        cfg = json.loads(Path(os.environ["OPENWISH_TASK_DIR"], "arch.json").read_text())
        req = json.loads(Path(os.environ["OPENWISH_REQ"]).read_text())
        out = run_request(load_params(os.environ["OPENWISH_NET"]), cfg, req)
    except (ValueError, json.JSONDecodeError) as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return
    path = Path(os.environ["OPENWISH_OUT"])
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "run.npz", logits=out["logits"].astype(np.float32), **{k: v.astype(np.float32) for k, v in out["cache"].items()})
    print(json.dumps({"ok": True, "predictions": out["predictions"], "file": "run.npz",
                      "arrays": {"logits": list(out["logits"].shape), **{k: list(v.shape) for k, v in out["cache"].items()}}}))


if __name__ == "__main__":
    main()
