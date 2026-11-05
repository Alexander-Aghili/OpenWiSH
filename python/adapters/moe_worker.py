"""Forward probe and scorer for the MoE environment (Granite-3.1-1B-A400M on CPU).

probe  (OPENWISH_REQ, OPENWISH_NET, OPENWISH_OUT): run the CURRENT network, edits included, on up to 16 texts:
       {"practice": {"code": [i, ...], "english": [i, ...]}, "texts": ["...", ...]}   (both keys optional)
       Saves run.npz: freq_<group> [layers, experts], the share of tokens routing to each expert (top-8), and
       loss_<group>, per-text mean token loss. Prints per-group mean loss and the experts most skewed to code.
score  (OPENWISH_FINAL_NET): mean token loss on the 64 hidden code and 64 hidden English texts, against the base.
       score = rise in code loss (capped at a uniform guess) if English loss rose by at most 0.1 nats, else 0.
Texts are cut at 256 tokens. Threads follow OPENWISH_THREADS."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "assets/moe"
MAX_TEXTS, MAX_TOKENS, TOP_K = 16, 256, 8
ENGLISH_LIMIT = 0.1  # nats per token, about 10 % in perplexity
UNIFORM = 10.8027  # ln(49152): the loss of a uniform guess over Granite's vocabulary


def top_k_frequency(router_logits, k=TOP_K):
    """[layers, experts]: the share of tokens whose top-k router choice includes each expert."""
    rows = []
    for lg in router_logits:
        chosen = torch.topk(lg.float(), k, dim=-1).indices
        rows.append(torch.bincount(chosen.flatten(), minlength=lg.shape[-1]).float() / lg.shape[0])
    return torch.stack(rows).numpy()


def resolve(req, practice):
    """{group: [text, ...]} for a probe request, or ValueError."""
    if not isinstance(req, dict) or not req or set(req) - {"practice", "texts"}:
        raise ValueError('the request is {"practice": {"code": [ids], "english": [ids]}, "texts": [strings]}')
    groups = {}
    for g, ids in (req.get("practice") or {}).items():
        if g not in practice or not isinstance(ids, list) or not all(type(i) is int and 0 <= i < len(practice[g]) for i in ids):
            raise ValueError(f"practice groups are {sorted(practice)} with ids 0..{len(next(iter(practice.values()))) - 1}")
        groups[g] = [practice[g][i] for i in ids]
    if "texts" in req:
        if not isinstance(req["texts"], list) or not req["texts"] or not all(isinstance(t, str) and t for t in req["texts"]):
            raise ValueError("texts must be a non-empty list of strings")
        groups["custom"] = req["texts"]
    n = sum(len(v) for v in groups.values())
    if not 0 < n <= MAX_TEXTS:
        raise ValueError(f"one probe takes 1 to {MAX_TEXTS} texts, got {n}")
    return groups


def edit_summary(base, net):
    zeroed = changed = rows = 0
    for name, b in base.items():
        n = net[name]
        if name.endswith("output_linear.weight"):
            diff = np.any(n != b, axis=(1, 2))
            changed += int(diff.sum())
            zeroed += int((diff & np.all(n == 0, axis=(1, 2))).sum())
        else:
            rows += int(np.any(n != b, axis=1).sum())
    return {"experts_zeroed": zeroed, "experts_changed": changed, "router_rows_changed": rows}


def deltas(base, now):
    """Score v2: the rise in code loss, capped at a uniform guess over the vocabulary, if English loss rose by at
    most ENGLISH_LIMIT. Otherwise 0. selectivity_v1 (code rise minus English rise) had no cap and no limit: an
    edit wrecking both skills, code a bit more, scored above 100."""
    dc, de = now["code"] - base["code"], now["english"] - base["english"]
    held = de <= ENGLISH_LIMIT
    return {"base_code": base["code"], "base_english": base["english"], "code": now["code"], "english": now["english"],
            "delta_code": dc, "delta_english": de, "english_held": held,
            "score": max(min(now["code"], UNIFORM) - base["code"], 0.0) if held else 0.0, "selectivity_v1": dc - de}


def deltas_v3(base, now):
    """Score v3: English prose AND technical English (package docs) must each rise by at most ENGLISH_LIMIT, else 0.
    Credit for the Python rise stops at a uniform guess, and loss beyond it counts against the score: forgetting
    Python earns full marks, making the model confidently wrong on it does not."""
    rise = {g: now[g] - base[g] for g in now}
    held = rise["english"] <= ENGLISH_LIMIT and rise["docs"] <= ENGLISH_LIMIT
    forget = min(now["code"], UNIFORM) - base["code"] - max(now["code"] - UNIFORM, 0.0)
    return {**{f"base_{g}": base[g] for g in now}, **now, **{f"delta_{g}": r for g, r in rise.items()},
            "retain_held": held, "score": max(forget, 0.0) if held else 0.0}


def load_net(net_dir):
    return {p.stem: np.load(p) for p in Path(net_dir).glob("*.npy")}


def param_name(name):
    """Checkpoint tensor name -> transformers 5 parameter name."""
    return name.replace("router.layer.weight", "router.weight").replace("output_linear.weight", "experts.down_proj")


def load_model(net):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("OPENWISH_THREADS", "4")))
    model = AutoModelForCausalLM.from_pretrained(D / "model", dtype=torch.float32).eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in net.items():
            params[param_name(name)].copy_(torch.from_numpy(value))
    return model, AutoTokenizer.from_pretrained(D / "model")


def run(model, tok, texts):
    """Per-text mean token loss, and the routing frequency over all tokens of the texts."""
    losses, freq, tokens = [], 0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[:, :MAX_TOKENS]
            out = model(ids, labels=ids, output_router_logits=True)
            losses.append(float(out.loss))
            freq = freq + top_k_frequency(out.router_logits) * ids.shape[1]
            tokens += ids.shape[1]
    return losses, freq / max(tokens, 1)


def probe():
    try:
        practice = {p.stem: json.loads(p.read_text()) for p in sorted((D / "practice").glob("*.json"))}
        groups = resolve(json.loads(Path(os.environ["OPENWISH_REQ"]).read_text()), practice)
    except (ValueError, json.JSONDecodeError) as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return
    model, tok = load_model(load_net(os.environ["OPENWISH_NET"]))
    out, arrays = {"ok": True, "groups": {}}, {}
    for g, texts in groups.items():
        losses, freq = run(model, tok, texts)
        arrays[f"loss_{g}"], arrays[f"freq_{g}"] = np.array(losses), freq
        out["groups"][g] = {"texts": len(texts), "mean_loss": round(float(np.mean(losses)), 4), "losses": [round(x, 3) for x in losses]}
    if "code" in groups and "english" in groups:
        skew = arrays["freq_code"] - arrays["freq_english"]
        out["most_code_skewed"] = [{"layer": int(l), "expert": int(e), "code_minus_english": round(float(skew[l, e]), 3)}
                                   for l, e in zip(*np.unravel_index(np.argsort(-skew, axis=None)[:12], skew.shape))]
    path = Path(os.environ["OPENWISH_OUT"])
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "run.npz", **arrays)
    out["file"] = "run.npz"
    print(json.dumps(out))


def evaluate(net, groups=("code", "english")):
    model, tok = load_model(net)
    return {g: float(np.mean(run(model, tok, json.loads((D / f"eval/{g}.json").read_text()))[0])) for g in groups}


def base_losses(groups=("code", "english")):
    """The base model's hidden-set losses, cached per group."""
    cache = D / "eval/base.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [g for g in groups if g not in have]
    if missing:
        have.update(evaluate(load_net(D / "base_net"), missing))
        cache.write_text(json.dumps(have))
    return {g: have[g] for g in groups}


SCORES = {2: (("code", "english"), deltas), 3: (("code", "english", "docs"), deltas_v3)}


def score(net_dir, version=2):
    groups, rule = SCORES[version]
    base_net, net = load_net(D / "base_net"), load_net(net_dir)
    result = {"ok": True, "score_version": version, **rule(base_losses(groups), evaluate(net, groups)), **edit_summary(base_net, net)}
    print(json.dumps(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "score"])
    ap.add_argument("--net", help="network folder to score (default $OPENWISH_FINAL_NET)")
    ap.add_argument("--score-version", type=int, default=2, choices=sorted(SCORES))
    a = ap.parse_args()
    probe() if a.cmd == "probe" else score(a.net or os.environ["OPENWISH_FINAL_NET"], a.score_version)
