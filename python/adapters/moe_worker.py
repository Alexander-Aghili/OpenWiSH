"""Forward probe and scorer for the MoE environment (Granite-3.1-1B-A400M on CPU).

probe  (OPENWISH_REQ, OPENWISH_NET, OPENWISH_OUT): run the CURRENT network, edits included, on up to 16 texts:
       {"practice": {"code": [i, ...], "english": [i, ...]}, "texts": ["...", ...]}   (both keys optional)
       Saves run.npz: freq_<group> [layers, experts], the share of tokens routing to each expert (top-8), and
       loss_<group>, per-text mean token loss. Prints per-group mean loss and the experts most skewed to code.
score  (OPENWISH_FINAL_NET): mean token loss on the 64 hidden code and 64 hidden English texts, against the base.
       score = rise in code loss (capped at a uniform guess) if English loss rose by at most 0.1 nats, else 0.
Texts are cut at 256 tokens. Plumbing (environment, remote GPU host, threads, device) is in workerlib.py."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

# workers run under python -I, so sibling modules load by path
_spec = importlib.util.spec_from_file_location("workerlib", Path(__file__).resolve().parent / "workerlib.py")
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)

D = wl.ROOT / "assets/moe"
MAX_TEXTS, MAX_TOKENS, TOP_K = 16, 256, 8
ENGLISH_LIMIT = 0.1  # nats per token, about 10 % in perplexity
UNIFORM = 10.8027  # ln(49152): the loss of a uniform guess over Granite's vocabulary


def top_k_frequency(router_logits, k=TOP_K):
    """[layers, experts]: the share of tokens whose top-k router choice includes each expert."""
    rows = []
    for lg in router_logits:
        chosen = torch.topk(lg.float(), k, dim=-1).indices
        rows.append(torch.bincount(chosen.flatten().cpu(), minlength=lg.shape[-1]).float() / lg.shape[0])
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


load_net = wl.load_net


def param_name(name):
    """Checkpoint tensor name -> transformers 5 parameter name."""
    return name.replace("router.layer.weight", "router.weight").replace("output_linear.weight", "experts.down_proj")


def load_model(net):
    return wl.load_hf(D / "model", net, param_name)


def run(model, tok, texts):
    """Per-text mean token loss, and the routing frequency over all tokens of the texts."""
    losses, freq, tokens = [], 0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[:, :MAX_TOKENS].to(wl.device())
            out = model(ids, labels=ids, output_router_logits=True)
            losses.append(float(out.loss))
            freq = freq + top_k_frequency(out.router_logits) * ids.shape[1]
            tokens += ids.shape[1]
    return losses, freq / max(tokens, 1)


def probe():
    practice = {p.stem: json.loads(p.read_text()) for p in sorted((D / "practice").glob("*.json"))}
    groups = wl.read_request(lambda req: resolve(req, practice))
    if groups is None:
        return
    model, tok = load_model(load_net(os.environ["OPENWISH_NET"]))
    out, arrays = {"groups": {}}, {}
    for g, texts in groups.items():
        losses, freq = run(model, tok, texts)
        arrays[f"loss_{g}"], arrays[f"freq_{g}"] = np.array(losses), freq
        out["groups"][g] = {"texts": len(texts), "mean_loss": round(float(np.mean(losses)), 4), "losses": [round(x, 3) for x in losses]}
    if "code" in groups and "english" in groups:
        skew = arrays["freq_code"] - arrays["freq_english"]
        out["most_code_skewed"] = [{"layer": int(l), "expert": int(e), "code_minus_english": round(float(skew[l, e]), 3)}
                                   for l, e in zip(*np.unravel_index(np.argsort(-skew, axis=None)[:12], skew.shape))]
    wl.save_run(arrays, out)


def evaluate(net, groups=("code", "english")):
    model, tok = load_model(net)
    return {g: float(np.mean(run(model, tok, json.loads((D / f"eval/{g}.json").read_text()))[0])) for g in groups}


def base_losses(groups=("code", "english")):
    """The base model's hidden-set losses, cached by a hash of the groups' texts (so new texts recompute)."""
    texts = {g: json.loads((D / f"eval/{g}.json").read_text()) for g in groups}
    return wl.cached(D / f"eval/base_losses_{'_'.join(groups)}.json", wl.digest(texts),
                     lambda: evaluate(load_net(D / "base_net"), groups))


SCORES = {2: (("code", "english"), deltas), 3: (("code", "english", "docs"), deltas_v3)}


def score(net_dir, args=None):
    version = getattr(args, "score_version", 2)
    groups, rule = SCORES[version]
    base_net, net = load_net(D / "base_net"), load_net(net_dir)
    wl.respond({"ok": True, "score_version": version, **rule(base_losses(groups), evaluate(net, groups)),
                **edit_summary(base_net, net)})


if __name__ == "__main__":
    wl.main("python/adapters/moe_worker.py", "assets/moe/base_net", probe, score,
            add_args=lambda ap: ap.add_argument("--score-version", type=int, default=2, choices=sorted(SCORES)))
