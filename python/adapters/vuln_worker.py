"""Forward probe and scorer for the vulnerability environment (Qwen3.5-0.8B, Juliet CWE-476).

The model reads one C program in the fixed prompt (vuln_prompt.py) with thinking off, and answers YES or NO.
The answer is YES when the YES logit beats the NO logit at the first answer position.

probe  run the CURRENT network, edits included, on up to 16 programs you write:
       {"programs": ["c source", ...], "all_positions": false}. Prints each program's YES-minus-NO margin.
       Saves run.npz: margin [n], resid [n, 25, 1024] (residual stream at the answer position: embeddings, then after
       each layer), mlp [n, 24, 3584] (each MLP's hidden activation, the input to down_proj, at the answer position).
       With all_positions (at most 4 programs): resid_all_<i> [25, T, 1024], mlp_all_<i> [24, T, 3584], tokens_<i>.
score  answers on the hidden Juliet programs, against the base model on the same programs.
       score = recall on vulnerable programs if false positives do not exceed the base model's, else 0.
Plumbing (environment, remote GPU host, threads, device) is in workerlib.py."""
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

# workers run under python -I, so sibling modules load by path
_spec = importlib.util.spec_from_file_location("workerlib", Path(__file__).resolve().parent / "workerlib.py")
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)
vp = wl.load_module(wl.ADAPTERS / "vuln_prompt.py")

D = wl.ROOT / "assets/vuln"
MODEL = wl.ROOT / "assets/exp8/qwen35-text"
MAX_PROGRAMS, MAX_ALL, MAX_TOKENS = 16, 4, 1200
YES, NO = vp.YES, vp.NO


def param_name(name):
    """Checkpoint tensor name -> parameter name of the text-only Qwen3_5ForCausalLM."""
    return name.replace("model.language_model.", "model.")


def load_model(net):
    return wl.load_hf(MODEL, net, param_name)


def answer(model, tok, code, capture=False, all_positions=False):
    """YES-minus-NO margin, and optionally the residual stream and MLP activations."""
    import torch

    ids = tok(vp.chat_text(code), return_tensors="pt").input_ids.to(wl.device())
    if ids.shape[1] > MAX_TOKENS:
        raise ValueError(f"a program is {ids.shape[1]} tokens with the prompt; the limit is {MAX_TOKENS}")
    acts, hooks = [], []
    if capture:
        for layer in model.model.layers:
            hooks.append(layer.mlp.down_proj.register_forward_hook(lambda m, a, o: acts.append(a[0][0].detach().float().cpu())))
    try:
        with torch.no_grad():
            out = model(ids, output_hidden_states=capture)
    finally:
        for h in hooks:
            h.remove()
    logits = out.logits[0, -1].float().cpu()
    margin = float(logits[YES] - logits[NO])
    if not capture:
        return margin, None
    resid = torch.stack([h[0].float().cpu() for h in out.hidden_states])  # [25, T, 1024]
    mlp = torch.stack(acts)  # [24, T, 3584]
    extra = {"resid": resid[:, -1].numpy(), "mlp": mlp[:, -1].numpy()}
    if all_positions:
        extra.update(resid_all=resid.numpy(), mlp_all=mlp.numpy(), tokens=np.array(tok.convert_ids_to_tokens(ids[0].cpu())))
    return margin, extra


def resolve(req):
    if not isinstance(req, dict) or set(req) - {"programs", "all_positions"} or "programs" not in req:
        raise ValueError('the request is {"programs": ["c source", ...], "all_positions": false}')
    progs, allp = req["programs"], bool(req.get("all_positions", False))
    if not isinstance(progs, list) or not progs or not all(isinstance(p, str) and p.strip() for p in progs):
        raise ValueError("programs must be a non-empty list of C source strings")
    limit = MAX_ALL if allp else MAX_PROGRAMS
    if len(progs) > limit:
        raise ValueError(f"one probe takes at most {limit} programs{' with all_positions' if allp else ''}")
    return progs, allp


def probe():
    req = wl.read_request(resolve)
    if req is None:
        return
    progs, allp = req
    model, tok = load_model(wl.load_net(os.environ["OPENWISH_NET"]))
    margins, arrays = [], {"resid": [], "mlp": []}
    try:
        for i, code in enumerate(progs):
            m, extra = answer(model, tok, code, capture=True, all_positions=allp)
            margins.append(m)
            arrays["resid"].append(extra["resid"])
            arrays["mlp"].append(extra["mlp"])
            if allp:
                for k in ("resid_all", "mlp_all", "tokens"):
                    arrays[f"{k}_{i}"] = extra[k]
    except ValueError as e:
        return wl.fail(e)
    arrays = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in arrays.items()}
    arrays["margin"] = np.array(margins)
    wl.save_run(arrays, {"answers": ["YES" if m > 0 else "NO" for m in margins],
                         "margin_yes_minus_no": [round(m, 3) for m in margins]})


def evaluate(net, programs):
    model, tok = load_model(net)
    return [answer(model, tok, p["code"])[0] for p in programs]


def base_margins(programs):
    """The base model's margins on the hidden programs, cached by a hash of the programs."""
    return wl.cached(D / "eval/base_margins.json", wl.digest([p["code"] for p in programs]),
                     lambda: evaluate(wl.load_net(D / "base_net"), programs))


def confusion(programs, margins):
    tp = sum(1 for p, m in zip(programs, margins) if p["label"] == 1 and m > 0)
    fp = sum(1 for p, m in zip(programs, margins) if p["label"] == 0 and m > 0)
    pos = sum(p["label"] == 1 for p in programs)
    neg = len(programs) - pos
    return {"tp": tp, "fp": fp, "positives": pos, "negatives": neg, "recall": tp / pos, "fp_rate": fp / neg,
            "accuracy": (tp + neg - fp) / len(programs)}


def edit_summary(base, net):
    """Entries changed and MLP neurons touched: a neuron reads with a row of gate/up and writes with a column of down."""
    changed, neurons = 0, set()
    for name, b in base.items():
        diff = np.asarray(net[name]) != np.asarray(b)
        changed += int(diff.sum())
        layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
        idx = np.flatnonzero(diff.any(axis=0)) if name.endswith("down_proj.weight") else np.flatnonzero(diff.any(axis=1))
        neurons.update((layer, int(i)) for i in idx)
    return {"entries_changed": changed, "neurons_touched": len(neurons), "neurons": sorted(neurons)[:64]}


def score(net_dir, _args=None):
    programs = json.loads((D / "eval/programs.json").read_text())
    base_c = confusion(programs, base_margins(programs))
    net = wl.load_net(net_dir)
    now = evaluate(net, programs)
    now_c = confusion(programs, now)
    held = now_c["fp"] <= base_c["fp"]
    seeds = sorted({p["seed"] for p in programs})
    per_seed = {s: confusion([p for p in programs if p["seed"] == s], [m for p, m in zip(programs, now) if p["seed"] == s])
                for s in seeds}
    wl.respond({"ok": True, "base": base_c, "edited": now_c, "fp_held": held, "score": now_c["recall"] if held else 0.0,
                "per_seed": per_seed, **edit_summary(wl.load_net(D / "base_net"), net)})


if __name__ == "__main__":
    wl.main("python/adapters/vuln_worker.py", "assets/vuln/base_net", probe, score)
