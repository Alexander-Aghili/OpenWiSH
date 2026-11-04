#!/usr/bin/env python3
"""MLP explorer: a static, self-contained HTML page for exploring a small OpenWiSH MLP unit by unit, following the
method of neuron-explorer (github.com/lyramakesmusic/neuron-explorer): read a unit's inputs and outputs as a pair,
see where it fires, steer it, ablate it, and treat weight links as leads rather than circuits.

Separate from the harness: nothing in crates/, python/ or tasks/ imports this, and it only reads files. It is for
people looking at networks and finished runs, not a tool for agents during a run (that would be an undeclared probe).

usage: python interp/mlp_explorer/explore.py --net assets/decompile/d1-a/net [--run runs/<run>] [--out page.html]
  --net   a folder with W1,b1,W2,b2,W3,b3 .npy (x -> relu -> relu -> logits)
  --run   a finished decompile run: overlays the agent's VARIABLES and its scores (from eval/score.json and the
          final answer's program block)
  --inputs  an .npy of inputs (rows); default: every binary input when the input has at most 16 bits
"""
import argparse
import ast
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(d):
    return {k: np.load(os.path.join(d, k + ".npy")).astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}


def forward(net, X, add=None):
    """add: (layer, unit, value) added to that unit after its ReLU (steering), or (layer, unit, None) to zero it."""
    def touch(name, h):
        if add and add[0] == name:
            h = h.copy()
            if add[2] is None:
                h[:, add[1]] = 0.0
            else:
                h[:, add[1]] += add[2]
        return h
    h1 = touch("h1", np.maximum(X @ net["W1"] + net["b1"], 0))
    h2 = touch("h2", np.maximum(h1 @ net["W2"] + net["b2"], 0))
    return h1, h2, h2 @ net["W3"] + net["b3"]


def bitstr(x):
    return "".join(str(int(round(b))) for b in x)


def links(W, rows_label, cols_label, j, incoming, top=12):
    v = W[:, j] if incoming else W[j, :]
    idx = [int(i) for i in np.argsort(-np.abs(v)) if v[i] != 0][:top]
    lab = rows_label if incoming else cols_label
    return [{"from" if incoming else "to": lab(i), "w": round(float(v[i]), 4)} for i in idx]


def unit_report(net, X, base_cls, layer, u, acts, binary):
    a = acts[:, u]
    on = a > 1e-9
    rep = {"id": f"{layer}[{u}]", "layer": layer, "unit": u, "max": round(float(a.max()), 4),
           "mean": round(float(a.mean()), 4), "fire_rate": round(float(on.mean()), 4)}
    order = np.argsort(-a)
    rep["top"] = [{"x": bitstr(X[i]), "act": round(float(a[i]), 4), "cls": int(base_cls[i])} for i in order[:8]]
    if binary and on.any() and (~on).any():
        rep["bits"] = [round(float(on[X[:, i] == 1].mean() - on[X[:, i] == 0].mean()), 3) for i in range(X.shape[1])]
        # near misses: inputs where the unit is off but one bit flip turns it on
        onset = {bitstr(X[i]) for i in np.nonzero(on)[0]}
        nm = []
        for i in np.nonzero(~on)[0]:
            s = bitstr(X[i])
            flips = [k for k in range(len(s)) if (s[:k] + ("1" if s[k] == "0" else "0") + s[k + 1:]) in onset]
            if flips:
                nm.append({"x": s, "flip_turns_on": flips})
            if len(nm) >= 6:
                break
        rep["near_misses"] = nm
    if layer == "h1":
        rep["reads"] = links(net["W1"], lambda i: f"x[{i}]", None, u, True)
        rep["writes"] = links(net["W2"], None, lambda k: f"h2[{k}]", u, False)
        rep["bias"] = round(float(net["b1"][u]), 4)
    else:
        rep["reads"] = links(net["W2"], lambda i: f"h1[{i}]", None, u, True)
        rep["writes"] = links(net["W3"], None, lambda k: f"logit {k}", u, False)
        rep["bias"] = round(float(net["b2"][u]), 4)
    _, _, lg = forward(net, X, (layer, u, None))
    c = lg.argmax(1)
    rep["ablate"] = {"flips": round(float((c != base_cls).mean()), 4),
                     "class_share": np.bincount(c, minlength=lg.shape[1]).tolist()}
    scale = max(float(a.max()), 1.0)
    rep["steer"] = []
    for dose in (-2.0, -1.0, 1.0, 2.0):
        _, _, lg = forward(net, X, (layer, u, dose * scale))
        c = lg.argmax(1)
        rep["steer"].append({"dose": dose, "add": round(dose * scale, 3), "flips": round(float((c != base_cls).mean()), 4),
                             "class_share": np.bincount(c, minlength=lg.shape[1]).tolist()})
    return rep


def program_block(text):
    body, cur = None, None
    for line in text.splitlines():
        s = line.strip()
        if cur is None:
            if s.startswith("```") and s[3:].strip().lower().startswith("program"):
                cur = []
        elif s.startswith("```") and not s[3:].strip():
            body, cur = "\n".join(cur), None
        else:
            cur.append(line)
    return body if cur is None else "\n".join(cur)


def run_overlay(run, widths):
    """The agent's VARIABLES (read statically from its final answer, never executed) and its scores."""
    out = {"run": os.path.abspath(run)}
    sp = os.path.join(run, "eval", "score.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        out["score"] = {k: s.get(k) for k in ("score", "behaviour", "mechanism", "exact", "iia_mean", "completeness",
                                              "quirk_agreement", "invalid")}
        out["per_variable"] = s.get("variables", {})
    ap = os.path.join(run, "final_answer.md")
    if os.path.exists(ap):
        src = program_block(open(ap).read()) or ""
        out["program"] = src
        try:
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VARIABLES" for t in node.targets):
                    V = ast.literal_eval(node.value)
                    loads = {}
                    for name, (layer, place) in V.items():
                        if isinstance(place, dict):
                            B = np.array(place["basis"], dtype=np.float64)
                            U, sv, _ = np.linalg.svd(B.T, full_matrices=False)
                            U = U[:, : int((sv > 1e-8 * sv.max()).sum())]
                            w = (U ** 2).sum(1)
                        else:
                            w = np.zeros(widths[layer])
                            w[list(place)] = 1.0
                        loads[name] = {"layer": layer, "load": np.round(w, 4).tolist()}
                    out["variables"] = loads
        except (SyntaxError, ValueError, KeyError, TypeError, IndexError) as e:
            out["variables_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--run")
    ap.add_argument("--inputs")
    ap.add_argument("--out", default="mlp_explorer.html")
    ap.add_argument("--title")
    a = ap.parse_args()
    net = load(a.net)
    n = net["W1"].shape[0]
    if a.inputs:
        X = np.load(a.inputs).astype(np.float64)
    elif n <= 16:
        X = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.float64)
    else:
        raise SystemExit(f"the input has {n} numbers; give --inputs")
    binary = bool(np.isin(X, (0.0, 1.0)).all())
    h1, h2, lg = forward(net, X)
    base_cls = lg.argmax(1)
    widths = {"h1": h1.shape[1], "h2": h2.shape[1]}
    data = {
        "title": a.title or os.path.abspath(a.net),
        "shape": [n, widths["h1"], widths["h2"], lg.shape[1]],
        "n_inputs": len(X), "binary": binary,
        "class_share": np.bincount(base_cls, minlength=lg.shape[1]).tolist(),
        "W1": np.round(net["W1"], 4).tolist(), "W2": np.round(net["W2"], 4).tolist(), "W3": np.round(net["W3"], 4).tolist(),
        "units": [unit_report(net, X, base_cls, "h1", u, h1, binary) for u in range(widths["h1"])]
                 + [unit_report(net, X, base_cls, "h2", u, h2, binary) for u in range(widths["h2"])],
        "overlay": run_overlay(a.run, widths) if a.run else None,
    }
    page = open(os.path.join(HERE, "template.html")).read().replace("/*DATA*/null", json.dumps(data))
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({len(page) // 1024} KB): {sum(widths.values())} units over {len(X)} inputs")


if __name__ == "__main__":
    main()
