#!/usr/bin/env python3
"""Harness-side calibration of the decompile networks (search is fine here; never for the agent).

For each secret variable, the best interchange-intervention accuracy (IIA) of the *secret* program's variable over
every set of at most K units of one hidden layer. It tells how localised the planted variables are, a reference for
agents' mechanism scores (an agent may find better variables than the secret ones, so it is not a ceiling).

usage: assets/venvs/sandbox/bin/python -I scripts/calibrate_decompile.py [--k 3] [--only d1-a,...]
writes assets/decompile/<id>/reference.json (hidden with the rest of the folder's scoring assets)
"""
import argparse
import itertools
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "python", "adapters"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

import decompile_lib as L  # noqa: E402


def calibrate(d, k, pairs=3000, seed=0):
    net = L.load_net(os.path.join(d, "net"))
    s = json.load(open(os.path.join(d, "secret.json")))
    Xl = L.all_inputs(s["n_bits"])
    X = np.array(Xl, dtype=np.float64)
    vals = np.array([L.eval_secret(s, x)[1] for x in Xl])
    netc = L.predict(L.acts(net, X)["logits"])
    secc = np.array([L.eval_secret(s, x)[0] for x in Xl])
    rng = np.random.default_rng(seed)
    B, S = rng.integers(len(Xl), size=pairs), rng.integers(len(Xl), size=pairs)
    out = {"k": k, "secret_behaviour": float(np.mean(netc == secc)), "vars": []}
    for j, v in enumerate(s["vars"]):
        fixed = vals[B].copy()
        fixed[:, j] = vals[S, j]
        target = np.array([s["table"][",".join(map(str, r))] for r in fixed])
        best = (0.0, None, None)
        for layer, w in (("h1", net["W1"].shape[1]), ("h2", net["W2"].shape[1])):
            for size in range(1, k + 1):
                for U in itertools.combinations(range(w), size):
                    iia = float((L.predict(L.patched_logits(net, X[B], layer, list(U), X[S])) == target).mean())
                    if iia > best[0]:
                        best = (iia, layer, list(U))
        out["vars"].append({"var": v, "best_iia": best[0], "layer": best[1], "units": best[2]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    base = os.path.join(ROOT, "assets", "decompile")
    only = [x for x in a.only.split(",") if x]
    for nid in sorted(os.listdir(base)):
        if only and nid not in only or not os.path.exists(os.path.join(base, nid, "secret.json")):
            continue
        r = calibrate(os.path.join(base, nid), a.k)
        json.dump(r, open(os.path.join(base, nid, "reference.json"), "w"), indent=1)
        print(nid, f"secret behaviour {r['secret_behaviour']:.3f};",
              "; ".join(f"{v['var']['op']} IIA {v['best_iia']:.3f} at {v['layer']} {v['units']}" for v in r["vars"]),
              flush=True)


if __name__ == "__main__":
    main()
