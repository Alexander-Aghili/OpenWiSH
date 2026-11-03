"""Experiment 8 conformance: score the base model and the published edits on the hidden MATH-500 set with the
published sampling protocol, on the configured GPU machine (assets/exp8/remote.json, or this computer).

usage: python3 scripts/exp8_conformance.py [base p50c p10c p25c] [--limit 500] [--samples 1]
Writes runs/exp8_conformance/<name>/ and prints one JSON line per configuration."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python", "adapters"))
import qwen_pass as Q  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="*", default=["base", "p50c", "p10c", "p25c"])
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--set", default="math500")
a = ap.parse_args()
empty = os.path.join(ROOT, "assets", "exp8", "published", "_none")
os.makedirs(empty, exist_ok=True)
for name in a.configs:
    edits = empty if name == "base" else os.path.join(ROOT, "assets", "exp8", "published", name)
    out = os.path.join(ROOT, "runs", "exp8_conformance", f"{name}_{a.set}")
    done = os.path.join(out, "summary.json")
    if os.path.exists(done) and json.load(open(done)).get("ok"):  # failed records rerun
        print(open(done).read().strip())
        continue
    res = Q.run_worker(["evaluate", "--model", "{model}", "--edits", "{edits}", "--set", a.set, "--data", "{data}",
                        "--limit", str(a.limit), "--samples", str(a.samples), "--out", "{out}"],
                       {"edits": edits, "tag": name}, out)
    res["config"] = name
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("config", "ok", "score", "accuracy_se", "share_cut_at_max_tokens", "mean_tokens", "problems", "error")}), flush=True)
