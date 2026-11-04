#!/usr/bin/env python3
"""Publish decompile runs and write their table into docs/benchmarks.md.

usage: python3 scripts/report_decompile.py <label> <run dir> [<run dir> ...]
Copies each run's score, final answer, log, rules and state to results/bench/<label>/<task>/ and rewrites the
section between <!-- BENCH:DECOMPILE:START --> and <!-- BENCH:DECOMPILE:END --> from every results/bench/*/decompile-*.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ["eval/score.json", "final_answer.md", "loop.log", "rules.txt", "state.json"]
TIER = {"native": 0, "read": 1, "probe": 2}


def publish(label, run):
    spec = json.load(open(os.path.join(run, "spec.json")))["spec"]
    task = spec["id"]
    dst = os.path.join(ROOT, "results", "bench", label, task)
    for f in KEEP:
        src = os.path.join(run, f)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, f))
    return task


def table():
    rows = []
    base = os.path.join(ROOT, "results", "bench")
    for label in sorted(os.listdir(base)):
        for task in sorted(os.listdir(os.path.join(base, label))):
            if not task.startswith("decompile-"):
                continue
            d = os.path.join(base, label, task)
            s = json.load(open(os.path.join(d, "eval", "score.json")))
            st = json.load(open(os.path.join(d, "state.json")))
            net, tier = task[len("decompile-"):].rsplit("-", 1)
            best = {}
            for r in (s.get("secret_variable_recovery") or {}).values():
                for k, v in r.items():
                    best[k] = max(best.get(k, 0), v)
            rec = " / ".join(f"{best[k]:.2f}" for k in sorted(best)) or "-"
            q = s.get("quirk_agreement")
            rows.append((net, TIER.get(tier, 9), tier, label, s, st, rec, len(s.get("secret_vars") or []), q))
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    out = ["| Network | Tier | Access | Score | Behaviour | Mechanism | Quirks | Secret variables (best match) | Literals | Turns | Python runs | Passes |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for net, _, tier, label, s, st, rec, nsec, q in rows:
        size = s.get("size") or {}
        passes = sum((st.get("passes_used") or {}).values())
        out.append(f"| {net} | `{tier}` | {s.get('access', {}).get('label', '')} | {s['score']:.3f} | {s['behaviour']:.3f} | "
                   f"{s['mechanism']:.3f} | {round((q or 0) * s.get('quirks', 0))}/{s.get('quirks', 0)} | {rec} | "
                   f"{size.get('literal_chars')}/{size.get('literal_budget')} | {st.get('turn')} | {st.get('runs_used')} | {passes} |")
    return "\n".join(out)


def main():
    label, runs = sys.argv[1], sys.argv[2:]
    for r in runs:
        print("published", publish(label, r))
    p = os.path.join(ROOT, "docs", "benchmarks.md")
    s = open(p).read()
    a, b = "<!-- BENCH:DECOMPILE:START -->", "<!-- BENCH:DECOMPILE:END -->"
    i, j = s.index(a) + len(a), s.index(b)
    open(p, "w").write(s[:i] + "\n" + table() + "\n" + s[j:])
    print("wrote the decompile table to docs/benchmarks.md")


if __name__ == "__main__":
    main()
