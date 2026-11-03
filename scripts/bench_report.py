"""Build the benchmark tables from run folders, so every published number comes from a file in the repo's runs.

usage: python3 scripts/bench_report.py [runs/bench/<label>] > table.md
Reads: runs/bench/<label>/*/eval/score.json and state.json, runs/exp8_conformance/*/summary.json,
       and the published results in assets/lm_understand-logs (for the comparison column)."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = f"{ROOT}/assets/lm_understand-logs"
label_dir = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/bench/claude-opus-5-5"


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def published():
    pub = {}
    for r in load(f"{LOGS}/exp1_which_game/eval/scores.json") or []:
        pub[f"exp1-{r['id']}"] = f"{r['guess']} ({r['confidence']} %)"
    for r in load(f"{LOGS}/exp2_diffusion/eval/scores.json") or []:
        pub[f"exp2-{r['id']}"] = f"{r['guess']} ({r['confidence']} %)"
    m = load(f"{LOGS}/exp5_write_from_scratch/eval/minatar_eval.json") or {}
    for g in ("breakout", "seaquest", "asterix"):
        if g in m:
            pub[f"exp5-{g}"] = f"{m[g]['hand']['mean']:.1f}"
    pub["exp4-task"] = "1,024 / 1,024"
    pub["exp3-net-a"] = "python; heads 2.1 3.1 3.5"
    e6 = load(f"{LOGS}/exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json") or {}
    vals = [e6.get(k, {}).get("score") for k in ("v1_b0_vs_lr3e-5", "v1_b0_vs_lr3e-5_s1", "v1_b0_vs_lr3e-5_s2")]
    if all(v is not None for v in vals):
        pub["exp6-budget_0"] = " / ".join(f"{100 * v:.1f}" for v in vals) + " %"
    return pub


def ours(d):
    s = load(os.path.join(d, "eval", "score.json"))
    st = load(os.path.join(d, "state.json")) or {}
    if not s:
        return None, st
    name = os.path.basename(d)
    if name.startswith(("exp1", "exp2")):
        v = f"{s.get('guess')} ({s.get('confidence')} %)" + ("" if s.get("correct") else ", wrong")
    elif name.startswith("exp3"):
        v = f"{s.get('corpus_guess')}; heads {' '.join(h[1] + '.' + h[3] for h in s.get('copy_named', []))}; P {s.get('copy_precision')}, R {s.get('copy_recall')}"
    elif name.startswith("exp4"):
        v = f"{s.get('success')} / {s.get('episodes')} ({s.get('changed')} weights changed)"
    elif name.startswith("exp5"):
        v = f"{s.get('mean', 0):.1f} ± {s.get('se', 0):.1f}"
    elif name.startswith("exp6"):
        v = " / ".join(f"{100 * s[k]['score']:.1f}" for k in ("ppo_same_size_s0", "ppo_same_size_s1", "ppo_same_size_s2") if k in s) + " %"
    else:
        v = str(s.get("score"))
    return v, st


def main():
    pub = published()
    print("| Environment | Published agent (GPT-6 Astra) | OpenWiSH + Claude Opus 5.5 | Turns | Python runs | Cost (USD) | Audit |")
    print("|---|---|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(label_dir, "*/"))):
        d = d.rstrip("/")
        v, st = ours(d)
        if v is None:
            continue
        name = os.path.basename(d)
        audit = load(os.path.join(d, "audit_summary.json")) or {}
        status = {True: "valid", False: "invalid"}.get(audit.get("valid"), "not audited")
        print(f"| {name} | {pub.get(name, '-')} | {v} | {st.get('turn', '-')} | {st.get('runs_used', '-')} | {st.get('cost_usd', 0):.2f} | {status} |")
    rows = sorted(glob.glob(f"{ROOT}/runs/exp8_conformance/*/summary.json"))
    if rows:
        print()
        print("| Qwen3.5-0.8B on MATH-500 (1 sample, 32,768 tokens) | Published (4 samples) | OpenWiSH | Share cut at 32,768 |")
        print("|---|---|---|---|")
        published8 = {"base": "7.9 %", "p10c": "12.3 %", "p25c": "14.1 %", "p50c": "48.1 %"}
        for p in rows:
            s = load(p)
            if s and s.get("ok"):
                c = s["config"]
                print(f"| {c} | {published8.get(c, '-')} | {100 * s['score']:.1f} ± {100 * s.get('accuracy_se', 0):.1f} % | {100 * s.get('share_cut_at_max_tokens', 0):.1f} % |")


PILOT = {  # WSPBench v0.2.1 pilot, condition A (reports/pilot-v0.2.1): read agreement, read exact, write accuracy, write exact
    "gpt-6-luna": (0.441, 0, 0.719, 0), "gpt-6.1-terra": (0.660, 0, 0.973, 1), "gpt-6.1-sol": (0.877, 0, 0.958, 1)}


def wspbench_row(label_dir):
    read, write = [], []
    for f in glob.glob(os.path.join(label_dir, "*", "eval", "score.json")):
        s = load(f)
        if not s:
            continue
        (write if s["family"] == "compositional_synthesis" else read).append((s["score"], s["exact"]))
    if len(read) + len(write) < 6:
        return None
    mean = lambda v: sum(x for x, _ in v) / len(v)
    return mean(read), sum(e for _, e in read), mean(write), sum(e for _, e in write)


def wspbench():
    print()
    print("| Model | Condition | Read agreement | Read exact | Write accuracy | Write exact | Source |")
    print("|---|---|---:|---:|---:|---:|---|")
    for model in ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "claude-opus-5-5"):
        if model in PILOT:
            r, re_, w, we = PILOT[model]
            print(f"| {model} | A | {100 * r:.1f} % | {re_}/4 | {100 * w:.1f} % | {we}/2 | WSPBench pilot |")
        for cond, label in (("A", "A, through OpenWiSH"), ("A-reparsed", "A, final-message replay"),
                            ("OW", "OpenWiSH sandbox"), ("OW-v2", "OpenWiSH sandbox, corrected backend")):
            row = wspbench_row(f"{ROOT}/runs/bench/wspbench-{cond}-{model}")
            if row:
                r, re_, w, we = row
                print(f"| {model} | {label} | {100 * r:.1f} % | {re_}/4 | {100 * w:.1f} % | {we}/2 | this repo |")


if __name__ == "__main__":
    main()
    wspbench()
