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
    pub["exp3-net-a"] = "python, heads 2.1 3.1 3.5"
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
        v = f"{s.get('corpus_guess')}, heads {' '.join(h[1] + '.' + h[3] for h in s.get('copy_named', []))}, P {s.get('copy_precision')}, R {s.get('copy_recall')}"
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
    rows = qwen_data()
    if rows:
        print()
        print("| Qwen3.5-0.8B on MATH-500 (1 sample, 32,768 tokens) | Published (4 samples) | OpenWiSH | Share cut at 32,768 |")
        print("|---|---|---|---|")
        for r in rows:
            pub = f"{100 * r['published']:.1f} %" if r["published"] is not None else "-"
            print(f"| {r['config']} | {pub} | {100 * r['score']:.1f} ± {100 * r['se']:.1f} % | {100 * r['cut']:.1f} % |")


PUBLISHED8 = {"base": 0.079, "p10c": 0.123, "p25c": 0.141, "p50c": 0.481}  # the blog's MATH-500 accuracy, 4 samples


def qwen_data():
    """Experiment 8 replays that finished: config, published accuracy, our accuracy, standard error, share cut at the limit."""
    out = []
    for p in sorted(glob.glob(f"{ROOT}/runs/exp8_conformance/*/summary.json")):
        s = load(p)
        if s and s.get("ok"):
            out.append({"config": s["config"], "published": PUBLISHED8.get(s["config"]), "score": s["score"],
                        "se": s.get("accuracy_se", 0), "cut": s.get("share_cut_at_max_tokens", 0)})
    order = list(PUBLISHED8)
    return sorted(out, key=lambda r: order.index(r["config"]) if r["config"] in order else 99)

PILOT = {  # WSPBench v0.2.1 pilot, condition A (reports/pilot-v0.2.1): read agreement, read exact, write accuracy, write exact
    "gpt-6-luna": (0.441, 0, 0.719, 0), "gpt-6.1-terra": (0.660, 0, 0.973, 1), "gpt-6.1-sol": (0.877, 0, 0.958, 1)}
MODELS = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "claude-opus-5-5")
PILOT_SIX = ("0ccf05ea24942ed4", "0e688589bf551b1f", "2b12b0f4688c3852", "62a9ccaf392d7c3f", "a1393d5b54c62b47", "f987f563d00f687f")
HELDOUT_SIX = ("136a89846d8b907e", "90051af877cf09fd", "97f31af465547846", "f0057721e75f1828", "94bf00ec34fc57d3", "cb64a3c92996f1ec")
TASK_SETS = {"pilot six": PILOT_SIX, "held-out six": HELDOUT_SIX}
HEADLINE = (("pilot six", "A-instructions-r1", "A, through OpenWiSH"), ("pilot six", "A", "A, through OpenWiSH"),
            ("pilot six", "B-inspect", "B, WSPBench tools"), ("pilot six", "C-inspect", "C, WSPBench tools"),
            ("pilot six", "OW-context-action-r5", "OpenWiSH sandbox"), ("pilot six", "OW", "OpenWiSH sandbox"),
            ("held-out six", "A-heldout", "A, through OpenWiSH"), ("held-out six", "B-inspect", "B, WSPBench tools"),
            ("held-out six", "C-inspect", "C, WSPBench tools"), ("held-out six", "OW-heldout", "OpenWiSH sandbox"))
ROUNDS = (("OW-instructions-r1", "r1: task instructions forwarded"), ("OW-context-r2", "r2: public task JSON in Python"),
          ("OW-validated-r3", "r3: format check at finish"), ("OW-guarded-r4", "r4: activation guard before execution"),
          ("OW-context-action-r5", "r5: context accounting, reply blocks kept"))
SUPERSEDED = (("A", "A, through OpenWiSH"), ("A-reparsed", "A, final-message replay"),
              ("OW", "OpenWiSH sandbox"), ("OW-v2", "OpenWiSH sandbox, corrected backend"))
CODEX = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol")


def audit_status(runs):
    """Audit verdicts over the runs that executed code. One-reply runs ran no code and need no audit."""
    states = [load(os.path.join(r, "state.json")) or {} for r in runs]
    coded = [r for r, st in zip(runs, states) if st.get("runs_used")]
    if not coded:
        return "no code run"
    verdicts = [(load(os.path.join(r, "audit_summary.json")) or {}).get("valid") for r in coded]
    status = f"{verdicts.count(True)}/{len(coded)} valid"
    if verdicts.count(False):
        status += f", {verdicts.count(False)} invalid"
    if verdicts.count(None):
        status += f", {verdicts.count(None)} not audited"
    return status


def wspbench_row(label_dir, ids=None):
    read, write, runs = [], [], []
    for f in sorted(glob.glob(os.path.join(label_dir, "*", "eval", "score.json"))):
        s = load(f)
        run = os.path.dirname(os.path.dirname(f))
        if not s or (ids and os.path.basename(run).split("-")[0] not in ids):
            continue
        runs.append(run)
        invalid = (load(os.path.join(run, "audit_summary.json")) or {}).get("valid") is False
        result = (0.0, False) if invalid else (s["score"], s["exact"])  # a run that fails its audit scores 0
        (write if s["family"] == "compositional_synthesis" else read).append(result)
    if len(read) + len(write) < 6:
        return None
    mean = lambda v: sum(x for x, _ in v) / len(v)
    return mean(read), sum(e for _, e in read), mean(write), sum(e for _, e in write), audit_status(runs)


def table(header, rows):
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---:" if h.startswith(("Read", "Write")) else "---" for h in header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def cells(row):
    r, re_, w, we, audit = row
    return [f"{100 * r:.1f} %", f"{re_}/4", f"{100 * w:.1f} %", f"{we}/2", audit]


def found(cond, model, tasks=None):
    return wspbench_row(f"{ROOT}/runs/bench/wspbench-{cond}-{model}", TASK_SETS.get(tasks))


def wspbench():
    metrics = ["Read agreement", "Read exact", "Write accuracy", "Write exact", "Audit"]
    rows = []
    for model in MODELS:
        if model in PILOT:
            rows.append([model, "pilot six", "A", *cells((*PILOT[model], "no code run")), "WSPBench pilot"])
        shown = set()
        for tasks, cond, label in HEADLINE:
            row = found(cond, model, tasks)
            if row and (tasks, label) not in shown:
                shown.add((tasks, label))
                rows.append([model, tasks, label, *cells(row), "this repo"])
    table(["Model", "Tasks", "Condition", *metrics, "Source"], rows)
    rounds = [[model, label, *cells(row)] for model in CODEX for cond, label in ROUNDS if (row := found(cond, model))]
    if rounds:
        table(["Model", "OW round, pilot six", *metrics], rounds)
    old = [[model, label, *cells(row)] for model in MODELS for cond, label in SUPERSEDED
           if model in CODEX and (row := found(cond, model))]
    if old:
        table(["Model", "Superseded transport record", *metrics], old)


HYVE = (("gpt-5.4", 0.74), ("claude-sonnet-4-6", 0.79), ("gemini-3.1-pro", 0.76), ("qwen-3-coder-30b-a3b", 0.67))
AI_CONDITIONS = (("A", "prompt only, one reply"), ("WO", "weights only, no execution"), ("P", "weights and forward probe"))


def agentic_data():
    """Rows of {model, kind, label, cases, accuracy, audit, source}. kind: A, WO, P, HyVE or baseline.
    Tag accuracy pools components (the benchmark's definition). A run that fails its audit scores 0."""
    rows = []
    for cond, label in AI_CONDITIONS:
        for d in sorted(glob.glob(f"{ROOT}/runs/bench/agentic-interp-{cond}-*")):
            model = os.path.basename(d)[len(f"agentic-interp-{cond}-"):]
            runs = [os.path.dirname(os.path.dirname(f)) for f in sorted(glob.glob(f"{d}/*/eval/score.json"))]
            scores = [(load(f"{r}/eval/score.json") or {}, (load(f"{r}/audit_summary.json") or {}).get("valid")) for r in runs]
            correct = sum(0 if valid is False else s.get("correct", 0) for s, valid in scores)
            total = sum(s.get("components", 0) for s, _ in scores)
            if total:
                rows.append({"model": model, "kind": cond, "label": label, "cases": len(runs), "accuracy": correct / total,
                             "audit": audit_status(runs), "source": "this repo"})
    if not rows:
        return rows
    rows += [{"model": m, "kind": "HyVE", "label": "HyVE: activations, ablation, patching", "cases": 84, "accuracy": a,
              "audit": "-", "source": "arXiv 2606.24026"} for m, a in HYVE]
    gold = [c for f in glob.glob(f"{ROOT}/assets/agentic_interp/hf/dataset/annotations/*.json") for c in (load(f) or {}).get("components", [])]
    if gold:
        tag = lambda c: c["role"]["tag"]
        kind = lambda c: "MAPPER" if c["id"].endswith("MLP") else "AGGREGATOR"
        for label, hit in (("always MAPPER", lambda c: tag(c) == "MAPPER"),
                           ("MLP MAPPER, attention AGGREGATOR (read off the gold)", lambda c: tag(c) == kind(c))):
            rows.append({"model": "-", "kind": "baseline", "label": label, "cases": 84, "accuracy": sum(map(hit, gold)) / len(gold),
                         "audit": "-", "source": "gold annotation"})
    return rows


def agentic_interp():
    rows = agentic_data()
    if rows:
        table(["Model", "AgenticInterpBench condition", "Cases", "Tag accuracy", "Audit", "Source"],
              [[r["model"], r["label"], str(r["cases"]), f"{100 * r['accuracy']:.1f} %", r["audit"], r["source"]] for r in rows])


def tools_data():
    """Exact solves on the 12 WSPBench tasks (pilot six and held-out six) per model: WSPBench B, C and the OpenWiSH sandbox.
    None marks a condition with no complete run."""
    out = {}
    for model in MODELS:
        row = {}
        for key, labels in (("B", ("B, WSPBench tools",)), ("C", ("C, WSPBench tools",)), ("OW", ("OpenWiSH sandbox",))):
            total = 0
            for tasks in TASK_SETS:
                cond = next((c for t, c, l in HEADLINE if t == tasks and l in labels and found(c, model, tasks)), None)
                if cond is None:
                    total = None
                    break
                r = found(cond, model, tasks)
                total += r[1] + r[3]
            row[key] = total
        if any(v is not None for v in row.values()):
            out[model] = row
    return out

if __name__ == "__main__":
    main()
    wspbench()
    agentic_interp()
