"""Compare every task in two saved run sets. Exit 1 for regressions or incomplete evidence.

usage: python3 scripts/bench_compare.py BASELINE CANDIDATE [--expected-task ID ...]
No scorer, prompt, submission or weight is changed. Means never hide a task-level regression.
"""
import argparse
import json
import math
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def records(root):
    out = {}
    for path in sorted(Path(root).glob("*/eval/score.json")):
        score = read(path)
        task = score.get("task", path.parent.parent.name)
        if task in out:
            raise ValueError(f"duplicate task {task} in {root}")
        run = path.parent.parent
        out[task] = (score, read(run / "state.json"), read(run / "audit_summary.json"))
    return out


def compare_runs(baseline, candidate, expected_tasks=None):
    base, cand = records(baseline), records(candidate)
    tasks = sorted(set(expected_tasks or ()) | base.keys() | cand.keys())
    report = {"passed": False, "regressions": [], "missing": [], "missing_baseline": [],
              "audit_pending": [], "invalid": [], "tasks": []}
    for task in tasks:
        if task not in base:
            report["missing_baseline"].append(task)
        if task not in cand:
            report["missing"].append(task)
        if task not in base or task not in cand:
            continue
        b, bs, _ = base[task]
        c, cs, audit = cand[task]
        scores = [b.get("score"), c.get("score")]
        external_tools = any("tool_free_turns" in state and state["tool_free_turns"] != state.get("turn")
                             for state in (bs, cs))
        rejected = any(state.get("end_reason") == "submission rejected" for state in (bs, cs))
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in scores)
                or not b.get("ok") or not c.get("ok") or not bs.get("done") or not cs.get("done") or external_tools or rejected):
            report["invalid"].append(task)
            continue
        if audit.get("valid") is False:
            report["invalid"].append(task)
        elif audit.get("suspicious", 0) > 0 or (cs.get("runs_used", 0) > 0 and audit.get("valid") is not True):
            report["audit_pending"].append(task)
        regressed = c["score"] < b["score"] or (b.get("exact") is True and c.get("exact") is not True)
        if regressed:
            report["regressions"].append(task)
        report["tasks"].append({"task": task, "baseline": b["score"], "candidate": c["score"],
                                "delta": c["score"] - b["score"], "regressed": regressed,
                                "candidate_audit": audit.get("valid")})
    report["passed"] = bool(tasks) and not any(report[k] for k in
        ("regressions", "missing", "missing_baseline", "audit_pending", "invalid"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-task", action="append")
    args = parser.parse_args()
    report = compare_runs(args.baseline, args.candidate, args.expected_task)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
