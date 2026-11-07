"""AgenticInterpBench adapter (github.com/Ziyu-Yao-NLP-Lab/LLM-Circuit-Explainer). The agent tags each localized
circuit component with one class of the benchmark's 5-class taxonomy. The score is the benchmark's tag accuracy.

  score    --case <id>                  grade the answer against the gold annotation; prints one JSON line
  validate --case <id> --components ..  format check of $OPENWISH_SUBMISSION, never grades

Gold annotations stay in assets/agentic_interp (hidden from the sandbox)."""
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# loaded by path: these scripts run under python -I or another project's interpreter
_spec = importlib.util.spec_from_file_location("subm", Path(__file__).resolve().parent / "submission.py")
subm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(subm)

GOLD = ROOT / "assets/agentic_interp/hf/dataset/annotations"
TAGS = ("INDICATOR", "AGGREGATOR", "ROUTER", "MAPPER", "COMBINER")


def check(sub, component_ids, protocol=False, requests=None):
    """Format check only. Returns "ok" or the first problem. With protocol=True every component also needs evidence:
    a hypothesis, and a finished probe call r<k> whose request ablated or patched that component."""
    keys = {"tags", "evidence"} if protocol else {"tags"}
    shape = '{"tags": {component_id: TAG}, "evidence": {component_id: {"hypothesis", "test", "supports"}}}' if protocol \
        else '{"tags": {component_id: TAG}}'
    if not isinstance(sub, dict) or set(sub) != keys or not isinstance(sub["tags"], dict):
        return f"the answer must be one JSON object {shape}"
    if set(sub["tags"]) != set(component_ids):
        return f"tags must name exactly the components {sorted(component_ids)}"
    wrong = sorted(k for k, v in sub["tags"].items() if v not in TAGS)
    if wrong:
        return f"unknown tag for {wrong}: use one of {', '.join(TAGS)}"
    return evidence_problem(sub["evidence"], component_ids, requests) or "ok" if protocol else "ok"


def tested_components(requests, k):
    """Components a finished probe call k ablated or patched, from the run's own request record."""
    req, done = Path(requests) / f"forward_{k}.json", Path(requests) / f"forward_{k}.done.json"
    if not (req.is_file() and done.is_file()) or req.is_symlink():
        return set()
    try:
        body = json.loads(req.read_text())
    except json.JSONDecodeError:
        return set()
    return set((body.get("ablate") or {})) | {p.get("component") for p in body.get("patch") or [] if isinstance(p, dict)}


def evidence_problem(evidence, component_ids, requests):
    if not isinstance(evidence, dict) or set(evidence) != set(component_ids):
        return f"evidence must name exactly the components {sorted(component_ids)}"
    for c, e in evidence.items():
        if not isinstance(e, dict) or set(e) != {"hypothesis", "test", "supports"} or not isinstance(e["supports"], bool):
            return f'evidence[{c!r}] needs exactly "hypothesis", "test" and "supports" (true or false)'
        if not isinstance(e["hypothesis"], str) or len(e["hypothesis"].split()) < 5:
            return f"evidence[{c!r}]: state the hypothesis in a sentence"
        m = re.fullmatch(r"r([0-9]{1,4})", e["test"]) if isinstance(e["test"], str) else None
        if not m or requests is None or c not in tested_components(requests, int(m.group(1))):
            return f"evidence[{c!r}]: test must name a finished forward call r<k> whose request ablated or patched {c}"
    return None


def parse(text):
    """The last JSON object in the text holding a "tags" key."""
    return subm.last_json_object(text, key="tags")


def grade(gold, sub):
    truth = {c["id"]: c["role"]["tag"] for c in gold["components"]}
    if isinstance(sub, dict) and "evidence" in sub:  # protocol answers: the tags are scored, the evidence was checked at finish
        sub = {"tags": sub["tags"]}
    error = check(sub, truth) if sub is not None else "no JSON object with tags found"
    if error != "ok":
        return {"valid": False, "error": error, "score": 0.0, "exact": False, "correct": 0, "components": len(truth)}
    hits = {k: sub["tags"][k] == v for k, v in truth.items()}
    correct = sum(hits.values())
    return {"valid": True, "score": correct / len(truth), "exact": correct == len(truth), "correct": correct,
            "components": len(truth), "per_component": {k: {"gold": truth[k], "answer": sub["tags"][k]} for k in truth}}


def answer_text():
    """A validated or sandbox-written submission wins over the reply text."""
    return subm.submission_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "validate"])
    ap.add_argument("--case", required=True, type=int)
    ap.add_argument("--components", nargs="*", default=[])
    ap.add_argument("--protocol", action="store_true", help="require per-component evidence from finished probe calls")
    a = ap.parse_args()
    if a.cmd == "validate":
        submission = Path(os.environ["OPENWISH_SUBMISSION"])  # <run>/requests/submission-<turn>.txt
        error = check(parse(submission.read_text()) or {}, a.components, protocol=a.protocol, requests=submission.parent)
        print(json.dumps({"ok": True} if error == "ok" else {"ok": False, "error": error}))
        return
    gold = json.loads((GOLD / f"case_{a.case}.json").read_text())
    print(json.dumps({"ok": True, "case": a.case, **grade(gold, parse(answer_text()))}))


if __name__ == "__main__":
    main()
