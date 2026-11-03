"""AgenticInterpBench adapter (github.com/Ziyu-Yao-NLP-Lab/LLM-Circuit-Explainer). The agent tags each localized
circuit component with one class of the benchmark's 5-class taxonomy. The score is the benchmark's tag accuracy.

  score    --case <id>                  grade the answer against the gold annotation; prints one JSON line
  validate --case <id> --components ..  format check of $OPENWISH_SUBMISSION, never grades

Gold annotations stay in assets/agentic_interp (hidden from the sandbox)."""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "assets/agentic_interp/hf/dataset/annotations"
TAGS = ("INDICATOR", "AGGREGATOR", "ROUTER", "MAPPER", "COMBINER")


def check(sub, component_ids):
    """Format check only. Returns "ok" or the first problem."""
    if not isinstance(sub, dict) or set(sub) != {"tags"} or not isinstance(sub["tags"], dict):
        return 'the answer must be one JSON object {"tags": {component_id: TAG}}'
    if set(sub["tags"]) != set(component_ids):
        return f"tags must name exactly the components {sorted(component_ids)}"
    wrong = sorted(k for k, v in sub["tags"].items() if v not in TAGS)
    return f"unknown tag for {wrong}: use one of {', '.join(TAGS)}" if wrong else "ok"


def parse(text):
    """The last JSON object in the text holding a "tags" key."""
    found, dec = None, json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = dec.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "tags" in value:
            found = value
    return found


def grade(gold, sub):
    truth = {c["id"]: c["role"]["tag"] for c in gold["components"]}
    error = check(sub, truth) if sub is not None else "no JSON object with tags found"
    if error != "ok":
        return {"valid": False, "error": error, "score": 0.0, "exact": False, "correct": 0, "components": len(truth)}
    hits = {k: sub["tags"][k] == v for k, v in truth.items()}
    correct = sum(hits.values())
    return {"valid": True, "score": correct / len(truth), "exact": correct == len(truth), "correct": correct,
            "components": len(truth), "per_component": {k: {"gold": truth[k], "answer": sub["tags"][k]} for k in truth}}


def answer_text():
    """A validated or sandbox-written submission wins over the reply text."""
    run = Path(os.environ.get("OPENWISH_RUN", ""))
    for path in (run / "validated_submission.json", run / "work/submission.json"):
        if path.is_file() and not path.is_symlink():
            return path.read_text()
    return Path(os.environ["OPENWISH_FINAL_ANSWER"]).read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "validate"])
    ap.add_argument("--case", required=True, type=int)
    ap.add_argument("--components", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "validate":
        error = check(parse(Path(os.environ["OPENWISH_SUBMISSION"]).read_text()) or {}, a.components)
        print(json.dumps({"ok": True} if error == "ok" else {"ok": False, "error": error}))
        return
    gold = json.loads((GOLD / f"case_{a.case}.json").read_text())
    print(json.dumps({"ok": True, "case": a.case, **grade(gold, parse(answer_text()))}))


if __name__ == "__main__":
    main()
