"""The one-line text baseline for tasks/vuln: answer YES unless the code compares a pointer with NULL.

usage: python3 scripts/vuln_text_rule.py   (reads the hidden set assets/vuln/eval/programs.json, writes
runs/vuln_text_rule.json). No model runs: this measures how far a surface rule gets on the scrubbed programs."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NULL_CMP = re.compile(r"(==|!=)\s*NULL|NULL\s*(==|!=)")


def says_yes(code):
    return not NULL_CMP.search(code)


def counts(programs):
    pos = [p for p in programs if p["label"] == 1]
    neg = [p for p in programs if p["label"] == 0]
    tp = sum(says_yes(p["code"]) for p in pos)
    fp = sum(says_yes(p["code"]) for p in neg)
    return {"tp": tp, "fp": fp, "positives": len(pos), "negatives": len(neg),
            "accuracy": (tp + len(neg) - fp) / (len(pos) + len(neg))}


def main():
    data = json.load(open(os.path.join(ROOT, "assets/vuln/eval/programs.json")))
    out = {"rule": "YES unless the code compares a pointer with NULL", **counts(data["programs"] if isinstance(data, dict) else data)}
    with open(os.path.join(ROOT, "runs/vuln_text_rule.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
