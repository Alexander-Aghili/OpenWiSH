"""Format-only finish validator using the same public grammar as the Python helper."""
import argparse
import json
import os
from pathlib import Path
import runpy

check_submission = runpy.run_path(str(Path(__file__).with_name("wspbench_format.py")))["check_submission"]


def validate_submission(text, family, task):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    try:
        submission = json.loads(text)
    except (ValueError, TypeError):
        return {"ok": False, "error": "Expected one JSON submission object. Prose is not a submission. Use the declared Python tool for weight algebra and check_submission(sub) for format checks."}
    try:
        error = check_submission(submission, family, task["architecture"]["shapes"])
    except (TypeError, ValueError, KeyError, RecursionError) as exc:
        return {"ok": False, "error": f"Malformed submission: {exc}"}
    return {"ok": True} if error == "ok" else {"ok": False, "error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--family", choices=["rule_recovery", "edit_effects", "compositional_synthesis"], required=True)
    args = parser.parse_args()
    task = json.loads(args.public.read_text())["task"]
    text = Path(os.environ["OPENWISH_SUBMISSION"]).read_text()
    print(json.dumps(validate_submission(text, args.family, task)))


if __name__ == "__main__":
    main()
