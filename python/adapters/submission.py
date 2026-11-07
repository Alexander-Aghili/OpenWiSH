"""Reading an agent's submission: shared by the benchmark scorers (agentic_interp.py, wspbench_bridge.py) and the
task generators. Standard library only, so any interpreter can load it by path."""
import json
import os
import re
from pathlib import Path


def submission_text(use_files=True):
    """The text to grade: a validated submission, else one the agent wrote to /work/submission.json, else the
    final reply. Symlinks are ignored, so the sandbox cannot point the scorer at another file."""
    if use_files:
        run = Path(os.environ.get("OPENWISH_RUN", ""))
        for path in (run / "validated_submission.json", run / "work/submission.json"):
            if path.is_file() and not path.is_symlink():
                return path.read_text()
    return Path(os.environ["OPENWISH_FINAL_ANSWER"]).read_text()


def last_json_object(text, key=None, prefer_fenced=False):
    """The last JSON object in `text` (holding `key`, if given). With prefer_fenced, the last ```json block
    that parses as an object wins over objects in the prose."""
    def fits(v):
        return isinstance(v, dict) and (key is None or key in v)

    if prefer_fenced:
        for block in reversed(re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)):
            try:
                v = json.loads(block)
            except json.JSONDecodeError:
                continue
            if fits(v):
                return v
    found, dec = None, json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            v, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if fits(v):
            found = v
    return found


def toml_string(s):
    """`s` as a TOML multi-line basic string."""
    return '"""' + s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'
