"""Helpers shared by the experiment scorers (plain Python, no dependencies)."""
import json
import os
import re
import sys


def clean(s):
    return re.sub(r"[*_`#]", "", s).strip()


def field(text, name):
    """The value of the first 'Name: value' line, or of a 'Name' / '## Name' heading followed by the value
    on the next non-empty line (markdown emphasis ignored)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\W*" + name + r"\s*(?::\s*(.*))?$", clean(line), re.I)
        if not m:
            continue
        if m.group(1):
            return m.group(1).strip()
        for nxt in lines[i + 1:]:
            if clean(nxt):
                return clean(nxt)
    return ""


def confidence(text):
    m = re.search(r"(\d+(?:\.\d+)?)", field(text, "Confidence"))
    return int(round(float(m.group(1)))) if m else None


def earliest(text, table):
    """The key of `table` (key -> list of patterns) whose pattern occurs first in `text`."""
    best, pos = "other", len(text) + 1
    low = text.lower()
    for key, pats in table.items():
        for p in pats:
            m = re.search(p, low)
            if m and m.start() < pos:
                best, pos = key, m.start()
    return best


def emit(result):
    print(json.dumps({"ok": True, **result}))


def final_answer():
    return open(os.environ["OPENWISH_FINAL_ANSWER"]).read()
