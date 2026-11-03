import json
import numpy as np

W = {n: load(n) for n in NAMES}
FAMILY = "compositional_synthesis"
_BINARY = {"and", "or", "xor", "nand", "nor"}
_NODE_KEYS = {"input": {"op", "index"}, "not": {"op", "arg"}, "mux": {"op", "condition", "then", "else"}, "majority": {"op", "args"}}
_PATCH_OPS = {"set", "array", "scale", "zero", "rank_one", "rank1"}


def _program(node, depth=0, count=None):
    count = count if count is not None else [0]
    count[0] += 1
    if not isinstance(node, dict) or count[0] > 128 or depth > 6:
        return "the expression exceeds the grammar (at most 128 nodes, depth 6)"
    op = node.get("op")
    keys = {"op", "left", "right"} if op in _BINARY else _NODE_KEYS.get(op)
    if keys is None or set(node) != keys:
        return f"node {node.get('op')!r} has keys {sorted(node)}; expected {sorted(keys) if keys else 'a known op'}"
    if op == "input" and not (type(node["index"]) is int and 0 <= node["index"] < 10):
        return "input index must be an integer 0..9"
    kids = [node["left"], node["right"]] if op in _BINARY else [node["arg"]] if op == "not" else \
        [node["condition"], node["then"], node["else"]] if op == "mux" else node["args"] if op == "majority" else []
    if op == "majority" and (not isinstance(kids, list) or len(kids) != 3):
        return "majority needs exactly three args"
    for k in kids:
        err = _program(k, depth + 1, count)
        if err:
            return err
    return None


def _table(v):
    return None if isinstance(v, list) and len(v) == 1024 and all(type(x) is int and x in (0, 1) for x in v) else \
        "an output table must be a list of exactly 1024 integers 0 or 1"


def check_submission(sub):
    """Format check only (keys, grammar, sizes). It never grades. Returns "ok" or the first problem."""
    if not isinstance(sub, dict):
        return "the submission must be one JSON object"
    if FAMILY == "rule_recovery":
        if set(sub) == {"program"}:
            return _program(sub["program"]) or "ok"
        if set(sub) == {"outputs"}:
            return _table(sub["outputs"]) or "ok"
        return f"keys {sorted(sub)}: use exactly {{program}} or {{outputs}}"
    if FAMILY == "edit_effects":
        if set(sub) == {"before_program", "after_program"}:
            return _program(sub["before_program"]) or _program(sub["after_program"]) or "ok"
        if set(sub) == {"before_outputs", "after_outputs"}:
            return _table(sub["before_outputs"]) or _table(sub["after_outputs"]) or "ok"
        return f"keys {sorted(sub)}: use exactly {{before_program, after_program}} or {{before_outputs, after_outputs}}"
    if set(sub) != {"patch"}:
        return f"keys {sorted(sub)}: use exactly {{patch}}"
    if not isinstance(sub["patch"], list) or len(sub["patch"]) > 8192:
        return "patch must be a list of at most 8192 operations"
    bad = [o for o in sub["patch"] if not isinstance(o, dict) or o.get("op") not in _PATCH_OPS or "tensor" not in o]
    if bad:
        return f"{len(bad)} patch operations lack a known op or a tensor"
    tensors = sorted({n.split(".", 1)[1] for n in NAMES})
    wrong = sorted({o["tensor"] for o in sub["patch"] if o["tensor"] not in tensors})
    return f"unknown tensor names {wrong}: patches name tensors without the checkpoint prefix, one of {tensors}" if wrong else "ok"
