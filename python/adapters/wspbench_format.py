"""Submission grammar and public shape checks. No weights, target execution or scoring."""
import math

_BINARY = {"and", "or", "xor", "nand", "nor"}
_NODE_KEYS = {"input": {"op", "index"}, "not": {"op", "arg"}, "mux": {"op", "condition", "then", "else"}, "majority": {"op", "args"}}
_PATCH_OPS = {"set", "replace", "add", "scale", "zero", "rank_one"}


def _program(node, depth=0, count=None):
    count = count if count is not None else [0]
    count[0] += 1
    if not isinstance(node, dict) or count[0] > 128 or depth > 6:
        return "the expression exceeds the grammar (at most 128 nodes, depth 6)"
    op = node.get("op")
    if not isinstance(op, str):
        return "program op must be a string"
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


def check_submission(sub, family, shapes, forecast_ids=()):
    """Format check only (keys, grammar, sizes). It never grades. Returns "ok" or the first problem."""
    if not isinstance(sub, dict):
        return "the submission must be one JSON object"
    if family == "rule_recovery":
        if set(sub) == {"program"}:
            return _program(sub["program"]) or "ok"
        if set(sub) == {"outputs"}:
            return _table(sub["outputs"]) or "ok"
        return f"keys {sorted(sub)}: use exactly {{program}} or {{outputs}}"
    if family == "edit_effects":
        if set(sub) == {"before_program", "after_program"}:
            return _program(sub["before_program"]) or _program(sub["after_program"]) or "ok"
        if set(sub) == {"before_outputs", "after_outputs"}:
            return _table(sub["before_outputs"]) or _table(sub["after_outputs"]) or "ok"
        return f"keys {sorted(sub)}: use exactly {{before_program, after_program}} or {{before_outputs, after_outputs}}"
    if family == "causal_forecast":
        return _effects(sub, forecast_ids)
    if family not in ("compositional_synthesis", "coupled_repair"):
        return f"unknown task family {family!r}"
    if set(sub) != {"patch"}:
        return f"keys {sorted(sub)}: use exactly {{patch}}"
    if not isinstance(sub["patch"], list) or len(sub["patch"]) > 8192:
        return "patch must be a list of at most 8192 operations"
    for operation in sub["patch"]:
        error = _patch_operation(operation, shapes)
        if error:
            return error
    return "ok"


def forecast_ids(task):
    """The public forecast intervention IDs of a task (empty for other families)."""
    return [f["id"] for f in task.get("public", {}).get("forecast_interventions", [])]


def _effects(sub, forecast_ids):
    if set(sub) != {"effects"} or not isinstance(sub["effects"], dict):
        return f"keys {sorted(sub)}: use exactly {{effects}}"
    if set(sub["effects"]) != set(forecast_ids):
        return f"effects must name exactly the forecast IDs {sorted(forecast_ids)}"
    for fid, pair in sub["effects"].items():
        if not (isinstance(pair, list) and len(pair) == 2 and all(_number(v) and 0 <= v <= 1 for v in pair)):
            return f"effects[{fid!r}] must be two fractions in [0, 1], one per output head"
    return "ok"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e6


def _array(value, shape):
    if not shape:
        return _number(value)
    return isinstance(value, list) and len(value) == shape[0] and all(_array(v, shape[1:]) for v in value)


def _indices(value, size):
    return (isinstance(value, list) and bool(value) and
            all(type(i) is int and 0 <= i < size for i in value) and len(set(value)) == len(value))


def _patch_operation(op, shapes):
    if not isinstance(op, dict) or not isinstance(op.get("tensor"), str) or op["tensor"] not in shapes:
        return "patch names an unknown tensor"
    kind, shape = op.get("op"), shapes[op["tensor"]]
    if not isinstance(kind, str) or kind not in _PATCH_OPS:
        return "patch operation must be set, replace, add, scale, zero or rank_one"
    if kind == "set":
        index = op.get("index")
        if set(op) != {"op", "tensor", "index", "value"}:
            return "set needs exactly op, tensor, index and value"
        if not isinstance(index, list) or len(index) != len(shape) or not all(type(i) is int and 0 <= i < n for i, n in zip(index, shape)):
            return "patch index does not match the public tensor shape"
        if not _number(op["value"]):
            return "patch value must be a finite number of magnitude at most 1e6"
    elif kind in {"replace", "add"}:
        if set(op) != {"op", "tensor", "values"} or not _array(op.get("values"), shape):
            return "replace/add needs bounded numeric values matching the public tensor shape"
    elif kind in {"scale", "zero"}:
        allowed = {"op", "tensor", "axis", "indices"} | ({"factor"} if kind == "scale" else set())
        if set(op) - allowed or (kind == "scale" and not _number(op.get("factor"))):
            return "invalid scale/zero fields or factor"
        if "axis" in op or "indices" in op:
            axis = op.get("axis")
            if type(axis) is not int or not 0 <= axis < len(shape) or not _indices(op.get("indices"), shape[axis]):
                return "invalid scale/zero axis or indices"
    elif kind == "rank_one":
        allowed = {"op", "tensor", "left", "right", "alpha", "columns"}
        if set(op) - allowed or not {"left", "right", "alpha"} <= set(op) or len(shape) != 2:
            return "invalid rank_one fields or tensor rank"
        if not _array(op["left"], [shape[0]]) or not _array(op["right"], [shape[1]]) or not _number(op["alpha"]):
            return "rank_one vectors must match the public tensor shape and contain bounded finite numbers"
        if "columns" in op and not _indices(op["columns"], shape[1]):
            return "invalid rank_one columns"
    return None
