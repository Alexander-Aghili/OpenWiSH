"""Shared code for the decompile environments (harness side only; never mounted into the sandbox).

- the secret-program DSL (evaluation only; generation lives in scripts/prepare_decompile.py)
- the network: ReLU MLP x -> h1 -> h2 -> logits, stored as W1,b1,W2,b2,W3,b3 .npy files
- forward passes with unit patches (interchange interventions)
- the program checker and a sandboxed runner for submitted programs
"""
import ast
import itertools
import json
import os
import subprocess
import sys

import numpy as np

LAYERS = ("h1", "h2")
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")


# ----------------------------------------------------------------------------- secret programs

def eval_var(v, x):
    op = v["op"]
    if op == "parity":
        return sum(x[i] for i in v["bits"]) % 2
    if op in ("and", "or"):
        lits = [x[i] ^ neg for i, neg in v["lits"]]
        return int(all(lits)) if op == "and" else int(any(lits))
    if op == "count":
        return sum(x[i] for i in v["bits"])
    if op == "gt":
        a = x[v["a"][0]] * 2 + x[v["a"][1]]
        b = x[v["b"][0]] * 2 + x[v["b"][1]]
        return int(a > b)
    raise ValueError(op)


def var_range(v):
    return len(v["bits"]) + 1 if v["op"] == "count" else 2


def eval_secret(secret, x):
    vals = [eval_var(v, x) for v in secret["vars"]]
    return secret["table"][",".join(map(str, vals))], vals


def all_inputs(n):
    return [tuple(int(b) for b in bits) for bits in itertools.product((0, 1), repeat=n)]


# ----------------------------------------------------------------------------- the network

def load_net(d):
    return {k: np.load(os.path.join(d, k + ".npy")).astype(np.float64) for k in TENSORS}


def acts(net, X):
    """Every layer's activations for inputs X (rows)."""
    X = np.asarray(X, dtype=np.float64)
    h1 = np.maximum(X @ net["W1"] + net["b1"], 0)
    h2 = np.maximum(h1 @ net["W2"] + net["b2"], 0)
    return {"h1": h1, "h2": h2, "logits": h2 @ net["W3"] + net["b3"]}


def run_from(net, layer, h):
    """Continue the forward pass from a (possibly patched) hidden layer."""
    if layer == "h1":
        h = np.maximum(h @ net["W2"] + net["b2"], 0)
    return h @ net["W3"] + net["b3"]


def projector(width, where):
    """The orthogonal projector onto a variable's place in a layer: a list of unit indices, or
    {"basis": [vector, ...]} (vectors of the layer's width; any spanning set, orthonormalised here).
    Raises ValueError with a message for the agent."""
    if isinstance(where, dict):
        if set(where) != {"basis"} or not isinstance(where["basis"], list) or not where["basis"]:
            raise ValueError('a subspace is {"basis": [[...], ...]} with at least one vector')
        B = np.array(where["basis"], dtype=np.float64)
        if B.ndim != 2 or B.shape[1] != width or not np.all(np.isfinite(B)):
            raise ValueError(f"basis vectors must be finite and have the layer's width {width}")
        U, sv, _ = np.linalg.svd(B.T, full_matrices=False)
        r = int((sv > 1e-8 * max(sv.max(), 1e-300)).sum())
        if r == 0:
            raise ValueError("the basis spans nothing")
        U = U[:, :r]
        return U @ U.T
    if not isinstance(where, (list, tuple)) or not where:
        raise ValueError("units must be a non-empty list")
    if not all(isinstance(u, int) and not isinstance(u, bool) and 0 <= u < width for u in where) \
            or len(set(where)) != len(where):
        raise ValueError(f"units must be distinct integers in 0..{width - 1}")
    P = np.zeros((width, width))
    P[list(where), list(where)] = 1.0
    return P


def patched_logits(net, Xbase, layer, P, Xsrc):
    """Logits on Xbase with the component of `layer` in the subspace of projector P (or a list of units)
    replaced by its value on the matching rows of Xsrc: h = h_base + (h_source - h_base) P."""
    hb = acts(net, Xbase)[layer]
    if not isinstance(P, np.ndarray):
        P = projector(hb.shape[1], list(P)) if len(P) else np.zeros((hb.shape[1], hb.shape[1]))
    hs = acts(net, Xsrc)[layer]
    return run_from(net, layer, hb + (hs - hb) @ P)


def predict(logits):
    return np.argmax(logits, axis=1)


# ----------------------------------------------------------------------------- submitted programs

SAFE_BUILTINS = ["abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float",
                 "frozenset", "hex", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "ord",
                 "pow", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "True",
                 "False", "None", "ValueError", "KeyError", "IndexError", "Exception"]
BANNED_NAMES = {"open", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                "input", "breakpoint", "help", "memoryview", "type", "object", "super", "classmethod",
                "staticmethod", "property", "dir", "id", "hash", "print", "exit", "quit"}
# Size rules. The program must be much smaller than the network: its literals (every character of every number and
# string, as written; the VARIABLES mapping, docstrings and variable names excepted) may not exceed one character per
# nonzero weight of the network, with a floor of MIN_LITERAL_CHARS. Copying the weights costs several characters each.
MIN_LITERAL_CHARS = 64
MAX_PROGRAM_BYTES = 3000      # after ast.unparse, without the mapping
MAX_VALUES_PER_VARIABLE = 16  # variables are discrete (int or bool) causal variables, not activations


def literal_budget(nonzero_weights):
    return max(MIN_LITERAL_CHARS, int(nonzero_weights))


def last_program_block(text):
    """Body of the last ```program block in a reply (None if there is none)."""
    body, cur = None, None
    for line in text.splitlines():
        s = line.strip()
        if cur is None:
            if s.startswith("```") and s[3:].strip().lower().startswith("program"):
                cur = []
        elif s.startswith("```") and not s[3:].strip():
            body, cur = "\n".join(cur) + "\n", None
        else:
            cur.append(line)
    if cur is not None:
        body = "\n".join(cur) + "\n"
    return body


def check_program(src, budget):
    """Static checks against a literal budget (characters). Returns (problems, stats)."""
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"], {}
    # the VARIABLES mapping (units or basis vectors) is not program payload: its literals are free, and the
    # program may not read it
    exempt, names = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VARIABLES" for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
                if isinstance(v, dict):
                    names |= {k for k in v if isinstance(k, str)}
            except Exception:  # noqa: BLE001
                problems.append(f"line {node.lineno}: VARIABLES must be a plain literal (dicts, tuples, lists, numbers)")
            exempt |= {id(n) for n in ast.walk(node.value)}
            exempt |= {id(t) for t in node.targets}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "VARIABLES" and id(node) not in exempt:
            problems.append(f"line {node.lineno}: the program may not read VARIABLES")
    # docstrings are free text, not payload
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                exempt.add(id(first.value))
    nums, chars = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append(f"line {node.lineno}: imports are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            problems.append(f"line {node.lineno}: attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            problems.append(f"line {node.lineno}: name {node.id!r} is not allowed")
        elif isinstance(node, ast.Constant):
            if id(node) in exempt or isinstance(node.value, bool) or node.value is None:
                continue
            if isinstance(node.value, (int, float, complex)):
                nums += 1
                seg = ast.get_source_segment(src, node)
                chars += len(seg) if seg else len(repr(node.value))
            elif isinstance(node.value, (str, bytes)) and node.value not in names:
                chars += len(node.value)
    body = ast.Module(body=[n for n in tree.body if not (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "VARIABLES" for t in n.targets))], type_ignores=[])
    nbytes = len(ast.unparse(body).encode())
    if chars > budget:
        problems.append(f"{chars} characters of literals (limit {budget}, one per nonzero weight of the network): "
                        "the program must be much smaller than the network, not a copy of it")
    if nbytes > MAX_PROGRAM_BYTES:
        problems.append(f"the program is {nbytes} bytes without its VARIABLES (limit {MAX_PROGRAM_BYTES})")
    stats = {"bytes_raw": len(src.encode()), "bytes": len(ast.unparse(tree).encode()), "bytes_without_mapping": nbytes,
             "numeric_literals": nums, "literal_chars": chars, "literal_budget": budget}
    return problems, stats


RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
src = open(sys.argv[1]).read()
calls = json.load(open(sys.argv[2]))
import builtins
safe = {k: getattr(builtins, k) for k in json.loads(sys.argv[3])}
g = {"__builtins__": safe, "__name__": "program"}
try:
    exec(compile(src, "program", "exec"), g)
except Exception as e:
    print(json.dumps({"ok": False, "error": "loading the program: %s: %s" % (type(e).__name__, e)})); sys.exit()
if not callable(g.get("program")):
    print(json.dumps({"ok": False, "error": "the program defines no function program(x, fix)"})); sys.exit()
V = g.get("VARIABLES", {})
out = []
for x, fix in calls:
    try:
        r = g["program"](tuple(x), dict(fix))
        cls, vals = r
        cls = int(cls)
        vals = {str(k): v for k, v in dict(vals).items()}
        json.dumps(vals)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "program(%s, %s) failed: %s: %s" % (x, fix, type(e).__name__, e)})); sys.exit()
    out.append([cls, vals])
try:
    json.dumps(V)
except Exception:
    V = "unserialisable"
print(json.dumps({"ok": True, "out": out, "VARIABLES": V}))
'''


def run_program(src, calls, workdir, timeout_s=600):
    """Run program(x, fix) for every (x, fix) in calls in a separate, restricted Python process."""
    os.makedirs(workdir, exist_ok=True)
    p_src, p_calls, p_run = (os.path.join(workdir, f) for f in ("program.py", "calls.json", "runner.py"))
    open(p_src, "w").write(src)
    json.dump([[list(x), f] for x, f in calls], open(p_calls, "w"))
    open(p_run, "w").write(RUNNER)
    try:
        r = subprocess.run([sys.executable, "-I", "-S", p_run, p_src, p_calls, json.dumps(SAFE_BUILTINS)],
                           capture_output=True, text=True, timeout=timeout_s, cwd=workdir)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the program took longer than {timeout_s} s"}
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        return {"ok": False, "error": "the program crashed: " + r.stderr[-2000:]}
    return json.loads(line)
