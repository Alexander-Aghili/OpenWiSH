#!/usr/bin/env python3
"""Build the decompile environments: secret programs over 10 input bits, small ReLU MLPs trained on half of the
inputs with a few labels flipped (harness side; training is allowed here, never for the agent), task specs and prompts.

usage: assets/venvs/sandbox/bin/python -I scripts/prepare_decompile.py [--root .] [--only d1-a,...]

Deterministic: the same seeds give the same programs and networks. Writes
  assets/decompile/<id>/net/{W1,b1,W2,b2,W3,b3}.npy   the network (what the agent reads)
  assets/decompile/<id>/secret.json                   the program, split and quirks (hidden; scoring only)
  tasks/decompile/<id>-{read,probe}.toml, tasks/decompile/prompts/<id>-{read,probe}.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "adapters"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

import decompile_lib as L  # noqa: E402

N_BITS = 10
# id: (seed, number of intermediate variables, allowed ops, classes, h1, h2)
LEVELS = {
    "d1-a": (101, 2, ("and", "or", "parity2"), 2, 12, 6),
    "d1-b": (102, 2, ("and", "or", "parity2"), 2, 12, 6),
    "d2-a": (201, 2, ("and", "or", "parity2", "parity3", "count", "gt"), 3, 16, 8),
    "d2-b": (202, 2, ("and", "or", "parity2", "parity3", "count", "gt"), 3, 16, 8),
    "d3-a": (301, 3, ("and", "or", "parity2", "parity3", "count", "gt"), 3, 24, 12),
    "d3-b": (302, 3, ("and", "or", "parity2", "parity3", "count", "gt"), 3, 24, 12),
}
TRAIN_FRAC = 0.5
FLIPS = 4  # training labels changed to another class: the network memorises them as exceptions
PRUNE = 0.02


def draw_var(rng, ops, used):
    op = ops[rng.integers(len(ops))]
    free = [i for i in range(N_BITS) if i not in used] or list(range(N_BITS))

    def pick(k):
        k = min(k, len(free))
        return [int(b) for b in rng.choice(free, size=k, replace=False)]

    if op.startswith("parity"):
        return {"op": "parity", "bits": sorted(pick(int(op[-1])))}
    if op in ("and", "or"):
        return {"op": op, "lits": [[b, int(rng.integers(2))] for b in sorted(pick(int(rng.integers(2, 4))))]}
    if op == "count":
        return {"op": "count", "bits": sorted(pick(int(rng.integers(3, 5))))}
    b = pick(4)
    return {"op": "gt", "a": b[:2], "b": b[2:]}


def draw_secret(rng, nvars, ops, classes):
    for _ in range(1000):
        used, vs = set(), []
        for _ in range(nvars):
            v = draw_var(rng, ops, used)
            vs.append(v)
            used |= set(v.get("bits", [])) | {l[0] for l in v.get("lits", [])} | set(v.get("a", [])) | set(v.get("b", []))
        ranges = [L.var_range(v) for v in vs]
        keys = list(np.ndindex(*ranges))
        table = {",".join(map(str, k)): int(rng.integers(classes)) for k in keys}
        secret = {"n_bits": N_BITS, "classes": classes, "vars": vs, "table": table}
        X = L.all_inputs(N_BITS)
        y = np.array([L.eval_secret(secret, x)[0] for x in X])
        if min(np.bincount(y, minlength=classes)) < 0.1 * len(X):
            continue
        if not all(var_matters(table, ranges, i) for i in range(nvars)):
            continue
        return secret
    raise RuntimeError("no balanced program found")


def var_matters(table, ranges, i):
    for k in np.ndindex(*ranges):
        for alt in range(ranges[i]):
            k2 = list(k)
            k2[i] = alt
            if table[",".join(map(str, k))] != table[",".join(map(str, k2))]:
                return True
    return False


def train(rng, X, y, classes, h1, h2, steps=6000, lr=0.01, l1=3e-4):
    """Full-batch Adam on cross-entropy + L1 (harness side)."""
    shapes = {"W1": (N_BITS, h1), "b1": (h1,), "W2": (h1, h2), "b2": (h2,), "W3": (h2, classes), "b3": (classes,)}
    P = {k: (rng.normal(0, np.sqrt(2 / s[0]), s) if len(s) == 2 else np.full(s, 0.05)) for k, s in shapes.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    Y = np.eye(classes)[y]
    for t in range(1, steps + 1):
        a1 = X @ P["W1"] + P["b1"]
        r1 = np.maximum(a1, 0)
        a2 = r1 @ P["W2"] + P["b2"]
        r2 = np.maximum(a2, 0)
        z = r2 @ P["W3"] + P["b3"]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(X)
        G = {"W3": r2.T @ g, "b3": g.sum(0)}
        d2 = (g @ P["W3"].T) * (a2 > 0)
        G["W2"], G["b2"] = r1.T @ d2, d2.sum(0)
        d1 = (d2 @ P["W2"].T) * (a1 > 0)
        G["W1"], G["b1"] = X.T @ d1, d1.sum(0)
        for k in P:
            if k.startswith("W"):
                G[k] = G[k] + l1 * np.sign(P[k])
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
    for k in P:
        if k.startswith("W"):
            P[k][np.abs(P[k]) < PRUNE] = 0.0
        P[k] = P[k].astype(np.float32)
    return P


def build(nid, cfg):
    seed, nvars, ops, classes, h1, h2 = cfg
    rng = np.random.default_rng(seed)
    secret = draw_secret(rng, nvars, ops, classes)
    Xl = L.all_inputs(N_BITS)
    X = np.array(Xl, dtype=np.float64)
    y = np.array([L.eval_secret(secret, x)[0] for x in Xl])
    for attempt in range(40):
        perm = rng.permutation(len(X))
        tr = np.sort(perm[: int(TRAIN_FRAC * len(X))])
        yt = y[tr].copy()
        fl = rng.choice(len(tr), FLIPS, replace=False)
        yt[fl] = (yt[fl] + 1 + rng.integers(classes - 1, size=FLIPS)) % classes
        P = train(rng, X[tr], yt, classes, h1, h2)
        net = {k: v.astype(np.float64) for k, v in P.items()}
        pred = L.predict(L.acts(net, X)["logits"])
        train_acc = float((pred[tr] == yt).mean())
        quirks = [int(i) for i in np.nonzero(pred != y)[0]]
        print(f"  {nid} attempt {attempt}: train acc {train_acc:.4f}, quirks {len(quirks)}", flush=True)
        if train_acc == 1.0 and 4 <= len(quirks) <= 60:
            break
    else:
        raise RuntimeError(f"{nid}: no network met the criteria")
    secret.update({"id": nid, "seed": seed, "train_idx": [int(i) for i in tr], "quirks": quirks,
                   "flipped": [{"input": int(tr[i]), "label": int(yt[i])} for i in fl],
                   "net_classes": [int(c) for c in pred], "shapes": {k: list(v.shape) for k, v in P.items()},
                   "nonzero_weights": int(sum(int((P[k] != 0).sum()) for k in P))})
    return P, secret


PROMPT = """TASK. Decompile a neural network into a program. The network below was trained on some unknown rule over
10 input bits, using only part of the 1,024 possible inputs, so on some inputs it may not follow any clean rule.
Write a short, deterministic Python program that is the same system as the network: it must give the network's
answer on every one of the 1,024 inputs, and its intermediate variables must correspond to the network's hidden
units, as tested by interventions (below). Reproduce the network, not the rule you think it was trained on.

THE NETWORK. Input x: 10 numbers, each 0 or 1 (x[0] .. x[9]). Then
  h1 = relu(x @ W1 + b1)        W1: {W1}, b1: {b1}
  h2 = relu(h1 @ W2 + b2)       W2: {W2}, b2: {b2}
  logits = h2 @ W3 + b3         W3: {W3}, b3: {b3}
and the answer is the class with the largest logit (0 .. {cmax}). {nz} weights are nonzero. Units are numbered
from 0 within each layer.

{tools}
YOUR ANSWER. {answer_when} Include exactly one block that starts with
```program, containing pure Python (no imports, no file or system access, deterministic) that defines the two
names below. It runs with these builtins only: {builtins}.

  VARIABLES = {{"name": ("h1" or "h2", PLACE), ...}}
      # PLACE is a list of unit indices, e.g. [3, 7], or a subspace of the layer, {{"basis": [[...], ...]}}: one or
      # more vectors of the layer's width (any spanning set; it is orthonormalised). Places of variables in the same
      # layer must not overlap (no shared units; subspaces orthogonal). VARIABLES must be a plain literal; its
      # numbers do not count toward the literal limit, and the program may not read it.
  def program(x, fix={{}}):
      # x: a tuple of 10 ints (0 or 1). fix: a dict that may set some of your variables.
      # Every variable in VARIABLES must be computed here and must be replaced by fix[name] when name is in fix,
      # and everything downstream must use the replaced value.
      ...
      return cls, {{"name": value, ...}}   # cls: the class (int); value: an int or bool
  Each variable is a discrete causal variable: an int or bool taking at most {maxvals} different values over the
  1,024 inputs (not an activation level).

For example:  a = fix["a"] if "a" in fix else (x[0] ^ x[4])

HOW IT IS SCORED (all on the full set of inputs; the network is the one above).
- behaviour: the fraction of the 1,024 inputs on which program(x)'s class equals the network's.
- interventions: for each variable v placed in layer L, for many pairs of inputs (base, source): the network is
  run on base with v's place in L replaced by its value on source (h = h_base + (h_source - h_base) P, where P
  projects onto the units or subspace), and program(base, fix={{v: program(source)'s value of v}}) is run.
  Agreement of the two classes is v's intervention accuracy (IIA).
- completeness: in each layer that has variables, the part of the layer outside your variables' places (the
  remaining units, or the orthogonal complement) is replaced by its value on another input on which your variables
  in that layer have the same values; the network's answer must not change.
- Not counted toward the mechanism score (but reported): variables whose intervention changes neither answer, and
  variables that alone fix the answer (for example "all of h2 = the class").
- mechanism = mean IIA over counted variables x mean completeness; score = behaviour x mechanism.
- Size: the program must be much smaller than the network. All its literals together (every character of every
  number and string as written, e.g. 0.25 is 4 characters; VARIABLES, docstrings and your variables' names are
  free) may have at most {budget} characters (one per nonzero weight of this network), and the program at most
  {maxbytes} bytes. An answer over either limit, or one that copies the network's weights, scores 0.
{weights_text}"""

TOOLS_NATIVE = """TOOLS. None: no code, and the network is never run. Every nonzero weight is listed at the end of this message
(all other entries are 0); work from them alone, and answer in a single reply.
"""

TOOLS_CODE = """TOOLS.
1. A Python sandbox (no internet, no GPU) with a working folder /work kept between runs. Available: numpy, scipy,
   scikit-learn, pandas, sympy, networkx. Before your code runs these are defined: NAMES and SHAPES (the tensors),
   load(name) (a float32 numpy array), W (a dict of all six tensors) and check_program(source), which checks a
   program's format and size against the limits below with the scorer's own rules (it never runs the network and
   does not grade). To run code, reply with exactly one block that starts with ```python and nothing else. Its printed output (cut at {max_out:,} characters) is sent back to
   you. You may run code up to {runs} times. The weights are read-only for this task.
{probe}
RULES. Not allowed: (a) running the network outside the declared tools: do not compute its output or activations
on any input in the sandbox (no forward passes, whole or partial, over inputs); (b) fitting: no regression,
decision-tree or program search, or any other fitting of a program or its parameters to the network's outputs;
(c) internet access (there is none). Allowed: any analysis of the weights themselves (reading, sorting, comparing,
linear algebra on the weight matrices, reasoning about which inputs can switch a unit on), and any analysis of
the outputs of the declared tools.
"""


PRELUDE_HEAD = """# Generated by scripts/prepare_decompile.py: the six tensors (read-only) and a format-only program check.
import ast

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=160)
W = {n: load(n) for n in NAMES}
SHAPES = {n: tuple(a.shape) for n, a in W.items()}
"""

PRELUDE_TAIL = """

def check_program(source):
    \"\"\"Format and size only, with the scorer's rules: literal characters against this network's budget, program
    bytes, forbidden constructs. It never runs the network and does not grade.\"\"\"
    budget = literal_budget(sum(int((a != 0).sum()) for a in W.values()))
    problems, stats = _check(source, budget)
    print("OK" if not problems else "PROBLEMS:\\n  " + "\\n  ".join(problems))
    print(stats)
    return not problems
"""


def write_prelude(root):
    """The sandbox prelude, with the scorer's static checks copied in (the sandbox cannot import harness code)."""
    import inspect
    parts = [PRELUDE_HEAD,
             f"BANNED_NAMES = {sorted(L.BANNED_NAMES)!r}",
             f"MIN_LITERAL_CHARS = {L.MIN_LITERAL_CHARS}",
             f"MAX_PROGRAM_BYTES = {L.MAX_PROGRAM_BYTES}",
             "", inspect.getsource(L.literal_budget),
             inspect.getsource(L.check_program).replace("def check_program(src, budget):", "def _check(src, budget):"),
             PRELUDE_TAIL]
    open(os.path.join(root, "tasks", "decompile", "prelude.py"), "w").write("\n".join(parts))


def weights_text(net_dir):
    """Every nonzero weight, unit by unit: a unit's bias and its incoming weights."""
    W = {k: np.load(os.path.join(net_dir, k + ".npy")) for k in L.TENSORS}
    out = ["", "THE WEIGHTS (every nonzero entry; a unit's bias, then its incoming weights from the previous layer)."]
    for layer, Wm, b, src in (("h1", W["W1"], W["b1"], "x"), ("h2", W["W2"], W["b2"], "h1"),
                              ("logit", W["W3"], W["b3"], "h2")):
        for j in range(Wm.shape[1]):
            ins = ", ".join(f"{src}[{i}] {Wm[i, j]:+.4f}" for i in range(Wm.shape[0]) if Wm[i, j] != 0)
            out.append(f"{layer}[{j}]: bias {b[j]:+.4f}" + (f"; {ins}" if ins else "; no incoming weights"))
    return "\n".join(out) + "\n"

PROBE = """2. A forward tool with a budget of {passes} passes: reply with exactly one block that starts with ```forward and
   contains JSON {{"inputs": ["0110010011", ...], "patch": [{{"layer": "h1", "units": [3, 7], "source": "1111100000"}}]}}
   (at most {per} inputs, each a string of 10 bits x[0]..x[9]; "patch" is optional: for every input, the listed
   units of that layer, or its component in a subspace given as "basis": [[...], ...] instead of "units", are
   replaced by their values on the source input before the pass continues; patches apply in order). You get the
   classes in reply and, in /work/fwd_<k>/acts.json, every input's h1, h2 and logits. A failed request is not
   counted.
"""


TIERS = ("native", "read", "probe")
ACCESS_TEXT = {"native": "all nonzero weights in the prompt, no tools, one reply",
               "read": "Python over the weights, network never run",
               "probe": "Python over the weights and {passes} forward passes of up to {per} inputs"}


def write_task(root, nid, secret, tier, passes=3, per=16, runs=60, max_out=20000):
    tdir = os.path.join(root, "tasks", "decompile")
    os.makedirs(os.path.join(tdir, "prompts"), exist_ok=True)
    s = secret["shapes"]
    fmt = lambda k: " x ".join(map(str, s[k]))  # noqa: E731
    prompt = PROMPT.format(W1=fmt("W1"), b1=fmt("b1"), W2=fmt("W2"), b2=fmt("b2"), W3=fmt("W3"), b3=fmt("b3"),
                           cmax=secret["classes"] - 1, nz=secret["nonzero_weights"],
                           tools=TOOLS_NATIVE if tier == "native" else TOOLS_CODE.format(
                               runs=runs, max_out=max_out, probe=PROBE.format(passes=passes, per=per) if tier == "probe" else ""),
                           answer_when="Reply once." if tier == "native" else
                           f"When you are done, reply WITHOUT a python{' or forward' if tier == 'probe' else ''} block.",
                           weights_text=weights_text(os.path.join(root, "assets", "decompile", nid, "net")) if tier == "native" else "",
                           budget=L.literal_budget(secret["nonzero_weights"]), maxbytes=L.MAX_PROGRAM_BYTES,
                           maxvals=L.MAX_VALUES_PER_VARIABLE,
                           builtins=", ".join(b for b in L.SAFE_BUILTINS if b not in ("True", "False", "None")))
    tid = f"{nid}-{tier}"
    open(os.path.join(tdir, "prompts", tid + ".txt"), "w").write(prompt)
    python = "" if tier == "native" else f"""
[python]
runs = {runs}
timeout_s = 300
mem_gb = 4
max_out = {max_out}
prelude = "prelude.py"
"""
    fwd = ""
    if tier == "probe":
        fwd = f"""
[forward.forward]
command = ["{{root}}/assets/venvs/sandbox/bin/python", "-I", "{{root}}/python/adapters/decompile_probe.py", "--max-inputs", "{per}"]
passes = {passes}
out_dir = "fwd_{{k}}"
max_out = {max_out}
timeout_s = 120
"""
    spec = f"""# Generated by scripts/prepare_decompile.py.
id = "decompile-{tid}"
description = "Decompile: rewrite a {s['W1'][0]}-{s['W1'][1]}-{s['W2'][1]}-{s['W3'][1]} ReLU MLP trained on a secret 10-bit rule as a program whose variables map to its units; {ACCESS_TEXT[tier].format(passes=passes, per=per)}."
prompt = "prompts/{tid}.txt"
hidden = ["../../assets/decompile/{nid}/secret.json", "../../assets/decompile/{nid}/reference.json"]

[network]
base = "../../assets/decompile/{nid}/net"

{python}{fwd}
[edits.policy]
forbidden = ["*"]

[finish]
max_turns = {1 if tier == "native" else 200}

[editor]
effort = "high"

[score]
command = ["{{root}}/assets/venvs/sandbox/bin/python", "-I", "{{root}}/python/adapters/decompile_score.py", "--net", "{{root}}/assets/decompile/{nid}/net", "--secret", "{{root}}/assets/decompile/{nid}/secret.json"]

[access]
weights = "{'prompt' if tier == 'native' else 'code'}"
"""
    open(os.path.join(tdir, tid + ".toml"), "w").write(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--only", default="")
    ap.add_argument("--specs-only", action="store_true", help="rewrite task specs and prompts from existing assets")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    only = [s for s in a.only.split(",") if s]
    write_prelude(root)
    for nid, cfg in LEVELS.items():
        if only and nid not in only:
            continue
        d = os.path.join(root, "assets", "decompile", nid)
        if a.specs_only:
            secret = json.load(open(os.path.join(d, "secret.json")))
            for tier in TIERS:
                write_task(root, nid, secret, tier)
            continue
        P, secret = build(nid, cfg)
        os.makedirs(os.path.join(d, "net"), exist_ok=True)
        for k, v in P.items():
            np.save(os.path.join(d, "net", k + ".npy"), v)
        json.dump(secret, open(os.path.join(d, "secret.json"), "w"), indent=1)
        for tier in TIERS:
            write_task(root, nid, secret, tier)
        print(f"{nid}: {len(secret['quirks'])} quirks, {secret['nonzero_weights']} nonzero weights, vars "
              + "; ".join(json.dumps(v) for v in secret["vars"]))


if __name__ == "__main__":
    main()
