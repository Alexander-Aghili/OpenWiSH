"""Score a decompile run: behaviour on every input, interchange interventions per variable, completeness.

usage: decompile_score.py --net <net dir> --secret <secret.json> [--answer <final_answer.md>] [--program <file.py>]
       [--pairs 8] [--seed 0]
The answer defaults to $OPENWISH_FINAL_ANSWER. Prints one JSON line.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import decompile_lib as L  # noqa: E402

INERT = 0.01      # a variable whose intervention changes either answer on fewer pairs than this is inert
TRIVIAL = 0.99    # a variable that alone fixes the program's answer on at least this share of pairs is trivial


def fail(msg, **extra):
    print(json.dumps({"ok": True, "score": 0.0, "behaviour": 0.0, "mechanism": 0.0, "exact": False,
                      "invalid": msg, **extra}))
    sys.exit(0)


def check_variables(V, shapes):
    """(error, {name: (layer, projector)}): every variable maps to units or a subspace of one hidden layer, and
    the places of variables in the same layer must not overlap."""
    if not isinstance(V, dict) or not V:
        return "VARIABLES must be a non-empty dict name -> (layer, [units] or {\"basis\": [...]})", None
    proj = {}
    for name, m in V.items():
        if not (isinstance(m, (list, tuple)) and len(m) == 2 and m[0] in L.LAYERS):
            return f"VARIABLES[{name!r}] must be (\"h1\" or \"h2\", [unit indices] or {{\"basis\": [vectors]}})", None
        try:
            proj[name] = (m[0], L.projector(shapes[m[0]], m[1]))
        except ValueError as e:
            return f"VARIABLES[{name!r}]: {e}", None
    names = list(proj)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if proj[a][0] == proj[b][0] and np.abs(proj[a][1] @ proj[b][1]).max() > 1e-6:
                return f"the places of {a!r} and {b!r} in {proj[a][0]} overlap (units shared or subspaces not orthogonal)", None
    return None, proj


def key(v):
    return json.dumps(v, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--answer", default=os.environ.get("OPENWISH_FINAL_ANSWER"))
    ap.add_argument("--program", help="score this program file instead of the answer's ```program block")
    ap.add_argument("--pairs", type=int, default=8, help="sources per base input for interventions and completeness")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    net = L.load_net(a.net)
    secret = json.load(open(a.secret))
    n = secret["n_bits"]
    Xl = L.all_inputs(n)
    X = np.array(Xl, dtype=np.float64)
    base_acts = L.acts(net, X)
    net_cls = L.predict(base_acts["logits"])
    shapes = {"h1": net["W1"].shape[1], "h2": net["W2"].shape[1]}

    if a.program:
        src = open(a.program).read()
    else:
        if not a.answer or not os.path.exists(a.answer):
            fail("no final answer")
        src = L.last_program_block(open(a.answer).read())
        if src is None:
            fail("the final answer has no ```program block")
    budget = L.literal_budget(secret.get("nonzero_weights") or sum(int((net[k] != 0).sum()) for k in L.TENSORS))
    problems, size = L.check_program(src, budget)
    if problems:
        fail("; ".join(problems[:10]), size=size)

    work = tempfile.mkdtemp(prefix="decompile_")
    calls = [(x, {}) for x in Xl]
    r1 = L.run_program(src, calls, work)
    if not r1["ok"]:
        fail(r1["error"], size=size)
    r2 = L.run_program(src, calls, work)
    if not r2["ok"] or r2["out"] != r1["out"]:
        fail("the program is not deterministic (two runs over the inputs differ)", size=size)
    V = r1["VARIABLES"]
    bad, proj = check_variables(V, shapes)
    if bad:
        fail(bad, size=size)
    prog_cls = np.array([o[0] for o in r1["out"]])
    prog_vals = [o[1] for o in r1["out"]]
    missing = [v for v in V if any(v not in pv for pv in prog_vals)]
    if missing:
        fail(f"program(x) does not return a value for {missing} on every input", size=size)
    for v in V:
        seen = {key(pv[v]) for pv in prog_vals}
        if not all(isinstance(pv[v], int) for pv in prog_vals):
            fail(f"variable {v!r} must be an int or bool on every input (it is a causal variable, not an activation)",
                 size=size)
        if len(seen) > L.MAX_VALUES_PER_VARIABLE:
            fail(f"variable {v!r} takes {len(seen)} values over the inputs (limit {L.MAX_VALUES_PER_VARIABLE}): "
                 "variables are discrete causal variables, not activations", size=size)

    agree = prog_cls == net_cls
    behaviour = float(agree.mean())
    quirks = secret["quirks"]
    quirk_agree = float(agree[quirks].mean()) if quirks else None

    rng = np.random.default_rng(a.seed)
    N = len(Xl)
    bases = np.repeat(np.arange(N), a.pairs)
    srcs = rng.integers(N, size=len(bases))

    # interventions: one variable at a time
    names = list(V)
    icalls = []
    for v in names:
        icalls += [(Xl[b], {v: prog_vals[s][v]}) for b, s in zip(bases, srcs)]
    r3 = L.run_program(src, icalls, work)
    if not r3["ok"]:
        fail("during interventions: " + r3["error"], size=size, behaviour_only=behaviour)
    per = {}
    for i, v in enumerate(names):
        out = np.array([o[0] for o in r3["out"][i * len(bases):(i + 1) * len(bases)]])
        layer, P = proj[v]
        net_out = L.predict(L.patched_logits(net, X[bases], layer, P, X[srcs]))
        iia = float((out == net_out).mean())
        changes_net = float((net_out != net_cls[bases]).mean())
        changes_prog = float((out != prog_cls[bases]).mean())
        follows_source = float((out == prog_cls[srcs]).mean())
        inert = changes_net < INERT and changes_prog < INERT
        trivial = follows_source >= TRIVIAL
        per[v] = {"layer": layer, "place": V[v][1] if isinstance(V[v][1], list) else f"subspace of dim {int(round(np.trace(P)))}", "iia": iia, "changes_net": changes_net,
                  "changes_program": changes_prog, "inert": inert, "trivial": trivial,
                  "counted": not inert and not trivial}

    # completeness: resample the unmapped part of each layer that has variables (the orthogonal complement of
    # their places), from an input whose variables in that layer have the same values
    comp = {}
    for layer in L.LAYERS:
        lv = [v for v in names if proj[v][0] == layer]
        if not lv:
            continue
        C = np.eye(shapes[layer]) - sum(proj[v][1] for v in lv)
        rest = int(round(np.trace(C)))
        if rest == 0:
            comp[layer] = {"completeness": 1.0, "unmapped_dims": 0}
            continue
        groups = {}
        for i in range(N):
            groups.setdefault(key([prog_vals[i][v] for v in lv]), []).append(i)
        donors = np.array([rng.choice(groups[key([prog_vals[b][v] for v in lv])]) for b in bases])
        out = L.predict(L.patched_logits(net, X[bases], layer, C, X[donors]))
        comp[layer] = {"completeness": float((out == net_cls[bases]).mean()), "unmapped_dims": rest}

    counted = [per[v]["iia"] for v in names if per[v]["counted"]]
    iia_mean = float(np.mean(counted)) if counted else 0.0
    completeness = float(np.mean([c["completeness"] for c in comp.values()])) if comp else 0.0
    mechanism = iia_mean * completeness
    exact = bool(behaviour == 1.0 and counted and all(per[v]["iia"] == 1.0 for v in names)
                 and completeness == 1.0)

    # diagnostic: how well each declared variable predicts each secret variable (share of inputs explained by
    # the best map from the declared variable's values to the secret variable's values)
    secret_vals = [L.eval_secret(secret, x)[1] for x in Xl]
    recovery = {}
    for v in names:
        best = {}
        for j in range(len(secret["vars"])):
            tab = {}
            for i in range(N):
                tab.setdefault(key(prog_vals[i][v]), {}).setdefault(secret_vals[i][j], 0)
                tab[key(prog_vals[i][v])][secret_vals[i][j]] += 1
            best[f"s{j}"] = round(sum(max(c.values()) for c in tab.values()) / N, 4)
        recovery[v] = best

    print(json.dumps({
        "ok": True, "score": behaviour * mechanism, "behaviour": behaviour, "mechanism": mechanism, "exact": exact,
        "quirk_agreement": quirk_agree, "quirks": len(quirks), "iia_mean": iia_mean, "completeness": completeness,
        "variables": per, "layers": comp, "size": size, "secret_variable_recovery": recovery,
        "secret_vars": secret["vars"], "reference": reference(a.secret),
    }))


def reference(secret_path):
    """scripts/calibrate_decompile.py's result, if present: how localised the secret variables are."""
    p = os.path.join(os.path.dirname(secret_path), "reference.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {"secret_behaviour": r["secret_behaviour"], "units_at_most": r["k"],
            "secret_var_best_iia": [v["best_iia"] for v in r["vars"]]}


if __name__ == "__main__":
    main()
