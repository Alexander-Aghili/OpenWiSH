"""The decompile environments' forward tool: run the network on chosen inputs, optionally with unit patches.

forward pass (harness): OPENWISH_REQ (the ```forward block: JSON), OPENWISH_NET, OPENWISH_OUT.
Request: {"inputs": ["0110010011", ...], "patch": [{"layer": "h1", "units": [3, 7], "source": "1111100000"}]}
(a patch may give "basis": [[...], ...] instead of "units": the layer's component in that subspace is replaced)
Writes OPENWISH_OUT/acts.json (every input's h1, h2, logits, class) and prints a JSON summary with the classes.
usage: decompile_probe.py [--max-inputs 64]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import decompile_lib as L  # noqa: E402


def bits(s, n):
    if isinstance(s, list):
        s = "".join(str(b) for b in s)
    s = str(s).strip()
    if len(s) != n or set(s) - {"0", "1"}:
        raise ValueError(f"input {s!r} is not a string of {n} bits")
    return [int(c) for c in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-inputs", type=int, default=64)
    a = ap.parse_args()
    try:
        req = json.loads(open(os.environ["OPENWISH_REQ"]).read())
        net = L.load_net(os.environ["OPENWISH_NET"])
        n = net["W1"].shape[0]
        widths = {"h1": net["W1"].shape[1], "h2": net["W2"].shape[1]}
        if not isinstance(req, dict) or not isinstance(req.get("inputs"), list) or not req["inputs"]:
            raise ValueError('the request must be JSON {"inputs": [...], "patch": [...] (optional)}')
        if set(req) - {"inputs", "patch"}:
            raise ValueError(f"unknown keys {sorted(set(req) - {'inputs', 'patch'})}")
        if len(req["inputs"]) > a.max_inputs:
            raise ValueError(f"{len(req['inputs'])} inputs; at most {a.max_inputs} per pass")
        X = np.array([bits(s, n) for s in req["inputs"]], dtype=np.float64)
        patches = req.get("patch") or []
        if not isinstance(patches, list):
            raise ValueError('"patch" must be a list')
        for p in patches:
            if not isinstance(p, dict) or p.get("layer") not in L.LAYERS or \
                    set(p) not in ({"layer", "units", "source"}, {"layer", "basis", "source"}):
                raise ValueError('each patch is {"layer": "h1" or "h2", "units": [...] or "basis": [[...], ...], '
                                 '"source": "<bits>"}')
            p["_P"] = L.projector(widths[p["layer"]], p["units"] if "units" in p else {"basis": p["basis"]})
            bits(p["source"], n)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}))
        return

    def layer_vals(name, h):
        out = h.copy()
        for p in patches:
            if p["layer"] == name:
                src = L.acts(net, np.array([bits(p["source"], n)], dtype=np.float64))[name][0]
                out = out + (src[None, :] - out) @ p["_P"]
        return out

    X1 = np.maximum(X @ net["W1"] + net["b1"], 0)
    h1 = layer_vals("h1", X1)
    h2 = layer_vals("h2", np.maximum(h1 @ net["W2"] + net["b2"], 0))
    logits = h2 @ net["W3"] + net["b3"]
    cls = [int(c) for c in L.predict(logits)]
    rows = [{"input": "".join(str(int(b)) for b in x), "h1": np.round(h1[i], 6).tolist(),
             "h2": np.round(h2[i], 6).tolist(), "logits": np.round(logits[i], 6).tolist(), "class": cls[i]}
            for i, x in enumerate(X)]
    out = os.environ["OPENWISH_OUT"]
    os.makedirs(out, exist_ok=True)
    patches = [{k: v for k, v in p.items() if k != "_P"} for p in patches]
    json.dump({"patch": patches, "rows": rows}, open(os.path.join(out, "acts.json"), "w"), indent=1)
    print(json.dumps({"ok": True, "n": len(rows), "patched": bool(patches),
                      "classes": {r["input"]: r["class"] for r in rows}, "file": "acts.json"}))


if __name__ == "__main__":
    main()
