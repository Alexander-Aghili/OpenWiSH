"""Shared plumbing for probe and scorer workers (python/adapters/*_worker.py).

A worker is called by the harness with environment variables:
  probe  OPENWISH_REQ (request JSON), OPENWISH_NET (current network), OPENWISH_OUT (output folder)
  score  OPENWISH_FINAL_NET (the edited network), or --net
  both   OPENWISH_THREADS, OPENWISH_DEVICE ("cpu" or "cuda"), OPENWISH_REMOTE (run on a GPU host, see remote.py)
It prints one JSON line: {"ok": true, ...} or {"ok": false, "error": "..."}.

Workers run with `python -I`, which drops their own folder from sys.path: load sibling modules with load_module.
numpy is the only hard dependency here; torch loads only when a worker builds a model."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ADAPTERS = Path(__file__).resolve().parent
ROOT = ADAPTERS.parents[1]


def load_module(path, name=None):
    """Import a .py file by path (works under python -I)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name or path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_to_name(stem):
    """A network folder stores tensor name '/' as '__' (crates/openwish-tensor/src/network.rs)."""
    return stem.replace("__", "/")


def load_net(net_dir, dtype=None):
    """{tensor name: array} for a folder of .npy files. Arrays are memory-mapped and read-only, so a 1 GB
    network costs no RAM until a tensor is read; pass dtype to get converted in-memory copies."""
    out = {}
    for p in sorted(Path(net_dir).glob("*.npy")):
        a = np.load(p, mmap_mode="r")
        out[file_to_name(p.stem)] = a.astype(dtype) if dtype is not None else a
    return out


def device():
    return os.environ.get("OPENWISH_DEVICE", "cpu")


def threads():
    return int(os.environ.get("OPENWISH_THREADS", "4"))


def load_hf(model_dir, net, param_name=lambda n: n):
    """A Hugging Face causal LM from `model_dir` in float32, with the tensors of `net` copied in (one tensor
    in memory at a time), on device(), in eval mode. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads())
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in net.items():
            params[param_name(name)].copy_(torch.from_numpy(np.array(value, dtype=np.float32)))
    return model.to(device()), AutoTokenizer.from_pretrained(model_dir)


def respond(obj):
    print(json.dumps(obj), flush=True)


def fail(msg):
    respond({"ok": False, "error": str(msg)})


def read_request(resolve):
    """Parse $OPENWISH_REQ and pass it to `resolve`, which raises ValueError on a bad request.
    Returns resolve's result, or None after printing the error."""
    try:
        return resolve(json.loads(Path(os.environ["OPENWISH_REQ"]).read_text()))
    except (ValueError, json.JSONDecodeError) as e:
        fail(e)
        return None


def save_run(arrays, summary):
    """Write arrays to $OPENWISH_OUT/run.npz and print the summary with "ok" and "file"."""
    path = Path(os.environ["OPENWISH_OUT"])
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "run.npz", **arrays)
    respond({"ok": True, **summary, "file": "run.npz"})


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def cached(path, key, compute):
    """compute() stored in the JSON file at `path` under `key`; recomputed when the key changes."""
    path = Path(path)
    if path.exists():
        have = json.loads(path.read_text())
        if have.get("key") == key and "value" in have:
            return have["value"]
    value = compute()
    path.write_text(json.dumps({"key": key, "value": value}))
    return value


def main(worker, base, probe, score, add_args=None):
    """The command line every worker shares: `probe` or `score [--net DIR]`. With OPENWISH_REMOTE set,
    the call runs on the GPU host instead (remote.py). `worker` and `base` are repo-relative paths
    (this worker's file and its task's base network); add_args(parser) adds worker options, and
    score(net_dir, args) receives them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "score"])
    ap.add_argument("--net", help="network folder to score (default $OPENWISH_FINAL_NET)")
    if add_args:
        add_args(ap)
    a = ap.parse_args()
    if a.cmd == "score" and a.net:
        os.environ["OPENWISH_FINAL_NET"] = a.net
    remote = load_module(ADAPTERS / "remote.py")
    if remote.maybe_forward(worker, base, sys.argv[1:]):
        return
    if a.cmd == "probe":
        probe()
    else:
        score(os.environ["OPENWISH_FINAL_NET"], a)
