"""Compact a finished, scored run: store every tensor file (edit blobs, edits/current, final/net, final/editset/edit.npz)
as a sparse delta from the base network, and delete the full copies. `restore` rebuilds each .npy file byte for byte
(checked against its SHA-256) and the edit set with the same arrays. A MoE run shrinks from about 7 GB to well under 1 GB.

usage: python3 scripts/compact_run.py compact RUN_DIR --base assets/moe/base_net
       python3 scripts/compact_run.py restore RUN_DIR --base assets/moe/base_net
Rescoring or replaying needs the full files: restore first."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

STORE = "compact"


def _bits(a):
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))


def _npy_parts(path):
    """Header bytes and the array of an .npy file (C order only)."""
    raw = Path(path).read_bytes()
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = read(f)
        offset = f.tell()
    if fortran:
        raise ValueError(f"{path}: Fortran order is not supported")
    arr = np.frombuffer(raw, dtype=dtype, offset=offset).reshape(shape)
    return raw[:offset], arr, hashlib.sha256(raw).hexdigest()


def _tensor_of_blobs(run):
    names = {}
    ledger = run / "edits/ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            d = json.loads(line)
            op = d.get("op") or {}
            t = op.get("tensor")
            if not t:
                continue
            if op.get("blob"):
                names[op["blob"]] = t
            for k in ("sha_before", "sha_after"):
                if d.get(k):
                    names[d[k] + ".npy"] = t
    return names


def _delta(arr, base):
    if base.shape != arr.shape or base.dtype != arr.dtype:
        raise ValueError("shape or dtype differs from base")
    idx = np.flatnonzero(_bits(arr).ravel() != _bits(base).ravel())
    return idx.astype(np.int64), arr.ravel()[idx]


def _apply(base, idx, val):
    out = np.array(base, copy=True)
    flat = out.reshape(-1)
    _bits(flat)[idx] = _bits(val)
    return out


def compact(run, base_dir):
    run, base_dir = Path(run), Path(base_dir)
    if (run / "compact.json").exists():
        raise SystemExit(f"{run}: already compact")
    store = run / STORE
    store.mkdir(exist_ok=True)
    blob_tensor = _tensor_of_blobs(run)
    entries, cache = [], {}

    def base_of(t):
        if t not in cache:
            cache.clear()
            cache[t] = np.load(base_dir / f"{t}.npy")
        return cache[t]

    files = []
    for sub in ("edits/blobs", "edits/current", "final/net"):
        for p in sorted((run / sub).glob("*.npy")) if (run / sub).is_dir() else []:
            t = blob_tensor.get(p.name) if sub == "edits/blobs" else p.name[:-4]
            if t is None:
                raise SystemExit(f"{p}: no tensor name in the ledger")
            files.append((p, t))
    for k, (p, t) in enumerate(files):
        header, arr, digest = _npy_parts(p)
        idx, val = _delta(arr, base_of(t))
        key = f"{k:05d}.npz"
        np.savez_compressed(store / key, header=np.frombuffer(header, dtype=np.uint8), idx=idx, val=val)
        back = _apply(base_of(t), idx, val)
        if hashlib.sha256(header + back.tobytes()).hexdigest() != digest:
            raise SystemExit(f"{p}: delta does not rebuild the file")
        entries.append({"file": str(p.relative_to(run)), "tensor": t, "sha256": digest, "delta": key, "changed": int(idx.size)})
    editset = run / "final/editset/edit.npz"
    members = []
    if editset.exists():
        with zipfile.ZipFile(editset) as z:
            names = [n[:-4] for n in z.namelist()]
        final = {e["tensor"]: e for e in entries if e["file"].startswith("final/net/")}
        z = np.load(editset)
        for n in names:
            e = final.get(n)
            if e is None or hashlib.sha256(z[n].tobytes()).hexdigest() != hashlib.sha256(_npy_parts(run / e["file"])[1].tobytes()).hexdigest():
                raise SystemExit(f"{editset}: {n} does not match final/net")
            members.append(n)
    (run / "compact.json").write_text(json.dumps({"base": str(base_dir), "files": entries, "editset": members}, indent=1))
    for e in entries:
        (run / e["file"]).unlink()
    if editset.exists():
        editset.unlink()
    return entries


def restore(run, base_dir):
    run, base_dir = Path(run), Path(base_dir)
    meta = json.loads((run / "compact.json").read_text())
    rebuilt = {}
    for e in meta["files"]:
        z = np.load(run / STORE / e["delta"])
        arr = _apply(np.load(base_dir / f"{e['tensor']}.npy"), z["idx"], z["val"])
        data = z["header"].tobytes() + arr.tobytes()
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            raise SystemExit(f"{e['file']}: restored bytes do not match")
        out = run / e["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        if e["file"].startswith("final/net/"):
            rebuilt[e["tensor"]] = arr
    if meta["editset"]:
        np.savez(run / "final/editset/edit.npz", **{n: rebuilt[n] for n in meta["editset"]})
    for e in meta["files"]:
        (run / STORE / e["delta"]).unlink()
    (run / STORE).rmdir()
    (run / "compact.json").unlink()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["compact", "restore"])
    ap.add_argument("run")
    ap.add_argument("--base", required=True, help="the task's base network folder")
    a = ap.parse_args(argv)
    run = Path(a.run)
    if a.action == "compact":
        if not (run / "eval/score.json").exists():
            raise SystemExit(f"{run}: no eval/score.json; compact only finished, scored runs")
        e = compact(run, a.base)
        print(json.dumps({"run": str(run), "files": len(e), "changed": sum(x["changed"] for x in e)}))
    else:
        restore(run, a.base)
        print(json.dumps({"run": str(run), "restored": True}))


if __name__ == "__main__":
    main(sys.argv[1:])
