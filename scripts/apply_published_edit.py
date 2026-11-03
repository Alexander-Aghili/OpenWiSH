"""Rebuild a published Experiment 8 weight edit (edit/manifest.json + edit/edit.npz) on top of our text-only base
checkpoint, checking the base and result hashes the publication records. Writes one float32 .npy per edited tensor
into <out dir> (the exact bfloat16 values, as evaluated).

usage: apply_published_edit.py <base model.safetensors> <published edit dir> <out dir>"""
import hashlib
import json
import os
import struct
import sys

import numpy as np


def read_bf16_tensors(path, names):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for k in names:
        v = header[k]
        assert v["dtype"] == "BF16", (k, v["dtype"])
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(bytes(raw[a:b]), dtype=np.uint16).reshape(v["shape"])
    return out


def bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bits(x):
    """Round-to-nearest-even float32 -> bfloat16 bits."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main(base, edit_dir, out):
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"))
    names = [t["name"] for t in man["tensors"]]
    base_t = read_bf16_tensors(base, names)
    os.makedirs(out, exist_ok=True)
    for i, t in enumerate(man["tensors"]):
        b = base_t[t["name"]]
        if hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() != t["base_sha256"]:
            sys.exit(f"{t['name']}: our base tensor differs from the one the edit was made on")
        if t["form"] == "lowrank":
            w = bits_to_f32(b).astype(np.float64)
            a, bb = z[f"t{i}_a"], z[f"t{i}_b"]
            w = w + a @ bb
            e = f32_to_bits(w.astype(np.float32)).reshape(-1)
        else:
            e = b.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(b.shape)
        if hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() != t["edited_sha256"]:
            sys.exit(f"{t['name']}: the rebuilt tensor does not match the published hash")
        np.save(os.path.join(out, t["name"] + ".npy"), bits_to_f32(e))
        print(f"{t['name']}: rebuilt ({t['form']}, {t['changed_entries']:,} entries changed), both hashes match")
    print(f"{len(names)} tensors -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
