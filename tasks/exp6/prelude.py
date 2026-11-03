# The authors' Experiment 6 sandbox helpers, on top of the harness kernel (load / save_edit go through the edit gate).
import json
import os

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=160)
DEPTH, WIDTH = 6, 512
SHAPES = {}
for _k in range(1, DEPTH + 1):
    SHAPES[f"L{_k}.W"] = (243 if _k == 1 else WIDTH, WIDTH)
    SHAPES[f"L{_k}.b"] = (WIDTH,)
SHAPES["out.W"], SHAPES["out.b"] = (WIDTH, 82), (82,)
NAMES = list(SHAPES)
PASS = 81
_SAVED = "/work/.saved.json"


def inp(row, col, plane):
    return (row * 9 + col) * 3 + plane


def mv(row, col):
    return row * 9 + col


def saved():
    return json.load(open(_SAVED)) if os.path.exists(_SAVED) else []


def save(name, a):
    assert name in SHAPES, f"unknown tensor {name}; NAMES = {NAMES}"
    a = np.asarray(a, np.float32)
    assert a.shape == tuple(SHAPES[name]), (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), name
    save_edit(name, a, reason="save()")
    json.dump(sorted(set(saved()) | {name}), open(_SAVED, "w"))
