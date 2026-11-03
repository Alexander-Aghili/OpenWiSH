"""Non-LLM baselines for the bundled read environments, for comparison with agent runs.

exp3 (copying heads from the middle blocks only): K-composition (Elhage et al. 2021, "A Mathematical Framework for
Transformer Circuits"). For a later head h2 and an earlier head h1:
    K(h1 -> h2) = ||QK_h2 @ OV_h1||_F / (||QK_h2||_F ||OV_h1||_F)
with LayerNorm gains folded in. A head is labelled "copy" when its strongest K-composition is more than 2 standard
deviations above the mean over all candidate heads (a fixed rule, chosen before looking at the ground truth).
Truth: the published induction scores (> 0.2).

usage: python3 scripts/baselines.py   (needs assets/exp3 from scripts/prepare_tasks.py). Prints JSON."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS, D = 6, 384
HD = D // HEADS


def head_mats(z, l, h):
    g = z[f"ln1_w.{l}"]
    s = slice(h * HD, (h + 1) * HD)
    wq, wk, wv, wo = z[f"W_q.{l}"][s], z[f"W_k.{l}"][s], z[f"W_v.{l}"][s], z[f"W_o.{l}"][:, s]
    qk = (wq * g).T @ (wk * g)          # (in, in): x_q^T QK x_k
    ov = wo @ (wv * g)                   # (out, in)
    return qk, ov


def exp3():
    heads = json.load(open(f"{ROOT}/assets/exp3/heads.json"))
    out = {}
    for corpus in ("python", "romaji", "english"):
        z = np.load(f"{ROOT}/assets/exp3/networks/{corpus}/blocks.npz")
        mats = {(l, h): head_mats(z, l, h) for l in range(4) for h in range(HEADS)}
        score = {}
        for (l2, h2), (qk, _) in mats.items():
            if l2 == 0:
                continue
            best = 0.0
            for (l1, h1), (_, ov) in mats.items():
                if l1 < l2:
                    k = np.linalg.norm(qk @ ov) / (np.linalg.norm(qk) * np.linalg.norm(ov))
                    best = max(best, float(k))
            score[f"L{l2}H{h2}"] = best
        v = np.array(list(score.values()))
        cut = v.mean() + 2 * v.std()
        named = sorted(k for k, s in score.items() if s > cut)
        truth = sorted(k for k, s in heads[f"{corpus}_L4_H6_d384_s0"].items() if s["induct"] > 0.2)
        tp = len(set(named) & set(truth))
        out[corpus] = {"named": named, "truth": truth, "precision": tp / len(named) if named else None,
                       "recall": tp / len(truth) if truth else None,
                       "top3": sorted(score, key=score.get, reverse=True)[:3]}
    return out


if __name__ == "__main__":
    print(json.dumps({"exp3_k_composition": exp3()}, indent=1))
