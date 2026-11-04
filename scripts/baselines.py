"""Non-LLM baselines for the bundled read environments, for comparison with agent runs.

exp3 (copying heads from the middle blocks only): K-composition (Elhage et al. 2021, "A Mathematical Framework for
Transformer Circuits"). For a later head h2 and an earlier head h1:
    K(h1 -> h2) = ||QK_h2 @ OV_h1||_F / (||QK_h2||_F ||OV_h1||_F)
with LayerNorm gains folded in. A head is labelled "copy" when its strongest K-composition is more than 2 standard
deviations above the mean over all candidate heads (a fixed rule, chosen before looking at the ground truth).
Truth: the published induction scores (> 0.2).

exp3, second baseline: the OV copying score (Elhage et al. 2021). Without the embeddings (the environment gives the
middle blocks only) the proxy is the eigenvalue sign of the head's own OV matrix, sum(eig) / sum(|eig|): a head whose
OV maps residual directions onto themselves copies. Same fixed rule: more than 2 standard deviations above the mean.

exp1 (which game, from the weights alone): Goedecke's heatmap (seangoedecke.com, 2026-11-03): the norm of each grid
cell's first-layer weights for one input channel. A heatmap is not an answer, so the reader is a rule fixed from the
games' rules before running it, never tuned on the four labels:
  agent mass  = per playable cell, the agent-channel weight norm minus the median (cells the agent never visits keep
                their initial weights)
  Pac-Man     if no single row or column holds at least half of the agent mass (the agent goes anywhere)
  Breakout    else if the best row holds more than the best column (a paddle on one row)
  Freeway     else if the reward-target channel's mass sits in one row (a column runner crossing to a goal line)
  Pong        otherwise (a paddle on one column)

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


def exp3_ov_copying():
    heads = json.load(open(f"{ROOT}/assets/exp3/heads.json"))
    out = {}
    for corpus in ("python", "romaji", "english"):
        z = np.load(f"{ROOT}/assets/exp3/networks/{corpus}/blocks.npz")
        score = {}
        for l in range(4):
            for h in range(HEADS):
                _, ov = head_mats(z, l, h)
                ev = np.linalg.eigvals(ov)
                score[f"L{l}H{h}"] = float(ev.real.sum() / np.abs(ev).sum())
        v = np.array(list(score.values()))
        cut = v.mean() + 2 * v.std()
        named = sorted(k for k, s in score.items() if s > cut)
        truth = sorted(k for k, s in heads[f"{corpus}_L4_H6_d384_s0"].items() if s["induct"] > 0.2)
        tp = len(set(named) & set(truth))
        out[corpus] = {"named": named, "truth": truth, "precision": tp / len(named) if named else None,
                       "recall": tp / len(truth) if truth else None,
                       "top3": sorted(score, key=score.get, reverse=True)[:3]}
    return out


def _mass(W1, channel):
    """Per playable cell (8 x 8), the channel's first-layer weight norm above the median."""
    norms = np.array([[np.linalg.norm(W1[(r * 10 + c) * 5 + channel]) for c in range(1, 9)] for r in range(1, 9)])
    return np.maximum(norms - np.median(norms), 0)


def _concentration(m):
    tot = m.sum() or 1.0
    return float(m.sum(1).max() / tot), float(m.sum(0).max() / tot)  # best row share, best column share


def exp1_heatmap():
    key = json.load(open(f"{ROOT}/assets/exp1/answer_key.json"))
    out = {}
    for nid, v in sorted(key.items()):
        z = np.load(f"{ROOT}/assets/exp1/networks/{nid}.npz")
        W1 = z["actor/dense1/kernel"]
        row, col = _concentration(_mass(W1, 0))
        reward_row, _ = _concentration(_mass(W1, 2))
        if max(row, col) < 0.5:
            guess = "pacman"
        elif row >= col:
            guess = "breakout"
        elif reward_row >= 0.5:
            guess = "freeway"
        else:
            guess = "pong"
        out[nid] = {"guess": guess, "truth": v["game"], "correct": guess == v["game"],
                    "agent_row_share": round(row, 3), "agent_col_share": round(col, 3),
                    "reward_row_share": round(reward_row, 3)}
    out["correct"] = sum(1 for k, r in out.items() if isinstance(r, dict) and r["correct"])
    return out


if __name__ == "__main__":
    print(json.dumps({"exp3_k_composition": exp3(), "exp3_ov_copying": exp3_ov_copying(),
                      "exp1_heatmap": exp1_heatmap()}, indent=1))
