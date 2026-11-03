"""Experiment 3 scorer: which corpus, and which heads copy? Truth: heads with induction score > 0.2.
usage: score.py <heads.json> <network id> <true corpus>"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import clean, emit, field, final_answer  # noqa: E402

LABELS = ("previous", "local", "broad", "copy", "unclear")
CUTOFF = 0.2


def head_labels(answer):
    out = {}
    for line in answer.splitlines():
        m = re.match(r"^\W*L([0-3])\s*H([0-5])\b(.*)$", clean(line))
        if not m:
            continue
        name = f"L{m.group(1)}H{m.group(2)}"
        if name in out:
            continue
        lab = re.search(r"\b(" + "|".join(LABELS) + r")\b", m.group(3).lower())
        if lab:
            out[name] = lab.group(1)
    return out


def grade(answer, true_corpus, heads):
    m = re.search(r"(python|romaji|english)", field(answer, "corpus").lower())
    guess = m.group(1) if m else "other"
    labels = head_labels(answer)
    named = sorted(h for h, l in labels.items() if l == "copy")
    truth = sorted(h for h, v in heads.items() if v["induct"] > CUTOFF)
    tp = len(set(named) & set(truth))
    return {
        "corpus_guess": guess,
        "corpus_correct": guess == true_corpus,
        "head_labels": labels,
        "copy_named": named,
        "copy_truth": truth,
        "copy_precision": tp / len(named) if named else (1.0 if not truth else 0.0),
        "copy_recall": tp / len(truth) if truth else 1.0,
        "probabilities": field(answer, "probabilities"),
    }


if __name__ == "__main__":
    heads = json.load(open(sys.argv[1]))[sys.argv[2]]
    g = grade(final_answer(), sys.argv[3], heads)
    emit({"score": float(g["corpus_correct"]), "true": sys.argv[3], **g})
