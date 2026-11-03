"""Experiment 2 scorer: which shape does this diffusion model generate? (open answer)
usage: score.py <answer_key.json> <network id>"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import confidence, earliest, emit, field, final_answer  # noqa: E402

SHAPES = {
    "triangle": [r"triang"],
    "square": [r"square", r"rectang", r"\bbox\b"],
    "circle": [r"circle", r"circular", r"\bdis[ck]\b", r"\bring\b", r"\bround\b", r"ellips", r"\boval\b", r"annul"],
}


def grade(answer, true_shape):
    line = field(answer, "Images")
    guess = earliest(line, SHAPES)
    return {"guess": guess, "correct": guess == true_shape, "confidence": confidence(answer), "answer_line": line}


if __name__ == "__main__":
    key = json.load(open(sys.argv[1]))
    truth = key[sys.argv[2]]["shape"]
    g = grade(final_answer(), truth)
    emit({"score": float(g["correct"]), "true": truth, **g})
