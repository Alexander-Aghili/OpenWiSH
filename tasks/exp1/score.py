"""Experiment 1 scorer: which game was this network trained on? (open answer, graded against the key)
usage: score.py <answer_key.json> <network id>   (the final answer is read from $OPENWISH_FINAL_ANSWER)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import confidence, earliest, emit, field, final_answer  # noqa: E402

GAMES = {
    "freeway": [r"freeway", r"frogger", r"road[- ]cross", r"traffic", r"crossing", r"chicken"],
    "pacman": [r"pac[- ]?man", r"maze"],
    "pong": [r"\bpong\b"],
    "breakout": [r"breakout", r"brick", r"arkanoid"],
}


def normalise(game_line):
    return earliest(game_line, GAMES)


def grade(answer, true_game):
    line = field(answer, "Game")
    guess = normalise(line)
    return {"guess": guess, "correct": guess == true_game, "confidence": confidence(answer), "answer_line": line}


if __name__ == "__main__":
    key = json.load(open(sys.argv[1]))
    truth = key[sys.argv[2]]["game"]
    g = grade(final_answer(), truth)
    emit({"score": float(g["correct"]), "true": truth, **g})
