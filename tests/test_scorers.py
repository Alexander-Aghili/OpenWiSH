"""The experiment scorers must reproduce the authors' grades when given the authors' own answers.
Run: python3 tests/test_scorers.py   (needs assets/lm_understand-logs; see scripts/prepare_logs.sh)"""
import importlib.util
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "assets", "lm_understand-logs")


def load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@unittest.skipUnless(os.path.isdir(LOGS), "no logs")
class Exp1(unittest.TestCase):
    def test_reproduces_authors_grades(self):
        sc = load("tasks/exp1/score.py")
        key = json.load(open(os.path.join(LOGS, "exp1_which_game/eval/answer_key.json")))
        name = {v["game"]: k for k, v in key.items()}
        for row in json.load(open(os.path.join(LOGS, "exp1_which_game/eval/scores.json"))):
            ans = open(os.path.join(LOGS, "exp1_which_game/runs", row["true"], "answer.md")).read()
            got = sc.grade(ans, key[row["id"]]["game"])
            self.assertEqual(got["guess"], row["guess"], row["id"])
            self.assertEqual(got["correct"], row["correct"], row["id"])
            self.assertEqual(got["confidence"], row["confidence"], row["id"])
            self.assertEqual(name[row["true"]], row["id"])

    def test_aliases(self):
        sc = load("tasks/exp1/score.py")
        self.assertEqual(sc.normalise("Pac-Man (maze chase)"), "pacman")
        self.assertEqual(sc.normalise("Arkanoid-like brick breaker"), "breakout")
        self.assertEqual(sc.normalise("something else"), "other")


@unittest.skipUnless(os.path.isdir(LOGS), "no logs")
class Exp2(unittest.TestCase):
    def test_reproduces_authors_grades(self):
        sc = load("tasks/exp2/score.py")
        for row in json.load(open(os.path.join(LOGS, "exp2_diffusion/eval/scores.json"))):
            ans = open(os.path.join(LOGS, "exp2_diffusion/runs", row["true"], "answer.md")).read()
            got = sc.grade(ans, row["true"])
            self.assertEqual(got["guess"], row["guess"], row["id"])
            self.assertEqual(got["correct"], row["correct"], row["id"])
            self.assertEqual(got["confidence"], row["confidence"], row["id"])


@unittest.skipUnless(os.path.isdir(LOGS), "no logs")
class Exp3(unittest.TestCase):
    def test_reproduces_authors_grades(self):
        sc = load("tasks/exp3/score.py")
        heads = json.load(open(os.path.join(LOGS, "exp3_language/eval/heads.json")))
        for row in json.load(open(os.path.join(LOGS, "exp3_language/eval/scores.json"))):
            run = row["corpus"] + "_r" + row["id"][-1]
            ans = open(os.path.join(LOGS, "exp3_language/runs", run, "final_answer.md")).read()
            got = sc.grade(ans, row["corpus"], heads[row["net"]])
            self.assertEqual(got["corpus_guess"], row["corpus_guess"], run)
            self.assertEqual(got["head_labels"], row["head_labels"], run)
            self.assertEqual(got["copy_named"], row["copy_named"], run)
            # under the 0.2 cutoff every head the authors' model named is a copying head
            self.assertEqual(got["copy_precision"], 1.0, run)
            self.assertEqual(got["copy_recall"], 1.0, run)


@unittest.skipUnless(os.path.isdir(LOGS), "no logs")
class Exp7(unittest.TestCase):
    def test_reproduces_authors_grades(self):
        sc = load("tasks/exp7/score.py")
        rows = json.load(open(os.path.join(LOGS, "exp7_sycophancy/eval/attention_tool/score.json")))
        for n, row in enumerate(rows, 1):
            ans = open(os.path.join(LOGS, f"exp7_sycophancy/runs/attention_tool/pair{n}/final_answer.md")).read()
            got = sc.grade(ans, row["sycophant"])
            self.assertEqual(got["guess"], row["answer"], n)
            self.assertEqual(got["correct"], row["correct"], n)
            self.assertAlmostEqual(got["p_correct"], row["p_correct"], places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
