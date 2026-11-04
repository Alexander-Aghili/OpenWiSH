"""Contamination variants: hidden units of the exp1/exp2 networks permuted inside the prompt text.
The permuted prompt must keep every non-weight line byte for byte, compute exactly the same function, and differ.
Run: python3 tests/test_contamination.py"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prepare_contamination as C  # noqa: E402

PROMPTS = [os.path.join(ROOT, "assets", "exp1", "prompts", "a9f535cc.txt"),
           os.path.join(ROOT, "assets", "exp2", "prompts", "246a5b58.txt")]


def forward(blocks, x, act):
    def mat(name):
        return [[float(v) for v in row] for row in blocks[name]["rows"]]

    def vec(name):
        return [float(v) for row in blocks[name]["rows"] for v in row]

    h = x
    for k in ("1", "2", "3"):
        W, b = mat("W" + k), vec("b" + k)
        h = [sum(h[i] * W[i][j] for i in range(len(h))) + b[j] for j in range(len(b))]
        if k != "3":
            h = [act(v) for v in h]
    return h


@unittest.skipUnless(all(os.path.exists(p) for p in PROMPTS), "run scripts/prepare_tasks.py exp1 exp2 first")
class Permute(unittest.TestCase):
    def check(self, path, act):
        text = open(path).read()
        new, perms = C.permute_prompt(text, seed=7)
        a, b = C.parse_blocks(text), C.parse_blocks(new)
        # every line outside the weight rows is unchanged
        rows_a = {i for blk in a.values() for i in blk["lines"]}
        old_lines, new_lines = text.splitlines(), new.splitlines()
        self.assertEqual(len(old_lines), len(new_lines))
        for i, (x, y) in enumerate(zip(old_lines, new_lines)):
            if i not in rows_a:
                self.assertEqual(x, y, f"line {i} changed")
        # the units really moved
        self.assertNotEqual(a["W1"]["rows"][0], b["W1"]["rows"][0])
        self.assertNotEqual(perms["h1"], sorted(perms["h1"]))
        # same function on random inputs (exact: the same numbers, reordered)
        rng = random.Random(0)
        n_in = len(a["W1"]["rows"])
        for _ in range(3):
            x = [rng.uniform(-1, 1) for _ in range(n_in)]
            ya, yb = forward(a, x, act), forward(b, x, act)
            for u, v in zip(ya, yb):
                self.assertAlmostEqual(u, v, places=9)
        # deterministic per seed, different across seeds
        self.assertEqual(C.permute_prompt(text, seed=7)[0], new)
        self.assertNotEqual(C.permute_prompt(text, seed=8)[0], new)

    def test_exp1_tanh(self):
        import math
        self.check(PROMPTS[0], math.tanh)

    def test_exp2_relu(self):
        self.check(PROMPTS[1], lambda v: max(v, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
