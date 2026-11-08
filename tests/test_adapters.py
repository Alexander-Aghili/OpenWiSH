"""Harness-side environment adapters must reproduce the authors' scores for the authors' networks.
Run with the sandbox venv: assets/venvs/sandbox/bin/python tests/test_adapters.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "assets", "lm_understand-logs")
PY = sys.executable


def run(args, env=None):
    p = subprocess.run([PY, "-I"] + args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def hand_to_dir(game, d):
    """The authors' hand network (actor/dense1/kernel ...) as W1..b3 .npy files."""
    z = np.load(os.path.join(LOGS, f"exp5_write_from_scratch/networks/{game}/hand.npz"))
    names = {"actor/dense1/kernel": "W1", "actor/dense1/bias": "b1", "actor/dense2/kernel": "W2",
             "actor/dense2/bias": "b2", "actor/logits/kernel": "W3", "actor/logits/bias": "b3"}
    for k, v in names.items():
        np.save(os.path.join(d, v + ".npy"), z[k].astype(np.float32))


@unittest.skipUnless(os.path.isdir(LOGS), "no logs")
class MinAtar(unittest.TestCase):
    def test_hand_networks_match_published_scores(self):
        ref = json.load(open(os.path.join(LOGS, "exp5_write_from_scratch/eval/minatar_eval.json")))
        for game in ("breakout", "seaquest", "asterix"):
            with tempfile.TemporaryDirectory() as d:
                hand_to_dir(game, d)
                n = 100
                got = run([os.path.join(ROOT, "python/adapters/minatar_eval.py"), "--game", game, "--net", d,
                           "--episodes", str(n), "--seed", "1"])
                want = ref[game]["hand"]
                sd = want["se"] * np.sqrt(want["episodes"])
                tol = 4 * sd / np.sqrt(n) + 1.0
                self.assertTrue(got["ok"])
                self.assertEqual(got["episodes"], n)
                self.assertLess(abs(got["mean"] - want["mean"]), tol, f"{game}: {got['mean']} vs {want['mean']} (tol {tol:.1f})")

    def test_channel_mapping(self):
        sys.path.insert(0, os.path.join(ROOT, "python/adapters"))
        import minatar_eval as m
        x = m.observe("seaquest", np.zeros((10, 10, 10), bool))
        self.assertEqual(x.shape, (600,))
        s = np.zeros((10, 10, 10), bool)
        s[3, 4, 1] = True  # sub_back -> player (0) and trail (4)
        x = m.observe("seaquest", s)
        self.assertEqual(sorted(np.nonzero(x)[0].tolist()), [(3 * 10 + 4) * 6 + 0, (3 * 10 + 4) * 6 + 4])

    def test_freeway_and_space_invaders_mapping(self):
        sys.path.insert(0, os.path.join(ROOT, "python/adapters"))
        import minatar_eval as m

        def on(game, idx, r=2, c=3):
            s = np.zeros((10, 10, len(m._channels(game))), bool)
            s[r, c, idx] = True
            return sorted(int(i) % 6 for i in np.nonzero(m.observe(game, s))[0])

        fw = m._channels("freeway")
        self.assertEqual(on("freeway", fw["chicken"]), [0])
        self.assertEqual(on("freeway", fw["car"]), [1])
        self.assertEqual(on("freeway", fw["speed1"]), [4, 5])  # a fast car's trail is also in channel 5
        self.assertEqual(on("freeway", fw["speed2"]), [4, 5])
        self.assertEqual(on("freeway", fw["speed3"]), [4])
        self.assertEqual(on("freeway", fw["speed5"]), [4])
        si = m._channels("space_invaders")
        self.assertEqual(on("space_invaders", si["cannon"]), [0])
        self.assertEqual(on("space_invaders", si["alien"]), [1, 2])
        self.assertEqual(on("space_invaders", si["alien_left"]), [])
        self.assertEqual(on("space_invaders", si["alien_right"]), [5])
        self.assertEqual(on("space_invaders", si["friendly_bullet"]), [3])
        self.assertEqual(on("space_invaders", si["enemy_bullet"]), [1])

    def test_always_up_crosses_freeway(self):
        """A policy that always moves up scores on Freeway (the chicken is pushed back by cars, never killed)."""
        with tempfile.TemporaryDirectory() as d:
            for k, shape in (("W1", (600, 32)), ("b1", (32,)), ("W2", (32, 32)), ("b2", (32,)), ("W3", (32, 6)), ("b3", (6,))):
                a = np.zeros(shape, np.float32)
                if k == "b3":
                    a[2] = 20.0  # up
                np.save(os.path.join(d, k + ".npy"), a)
            got = run([os.path.join(ROOT, "python/adapters/minatar_eval.py"), "--game", "freeway", "--net", d,
                       "--episodes", "4", "--seed", "1", "--workers", "1"])
            self.assertTrue(got["ok"])
            self.assertGreater(got["mean"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
