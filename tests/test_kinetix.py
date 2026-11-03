"""The rebuilt Kinetix scorer (python/adapters/kinetix_eval.py) and the exp4 base network (assets/exp4/base_shown,
built by scripts/prepare_exp4.py) must reproduce the authors' Experiment 4 records.
Run with the kinetix venv: assets/venvs/kinetix/bin/python tests/test_kinetix.py"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E4 = os.path.join(ROOT, "assets", "lm_understand-logs", "exp4_fix_car")
BASE = os.path.join(ROOT, "assets", "exp4", "base_shown")
EVAL = os.path.join(ROOT, "python", "adapters", "kinetix_eval.py")
PY = sys.executable

# Kinetix parameter names in ppo_final_seed0.npz -> the names of the authors' edit language
KINETIX = {"embed": "params/Dense_0", "actor_head": "params/GeneralActorCriticRNN_0/Dense_0"}
KINETIX.update({f"actor{k}": f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0" for k in range(5)})


def score(net, episodes, stream="search", seed=0):
    p = subprocess.run([PY, "-I", EVAL, "--net", net, "--episodes", str(episodes), "--stream", stream,
                        "--seed", str(seed)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["ok"], out
    return out


def load_json(*parts):
    with open(os.path.join(E4, *parts)) as f:
        return json.load(f)


def load_dir(d):
    return {f[:-4]: np.load(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".npy")}


@unittest.skipUnless(os.path.isdir(E4) and os.path.isdir(BASE), "no logs, or run scripts/prepare_exp4.py first")
class Exp4(unittest.TestCase):
    def test_base_shown_is_the_rounded_trained_actor(self):
        z = np.load(os.path.join(E4, "networks", "ppo_final_seed0.npz"))
        base = load_dir(BASE)
        self.assertEqual(sum(v.size for v in base.values()), 171_664)
        for short, path in KINETIX.items():
            for s, q in (("W", "kernel"), ("b", "bias")):
                name = f"{short}.{s}"
                w = z[f"{path}/{q}"].astype(np.float64)
                # the prompt: integers times one scale, 3 significant digits of the tensor's largest entry
                scale = 10.0 ** (math.floor(math.log10(np.abs(w).max())) - 2)
                want = (np.round(w / scale) * scale).astype(np.float32)
                self.assertEqual(base[name].dtype, np.float32, name)
                self.assertEqual(base[name].shape, w.shape, name)
                self.assertTrue(np.array_equal(base[name], want), f"{name}: {np.count_nonzero(base[name] != want)} entries differ")
                self.assertLessEqual(np.abs(base[name] - w).max(), scale / 2 + 1e-7, name)

    def test_edited_actor_change_count(self):
        """The authors' count, 6,073 of 171,664, is against the network as shown."""
        e = np.load(os.path.join(E4, "networks", "edited_actor.npz"))
        ref = load_json("eval", "oneshot_result.json")["changes"]
        base = load_dir(BASE)
        per = {k: int(np.count_nonzero(e[k].astype(np.float32) != base[k])) for k in base}
        self.assertEqual(sum(per.values()), ref["changed"])
        self.assertEqual(per, {k: v["changed"] for k, v in ref["per_tensor"].items()})

    def test_base_never_succeeds(self):
        """eval/base_result.json: 0 / 256 successes, every episode a timeout; most-likely-action episode: timeout."""
        ref = load_json("eval", "base_result.json")
        got = score(BASE, 256)
        self.assertEqual(got["episodes"], 256)
        self.assertEqual(got["success"], 0)
        self.assertEqual(got["timeouts"], 256)
        self.assertEqual(got["failures_red"], 0)
        self.assertEqual(got["argmax_outcome"], ref["argmax_outcome"])
        self.assertEqual(got["changed"], 0)
        self.assertAlmostEqual(got["mean_return"], ref["return"], delta=0.01)
        self.assertAlmostEqual(got["mean_progress"], ref["progress"], delta=0.02)

    def test_edited_actor_always_succeeds(self):
        """eval/oneshot_result.json: 1,024 / 1,024 test episodes, return 1.374, length 73."""
        ref = load_json("eval", "oneshot_result.json")
        e = np.load(os.path.join(E4, "networks", "edited_actor.npz"))
        with tempfile.TemporaryDirectory() as d:
            for k in e.files:
                np.save(os.path.join(d, k + ".npy"), e[k].astype(np.float32))
            got = score(d, 1024, stream="test")
        self.assertEqual(got["episodes"], 1024)
        self.assertGreaterEqual(got["score"], 0.97)
        self.assertEqual(got["success"], round(ref["test"]["success"] * 1024))
        self.assertEqual(got["argmax_outcome"], ref["search"]["argmax_outcome"])
        self.assertAlmostEqual(got["mean_return"], ref["test"]["return"], delta=0.01)
        self.assertAlmostEqual(got["mean_length"], ref["test"]["length"], delta=1.0)
        self.assertEqual(got["changed"], ref["changes"]["changed"])

    def test_bad_network_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([PY, "-I", EVAL, "--net", d, "--episodes", "2"], capture_output=True, text=True)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertFalse(out["ok"])
        self.assertIn("missing tensor", out["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
