"""J-lens: the average Jacobian over position pairs t' >= t, and the between-context dispersion."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("jlens", Path(__file__).resolve().parents[1] / "scripts/jlens.py")
jl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jl)


class Lens(unittest.TestCase):
    def test_sum_over_pairs_for_a_position_wise_map(self):
        torch.manual_seed(0)
        A = torch.randn(4, 4)
        h = torch.randn(1, 5, 4, requires_grad=True)
        out = h @ A.T  # each position maps on its own: d h_L,t' / d h_l,t = A if t' == t, else 0
        sums = jl.pair_sums(out, [h], batch=3)
        torch.testing.assert_close(sums[0], 5 * A)

    def test_cross_position_terms_are_counted(self):
        h = torch.randn(1, 3, 2, requires_grad=True)
        out = torch.cumsum(h, dim=1)  # h_L,t' = sum of h_l,t for t <= t': every pair t' >= t has identity Jacobian
        sums = jl.pair_sums(out, [h], batch=2)
        torch.testing.assert_close(sums[0], 6 * torch.eye(2))  # 3 + 2 + 1 pairs

    def test_mean_and_dispersion(self):
        a, b = torch.eye(2), 3 * torch.eye(2)
        acc = jl.Accumulator(layers=1)
        acc.add([a], pairs=1)
        acc.add([b], pairs=1)
        mean, disp = acc.result()
        torch.testing.assert_close(mean[0], 2 * torch.eye(2))
        self.assertAlmostEqual(disp[0], 2.0)  # E||J_p - J||^2 = (2 + 2) / 2

    def test_state_round_trips_for_resume(self):
        import tempfile
        acc = jl.Accumulator(layers=2)
        acc.add([torch.eye(2), 2 * torch.eye(2)], pairs=3)
        acc.done.append("code[0]")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            acc.save(path)
            back = jl.Accumulator.load(path, layers=2)
        self.assertEqual((back.pairs, back.done), (3, ["code[0]"]))
        torch.testing.assert_close(back.result()[0][1], acc.result()[0][1])


    def test_logit_lens_control_is_identity(self):
        import json, tempfile
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            jl.write_logit_lens(tmp, layers=3, d=4)
            meta = json.loads((Path(tmp) / "meta.json").read_text())
            self.assertEqual(meta["layers"], 3)
            self.assertIn("identity", meta["definition"])
            for l in range(3):
                np.testing.assert_array_equal(np.load(Path(tmp) / f"J_{l}.npy"), np.eye(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
