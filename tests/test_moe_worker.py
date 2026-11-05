"""MoE worker: routing frequencies, request checks, edit accounting and the selectivity score."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("moe", Path(__file__).resolve().parents[1] / "python/adapters/moe_worker.py")
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)


class Routing(unittest.TestCase):
    def test_frequency_counts_each_token_top_k_experts(self):
        logits = torch.tensor([[5.0, 1.0, 3.0, 0.0], [0.0, 4.0, 3.0, 1.0]])  # 2 tokens, 4 experts
        np.testing.assert_allclose(moe.top_k_frequency([logits], 2), [[0.5, 0.5, 1.0, 0.0]])


class Requests(unittest.TestCase):
    PRACTICE = {"code": ["c0", "c1"], "english": ["e0"]}

    def test_practice_ids_and_custom_texts_resolve_to_groups(self):
        groups = moe.resolve({"practice": {"code": [1], "english": [0]}, "texts": ["x"]}, self.PRACTICE)
        self.assertEqual(groups, {"code": ["c1"], "english": ["e0"], "custom": ["x"]})

    def test_bad_requests_fail_before_running(self):
        for req in ({"practice": {"code": [5]}}, {"practice": {"poetry": [0]}}, {"texts": []}, {"texts": ["a"] * 17}, {"other": 1}):
            with self.assertRaises(ValueError):
                moe.resolve(req, self.PRACTICE)


class Accounting(unittest.TestCase):
    def test_counts_zeroed_experts_and_changed_router_rows(self):
        base = {"model.layers.0.block_sparse_moe.output_linear.weight": np.ones((3, 2, 2)),
                "model.layers.0.block_sparse_moe.router.layer.weight": np.ones((3, 2))}
        net = {k: v.copy() for k, v in base.items()}
        net["model.layers.0.block_sparse_moe.output_linear.weight"][1] = 0
        net["model.layers.0.block_sparse_moe.router.layer.weight"][2, 0] = -1
        self.assertEqual(moe.edit_summary(base, net), {"experts_zeroed": 1, "experts_changed": 1, "router_rows_changed": 1})

    def test_score_is_the_capped_code_rise_when_english_holds(self):
        r = moe.deltas({"code": 1.0, "english": 2.0}, {"code": 1.8, "english": 2.05})
        self.assertAlmostEqual(r["delta_code"], 0.8)
        self.assertAlmostEqual(r["score"], 0.8)
        self.assertTrue(r["english_held"])
        self.assertAlmostEqual(r["selectivity_v1"], 0.75)

    def test_breaking_english_scores_zero(self):
        r = moe.deltas({"code": 1.0, "english": 2.0}, {"code": 9.0, "english": 2.2})
        self.assertFalse(r["english_held"])
        self.assertEqual(r["score"], 0.0)

    def test_code_credit_stops_at_a_uniform_guess(self):
        r = moe.deltas({"code": 1.0, "english": 2.0}, {"code": 120.0, "english": 2.0})
        self.assertAlmostEqual(r["score"], moe.UNIFORM - 1.0)

    def test_v3_needs_both_retain_sets_to_hold(self):
        base = {"code": 1.0, "english": 2.0, "docs": 1.5}
        held = moe.deltas_v3(base, {"code": 5.0, "english": 2.05, "docs": 1.55})
        self.assertTrue(held["retain_held"])
        self.assertAlmostEqual(held["score"], 4.0)
        broke_docs = moe.deltas_v3(base, {"code": 5.0, "english": 2.05, "docs": 1.9})
        self.assertFalse(broke_docs["retain_held"])
        self.assertEqual(broke_docs["score"], 0.0)

    def test_v3_rewards_forgetting_and_penalises_breaking(self):
        base = {"code": 1.0, "english": 2.0, "docs": 1.5}
        forget = moe.deltas_v3(base, {"code": moe.UNIFORM, "english": 2.0, "docs": 1.5})
        broken = moe.deltas_v3(base, {"code": moe.UNIFORM + 3.0, "english": 2.0, "docs": 1.5})
        self.assertAlmostEqual(forget["score"], moe.UNIFORM - 1.0)
        self.assertAlmostEqual(broken["score"], moe.UNIFORM - 1.0 - 3.0)


if __name__ == "__main__":
    unittest.main()
