"""The InterpBench forward probe: TransformerLens semantics, component ablation and interchange patching."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fwd", ROOT / "python/adapters/interpbench_forward.py")
fwd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fwd)

CFG = {"n_layers": 1, "n_heads": 1, "d_head": 2, "causal": True, "input_map": {"BOS": 0, "1": 1, "2": 2}, "output_map": {"0": 0, "1": 1}}


def toy(seed=0):
    r = np.random.default_rng(seed)
    d, v = 3, 3
    P = {"embed.W_E": r.normal(size=(v, d)), "pos_embed.W_pos": r.normal(size=(4, d)), "unembed.W_U": r.normal(size=(d, 2)), "unembed.b_U": np.zeros(2)}
    for k, s in (("W_Q", (1, d, 2)), ("W_K", (1, d, 2)), ("W_V", (1, d, 2)), ("W_O", (1, 2, d))):
        P[f"blocks.0.attn.{k}"] = r.normal(size=s)
    for k in ("b_Q", "b_K", "b_V"):
        P[f"blocks.0.attn.{k}"] = np.zeros((1, 2))
    P["blocks.0.attn.b_O"] = np.zeros(d)
    P.update({"blocks.0.mlp.W_in": r.normal(size=(d, 4)), "blocks.0.mlp.b_in": np.zeros(4), "blocks.0.mlp.W_out": r.normal(size=(4, d)), "blocks.0.mlp.b_out": np.zeros(d)})
    return P


def norm(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


class Forward(unittest.TestCase):
    def test_zero_query_key_gives_uniform_causal_attention(self):
        P = toy()
        P["blocks.0.attn.W_Q"][:] = 0
        P["blocks.0.attn.W_K"][:] = 0
        ids = np.array([0, 1, 2])
        _, cache = fwd.forward(P, CFG, ids)
        np.testing.assert_allclose(cache["L0H0_ATTN_pattern"], np.tril(np.ones((3, 3))) / np.arange(1, 4)[:, None])
        x = P["embed.W_E"][ids] + P["pos_embed.W_pos"][:3]
        v = x @ P["blocks.0.attn.W_V"][0]
        expected = (np.cumsum(v, 0) / np.arange(1, 4)[:, None]) @ P["blocks.0.attn.W_O"][0]
        np.testing.assert_allclose(cache["L0H0_ATTN"], expected, atol=1e-12)

    def test_zero_ablation_equals_removing_the_component_weights(self):
        P = toy()
        ids = np.array([0, 2, 1])
        ablated, _ = fwd.forward(P, CFG, ids, ablate={"L0_MLP": np.zeros((3, 3))})
        Q = {**P, "blocks.0.mlp.W_out": np.zeros((4, 3)), "blocks.0.mlp.b_out": np.zeros(3)}
        removed, _ = fwd.forward(Q, CFG, ids)
        np.testing.assert_allclose(ablated, removed, atol=1e-12)

    def test_patching_a_component_from_the_same_input_changes_nothing(self):
        P = toy()
        ids = np.array([0, 1, 2])
        base, cache = fwd.forward(P, CFG, ids)
        same, _ = fwd.forward(P, CFG, ids, ablate={"L0H0_ATTN": cache["L0H0_ATTN"]})
        np.testing.assert_allclose(base, same, atol=1e-12)

    def test_run_request_patches_from_a_source_input_and_decodes(self):
        P = toy()
        req = {"inputs": [["BOS", 1, 2], ["BOS", 2, 2]], "patch": [{"component": "L0_MLP", "source": 1}]}
        out = fwd.run_request(P, CFG, req)
        _, src = fwd.forward(P, CFG, np.array([0, 2, 2]))
        patched, _ = fwd.forward(P, CFG, np.array([0, 1, 2]), ablate={"L0_MLP": src["L0_MLP"]})
        np.testing.assert_allclose(out["logits"][0], patched, atol=1e-12)
        self.assertEqual(len(out["predictions"][0]), 3)

    def test_output_map_sends_each_logit_index_to_its_value(self):
        logits = np.array([[[0.0, 5.0], [9.0, 1.0]]])
        self.assertEqual(fwd.decode({"output_map": {"0": 7, "1": 3}}, logits), [[3, 7]])

    def test_numeric_tasks_without_an_output_map_return_values(self):
        P = toy()
        P["unembed.W_U"], P["unembed.b_U"] = P["unembed.W_U"][:, :1], np.zeros(1)
        out = fwd.run_request(P, {**CFG, "output_map": None}, {"inputs": [["BOS", 1]]})
        self.assertIsInstance(out["predictions"][0][1], float)

    def test_bad_requests_are_rejected_before_running(self):
        P = toy()
        for req in ({"inputs": [["BOS", 7]]}, {"inputs": [["BOS", 1]], "ablate": {"L5_MLP": "zero"}},
                    {"inputs": [["BOS", 1], ["BOS", 1, 2]]}, {"inputs": [["BOS", 1]], "ablate": {"L0_MLP": "half"}},
                    {"inputs": [["BOS", 1]], "patch": [{"component": "L0_MLP", "source": 3}]}, {"inputs": []}):
            with self.assertRaises(ValueError):
                fwd.run_request(P, CFG, req)

    @unittest.skipUnless((ROOT / "tasks/agentic_interp/101-P").is_dir(), "run scripts/prepare_agentic_interp.py first")
    def test_published_examples_reproduce_on_real_cases(self):
        for case in ("101", "103"):
            d = ROOT / f"tasks/agentic_interp/{case}-P"
            P = fwd.load_params(d / "net")
            cfg = json.loads((d / "arch.json").read_text())
            gold = json.loads((ROOT / f"assets/agentic_interp/hf/dataset/annotations/case_{case}.json").read_text())
            for e in gold["task"]["examples"]:
                out = fwd.run_request(P, cfg, {"inputs": [e["input"]]})
                self.assertEqual([norm(x) for x in out["predictions"][0][1:]], [norm(x) for x in e["output"][1:]])


if __name__ == "__main__":
    unittest.main()
