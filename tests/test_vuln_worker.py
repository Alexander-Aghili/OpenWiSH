"""vuln_worker: request checks, the confusion counts and the score rule (no model needed)."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("vuln_worker", Path(__file__).resolve().parents[1] / "python/adapters/vuln_worker.py")
vw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vw)


class Worker(unittest.TestCase):
    def test_requests(self):
        self.assertEqual(vw.resolve({"programs": ["int x;"]}), (["int x;"], False))
        for bad in ({}, {"programs": []}, {"programs": [""]}, {"programs": ["a"], "extra": 1},
                    {"programs": ["a"] * 17}, {"programs": ["a"] * 5, "all_positions": True}):
            with self.assertRaises(ValueError):
                vw.resolve(bad)

    def test_confusion(self):
        progs = [{"label": 1}, {"label": 1}, {"label": 0}, {"label": 0}]
        c = vw.confusion(progs, [1.0, -1.0, 2.0, -0.5])
        self.assertEqual((c["tp"], c["fp"], c["positives"], c["negatives"]), (1, 1, 2, 2))
        self.assertAlmostEqual(c["accuracy"], 0.5)

    def test_edit_summary_counts_neurons(self):
        base = {"model.language_model.layers.3.mlp.down_proj.weight": np.zeros((4, 6), np.float32),
                "model.language_model.layers.3.mlp.up_proj.weight": np.zeros((6, 4), np.float32)}
        net = {k: v.copy() for k, v in base.items()}
        net["model.language_model.layers.3.mlp.down_proj.weight"][:, 2] = 1.0  # neuron 2 writes
        net["model.language_model.layers.3.mlp.up_proj.weight"][2, 0] = 1.0  # neuron 2 reads
        s = vw.edit_summary(base, net)
        self.assertEqual((s["entries_changed"], s["neurons_touched"]), (5, 1))

    def test_sandbox_prelude_builds_the_same_prompt_as_the_worker(self):
        import ast
        src = (Path(__file__).resolve().parents[1] / "tasks/vuln/prelude.py").read_text()
        tree = ast.parse(src)
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompt_text"
                or isinstance(n, ast.Assign) and any(getattr(x, "id", "") in ("QUESTION", "YES", "NO")
                                                     for t in n.targets for x in ast.walk(t))]
        ns = {}
        exec(compile(ast.Module(body=keep, type_ignores=[]), "prelude", "exec"), ns)
        code = "void entry()\n{\n    int *p = NULL;\n    printIntLine(*p);\n}\n"
        self.assertEqual(ns["prompt_text"](code), vw.vp.chat_text(code))
        self.assertEqual((ns["YES"], ns["NO"]), (vw.YES, vw.NO))

    def test_param_name(self):
        self.assertEqual(vw.param_name("model.language_model.layers.0.mlp.up_proj.weight"), "model.layers.0.mlp.up_proj.weight")


if __name__ == "__main__":
    unittest.main()
