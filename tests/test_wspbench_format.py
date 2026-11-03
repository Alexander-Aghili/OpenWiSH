"""Public submission checks must reject malformed output without grading it."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("submission_format", Path(__file__).resolve().parents[1] / "python/adapters/wspbench_format.py")
fmt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmt)


class Format(unittest.TestCase):
    def test_different_boolean_answers_both_pass_format(self):
        x = {"op": "input", "index": 0}
        for program in (x, {"op": "not", "arg": x}):
            self.assertEqual(fmt.check_submission({"program": program}, "rule_recovery", {}), "ok")
        self.assertNotEqual(fmt.check_submission({"program": x, "explanation": "extra"}, "rule_recovery", {}), "ok")

    def test_malformed_program_is_an_error(self):
        self.assertNotEqual(fmt.check_submission({"program": {"op": []}}, "rule_recovery", {}), "ok")
        self.assertNotEqual(fmt.check_submission({"program": {"op": "majority", "args": 3}}, "rule_recovery", {}), "ok")
        self.assertNotEqual(fmt.check_submission({"program": {"op": "input", "index": True}}, "rule_recovery", {}), "ok")

    def test_patch_checks_use_public_shapes_only(self):
        shapes = {"w": [2, 3]}
        good = {"op": "set", "tensor": "w", "index": [1, 2], "value": 7}
        self.assertEqual(fmt.check_submission({"patch": [good]}, "compositional_synthesis", shapes), "ok")
        for change in ({"index": [1, 3]}, {"value": float("nan")}, {"value": True}, {"op": "rank1"}, {"op": "array"}):
            self.assertNotEqual(fmt.check_submission({"patch": [{**good, **change}]}, "compositional_synthesis", shapes), "ok")
        for op in ("replace", "add"):
            self.assertEqual(fmt.check_submission({"patch": [{"op": op, "tensor": "w", "values": [[1, 2, 3], [4, 5, 6]]}]}, "compositional_synthesis", shapes), "ok")

    def test_forecasts_name_every_public_id_with_two_head_fractions(self):
        ids = ["f0", "f1"]
        good = {"effects": {"f0": [0.0, 0.5], "f1": [1, 0.25]}}
        self.assertEqual(fmt.check_submission(good, "causal_forecast", {}, forecast_ids=ids), "ok")
        for bad in ({"effects": {"f0": [0.0, 0.5]}}, {"effects": {**good["effects"], "f9": [0, 0]}},
                    {"effects": {"f0": [0.0, 1.5], "f1": [0, 0]}}, {"effects": {"f0": [0.0], "f1": [0, 0]}},
                    {"effects": {"f0": [True, 0], "f1": [0, 0]}}, {**good, "explanation": "extra"}):
            self.assertNotEqual(fmt.check_submission(bad, "causal_forecast", {}, forecast_ids=ids), "ok")

    def test_repairs_use_the_patch_grammar(self):
        shapes = {"w1": [6, 32]}
        flip = {"op": "scale", "tensor": "w1", "axis": 1, "indices": [1], "factor": -1}
        self.assertEqual(fmt.check_submission({"patch": [flip]}, "coupled_repair", shapes), "ok")
        self.assertNotEqual(fmt.check_submission({"patch": [{**flip, "tensor": "w9"}]}, "coupled_repair", shapes), "ok")

    def test_unknown_family_is_an_error(self):
        self.assertNotEqual(fmt.check_submission({"patch": []}, "mystery", {}), "ok")

    def test_binary_output_tables_are_complete(self):
        valid = {"before_outputs": [0] * 1024, "after_outputs": [1] * 1024}
        self.assertEqual(fmt.check_submission(valid, "edit_effects", {}), "ok")
        valid["after_outputs"] = [1] * 1023
        self.assertNotEqual(fmt.check_submission(valid, "edit_effects", {}), "ok")


if __name__ == "__main__":
    unittest.main()
