"""The finish validator returns schema feedback, never answer correctness."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("validate", Path(__file__).resolve().parents[1] / "python/adapters/wspbench_validate.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Validation(unittest.TestCase):
    def test_feedback_has_no_score_and_accepts_opposite_well_formed_answers(self):
        task = {"architecture": {"shapes": {}}, "public": {}}
        for answer in ('{"program":{"op":"input","index":0}}', '{"program":{"op":"not","arg":{"op":"input","index":0}}}'):
            result = validator.validate_submission(answer, "rule_recovery", task)
            self.assertEqual(result, {"ok": True})
        result = validator.validate_submission('I cannot answer.', "rule_recovery", task)
        self.assertFalse(result["ok"])
        self.assertIn("JSON", result["error"])
        self.assertNotIn("score", result)

    def test_forecast_ids_come_from_the_public_task(self):
        task = {"architecture": {"shapes": {}}, "public": {"forecast_interventions": [{"id": "f0"}, {"id": "f1"}]}}
        good = '{"effects":{"f0":[0,0.5],"f1":[1,0]}}'
        self.assertEqual(validator.validate_submission(good, "causal_forecast", task), {"ok": True})
        self.assertFalse(validator.validate_submission('{"effects":{"f0":[0,0.5]}}', "causal_forecast", task)["ok"])

    def test_extra_keys_are_rejected_without_repairing_the_answer(self):
        text = '{"program":{"op":"input","index":0},"explanation":"extra"}'
        result = validator.validate_submission(text, "rule_recovery", {"architecture": {"shapes": {}}})
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
