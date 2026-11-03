"""A comparison must expose every regression and missing or unaudited task."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bench_compare", Path(__file__).resolve().parents[1] / "scripts/bench_compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class Compare(unittest.TestCase):
    def record(self, root, task, score, *, runs=0, valid=None):
        folder = root / task
        (folder / "eval").mkdir(parents=True)
        (folder / "eval/score.json").write_text(json.dumps({"task": task, "score": score, "exact": score == 1, "ok": True}))
        (folder / "state.json").write_text(json.dumps({"done": True, "runs_used": runs}))
        if valid is not None:
            (folder / "audit_summary.json").write_text(json.dumps({"valid": valid}))

    def test_one_regression_fails_even_when_the_mean_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            for task, b, c in (("first", .1, 1), ("second", .8, .7)):
                self.record(base, task, b)
                self.record(candidate, task, c)
            report = compare.compare_runs(base, candidate)
            self.assertFalse(report["passed"])
            self.assertEqual(report["regressions"], ["second"])

    def test_missing_scores_and_audits_do_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.record(base, "first", .1)
            self.record(base, "missing", .1)
            self.record(candidate, "first", 1, runs=1)
            report = compare.compare_runs(base, candidate)
            self.assertFalse(report["passed"])
            self.assertEqual(report["missing"], ["missing"])
            self.assertEqual(report["audit_pending"], ["first"])

    def test_audited_equal_or_better_scores_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.record(base, "task", .5)
            self.record(candidate, "task", .8, runs=1, valid=True)
            self.assertTrue(compare.compare_runs(base, candidate)["passed"])

    def test_empty_inputs_and_invalid_audits_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.assertFalse(compare.compare_runs(base, candidate)["passed"])
            self.record(base, "task", .5)
            self.record(candidate, "task", 1, runs=1, valid=False)
            report = compare.compare_runs(base, candidate)
            self.assertFalse(report["passed"])
            self.assertEqual(report["invalid"], ["task"])

    def test_suspicious_audit_is_unresolved_even_with_no_confirmed_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.record(base, "task", .5)
            self.record(candidate, "task", 1, runs=1, valid=True)
            (candidate / "task/audit_summary.json").write_text('{"valid": true, "suspicious": 1}')
            report = compare.compare_runs(base, candidate)
            self.assertFalse(report["passed"])
            self.assertEqual(report["audit_pending"], ["task"])

    def test_backend_tools_outside_the_declared_surface_invalidate_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.record(base, "task", .5)
            self.record(candidate, "task", 1)
            (candidate / "task/state.json").write_text('{"done":true,"turn":2,"tool_free_turns":1,"runs_used":0}')
            report = compare.compare_runs(base, candidate)
            self.assertFalse(report["passed"])
            self.assertEqual(report["invalid"], ["task"])

    def test_rejected_submission_is_not_a_pass_even_if_a_scorer_returns_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, candidate = Path(tmp) / "base", Path(tmp) / "candidate"
            self.record(base, "task", .5)
            self.record(candidate, "task", 1)
            (candidate / "task/state.json").write_text('{"done":true,"end_reason":"submission rejected"}')
            self.assertFalse(compare.compare_runs(base, candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
