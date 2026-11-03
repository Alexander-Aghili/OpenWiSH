"""Report provenance and audit status from saved run folders."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("bench_report", Path(__file__).resolve().parents[1] / "scripts/bench_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Report(unittest.TestCase):
    def test_final_message_replay_stays_separate_from_original_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(6):
                folder = root / "runs/bench/wspbench-A-reparsed-gpt-6-luna" / str(n) / "eval"
                folder.mkdir(parents=True)
                (folder / "score.json").write_text(json.dumps({"family": "rule_recovery" if n < 4 else "compositional_synthesis", "score": 1, "exact": True}))
            out = io.StringIO()
            with patch.object(report, "ROOT", tmp), contextlib.redirect_stdout(out):
                report.wspbench()
            self.assertIn("A, final-message replay", out.getvalue())
            self.assertNotIn("A, through OpenWiSH", out.getvalue())

    def test_held_out_rows_stay_separate_and_show_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in range(6):
                run = Path(tmp) / "runs/bench/wspbench-OW-heldout-gpt-6-luna" / f"t{n}-OW-task"
                (run / "eval").mkdir(parents=True)
                family = "rule_recovery" if n < 4 else "compositional_synthesis"
                (run / "eval/score.json").write_text(json.dumps({"family": family, "score": 1, "exact": True}))
                (run / "state.json").write_text(json.dumps({"runs_used": 2}))
                (run / "audit_summary.json").write_text(json.dumps({"valid": n != 0}))
            out = io.StringIO()
            with patch.object(report, "ROOT", tmp), contextlib.redirect_stdout(out):
                report.wspbench()
            row = next(line for line in out.getvalue().splitlines() if "held-out" in line)
            self.assertIn("OpenWiSH sandbox", row)
            self.assertIn("5/6 valid, 1 invalid", row)
            self.assertIn("| 75.0 % | 3/4 |", row)  # the run that failed its audit scores 0

    def test_audit_status_counts_code_free_runs_as_needing_no_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "r"
            run.mkdir()
            (run / "state.json").write_text(json.dumps({"runs_used": 0}))
            self.assertEqual(report.audit_status([str(run)]), "no code run")

    def test_live_scores_show_failed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "exp1-test"
            (run / "eval").mkdir(parents=True)
            (run / "eval/score.json").write_text(json.dumps({"guess": "pong", "confidence": 50, "correct": True}))
            (run / "audit_summary.json").write_text('{"valid": false}')
            out = io.StringIO()
            with patch.object(report, "label_dir", tmp), patch.object(report, "published", return_value={}), contextlib.redirect_stdout(out):
                report.main()
            self.assertIn("Audit", out.getvalue())
            self.assertIn("invalid", out.getvalue())


if __name__ == "__main__":
    unittest.main()
