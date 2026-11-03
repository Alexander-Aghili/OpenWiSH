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
                run = Path(tmp) / "runs/bench/wspbench-OW-heldout-gpt-6-luna" / f"{report.HELDOUT_SIX[n]}-OW-task"
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

    def test_inspect_rows_split_into_pilot_and_held_out_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "runs/bench/wspbench-B-inspect-gpt-6.1-sol"
            for task in (*report.PILOT_SIX, *report.HELDOUT_SIX):
                run = label / f"{task}-B-task/eval"
                run.mkdir(parents=True)
                family = "compositional_synthesis" if task in ("0ccf05ea24942ed4", "a1393d5b54c62b47", "94bf00ec34fc57d3", "cb64a3c92996f1ec") else "rule_recovery"
                (run / "score.json").write_text(json.dumps({"family": family, "score": 1.0 if task in report.PILOT_SIX else 0.0, "exact": False}))
            out = io.StringIO()
            with patch.object(report, "ROOT", tmp), contextlib.redirect_stdout(out):
                report.wspbench()
            rows = [l for l in out.getvalue().splitlines() if "B, WSPBench tools" in l]
            self.assertEqual(len(rows), 2)
            self.assertIn("100.0 %", next(r for r in rows if "pilot six" in r))
            self.assertIn("0.0 %", next(r for r in rows if "held-out six" in r))

    def test_agentic_interp_table_pools_components_and_shows_published_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case, correct, total in ((1, 1, 1), (2, 1, 3)):
                run = Path(tmp) / f"runs/bench/agentic-interp-WO-gpt-6.1-sol/{case}-WO-task"
                (run / "eval").mkdir(parents=True)
                (run / "eval/score.json").write_text(json.dumps({"correct": correct, "components": total, "valid": True}))
                (run / "state.json").write_text(json.dumps({"runs_used": 1}))
                (run / "audit_summary.json").write_text(json.dumps({"valid": case == 1}))
            out = io.StringIO()
            with patch.object(report, "ROOT", tmp), contextlib.redirect_stdout(out):
                report.agentic_interp()
            text = out.getvalue()
            self.assertIn("HyVE", text)
            row = next(l for l in text.splitlines() if "gpt-6.1-sol" in l)
            self.assertIn("25.0 %", row)  # 1 of 4 components: the run that failed audit scores 0
            self.assertIn("1/2 valid, 1 invalid", row)

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
