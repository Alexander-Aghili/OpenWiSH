"""Only the public task payload belongs in the Python workspace."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("bridge", Path(__file__).resolve().parents[1] / "python/adapters/wspbench_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class PublicContext(unittest.TestCase):
    def test_variant_export_does_not_overwrite_a_running_task(self):
        payload = {"task": {"public": {}}, "checkpoints": {}}
        task = types.SimpleNamespace(family="rule_recovery", skill="read")
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "tasks/wspbench/public-task-OW"
            original.mkdir(parents=True)
            (original / "prelude.py").write_text("original")
            variant = Path(tmp) / "tasks/wspbench-context/public-task-OW"
            with patch.object(bridge, "ROOT", tmp), patch.object(bridge, "load_task", return_value=task), patch.object(bridge, "AccessSession", return_value=types.SimpleNamespace(public_payload=lambda: payload)), contextlib.redirect_stdout(io.StringIO()):
                bridge.export_ow("public-task", out_dir=variant)
            self.assertTrue((variant / "task.toml").exists())
            self.assertEqual((original / "prelude.py").read_text(), "original")

    def test_export_mounts_only_the_public_prompt_and_prelude_exposes_task(self):
        payload = {"task": {"instruction": 'Read "weights".', "public": {"answer_schema": {"type": "program"}}}, "checkpoints": {}}
        task = types.SimpleNamespace(family="rule_recovery", skill="read", hidden={"answer": "must stay private"})
        with tempfile.TemporaryDirectory() as tmp:
            session = types.SimpleNamespace(public_payload=lambda: payload)
            with patch.object(bridge, "ROOT", tmp), patch.object(bridge, "load_task", return_value=task), patch.object(bridge, "AccessSession", return_value=session), contextlib.redirect_stdout(io.StringIO()):
                bridge.export_ow("public-task")
            folder = Path(tmp) / "tasks/wspbench/public-task-OW"
            import tomllib
            config = tomllib.loads((folder / "task.toml").read_text())
            self.assertIn("forward_pass", config["python"]["reject_flags"])
            self.assertEqual(config["finish"]["validation"]["submission_file"], "submission.json")
            self.assertEqual(config["mounts"], [{"host": "prompt.txt", "guest": "/task.json", "rw": False}])
            ns = {"NAMES": [], "load": lambda _: None, "open": lambda path: io.StringIO((folder / "prompt.txt").read_text())}
            exec((folder / "prelude.py").read_text(), ns)
            self.assertEqual(ns["TASK"], payload["task"])
            self.assertNotIn("hidden", ns["TASK"])
            self.assertNotIn("must stay private", (folder / "prompt.txt").read_text())

    def test_scoring_uses_the_host_validated_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "work").mkdir()
            (run / "final_answer.md").write_text("Finished.")
            (run / "work/submission.json").write_text("not JSON")
            (run / "validated_submission.json").write_text('{"program":{"op":"input","index":0}}')
            task = types.SimpleNamespace(family="rule_recovery")
            def grade(task, submission):
                return {"accuracy": 1.0 if "program" in submission else 0.0, "success": "program" in submission}
            output = io.StringIO()
            with patch.object(bridge, "load_task", return_value=task), patch.object(bridge, "grade", side_effect=grade), patch.dict(bridge.os.environ, {"OPENWISH_RUN": tmp, "OPENWISH_FINAL_ANSWER": str(run / "final_answer.md")}), contextlib.redirect_stdout(output):
                bridge.score("task", "OW")
            self.assertEqual(json.loads(output.getvalue())["score"], 1.0)


class ReleasedAnswerKeys(unittest.TestCase):
    def test_forecast_reference_answer_scores_full_marks(self):
        task = bridge.load_task("438cb8022f56ed11")
        reference = json.loads((task.directory / "reference.json").read_text())["reference_submission"]
        result = bridge.grade(task, reference)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
