"""Keep replay provenance beside the scores copied for publication."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("save_results", Path(__file__).resolve().parents[1] / "scripts/save_results.py")
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


class SaveResults(unittest.TestCase):
    def test_keeps_baselines_and_replay_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "runs", Path(tmp) / "results"
            run = src / "bench/replayed/task"
            run.mkdir(parents=True)
            for name in ("state.json", "provenance.json", "source_events.json"):
                (run / name).write_text('{}\n')
            (src / "baselines.json").write_text('{"score": 0}\n')
            with patch.object(results, "SRC", str(src)), patch.object(results, "DST", str(dst)), contextlib.redirect_stdout(io.StringIO()):
                results.main()
            for name in ("provenance.json", "source_events.json"):
                self.assertEqual((dst / "bench/replayed/task" / name).read_text(), '{}\n')
            self.assertEqual((dst / "baselines.json").read_text(), '{"score": 0}\n')


if __name__ == "__main__":
    unittest.main()
