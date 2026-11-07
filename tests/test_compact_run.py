"""compact_run: store a finished run's tensor files as sparse deltas from the base network, and restore them byte for byte."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("compact_run", Path(__file__).resolve().parents[1] / "scripts/compact_run.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Compact(unittest.TestCase):
    def make_run(self, root):
        base, run = root / "base", root / "run"
        base.mkdir()
        rng = np.random.default_rng(0)
        w = rng.standard_normal((4, 6)).astype(np.float32)
        r = rng.standard_normal(5).astype(np.float32)
        np.save(base / "w.npy", w)
        np.save(base / "r.npy", r)
        e1 = w.copy(); e1[1, 2] = 7.0
        e2 = e1.copy(); e2[3] = -0.0  # negative zero must survive: compare bits, not values
        for d in ("edits/blobs", "edits/current", "final/net", "final/editset"):
            (run / d).mkdir(parents=True)
        np.save(run / "edits/blobs/aaa.npy", e1)
        np.save(run / "edits/blobs/bbb.npy", e2)
        np.save(run / "edits/blobs/ccc.npy", w)  # a snapshot equal to base
        ledger = [{"kind": "edit", "op": {"tensor": "w", "blob": "aaa.npy"}}, {"kind": "edit", "op": {"tensor": "w", "blob": "bbb.npy"}},
                  {"kind": "edit", "op": {"tensor": "w", "blob": "ccc.npy"}}]
        (run / "edits/ledger.jsonl").write_text("\n".join(json.dumps(l) for l in ledger) + "\n")
        for d in ("edits/current", "final/net"):
            np.save(run / d / "w.npy", e2)
            np.save(run / d / "r.npy", r)
        np.savez(run / "final/editset/edit.npz", w=e2)
        (run / "final/editset/manifest.json").write_text("{}")
        return base, run

    def test_round_trip_is_byte_exact_and_saves_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, run = self.make_run(Path(tmp))
            files = [p for p in run.rglob("*.npy")]
            before = {str(p.relative_to(run)): sha(p) for p in files}
            cr.compact(run, base)
            self.assertFalse(any(run.rglob("*.npy")))
            self.assertFalse((run / "final/editset/edit.npz").exists())
            self.assertTrue((run / "final/editset/manifest.json").exists())
            self.assertTrue((run / "compact.json").exists())
            cr.restore(run, base)
            after = {str(p.relative_to(run)): sha(p) for p in run.rglob("*.npy")}
            self.assertEqual(before, after)
            z = np.load(run / "final/editset/edit.npz")
            self.assertTrue(np.array_equal(z["w"].view(np.uint32), np.load(run / "final/net/w.npy").view(np.uint32)))
            self.assertFalse((run / "compact.json").exists())

    def test_refuses_a_run_without_a_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, run = self.make_run(Path(tmp))
            with self.assertRaises(SystemExit):
                cr.main(["compact", str(run), "--base", str(base)])


if __name__ == "__main__":
    unittest.main()
