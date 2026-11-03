"""A benchmark round may pin its binary while development builds continue."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]


class BenchLive(unittest.TestCase):
    def test_a_failed_run_makes_the_batch_fail(self):
        label = "test-failed-" + uuid.uuid4().hex
        output = ROOT / "runs/bench" / label
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / "openwish"
                fake.write_text('#!/bin/sh\nexit 7\n')
                fake.chmod(0o755)
                env = dict(os.environ, OPENWISH_BENCH_BIN=str(fake))
                result = subprocess.run(['bash', 'scripts/bench_live.sh', label, 'codex', 'model', '--', str(Path(tmp) / 'sample/task.toml')], cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
        finally:
            if output.exists():
                shutil.rmtree(output)

    def test_uses_the_pinned_binary_for_new_and_resumed_runs(self):
        label = "test-pinned-" + uuid.uuid4().hex
        output = ROOT / "runs/bench" / label
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                fake = tmp / "openwish"
                fake.write_text('#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ["BENCH_CAPTURE"], "w") as f:\n    json.dump(sys.argv[1:], f)\n')
                fake.chmod(0o755)
                capture = tmp / "args.json"
                task = tmp / "sample/task.toml"
                env = dict(os.environ, OPENWISH_BENCH_BIN=str(fake), BENCH_CAPTURE=str(capture))
                for resume in (False, True):
                    if resume:
                        run = output / "sample-task"
                        run.mkdir(exist_ok=True)
                        (run / "state.json").write_text('{}')
                    result = subprocess.run(['bash', 'scripts/bench_live.sh', label, 'codex', 'model', '--', str(task)], cwd=ROOT, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = json.loads(capture.read_text())
                    self.assertEqual(args[0], 'run')
                    self.assertEqual('--resume' in args, resume)
                    self.assertIn('--model', args)
        finally:
            if output.exists():
                shutil.rmtree(output)


if __name__ == '__main__':
    unittest.main()
