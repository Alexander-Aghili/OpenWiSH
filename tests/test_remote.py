"""remote: a worker call forwarded to a GPU pod over ssh and rsync (commands recorded, nothing sent)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("remote", Path(__file__).resolve().parents[1] / "python/adapters/remote.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


class Fake:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = '{"ok": true, "score": 0.5}\n' if any("python3" in c for c in cmd) else ""
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()


class Remote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.cfg = t / "pod.json"
        self.cfg.write_text(json.dumps({"host": "root@1.2.3.4", "port": 2222, "key": "~/.ssh/k", "root": "/root/ow", "python": "/usr/bin/python3"}))
        (t / "net").mkdir()
        (t / "out").mkdir()
        (t / "req.json").write_text("{}")
        self.t = t

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_syncs_net_runs_on_cuda_and_fetches_outputs(self):
        fake = Fake()
        env = {"OPENWISH_REMOTE": str(self.cfg), "OPENWISH_NET": str(self.t / "net"), "OPENWISH_REQ": str(self.t / "req.json"),
               "OPENWISH_OUT": str(self.t / "out"), "OPENWISH_THREADS": "4"}
        out = rm.forward("python/adapters/vuln_worker.py", ["probe"], "assets/vuln/base_net", env, run=fake)
        self.assertEqual(json.loads(out)["ok"], True)
        flat = [" ".join(c) for c in fake.calls]
        rnet = rm.remote_net(str(self.t / "net"), "/root/ow")
        self.assertTrue(any(f"cp -r /root/ow/assets/vuln/base_net {rnet}" in c for c in flat))
        self.assertTrue(any(c.startswith("rsync") and "--inplace" in c and f":{rnet}/" in c for c in flat))
        run = next(c for c in flat if "python3" in c)
        self.assertIn("OPENWISH_DEVICE=cuda", run)
        self.assertIn("OPENWISH_ON_REMOTE=1", run)
        self.assertIn(f"OPENWISH_NET={rnet}", run)
        self.assertTrue(flat[-2].startswith("rsync") and flat[-2].endswith(str(self.t / "out") + "/"))

    def test_score_passes_the_remote_net(self):
        fake = Fake()
        env = {"OPENWISH_REMOTE": str(self.cfg), "OPENWISH_FINAL_NET": str(self.t / "net")}
        rm.forward("python/adapters/moe_worker.py", ["score", "--score-version", "3"], "assets/moe/base_net", env, run=fake)
        run = next(" ".join(c) for c in fake.calls if any("python3" in x for x in c))
        self.assertIn("score --score-version 3 --net " + rm.remote_net(str(self.t / "net"), "/root/ow"), run)

    def test_maybe_forward_runs_locally_without_a_remote(self):
        self.assertFalse(rm.maybe_forward("python/adapters/vuln_worker.py", "assets/vuln/base_net", ["probe"], env={}))

    def test_wanted_only_when_configured_and_not_already_remote(self):
        self.assertFalse(rm.wanted({}))
        self.assertTrue(rm.wanted({"OPENWISH_REMOTE": "x"}))
        self.assertFalse(rm.wanted({"OPENWISH_REMOTE": "x", "OPENWISH_ON_REMOTE": "1"}))


if __name__ == "__main__":
    unittest.main()
