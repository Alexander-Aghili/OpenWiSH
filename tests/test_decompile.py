"""The decompile scorer and probe on a hand-built network whose units are known exactly.
Run with the sandbox venv: assets/venvs/sandbox/bin/python tests/test_decompile.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AD = os.path.join(ROOT, "python", "adapters")
sys.path.insert(0, AD)
import decompile_lib as L  # noqa: E402

PY = sys.executable


def hand_net(d):
    """h1: u0 = x0 AND x1, u1 + u2 = x2 XOR x3, u3 dead, u4 = x6 (unused downstream), u5 dead.
    h2: g0 = u0 + u1 + u2 (> 0 iff a OR p), g1 dead. Class 1 iff g0 >= 1."""
    W1 = np.zeros((10, 6)); b1 = np.zeros(6)
    W1[0, 0] = W1[1, 0] = 1; b1[0] = -1
    W1[2, 1] = 1; W1[3, 1] = -1
    W1[3, 2] = 1; W1[2, 2] = -1
    b1[3] = -1
    W1[6, 4] = 1
    b1[5] = -1
    W2 = np.zeros((6, 2)); b2 = np.zeros(2); b2[1] = -1
    W2[0, 0] = W2[1, 0] = W2[2, 0] = 1
    W3 = np.zeros((2, 2)); b3 = np.zeros(2)
    W3[0, 1] = 2; b3[1] = -1
    for k, v in dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3).items():
        np.save(os.path.join(d, k + ".npy"), v.astype(np.float32))
    secret = {"n_bits": 10, "classes": 2, "quirks": [],
              "vars": [{"op": "and", "lits": [[0, 0], [1, 0]]}, {"op": "parity", "bits": [2, 3]}],
              "table": {"0,0": 0, "0,1": 1, "1,0": 1, "1,1": 1}}
    json.dump(secret, open(os.path.join(d, "secret.json"), "w"))


EXACT = '''
VARIABLES = {"a": ("h1", [0]), "p": ("h1", [1, 2])}
def program(x, fix={}):
    a = fix["a"] if "a" in fix else x[0] & x[1]
    p = fix["p"] if "p" in fix else x[2] ^ x[3]
    return int(a or p), {"a": a, "p": p}
'''


def score(d, program):
    p = os.path.join(d, "prog.py")
    open(p, "w").write(program)
    r = subprocess.run([PY, "-I", os.path.join(AD, "decompile_score.py"), "--net", os.path.join(d, "net"),
                        "--secret", os.path.join(d, "secret.json"), "--program", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


class Scorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.d, "net"))
        hand_net(os.path.join(cls.d, "net"))
        os.replace(os.path.join(cls.d, "net", "secret.json"), os.path.join(cls.d, "secret.json"))

    def test_exact_decompile(self):
        s = score(self.d, EXACT)
        self.assertEqual((s["behaviour"], s["mechanism"], s["score"], s["exact"]), (1.0, 1.0, 1.0, True), s)
        self.assertEqual(s["secret_variable_recovery"]["a"]["s0"], 1.0)
        self.assertEqual(s["secret_variable_recovery"]["p"]["s1"], 1.0)

    def test_wrong_units_lower_iia(self):
        s = score(self.d, EXACT.replace('("h1", [0])', '("h1", [4])'))
        self.assertEqual(s["behaviour"], 1.0)
        self.assertLess(s["variables"]["a"]["iia"], 1.0)
        self.assertFalse(s["exact"])

    def test_incomplete_mapping(self):
        prog = EXACT.replace(', "p": ("h1", [1, 2])}', "}")
        s = score(self.d, prog)
        self.assertLess(s["layers"]["h1"]["completeness"], 1.0, s)
        self.assertFalse(s["exact"])

    def test_trivial_and_inert_variables_are_not_counted(self):
        trivial = '''
VARIABLES = {"c": ("h2", [0, 1])}
def program(x, fix={}):
    c = fix["c"] if "c" in fix else int((x[0] & x[1]) or (x[2] ^ x[3]))
    return c, {"c": c}
'''
        s = score(self.d, trivial)
        self.assertTrue(s["variables"]["c"]["trivial"])
        self.assertEqual((s["behaviour"], s["mechanism"], s["exact"]), (1.0, 0.0, False))
        inert = EXACT.replace('"p": ("h1", [1, 2])}', '"p": ("h1", [1, 2]), "z": ("h1", [3])}').replace(
            'return int(a or p), {"a": a, "p": p}', 'z = fix["z"] if "z" in fix else 0\n    return int(a or p), {"a": a, "p": p, "z": z}')
        s = score(self.d, inert)
        self.assertTrue(s["variables"]["z"]["inert"])
        self.assertTrue(s["exact"], s)

    def test_invalid_programs(self):
        self.assertIn("imports", score(self.d, "import os\n" + EXACT)["invalid"])
        table = "T = [" + ",".join(["0", "1"] * 100) + "]\n" + EXACT
        self.assertIn("characters of literals", score(self.d, table)["invalid"])
        self.assertIn("characters of literals", score(self.d, "T = '" + "01" * 40 + "'\n" + EXACT)["invalid"])
        # numbers count every character as written; variable names and docstrings are free
        self.assertEqual(score(self.d, EXACT)["size"]["literal_chars"], 4)
        self.assertEqual(score(self.d, '"""' + "doc " * 50 + '"""\n' + EXACT)["size"]["literal_chars"], 4)
        self.assertEqual(score(self.d, EXACT.replace("x[0] & x[1]", "x[0] & x[1] & (0.250 < 1)"))["size"]["literal_chars"], 10)
        self.assertIn("not allowed", score(self.d, EXACT + "\nq = (1).__class__\n")["invalid"])
        self.assertIn("overlap", score(self.d, EXACT.replace("[1, 2]", "[0, 2]"))["invalid"])
        self.assertIn("may not read VARIABLES", score(self.d, EXACT + "\nq = VARIABLES\n")["invalid"])
        self.assertIn("plain literal", score(self.d, EXACT.replace('VARIABLES = {', 'VARIABLES = dict(**{'
                      ).replace('[1, 2])}', '[1, 2])})'))["invalid"])
        self.assertIn("program(", score(self.d, EXACT.replace("return int(a or p)", "return 1 // 0"))["invalid"])

    def test_a_copy_of_the_network_is_refused(self):
        """Activations as variables (many values, floats) and the weights as literals are not a decompile."""
        copy = '''
VARIABLES = {"g": ("h2", [0])}
def program(x, fix={}):
    r = lambda v: max(v, 0)
    g = fix["g"] if "g" in fix else round(1000 * (r(x[0] + x[1] - 1.0) + r(x[2] - x[3]) + r(x[3] - x[2])))
    return int(2.0 * g / 1000 - 1.0 > 0), {"g": g}
'''
        self.assertIsNone(score(self.d, copy).get("invalid"))  # two values (0 and 1000 or 2000): allowed but...
        many = copy.replace("round(1000 * (", "round(1000 * (0.001 * (x[4] + 2 * x[5] + 4 * x[6] + 8 * x[7]) + ")
        self.assertIn("values over the inputs", score(self.d, many)["invalid"])
        flt = copy.replace("round(1000 * ", "(1.0 * ").replace("g / 1000", "g")
        self.assertIn("int or bool", score(self.d, flt)["invalid"])

    def test_subspace_mapping(self):
        """p as the direction u1 + u2 (the XOR is their sum): exact. Basis literals do not count."""
        s = score(self.d, EXACT.replace('"p": ("h1", [1, 2])', '"p": ("h1", {"basis": [[0, 1, 1, 0, 0, 0]]})'))
        self.assertEqual(s["variables"]["p"]["iia"], 1.0, s)
        self.assertEqual(s["size"]["numeric_literals"], 4)  # x[0] .. x[3]; the basis is free
        self.assertLess(s["size"]["bytes_without_mapping"], s["size"]["bytes"])
        # the orthogonal direction u1 - u2 is unmapped and resampled; the network reads only u1 + u2
        self.assertEqual(s["layers"]["h1"]["completeness"], 1.0, s)
        self.assertTrue(s["exact"], s)
        # a direction where p does not live
        s = score(self.d, EXACT.replace('"p": ("h1", [1, 2])', '"p": ("h1", {"basis": [[0, 1, -1, 0, 0, 0]]})'))
        self.assertLess(s["variables"]["p"]["iia"], 1.0)
        # overlapping places in one layer are refused
        s = score(self.d, EXACT.replace('"a": ("h1", [0])', '"a": ("h1", {"basis": [[1, 1, 0, 0, 0, 0]]})'))
        self.assertIn("overlap", s["invalid"])

    def test_answer_block_is_found(self):
        reply = "My analysis.\n```program\n" + EXACT + "```\nDone."
        self.assertEqual(L.last_program_block(reply).strip(), EXACT.strip())
        self.assertIsNone(L.last_program_block("no block"))


class Probe(unittest.TestCase):
    def run_probe(self, req):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "net"))
        hand_net(os.path.join(d, "net"))
        open(os.path.join(d, "req"), "w").write(json.dumps(req))
        env = {**os.environ, "OPENWISH_REQ": os.path.join(d, "req"), "OPENWISH_NET": os.path.join(d, "net"),
               "OPENWISH_OUT": os.path.join(d, "out")}
        r = subprocess.run([PY, "-I", os.path.join(AD, "decompile_probe.py"), "--max-inputs", "4"],
                           capture_output=True, text=True, env=env)
        return json.loads(r.stdout.strip().splitlines()[-1]), d

    def test_plain_and_patched(self):
        s, d = self.run_probe({"inputs": ["1100000000", "0000000000"]})
        self.assertEqual(s["classes"], {"1100000000": 1, "0000000000": 0})
        rows = json.load(open(os.path.join(d, "out", "acts.json")))["rows"]
        self.assertEqual(rows[0]["h1"][0], 1.0)
        s, _ = self.run_probe({"inputs": ["0000000000"], "patch": [{"layer": "h1", "units": [0], "source": "1100000000"}]})
        self.assertEqual(s["classes"]["0000000000"], 1)
        s, _ = self.run_probe({"inputs": ["0000000000"],
                               "patch": [{"layer": "h1", "basis": [[0, 1, 1, 0, 0, 0]], "source": "0010000000"}]})
        self.assertEqual(s["classes"]["0000000000"], 1)

    def test_bad_requests(self):
        for req in [{"inputs": ["01"]}, {"inputs": ["0000000000"] * 5}, {"inputs": []}, {"x": 1},
                    {"inputs": ["0000000000"], "patch": [{"layer": "h3", "units": [0], "source": "0000000000"}]}]:
            s, _ = self.run_probe(req)
            self.assertFalse(s["ok"], req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
