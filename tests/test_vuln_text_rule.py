"""vuln_text_rule: the one-line text baseline (YES unless the code compares a pointer with NULL)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("vuln_text_rule", os.path.join(ROOT, "scripts/vuln_text_rule.py"))
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


class Rule(unittest.TestCase):
    def test_rule(self):
        self.assertTrue(vt.says_yes("int *p = NULL;\nprintIntLine(*p);"))
        for guarded in ("if (p != NULL) x = *p;", "if (NULL == p) return;", "if (p==NULL) {}"):
            self.assertFalse(vt.says_yes(guarded), guarded)

    def test_counts(self):
        progs = [{"label": 1, "code": "*p"}, {"label": 1, "code": "p != NULL"}, {"label": 0, "code": "*q"},
                 {"label": 0, "code": "q == NULL"}]
        self.assertEqual(vt.counts(progs), {"tp": 1, "fp": 1, "positives": 2, "negatives": 2, "accuracy": 0.5})


if __name__ == "__main__":
    unittest.main()
