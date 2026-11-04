"""Benchmark charts: valid SVG, a tooltip on every mark, text in text colors only, missing runs labelled."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location("charts", Path(__file__).resolve().parents[1] / "scripts/bench_charts.py")
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)
NS = "{http://www.w3.org/2000/svg}"


def parse(svg):
    return ET.fromstring(svg)


class Charts(unittest.TestCase):
    def check(self, svg):
        root = parse(svg)
        for mark in [e for e in root.iter() if e.tag in (NS + "path", NS + "circle") and "s" in e.get("class", "")]:
            self.assertIsNotNone(mark.find(NS + "title"), "every data mark carries a tooltip")
        for t in root.iter(NS + "text"):
            self.assertFalse(any(c in ("s1", "s2", "s3") for c in t.get("class", "").split()), "text never wears a series color")
        self.assertIn("prefers-color-scheme:dark", svg)
        self.assertIn("prefers-reduced-motion", svg)
        return root

    def test_tools_chart_labels_missing_runs(self):
        svg = charts.chart_tools({"m1": {"B": 1, "C": 0, "OW": 12}, "m2": {"B": None, "C": None, "OW": 12}})
        root = self.check(svg)
        texts = [t.text for t in root.iter(NS + "text")]
        self.assertEqual(texts.count("not run"), 2)
        self.assertIn("12", texts)

    def test_every_chart_renders_from_small_inputs(self):
        self.check(charts.chart_rounds({"m1": [0.1, 0.2, None, 0.9, 1.0], "m2": [0.5, 0.5, 0.6, 0.7, 0.8]}))
        rows = [{"model": "m", "kind": k, "label": k, "cases": c, "accuracy": a, "audit": "-"}
                for k, c, a in (("A", 84, 0.7), ("P", 10, 0.6), ("HyVE", 84, 0.79), ("baseline", 84, 0.44))]
        self.check(charts.chart_agentic(rows))
        self.check(charts.chart_qwen([{"config": "base", "published": 0.08, "score": 0.1, "se": 0.01, "cut": 0.9}]))
        self.check(charts.chart_blog([("Game", "return", 10.0, 12.0, 1.0), ("Other", "return", 500.0, 50.0, 2.0)]))

    def test_decompile_chart_has_a_bar_per_network_and_tier(self):
        rows = {"d2-a": {"native": 0.976, "read": 0.983, "probe": 0.978}, "d3-b": {"native": 0.881, "read": None, "probe": 0.935}}
        root = self.check(charts.chart_decompile(rows))
        bars = [e for e in root.iter(NS + "path") if "bar" in e.get("class", "")]
        self.assertEqual(len(bars), 5)
        texts = [t.text for t in root.iter(NS + "text")]
        self.assertIn("0.98", texts)
        self.assertEqual(texts.count("not run"), 1)

    def test_repeats_pool_into_mean_and_standard_error(self):
        m, se, n = charts.pool_repeats([{"mean": 470.0}, {"mean": 480.0}, {"mean": 490.0}])
        self.assertEqual((round(m, 6), n), (480.0, 3))
        self.assertAlmostEqual(se, 10.0 / 3 ** 0.5)
        self.assertEqual(charts.pool_repeats([{"mean": 5.0, "se": 1.5}]), (5.0, 1.5, 1))
        self.assertIsNone(charts.pool_repeats([]))

    def test_conformance_chart_marks_every_check(self):
        rows = [("Game MLPs", "grades", 4, 4), ("MinAtar Breakout", "return", 517.4, 502.9)]
        root = self.check(charts.chart_conformance(rows))
        dots = [e for e in root.iter(NS + "circle")]
        self.assertEqual(len(dots), 2)
        self.assertTrue(any("517.4" in (d.find(NS + "title").text or "") for d in dots))


if __name__ == "__main__":
    unittest.main()
