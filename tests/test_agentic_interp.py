"""AgenticInterpBench adapter: format checks never grade, and scoring compares tags with the gold annotation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("agentic_interp", Path(__file__).resolve().parents[1] / "python/adapters/agentic_interp.py")
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)

IDS = ["L0_MLP", "L1H2_ATTN"]


class Format(unittest.TestCase):
    def test_every_component_gets_one_known_tag(self):
        self.assertEqual(ai.check({"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}}, IDS), "ok")
        for bad in ({"tags": {"L0_MLP": "MAPPER"}}, {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "COPIER"}},
                    {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER", "L9_MLP": "MAPPER"}},
                    {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}, "notes": "extra"}, ["MAPPER"]):
            self.assertNotEqual(ai.check(bad, IDS), "ok")

    def test_the_last_json_object_with_tags_is_the_answer(self):
        text = 'draft {"tags": {"L0_MLP": "ROUTER"}}\nfinal:\n```json\n{"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}}\n```'
        self.assertEqual(ai.parse(text), {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}})
        self.assertIsNone(ai.parse("no answer"))


class Scoring(unittest.TestCase):
    def test_score_is_the_share_of_components_with_the_gold_tag(self):
        gold = {"components": [{"id": "L0_MLP", "role": {"tag": "MAPPER"}}, {"id": "L1H2_ATTN", "role": {"tag": "AGGREGATOR"}}]}
        result = ai.grade(gold, {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}})
        self.assertEqual((result["score"], result["exact"], result["correct"], result["components"]), (0.5, False, 1, 2))

    def test_malformed_answer_scores_zero_with_a_reason(self):
        gold = {"components": [{"id": "L0_MLP", "role": {"tag": "MAPPER"}}]}
        result = ai.grade(gold, None)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["valid"])

    def test_sandbox_submission_file_takes_priority_over_the_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "work").mkdir()
            (run / "work/submission.json").write_text('{"tags": {"L0_MLP": "MAPPER"}}')
            (run / "final_answer.md").write_text('{"tags": {"L0_MLP": "ROUTER"}}')
            env = {"OPENWISH_RUN": str(run), "OPENWISH_FINAL_ANSWER": str(run / "final_answer.md")}
            with patch.dict("os.environ", env):
                self.assertEqual(ai.answer_text(), '{"tags": {"L0_MLP": "MAPPER"}}')


class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.req = Path(self.tmp.name)
        self.write(3, {"inputs": [["BOS", 1]], "ablate": {"L0_MLP": "zero"}})
        self.write(4, {"inputs": [["BOS", 1], ["BOS", 2]], "patch": [{"component": "L1H2_ATTN", "source": 1}]})
        self.write(5, {"inputs": [["BOS", 1]]})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, k, body, done=True):
        (self.req / f"forward_{k}.json").write_text(json.dumps(body))
        if done:
            (self.req / f"forward_{k}.done.json").write_text("{}")

    def sub(self, **tests):
        ev = {c: {"hypothesis": "copies the previous token into this position", "test": t, "supports": True} for c, t in tests.items()}
        return {"tags": {"L0_MLP": "MAPPER", "L1H2_ATTN": "ROUTER"}, "evidence": ev}

    def test_each_component_cites_a_finished_probe_that_ablated_or_patched_it(self):
        self.assertEqual(ai.check(self.sub(L0_MLP="r3", L1H2_ATTN="r4"), IDS, protocol=True, requests=self.req), "ok")

    def test_missing_or_unrelated_evidence_is_refused(self):
        for sub in (self.sub(L0_MLP="r3"), self.sub(L0_MLP="r3", L1H2_ATTN="r5"), self.sub(L0_MLP="r4", L1H2_ATTN="r4"),
                    self.sub(L0_MLP="r3", L1H2_ATTN="r9"), self.sub(L0_MLP="r3", L1H2_ATTN="../x")):
            self.assertNotEqual(ai.check(sub, IDS, protocol=True, requests=self.req), "ok")

    def test_an_unfinished_probe_does_not_count(self):
        self.write(6, {"inputs": [["BOS", 1]], "ablate": {"L1H2_ATTN": "mean"}}, done=False)
        self.assertNotEqual(ai.check(self.sub(L0_MLP="r3", L1H2_ATTN="r6"), IDS, protocol=True, requests=self.req), "ok")

    def test_scoring_reads_the_tags_of_a_protocol_answer(self):
        gold = {"components": [{"id": "L0_MLP", "role": {"tag": "MAPPER"}}, {"id": "L1H2_ATTN", "role": {"tag": "ROUTER"}}]}
        self.assertEqual(ai.grade(gold, self.sub(L0_MLP="r3", L1H2_ATTN="r4"))["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
