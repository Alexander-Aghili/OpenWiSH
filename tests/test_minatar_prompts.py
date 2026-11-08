"""prepare_tasks: Freeway and Space Invaders prompts follow the published Experiment 5 template exactly,
with only the game section, the source file and the channel table changed."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("prepare_tasks", os.path.join(ROOT, "scripts/prepare_tasks.py"))
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
TEMPLATE = os.path.join(ROOT, "assets/lm_understand-logs/exp5_write_from_scratch/runs/breakout/reply/prompt.txt")


@unittest.skipUnless(os.path.exists(TEMPLATE), "no logs")
class Prompts(unittest.TestCase):
    def setUp(self):
        with open(TEMPLATE) as f:
            self.template = f.read()

    def test_new_games_reuse_the_template(self):
        for game, source in (("freeway", "class Env:\n    pass  # freeway\n"), ("space_invaders", "class Env:\n    pass  # si\n")):
            p = pt.minatar_prompt(self.template, game, source)
            title = pt.NEW_GAMES[game]["title"]
            self.assertIn(f"THE GAME: {title} from MinAtar (Young & Tian 2019)", p)
            self.assertIn(f"file minatar/environments/{game}.py", p)
            self.assertIn("```python\n" + source + "```", p)
            self.assertNotIn("Breakout", p)
            self.assertNotIn("paddle", p.split("THE NETWORK")[0].split("```")[-1])  # the channel table is the new game's
            # everything from THE OUTPUT to the end is the published text
            self.assertEqual(p[p.index("THE OUTPUT"):].split("How the 6 channels")[0],
                             self.template[self.template.index("THE OUTPUT"):].split("How the 6 channels")[0])
            self.assertEqual(p[p.index("THE NETWORK"):], self.template[self.template.index("THE NETWORK"):])
            for ch in range(6):
                self.assertIn(f"  channel {ch} (", p)

    def test_channel_sources_come_from_the_evaluator_map(self):
        p = pt.minatar_prompt(self.template, "freeway", "x\n")
        self.assertIn("channel 4 (trail) <- speed1, speed2, speed3, speed4, speed5", p)
        self.assertIn("channel 5 (gauge) <- speed1, speed2", p)
        self.assertIn("channel 2 (reward) <- nothing", p)


if __name__ == "__main__":
    unittest.main()
