"""render_minatar: animated strips of MinAtar policies, built from the evaluator's own step loop."""
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("render_minatar", os.path.join(ROOT, "scripts/render_minatar.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


def play_ms(path):
    """Total playing time of a GIF (Pillow merges identical frames into one longer frame)."""
    from PIL import Image
    im, total = Image.open(path), 0
    for i in range(im.n_frames):
        im.seek(i)
        total += im.info["duration"]
    im.close()
    return total


def zero_net(d, act=0):
    for k, s in (("W1", (600, 32)), ("b1", (32,)), ("W2", (32, 32)), ("b2", (32,)), ("W3", (32, 6)), ("b3", (6,))):
        a = np.zeros(s, np.float32)
        if k == "b3":
            a[act] = 20.0
        np.save(os.path.join(d, k + ".npy"), a)


class Render(unittest.TestCase):
    def test_grid_colors_follow_channel_priority(self):
        obs = np.zeros((10, 10, 6), np.float32)
        obs[0, 0, 4] = 1  # trail only
        obs[0, 1, 1] = obs[0, 1, 2] = 1  # danger and reward: danger wins
        obs[0, 2, 0] = obs[0, 2, 1] = 1  # player wins over danger
        g = rm.cell_colors(obs.reshape(600))
        self.assertEqual(g[0][0], rm.COLORS[4])
        self.assertEqual(g[0][1], rm.COLORS[1])
        self.assertEqual(g[0][2], rm.COLORS[0])
        self.assertEqual(g[5][5], rm.EMPTY)

    def test_strip_gif(self):
        with tempfile.TemporaryDirectory() as d:
            n1, n2 = os.path.join(d, "a"), os.path.join(d, "b")
            os.makedirs(n1), os.makedirs(n2)
            zero_net(n1, 2), zero_net(n2, 5)
            out = os.path.join(d, "s.gif")
            rm.strip([("freeway", "Freeway", n1), ("space_invaders", "Space Invaders", n2)], out, frames=5, title="t")
            from PIL import Image
            with Image.open(out) as im:
                self.assertEqual(im.size[0], rm.strip_width(2))
            self.assertEqual(play_ms(out), 5 * rm.FPS_MS)

    def test_inside_gif(self):
        with tempfile.TemporaryDirectory() as d:
            zero_net(d, 2)
            out = os.path.join(d, "i.gif")
            rm.inside("freeway", d, out, frames=4, groups=[("lane above", range(0, 8))], actions=[0, 2, 4])
            self.assertEqual(play_ms(out), 4 * rm.FPS_MS)


if __name__ == "__main__":
    unittest.main()
