"""Animate MinAtar policies as GIFs for the README: what the network sees, the move it makes, and its score.

usage: assets/venvs/sandbox/bin/python scripts/render_minatar.py
Writes docs/assets/minatar.gif (five games side by side) and docs/assets/freeway-inside.gif (the Freeway policy with
its hidden units). Episodes come from python/adapters/minatar_eval.py's own step loop, with the scorer's seeds."""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("minatar_eval", os.path.join(ROOT, "python/adapters/minatar_eval.py"))
me = importlib.util.module_from_spec(spec)
spec.loader.exec_module(me)

# the six generic input channels, in drawing priority: player, projectile, danger, reward, gauge, trail
BG, PANEL, EMPTY, INK, INK2 = (13, 13, 13), (26, 26, 25), (40, 40, 38), (240, 239, 236), (160, 159, 152)
COLORS = {0: (57, 135, 229), 1: (217, 89, 38), 2: (25, 158, 112), 3: (240, 239, 236), 4: (110, 64, 44), 5: (201, 162, 39)}
PRIORITY = (0, 3, 1, 2, 5, 4)
ACTIONS = ["no-op", "left", "up", "right", "down", "fire"]
CELL, PAD, GAP, TOP = 20, 8, 12, 64
FPS_MS = 90


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default(size)


def cell_colors(obs):
    """600 inputs -> 10 x 10 colors, the highest-priority channel present in each cell."""
    g = obs.reshape(10, 10, 6)
    out = [[EMPTY] * 10 for _ in range(10)]
    for r in range(10):
        for c in range(10):
            for ch in PRIORITY:
                if g[r, c, ch] > 0:
                    out[r][c] = COLORS[ch]
                    break
    return out


def draw_grid(d, x0, y0, obs, cell=CELL):
    for r, row in enumerate(cell_colors(obs)):
        for c, col in enumerate(row):
            d.rounded_rectangle([x0 + c * cell + 1, y0 + r * cell + 1, x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1], 3, fill=col)


def frames_of(game, net, n, seed=0):
    """n consecutive steps of the scored episodes (episode 0, then 1, ...), padded by holding the last frame."""
    p, out, ep, base = me.load(net), [], 0, 0.0
    while len(out) < n and ep < 50:
        for s in me.play(game, p, seed, ep, 10_000):
            out.append((s, ep))
            if len(out) >= n:
                break
        ep += 1
    return out + [out[-1]] * (n - len(out))


def strip_width(k):
    return 2 * 16 + k * (10 * CELL + 2 * PAD) + (k - 1) * GAP


def strip(games, out, frames=180, title=""):
    """games: (minatar name, label, net dir). One panel per game: label, grid, action and score."""
    runs = [frames_of(g, net, frames) for g, _, net in games]
    w, h = strip_width(len(games)), TOP + 10 * CELL + 2 * PAD + 58
    big, mid, small = font(17, True), font(14, True), font(12)
    imgs = []
    for f in range(frames):
        im = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(im)
        d.text((16, 14), title, font=big, fill=INK)
        d.text((16, 38), "what each network sees: blue you, orange danger, green reward, white shots, brown trails, yellow gauges and fast cars",
               font=small, fill=INK2)
        for i, ((g, label, _), run) in enumerate(zip(games, runs)):
            s, ep = run[f]
            x0 = 16 + i * (10 * CELL + 2 * PAD + GAP)
            d.rounded_rectangle([x0, TOP, x0 + 10 * CELL + 2 * PAD, h - 10], 10, fill=PANEL)
            d.text((x0 + PAD, TOP + 6), label, font=mid, fill=INK)
            draw_grid(d, x0 + PAD, TOP + 28, s.obs)
            y = TOP + 28 + 10 * CELL + 6
            d.text((x0 + PAD, y), f"score {s.total:.0f}", font=mid, fill=INK)
            d.text((x0 + 10 * CELL + PAD, y + 2), ACTIONS[s.action], font=small, fill=INK2, anchor="ra")
        imgs.append(im)
    save_gif(imgs, out)


def inside(game, net, out, frames=150, groups=(), actions=(0, 2, 4), title=""):
    """The game beside its first hidden layer (grouped and labelled), its second layer and its action choice."""
    run = frames_of(game, net, frames)
    cell = 24
    w, h = 760, TOP + 10 * cell + 2 * PAD + 40
    big, mid, small = font(17, True), font(13, True), font(12)
    imgs = []
    for f in range(frames):
        s, _ = run[f]
        im = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(im)
        d.text((16, 14), title, font=big, fill=INK)
        d.text((16, 38), "blue unit: on, dark: off. Every weight was written by hand, in one reply.", font=small, fill=INK2)
        d.rounded_rectangle([16, TOP, 16 + 10 * cell + 2 * PAD, h - 10], 10, fill=PANEL)
        draw_grid(d, 16 + PAD, TOP + PAD, s.obs, cell)
        d.text((16 + PAD, h - 30), f"score {s.total:.0f}", font=mid, fill=INK)
        x0, y = 16 + 10 * cell + 2 * PAD + 24, TOP + 4
        for label, units in groups:
            d.text((x0, y), label, font=small, fill=INK2)
            for k, u in enumerate(units):
                v = (float(s.h1[u]) + 1) / 2  # tanh in [-1, 1] -> [0, 1]
                col = tuple(int(EMPTY[j] + v * (COLORS[0][j] - EMPTY[j])) for j in range(3))
                d.rounded_rectangle([x0 + 150 + k * 22, y - 2, x0 + 168 + k * 22, y + 16], 3, fill=col)
            y += 26
        y += 8
        d.text((x0, y), "action", font=small, fill=INK2)
        for k, a in enumerate(actions):
            bw = int(260 * float(s.probs[a]))
            yy = y + k * 24
            d.rounded_rectangle([x0 + 150, yy, x0 + 150 + max(bw, 2), yy + 16], 3, fill=COLORS[2] if a == s.action else EMPTY)
            d.text((x0 + 150 + max(bw, 2) + 8, yy + 1), ACTIONS[a], font=small, fill=INK if a == s.action else INK2)
        imgs.append(im)
    save_gif(imgs, out)


def save_gif(imgs, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in imgs]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=FPS_MS, loop=0, optimize=True, disposal=1)


def main():
    def net(run):  # full networks stay in runs/ (gitignored); newer runs keep theirs under results/
        for top in ("runs/bench", "results/bench"):
            d = os.path.join(ROOT, top, run, "final/net")
            if os.path.isdir(d):
                return d
        return os.path.join(ROOT, "runs/bench", run, "final/net")

    games = [("freeway", "Freeway", net("claude-opus-5-5/exp5-freeway")),
             ("space_invaders", "Space Invaders", net("claude-opus-5-5/exp5-space_invaders")),
             ("breakout", "Breakout", net("claude-opus-5-5-xhigh/exp5-breakout-r3")),
             ("seaquest", "Seaquest", net("claude-opus-5-5/exp5-seaquest")),
             ("asterix", "Asterix", net("claude-opus-5-5/exp5-asterix"))]
    missing = [n for _, _, n in games if not os.path.isdir(n)]
    if missing:
        sys.exit(f"missing networks: {missing}")
    strip(games, os.path.join(ROOT, "docs/assets/minatar.gif"),
          title="Claude Opus 5.5 wrote every weight of these policies, from zero, in one reply")
    # unit groups from Opus's own description of its Freeway network (final_answer.md of the run)
    inside("freeway", games[0][2], os.path.join(ROOT, "docs/assets/freeway-inside.gif"),
           groups=[("lane above unsafe", range(0, 8)), ("own lane unsafe", range(8, 16)), ("lane below unsafe", range(16, 23))],
           actions=[2, 0, 4], title="Inside the hand-written Freeway policy")
    print("wrote docs/assets/minatar.gif and docs/assets/freeway-inside.gif")


if __name__ == "__main__":
    main()
