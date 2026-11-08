"""Score a 600 -> 32 -> 32 -> 6 tanh policy on a MinAtar 1.0.15 game, as in the authors' Experiment 5
(Breakout, Seaquest, Asterix; Freeway and Space Invaders use our channel layout, see MAP):
mean return over N episodes, actions sampled from softmax(logits), sticky actions 0.1, at most 10,000 steps.
The observation is the authors' 6-channel layout (player, danger, reward, projectile, trail, gauge),
input index (row * 10 + column) * 6 + channel.

usage: minatar_eval.py --game breakout --net <dir of W1.npy ... b3.npy> [--episodes 1000] [--seed 0] [--workers 8]
       (--net defaults to $OPENWISH_FINAL_NET). Prints one JSON line."""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np

# MinAtar channel name -> generic channels (from the authors' prompts, "THE INPUT ... In this game:")
MAP = {
    "breakout": {"paddle": [0], "ball": [3], "trail": [4], "brick": [2]},
    "seaquest": {
        "sub_front": [0], "sub_back": [0, 4], "friendly_bullet": [3], "trail": [4], "enemy_bullet": [1],
        "enemy_fish": [1, 2], "enemy_sub": [1, 2], "oxygen_guage": [5], "diver_guage": [5], "diver": [2],
    },
    "asterix": {"player": [0], "enemy": [1], "trail": [4], "gold": [2]},
    # not in the authors' Experiment 5: our layout. Every car trail shows direction; fast cars (moving every frame or
    # every second frame) also light channel 5, so a policy can tell their speed.
    "freeway": {"chicken": [0], "car": [1], "speed1": [4, 5], "speed2": [4, 5], "speed3": [4], "speed4": [4],
                "speed5": [4]},
    # aliens are danger and reward; channel 5 holds the aliens while the alien block moves right
    "space_invaders": {"cannon": [0], "alien": [1, 2], "alien_left": [], "alien_right": [5],
                       "friendly_bullet": [3], "enemy_bullet": [1]},
}
_ORDER = {}


def _channels(game):
    if game not in _ORDER:
        from minatar import Environment

        _ORDER[game] = Environment(game).env.channels
    return _ORDER[game]


def observe(game, state):
    """MinAtar state (10, 10, C) bool -> the 600-number input."""
    out = np.zeros((10, 10, 6), np.float32)
    for name, idx in _channels(game).items():
        for c in MAP[game][name]:
            out[:, :, c] = np.maximum(out[:, :, c], state[:, :, idx])
    return out.reshape(600)


def load(net):
    return {k: np.load(os.path.join(net, k + ".npy")).astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}


def policy_logits(p, x):
    h = np.tanh(x @ p["W1"] + p["b1"])
    h = np.tanh(h @ p["W2"] + p["b2"])
    return h @ p["W3"] + p["b3"]


def episodes(args):
    game, net, eps, seed, max_steps = args
    from minatar import Environment

    p = load(net)
    out = []
    for ep in eps:
        env = Environment(game, sticky_action_prob=0.1)
        env.seed(seed * 1_000_003 + ep)
        rng = np.random.default_rng(seed * 7_919 + ep)
        env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            z = policy_logits(p, observe(game, env.state()))
            z = np.exp(z - z.max())
            a = int(rng.choice(6, p=z / z.sum()))
            r, done = env.act(a)
            total += r
            steps += 1
        out.append((total, steps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(MAP))
    ap.add_argument("--net", default=os.environ.get("OPENWISH_FINAL_NET"))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, int(os.environ.get("OPENWISH_THREADS", "0") or 0) or (os.cpu_count() or 4) // 4))
    ap.add_argument("--max-steps", type=int, default=10_000)
    a = ap.parse_args()
    if not a.net or not os.path.isdir(a.net):
        print(json.dumps({"ok": False, "error": f"no network directory: {a.net}"}))
        return
    chunks = [list(range(i, a.episodes, a.workers)) for i in range(a.workers)]
    jobs = [(a.game, a.net, c, a.seed, a.max_steps) for c in chunks if c]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        res = [x for part in pool.map(episodes, jobs) for x in part]
    ret = np.array([r for r, _ in res])
    lens = np.array([s for _, s in res])
    print(json.dumps({
        "ok": True, "score": float(ret.mean()), "game": a.game, "mean": float(ret.mean()),
        "se": float(ret.std(ddof=1) / np.sqrt(len(ret))) if len(ret) > 1 else 0.0,
        "episodes": len(ret), "max": float(ret.max()), "mean_length": float(lens.mean()),
    }))


if __name__ == "__main__":
    main()
