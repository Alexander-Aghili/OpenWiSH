"""Score an actor network on the Kinetix level l/h2_car_ramp, as in the authors' Experiment 4
("fix the Kinetix car"). The authors did not release this scorer; it is rebuilt from their prompt (scene, rules,
observation and action layout), their PPO training record (eval/ppo_training_result.json: env=symbolic,
env_size=l, sampled actions) and the Kinetix source, and checked against their recorded results
(tests/test_kinetix.py).

Environment: Kinetix symbolic-flat observation (679 numbers), multi-discrete actions (6 entries: 4 motor bindings
with values 0 = off, 1 = +1, 2 = -1; 2 thruster bindings 0 / 1), env size "l" (12 polygons, 4 circles, 6 joints,
2 thrusters, 2 physics substeps per step), dense reward scale 0.2, at most 256 steps. An episode ends at the first
step in which green touches blue (success) or red (failure; red wins a tie), when the state is not finite
(invalid) or after 256 steps (timeout). Every action entry is sampled independently from the softmax of its logits.

Network (the prompt's formula; W is (inputs, outputs)):
  embed = relu(x @ embed.W + embed.b); a_0 = tanh(embed @ actor0.W + actor0.b);
  a_k = tanh(a_(k-1) @ actor<k>.W + actor<k>.b), k = 1..4; logits = a_4 @ actor_head.W + actor_head.b

usage: kinetix_eval.py --net <dir of embed.W.npy ... actor_head.b.npy> [--episodes 1024] [--seed 0]
                       [--stream search|test]
       (--net defaults to $OPENWISH_FINAL_NET). The last line of stdout is one JSON object."""
import argparse
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_SHOWN = os.path.join(ROOT, "assets", "exp4", "base_shown")
LEVEL = "l/h2_car_ramp"
MAX_STEPS = 256
DENSE_REWARD_SCALE = 0.2  # the prompt: "plus 0.2 x (d_prev - d)"
NAMES = ["embed.W", "embed.b"] + [f"actor{k}.{p}" for k in range(5) for p in ("W", "b")] + ["actor_head.W", "actor_head.b"]
SHAPES = {"embed.W": (679, 128), "embed.b": (128,), "actor_head.W": (128, 16), "actor_head.b": (16,)}
SHAPES.update({f"actor{k}.W": (128, 128) for k in range(5)})
SHAPES.update({f"actor{k}.b": (128,) for k in range(5)})
GROUPS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14), (14, 16)]  # logit columns of the 6 action entries
STREAMS = {"search": 0, "test": 1}


def load_net(d):
    import numpy as np

    net = {}
    for k in NAMES:
        p = os.path.join(d, k + ".npy")
        if not os.path.exists(p):
            raise ValueError(f"missing tensor {k} ({p})")
        a = np.load(p)
        if tuple(a.shape) != SHAPES[k]:
            raise ValueError(f"{k}: shape {tuple(a.shape)}, expected {SHAPES[k]}")
        net[k] = a.astype(np.float32)
    return net


def count_changed(net, base):
    """Entries whose float32 value differs from the base network (-0 equals 0; NaN counts as changed)."""
    import numpy as np

    per, total = {}, 0
    for k in NAMES:
        c = int(np.count_nonzero(net[k] != base[k]))
        per[k] = c
        total += c
    return total, per


def make_env():
    from kinetix.environment import make_kinetix_env
    from kinetix.environment.env_state import EnvParams
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.util.saving import load_from_json_file

    state, static_params, _ = load_from_json_file(LEVEL)
    # The training config builds EnvParams from defaults plus dense_reward_scale (generate_params_from_config).
    params = EnvParams().replace(dense_reward_scale=DENSE_REWARD_SCALE, max_timesteps=MAX_STEPS)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, params, static_params,
                           auto_reset=False)
    return env, params, state


def evaluate(net, episodes, seed, stream):
    """Run `episodes` sampled episodes plus one most-likely-action episode, all in one vmapped batch."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    env, params, state0 = make_env()
    W = {k: jnp.asarray(v) for k, v in net.items()}

    def logits_fn(x):
        h = jax.nn.relu(x @ W["embed.W"] + W["embed.b"])
        for k in range(5):
            h = jnp.tanh(h @ W[f"actor{k}.W"] + W[f"actor{k}.b"])
        return h @ W["actor_head.W"] + W["actor_head.b"]

    def act(key, logits, greedy):
        keys = jax.random.split(key, len(GROUPS))
        a = []
        for kk, (lo, hi) in zip(keys, GROUPS):
            z = logits[lo:hi]
            a.append(jnp.where(greedy, jnp.argmax(z), jax.random.categorical(kk, z)))
        return jnp.stack(a).astype(jnp.int32)

    def finite(tree):
        leaves = [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree) if jnp.issubdtype(x.dtype, jnp.floating)]
        return jnp.all(jnp.stack(leaves))

    n = episodes + 1
    greedy = jnp.arange(n) == episodes
    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), STREAMS[stream])
    ep_keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(jnp.arange(n))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), state0)
    obs0 = env.get_obs(state0)
    obs = jnp.broadcast_to(obs0, (n,) + obs0.shape)
    d0 = jnp.zeros(n)

    def one_step(key, state, ob, g):
        a = act(key, logits_fn(ob), g)
        prev = state.last_distance
        ob2, state2, r, done, info = env.step(key, state, a, params)
        dense = jnp.where(prev == -1.0, 0.0, -(info["distance"] - prev) * DENSE_REWARD_SCALE)
        sparse = r - jnp.nan_to_num(dense)
        ok = finite(state2)
        return ob2, state2, r, done, info["GoalR"], sparse < -0.5, ~ok, info["distance"], a[0]

    vstep = jax.vmap(one_step)
    zi = jnp.zeros(n, jnp.int32)
    zf = jnp.zeros(n, jnp.float32)
    zb = jnp.zeros(n, bool)
    carry = dict(t=jnp.int32(0), state=states, obs=obs, alive=~zb, ret=zf, length=zi, outcome=zi,
                 dist0=d0, dist=zf, dmin=zf + jnp.inf)

    def cond(c):
        return (c["t"] < MAX_STEPS) & jnp.any(c["alive"])

    def body(c):
        keys = jax.vmap(lambda k: jax.random.fold_in(k, c["t"]))(ep_keys)
        ob2, st2, r, done, goal, red, bad, dist, _ = vstep(keys, c["state"], c["obs"], greedy)
        alive = c["alive"]
        # outcome codes: 0 running/timeout, 1 success, 2 red, 3 invalid
        oc = jnp.where(red, 2, jnp.where(goal, 1, jnp.where(bad, 3, 0)))
        keep = lambda new, old: jax.tree.map(
            lambda x, y: jnp.where(alive.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), new, old)
        out = dict(
            t=c["t"] + 1,
            state=keep(st2, c["state"]),
            obs=keep(ob2, c["obs"]),
            alive=alive & ~done,
            ret=c["ret"] + jnp.where(alive, jnp.nan_to_num(r), 0.0),
            length=c["length"] + alive.astype(jnp.int32),
            outcome=jnp.where(alive & done, oc, c["outcome"]),
            dist0=jnp.where(c["t"] == 0, dist, c["dist0"]),
            dist=jnp.where(alive, dist, c["dist"]),
            dmin=jnp.where(alive, jnp.minimum(dist, c["dmin"]), c["dmin"]),
        )
        return out

    res = jax.jit(lambda c: jax.lax.while_loop(cond, body, c))(carry)
    res = jax.tree.map(np.asarray, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.environ.get("OPENWISH_FINAL_NET"))
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", choices=sorted(STREAMS), default="test")
    ap.add_argument("--base", default=BASE_SHOWN, help="network to count changed entries against")
    a = ap.parse_args()
    try:
        if not a.net or not os.path.isdir(a.net):
            raise ValueError(f"no network directory: {a.net}")
        if a.episodes < 1:
            raise ValueError("--episodes must be >= 1")
        import numpy as np

        net = load_net(a.net)
        out = {"ok": True}
        res = evaluate(net, a.episodes, a.seed, a.stream)
        n = a.episodes
        oc, ret, length = res["outcome"][:n], res["ret"][:n], res["length"][:n]
        succ = int((oc == 1).sum())
        # progress toward the goal: share of the green-blue distance (after the first step) closed at the closest
        # approach; 1 for a success (this definition reproduces the authors' 0.339 for the shown network)
        prog = np.where(oc == 1, 1.0, np.clip(1.0 - res["dmin"][:n] / np.maximum(res["dist0"][:n], 1e-9), 0.0, 1.0))
        names = {0: "timeout", 1: "success", 2: "red", 3: "invalid"}
        out.update({
            "score": succ / n, "success": succ, "episodes": n,
            "timeouts": int((oc == 0).sum()), "failures_red": int((oc == 2).sum()), "invalid": int((oc == 3).sum()),
            "mean_return": float(ret.mean()), "mean_length": float(length.mean()), "mean_progress": float(prog.mean()),
            "argmax_outcome": names[int(res["outcome"][n])], "stream": a.stream, "seed": a.seed,
            "level": LEVEL, "max_steps": MAX_STEPS,
        })
        if a.base and os.path.isdir(a.base):
            changed, per = count_changed(net, load_net(a.base))
            entries = sum(int(np.prod(s)) for s in SHAPES.values())
            out.update({"changed": changed, "entries": entries, "frac_changed": changed / entries,
                        "changed_per_tensor": per})
        else:
            out.update({"changed": None, "frac_changed": None})
        print(json.dumps(out))
    except Exception as e:  # noqa: BLE001 - the harness reads the JSON line
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
