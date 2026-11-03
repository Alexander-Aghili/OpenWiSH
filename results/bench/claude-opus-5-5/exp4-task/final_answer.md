# Fixing the car policy: add a run-up so the car jumps the red block

## Why the trained network fails

**The scene.** The car (body 5, green front wheel 12, rear wheel 13) starts at x ≈ 2.8, right at the foot of the static ramp (shape 6).
- The ramp rises leftward at about 35° up to its top corner at (1.98, 0.817).
- Beyond the ramp lies the red strip, x 1.18–2.065. The green wheel must not touch it.
- The blue target is at x 0.025–0.44.
- To succeed, the car must leave the ramp top at roughly ≥ 2.5 m/s, so the green wheel clears the red strip.

**What the network does.** The network has learned the right direction: value 1 rolls the wheels counter-clockwise, which drives the car left. But it drives left from rest, with no run-up, and flickers the motor between 0 and 1.

**Where it gets stuck.** It crawls up the slope at under 1 m/s. At step 32 it is stopped in this pose:
- The body is at (2.00, 0.92). Its underside, at y ≈ 0.81, rests on the ramp's top corner.
- The green wheel hangs in the air at (1.75, 1.00).
- The rear wheel sits on the slope.

The car stays in exactly this pose for the remaining 224 steps. The head still outputs only "off" or "left" (P(value 2) = 0.00 throughout). Pushing left cannot free a high-centered car, and the policy never backs off to try again.

**What is missing.** Throughout the episode, entry 0's output never chooses value 2. The missing piece is a run-up: back off to the right first, then charge left fast enough to jump.

## The fix

I replace entry 0's controller with a small dedicated circuit that reads the car-body observation:
- x is entry 52.
- vx is entry 54.

The circuit computes a score s:

s = (x − 3.5) + 3·relu(−vx) + 50·relu(1.6 − x)

- If s > 0, it drives left (value 1). Otherwise it drives right (value 2).
- The signal is carried through unit 0 of every layer with high gain, so it saturates to ±1.
- The head reads only that unit, which makes the decision near-deterministic. "Motor off" gets probability ≈ 1e-4 per step.

```edits
# embed units 0..2: e0 = x+1, e1 = relu(-vx), e2 = relu(1.6 - x)
embed.W[:, 0:3] = 0
embed.W[52, 0] = 1
embed.W[54, 1] = -1
embed.W[52, 2] = -1
embed.b[0:3] = [1, 0, 1.6]
# actor0 unit 0: pre = 4*((x-3.5) + 3*relu(-vx) + 50*relu(1.6-x))
actor0.W[:, 0] = 0
actor0.W[0, 0] = 4
actor0.W[1, 0] = 12
actor0.W[2, 0] = 200
actor0.b[0] = -18
# pass the sign through unit 0 of each later layer
actor1.W[:, 0] = 0
actor1.W[0, 0] = 5
actor1.b[0] = 0
actor2.W[:, 0] = 0
actor2.W[0, 0] = 5
actor2.b[0] = 0
actor3.W[:, 0] = 0
actor3.W[0, 0] = 5
actor3.b[0] = 0
actor4.W[:, 0] = 0
actor4.W[0, 0] = 5
actor4.b[0] = 0
# head, entry 0: value 1 (left) if signal > 0, value 2 (right) if < 0, off strongly suppressed
actor_head.W[:, 0:3] = 0
actor_head.W[0, 1] = 6
actor_head.W[0, 2] = -6
actor_head.b[0:3] = [-4, 0, 0]
```

## What the edits do

1. **Back off.** At the start (x = 2.8, at rest) the score is negative, so the car drives right.
2. **Turn around.** Once x > 3.5 it switches to driving left. It brakes, stopping around x ≈ 4.2, safely short of the wall at x = 5.
3. **Charge.** The term 3·relu(−vx) keeps "left" selected while the car moves left. This gives about 1.4 m of run-up, enough to reach the wheels' top speed of about 4 m/s.
4. **Jump.** At that speed the car flies off the ramp top. The green wheel clears the red strip and either lands near or directly hits the blue box.
5. **Keep going after landing.** The term 50·relu(1.6 − x) forces "left" once the body is past the ramp, so the car keeps driving into the blue box.
6. **Retry if stuck.** If the car stalls between x 1.6 and 3.5 (for example, high-centered on the ramp again), the score turns negative. It then reverses and retries the run-up instead of sitting stuck.

Entries 1–5 have no effect in this scene, so the rest of the network is left untouched.