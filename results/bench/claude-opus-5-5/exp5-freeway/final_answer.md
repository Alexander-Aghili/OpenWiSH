# Freeway policy: hand-set weights for a safe-lane chicken

## Strategy

The policy moves up whenever the lane above is safe for the next 3 frames. If the lane above is unsafe, it waits. If it is waiting and its own lane becomes unsafe, it steps down, but only when the lane below is safe. If all three options are unsafe, it waits. Waiting (no-op) leaves the move timer at 0, so the chicken can move the first frame a lane clears. Pressing down on the bottom row would restart the timer, so the policy never does that.

## Danger test

These timings come from the source code:
- A car with speed s moves once every s+1 frames.
- The collision check runs both before and after each car moves.
- After a move, the chicken stays in the new lane through 3 car updates.

So a lane is dangerous to **enter** if any of these hold:
- its car is in column 4 (certain hit);
- its car is one step away and moving toward column 4 (car in column 3 with its trail in column 2, or car in column 5 with its trail in column 6);
- a fast car (gauge channel) is two steps away and moving toward column 4 (car in column 2 with gauge in column 1, or car in column 6 with gauge in column 7).

Each lane has one car, so a single linear score separates these cases. Per lane, the weights are:

| Channel | Columns 0–9 |
|---|---|
| car | −8, −4, 0, 4, 16, 4, 0, −4, −8, 0 |
| trail | +4 at columns 2 and 6, −8 at column 4 |
| gauge | +8 at columns 1 and 7 |

Every dangerous configuration scores at least 8 and every safe one scores at most 0. I checked all 10 positions × 2 directions × fast/slow.

For the lane the chicken is **standing in**, only "one step away and approaching" matters. When it is waiting, the chicken sees the car the frame it arrives at that cell, and still has a full car period to leave. So this test drops the gauge terms.

## Layer 1 (23 units)

Each unit combines the chicken's row with the danger score of one lane: chicken weight 24, the lane weights above, bias −28. The pre-activation is about ±4, so each unit is close to a binary flag: +1 means the chicken is in that row and that lane is dangerous, −1 otherwise.
- **Units 0–7:** the lane above the chicken is dangerous (chicken rows 2–9).
- **Units 8–15:** the chicken's own lane is dangerous (chicken rows 1–8).
- **Units 16–22:** the lane below is dangerous (chicken rows 1–7). Row 9 has no cars, so it is always safe.

## Layer 2 (3 units)

Each unit is an OR over its group of layer-1 units, using weight 4 and bias 28, or 24 for the group of 7:
- **Gu:** the lane above is dangerous.
- **Gc:** the current lane is dangerous.
- **Gd:** the lane below is dangerous.

## Output logits

- up = −40·Gu
- no-op = −10·Gc − 10
- down = 20·Gc − 40·Gd − 40
- left, right and fire = −80

In every case the intended action wins by at least 20 logits:

| Situation | Action |
|---|---|
| Lane above safe | up |
| Lane above unsafe, own lane safe | wait |
| Above and own lane unsafe, lane below safe | down |
| All three unsafe | wait |

```edits
# ---------- Layer 1: "lane above dangerous" units 0..7 (chicken rows 2..9, lanes 1..8) ----------
b1[0:23] = -28
W1[144, 0] = 24
W1[61:120:6, 0] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[64:120:6, 0] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[65:120:6, 0] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[204, 1] = 24
W1[121:180:6, 1] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[124:180:6, 1] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[125:180:6, 1] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[264, 2] = 24
W1[181:240:6, 2] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[184:240:6, 2] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[185:240:6, 2] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[324, 3] = 24
W1[241:300:6, 3] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[244:300:6, 3] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[245:300:6, 3] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[384, 4] = 24
W1[301:360:6, 4] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[304:360:6, 4] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[305:360:6, 4] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[444, 5] = 24
W1[361:420:6, 5] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[364:420:6, 5] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[365:420:6, 5] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[504, 6] = 24
W1[421:480:6, 6] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[424:480:6, 6] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[425:480:6, 6] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[564, 7] = 24
W1[481:540:6, 7] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[484:540:6, 7] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[485:540:6, 7] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
# ---------- Layer 1: "own lane dangerous" units 8..15 (chicken rows 1..8) ----------
W1[84, 8] = 24
W1[61:120:6, 8] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[64:120:6, 8] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[144, 9] = 24
W1[121:180:6, 9] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[124:180:6, 9] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[204, 10] = 24
W1[181:240:6, 10] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[184:240:6, 10] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[264, 11] = 24
W1[241:300:6, 11] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[244:300:6, 11] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[324, 12] = 24
W1[301:360:6, 12] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[304:360:6, 12] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[384, 13] = 24
W1[361:420:6, 13] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[364:420:6, 13] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[444, 14] = 24
W1[421:480:6, 14] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[424:480:6, 14] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[504, 15] = 24
W1[481:540:6, 15] = [0, -4, 0, 4, 16, 4, 0, -4, 0, 0]
W1[484:540:6, 15] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
# ---------- Layer 1: "lane below dangerous" units 16..22 (chicken rows 1..7, lanes 2..8) ----------
W1[84, 16] = 24
W1[121:180:6, 16] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[124:180:6, 16] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[125:180:6, 16] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[144, 17] = 24
W1[181:240:6, 17] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[184:240:6, 17] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[185:240:6, 17] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[204, 18] = 24
W1[241:300:6, 18] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[244:300:6, 18] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[245:300:6, 18] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[264, 19] = 24
W1[301:360:6, 19] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[304:360:6, 19] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[305:360:6, 19] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[324, 20] = 24
W1[361:420:6, 20] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[364:420:6, 20] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[365:420:6, 20] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[384, 21] = 24
W1[421:480:6, 21] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[424:480:6, 21] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[425:480:6, 21] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
W1[444, 22] = 24
W1[481:540:6, 22] = [-8, -4, 0, 4, 16, 4, 0, -4, -8, 0]
W1[484:540:6, 22] = [0, 0, 4, 0, -8, 0, 4, 0, 0, 0]
W1[485:540:6, 22] = [0, 8, 0, 0, 0, 0, 0, 8, 0, 0]
# ---------- Layer 2: OR-pools -> Gu (unit 0), Gc (unit 1), Gd (unit 2) ----------
W2[0:8, 0] = 4
W2[8:16, 1] = 4
W2[16:23, 2] = 4
b2[0] = 28
b2[1] = 28
b2[2] = 24
# ---------- Output logits ----------
W3[0, 2] = -40          # up = -40*Gu
W3[1, 0] = -10          # noop = -10*Gc - 10
b3[0] = -10
W3[1, 4] = 20           # down = 20*Gc - 40*Gd - 40
W3[2, 4] = -40
b3[4] = -40
b3[2] = 0
b3[1] = -80             # left/right/fire suppressed
b3[3] = -80
b3[5] = -80
```