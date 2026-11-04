## Design

**Strategy:** Move the paddle to the column where the ball will reach row 8, then wait there. Being directly under the ball gives a catch that keeps its horizontal direction. Each ball flight then shifts the landing phase by 2 cells when no brick is hit, so the ball eventually reaches every brick and can't get stuck in a permanent loop.

**Key trick:** The ball's horizontal motion, including wall bounces, is a 20-state cycle, and each state is one (ball column, trail column) pair. For a fixed row and direction, the wall-folded landing column can be written exactly as `T = A(ball column) + B(trail column)`. This means one tanh unit can compare the target with the paddle without separate logic for wall reflections.

**First hidden layer**

| Unit | Computes | When it is used | Target |
|---|---|---|---|
| 0 | `tanh(2(T − paddle))` | Ball moving down | Exact landing column |
| 1 | `tanh(2(T − paddle))` | Ball moving up | Estimate from the possible turning rows; exact once the ball reaches row 1 or 0, where only the ceiling remains |
| 2 | `tanh(2(T − paddle))` | Ball and trail in the same row, just after a bounce or at the start | Landing column after a brick or ceiling bounce; upward estimate after a paddle bounce |
| 3 | Moving-down gate | Compares ball row with trail row | n/a |
| 4 | Moving-up gate | Compares ball row with trail row | n/a |

**Second hidden layer:** Six AND units pair a comparator with its phase gate, giving a "go right" and a "go left" unit for each of the down, up and flat phases.

**Output:** The right logit is about +15 when a "go right" unit is active and −15 otherwise, and the left logit works the same way. If the paddle is already on target, both are −15 and a no-op is chosen. The remaining hidden units stay at zero.

**Up-phase estimate:** While the ball is in rows 4–7, the target assumes it will turn at row 3, just below the second brick row. This keeps the paddle close to every possible landing column, leaving spare moves in most cases to absorb sticky actions. The tightest case is a short flight between opposite walls, which has only one spare move.

I couldn't run this, but I checked sample table entries and traced the opening moves by hand.

```edits
# ---------- Unit 0: down-phase comparator (ball row y, trail row y-1) ----------
W1[540:600:6, 0] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[483:540:6, 0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
W1[423:480:6, 0] = [0, 1, 3, 5, 7, 9, 11, 13, 15, 16]
W1[364:420:6, 0] = [1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
W1[363:420:6, 0] = [0, 1, 3, 6, 9, 12, 15, 18, 20, 21]
W1[304:360:6, 0] = [2, 1, -1, -3, -5, -7, -9, -11, -13, -14]
W1[303:360:6, 0] = [0, 1, 3, 6, 10, 14, 18, 21, 23, 24]
W1[244:300:6, 0] = [3, 2, 0, -3, -6, -9, -12, -15, -17, -18]
W1[243:300:6, 0] = [0, 1, 3, 6, 10, 15, 19, 22, 24, 25]
W1[184:240:6, 0] = [4, 3, 1, -2, -6, -10, -14, -17, -19, -20]
W1[183:240:6, 0] = [0, 1, 3, 6, 10, 14, 18, 21, 23, 24]
W1[124:180:6, 0] = [5, 4, 2, -1, -5, -10, -14, -17, -19, -20]
W1[123:180:6, 0] = [0, 1, 3, 6, 9, 12, 15, 18, 20, 21]
W1[64:120:6, 0] = [6, 5, 3, 0, -4, -8, -12, -15, -17, -18]
W1[63:120:6, 0] = [0, 1, 3, 5, 7, 9, 11, 13, 15, 16]
W1[4:60:6, 0] = [7, 6, 4, 1, -2, -5, -8, -11, -13, -14]
W1[:, 0] *= 2

# ---------- Unit 1: up-phase comparator (ball row y, trail row y+1) ----------
W1[540:600:6, 1] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[423:480:6, 1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[424:480:6, 1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[303:360:6, 1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
W1[364:420:6, 1] = [8, 7, 5, 3, 1, -1, -3, -5, -7, -8]
W1[243:300:6, 1] = [0, 1, 3, 5, 7, 9, 11, 13, 15, 16]
W1[304:360:6, 1] = [7, 6, 4, 1, -2, -5, -8, -11, -13, -14]
W1[183:240:6, 1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
W1[244:300:6, 1] = [8, 7, 5, 3, 1, -1, -3, -5, -7, -8]
W1[184:240:6, 1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[63:120:6, 1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[64:120:6, 1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[:, 1] *= 2

# ---------- Unit 2: flat-phase comparator (ball and trail in the same row) ----------
W1[540:600:6, 2] = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
W1[483:540:6, 2] = [0, -1, -3, -5, -7, -9, -11, -13, -15, -16]
W1[484:540:6, 2] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
W1[243:300:6, 2] = [0, 1, 3, 6, 10, 15, 19, 22, 24, 25]
W1[244:300:6, 2] = [4, 3, 1, -2, -6, -10, -14, -17, -19, -20]
W1[183:240:6, 2] = [0, 1, 3, 6, 10, 14, 18, 21, 23, 24]
W1[184:240:6, 2] = [5, 4, 2, -1, -5, -10, -14, -17, -19, -20]
W1[123:180:6, 2] = [0, 1, 3, 6, 9, 12, 15, 18, 20, 21]
W1[124:180:6, 2] = [6, 5, 3, 0, -4, -8, -12, -15, -17, -18]
W1[63:120:6, 2] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
W1[3:60:6, 2] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
W1[4:60:6, 2] = [8, 7, 5, 3, 1, -1, -3, -5, -7, -8]
W1[:, 2] *= 2

# ---------- Unit 3: moving-down gate ----------
W1[63:120:6, 3] = 10
W1[64:120:6, 3] = -10
W1[123:180:6, 3] = 20
W1[124:180:6, 3] = -20
W1[183:240:6, 3] = 30
W1[184:240:6, 3] = -30
W1[243:300:6, 3] = 40
W1[244:300:6, 3] = -40
W1[303:360:6, 3] = 50
W1[304:360:6, 3] = -50
W1[363:420:6, 3] = 60
W1[364:420:6, 3] = -60
W1[423:480:6, 3] = 70
W1[424:480:6, 3] = -70
W1[483:540:6, 3] = 80
W1[484:540:6, 3] = -80
b1[3] = -5

# ---------- Unit 4: moving-up gate ----------
W1[63:120:6, 4] = -10
W1[64:120:6, 4] = 10
W1[123:180:6, 4] = -20
W1[124:180:6, 4] = 20
W1[183:240:6, 4] = -30
W1[184:240:6, 4] = 30
W1[243:300:6, 4] = -40
W1[244:300:6, 4] = 40
W1[303:360:6, 4] = -50
W1[304:360:6, 4] = 50
W1[363:420:6, 4] = -60
W1[364:420:6, 4] = 60
W1[423:480:6, 4] = -70
W1[424:480:6, 4] = 70
W1[483:540:6, 4] = -80
W1[484:540:6, 4] = 80
b1[4] = -5

# ---------- Second layer: phase-gated right/left units ----------
# Unit 0: right, down phase
W2[0, 0] = 10
W2[3, 0] = 10
b2[0] = -15
# Unit 1: left, down phase
W2[0, 1] = -10
W2[3, 1] = 10
b2[1] = -15
# Unit 2: right, up phase
W2[1, 2] = 10
W2[4, 2] = 10
b2[2] = -15
# Unit 3: left, up phase
W2[1, 3] = -10
W2[4, 3] = 10
b2[3] = -15
# Unit 4: right, flat phase
W2[2, 4] = 10
W2[3, 4] = -10
W2[4, 4] = -10
b2[4] = -25
# Unit 5: left, flat phase
W2[2, 5] = -10
W2[3, 5] = -10
W2[4, 5] = -10
b2[5] = -25

# ---------- Output: about +15 when active, -15 otherwise; no-op logits stay at 0 ----------
W3[0, 3] = 15
W3[2, 3] = 15
W3[4, 3] = 15
b3[3] = 30
W3[1, 1] = 15
W3[3, 1] = 15
W3[5, 1] = 15
b3[1] = 30
```