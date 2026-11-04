## Design

**Strategy.** The ball's path is fully predictable, so the paddle goes straight to the landing column and waits there. The horizontal motion is a period-20 triangle wave (columns 0..9, then 9..0).

The catch window at the bottom is two cells: the ball's column `xf` on row 8, and the next column `xf+dx`.
- **Ball above row 5:** aim exactly at `xf`. This catch keeps the ball's horizontal direction, and with no bricks in the way that path passes through every cell within 180 steps, so leftover bricks get hit.
- **Ball on rows 5–8:** only move if outside the two-cell window. This avoids late corrections, which sticky actions could turn into a miss.
- **Ball rising:** assume it bounces off the nearest possible brick row. The real bounce can only be higher, and each extra step of delay shifts the landing point by at most one column, so the paddle can still keep up.
- **Just after a paddle hit:** the next landing window always contains about `9 - pos`, so the paddle simply moves toward the centre for that step.

**Key trick.** For a fixed pair of ball row and trail row, the ball/trail cells form a single 20-edge cycle. Any target whose even and odd unfolded states sum equally can be written as `f(ball cell) + g(trail cell)`. The reflected landing column satisfies this, so one linear unit can compute "landing column − paddle column" exactly, including wall bounces. I solved these tables by hand around each cycle and checked the edges.

**Hidden units (first layer, gain 10).**

| Unit | Computes |
|---|---|
| 0, 1 | Ball falling (dy=+1): target right of paddle / left of paddle |
| 2, 3 | Ball rising (dy=−1): same comparison |
| 4, 5 | Same row as trail (dy=0, just bounced): same comparison, with target ≈ centre on row 8 |
| 6 | Up flag: trail row − ball row − 0.5 |
| 7 | Down flag: ball row − trail row − 0.5 |

**Hidden units (second layer).** Each unit is an AND of a comparator and its mode:
- Units 0–2 mean "go right" in the falling, rising and just-bounced modes.
- Units 3–5 mean "go left" in the same three modes.

**Outputs.** Right and left get a logit of about +20 when one of their units fires and −20 otherwise. No-op stays at 0, and up, down and fire are held at −40.

Nothing has been run. I only checked the tables and several trajectories by hand, so I have no measured score.

```edits
# paddle: raw -column on all six comparators
W1[540:600:6, 0] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 1] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 4] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[540:600:6, 5] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]

# unit 0: falling, go right - ball rows 1..8
W1[63:120:6, 0] = [7,8,10,12,14,16,18,20,22,23]
W1[123:180:6, 0] = [6,7,9,12,15,18,21,24,26,27]
W1[183:240:6, 0] = [5,6,8,11,15,19,23,26,28,29]
W1[243:300:6, 0] = [4,5,7,10,14,19,23,26,28,29]
W1[303:360:6, 0] = [3,4,6,9,13,18,22,25,27,28]
W1[363:420:6, 0] = [2,3,5,8,12,15,19,22,24,25]
W1[423:480:6, 0] = [1,2,4,7,9,12,14,17,19,20]
W1[483:540:6, 0] = [0,1,3,4,6,7,9,10,12,13]
# unit 0: trail rows 0..7
W1[4:60:6, 0] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[64:120:6, 0] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[124:180:6, 0] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[184:240:6, 0] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[244:300:6, 0] = [0,-1,-3,-6,-10,-13,-17,-20,-22,-23]
W1[304:360:6, 0] = [0,-1,-3,-6,-8,-11,-13,-16,-18,-19]
W1[364:420:6, 0] = [0,-1,-3,-4,-6,-7,-9,-10,-12,-13]
W1[424:480:6, 0] = [0,-1,-1,-2,-2,-3,-3,-4,-4,-5]

# unit 1: falling, go left - ball rows 1..8
W1[63:120:6, 1] = [7,8,10,12,14,16,18,20,22,23]
W1[123:180:6, 1] = [6,7,9,12,15,18,21,24,26,27]
W1[183:240:6, 1] = [5,6,8,11,15,19,23,26,28,29]
W1[243:300:6, 1] = [4,5,7,10,14,19,23,26,28,29]
W1[303:360:6, 1] = [4,5,7,10,14,19,23,26,28,29]
W1[363:420:6, 1] = [3,4,6,9,13,16,20,23,25,26]
W1[423:480:6, 1] = [2,3,5,8,10,13,15,18,20,21]
W1[483:540:6, 1] = [1,2,4,5,7,8,10,11,13,14]
# unit 1: trail rows 0..7
W1[4:60:6, 1] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[64:120:6, 1] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[124:180:6, 1] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[184:240:6, 1] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[244:300:6, 1] = [0,-1,-3,-6,-10,-13,-17,-20,-22,-23]
W1[304:360:6, 1] = [0,-1,-3,-6,-8,-11,-13,-16,-18,-19]
W1[364:420:6, 1] = [0,-1,-3,-4,-6,-7,-9,-10,-12,-13]
W1[424:480:6, 1] = [0,-1,-1,-2,-2,-3,-3,-4,-4,-5]

# unit 2: rising, go right - ball rows 0..7
W1[3:60:6, 2] = [9,9,9,9,9,9,9,9,9,9]
W1[63:120:6, 2] = [9,8,7,6,5,4,3,2,1,0]
W1[123:180:6, 2] = [7,8,10,12,14,16,18,20,22,23]
W1[183:240:6, 2] = [6,7,9,12,15,18,21,24,26,27]
W1[243:300:6, 2] = [5,6,8,11,15,19,23,26,28,29]
W1[303:360:6, 2] = [6,7,9,12,15,18,21,24,26,27]
W1[363:420:6, 2] = [7,8,10,12,14,16,18,20,22,23]
W1[423:480:6, 2] = [8,9,10,11,12,13,14,15,16,17]
# unit 2: trail rows 1..8 (row 2 stays zero)
W1[64:120:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[184:240:6, 2] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[244:300:6, 2] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[304:360:6, 2] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[364:420:6, 2] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[424:480:6, 2] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[484:540:6, 2] = [0,-1,-3,-5,-7,-9,-11,-13,-15,-16]

# unit 3: rising, go left - ball rows 0..7
W1[3:60:6, 3] = [9,9,9,9,9,9,9,9,9,9]
W1[63:120:6, 3] = [9,8,7,6,5,4,3,2,1,0]
W1[123:180:6, 3] = [7,8,10,12,14,16,18,20,22,23]
W1[183:240:6, 3] = [6,7,9,12,15,18,21,24,26,27]
W1[243:300:6, 3] = [5,6,8,11,15,19,23,26,28,29]
W1[303:360:6, 3] = [6,7,9,12,15,18,21,24,26,27]
W1[363:420:6, 3] = [7,8,10,12,14,16,18,20,22,23]
W1[423:480:6, 3] = [8,9,10,11,12,13,14,15,16,17]
# unit 3: trail rows 1..8 (row 2 stays zero)
W1[64:120:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[184:240:6, 3] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[244:300:6, 3] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[304:360:6, 3] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[364:420:6, 3] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[424:480:6, 3] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[484:540:6, 3] = [0,-1,-3,-5,-7,-9,-11,-13,-15,-16]

# unit 4: just bounced, go right - ball rows 0..4 and 8
W1[3:60:6, 4] = [8,9,10,11,12,13,14,15,16,17]
W1[63:120:6, 4] = [7,8,10,12,14,16,18,20,22,23]
W1[123:180:6, 4] = [6,7,9,12,15,18,21,24,26,27]
W1[183:240:6, 4] = [5,6,8,11,15,19,23,26,28,29]
W1[243:300:6, 4] = [4,5,7,10,14,19,23,26,28,29]
W1[483:540:6, 4] = 4
# unit 4: trail rows 0..4
W1[4:60:6, 4] = [0,-1,-3,-5,-7,-9,-11,-13,-15,-16]
W1[64:120:6, 4] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[124:180:6, 4] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[184:240:6, 4] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[244:300:6, 4] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]

# unit 5: just bounced, go left - ball rows 0..4 and 8
W1[3:60:6, 5] = [8,9,10,11,12,13,14,15,16,17]
W1[63:120:6, 5] = [7,8,10,12,14,16,18,20,22,23]
W1[123:180:6, 5] = [6,7,9,12,15,18,21,24,26,27]
W1[183:240:6, 5] = [5,6,8,11,15,19,23,26,28,29]
W1[243:300:6, 5] = [4,5,7,10,14,19,23,26,28,29]
W1[483:540:6, 5] = 5
# unit 5: trail rows 0..4
W1[4:60:6, 5] = [0,-1,-3,-5,-7,-9,-11,-13,-15,-16]
W1[64:120:6, 5] = [0,-1,-3,-6,-9,-12,-15,-18,-20,-21]
W1[124:180:6, 5] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]
W1[184:240:6, 5] = [0,-1,-3,-6,-10,-15,-19,-22,-24,-25]
W1[244:300:6, 5] = [0,-1,-3,-6,-10,-14,-18,-21,-23,-24]

# unit 6: up flag (trail row - ball row)
W1[63:120:6, 6] = -1
W1[123:180:6, 6] = -2
W1[183:240:6, 6] = -3
W1[243:300:6, 6] = -4
W1[303:360:6, 6] = -5
W1[363:420:6, 6] = -6
W1[423:480:6, 6] = -7
W1[483:540:6, 6] = -8
W1[64:120:6, 6] = 1
W1[124:180:6, 6] = 2
W1[184:240:6, 6] = 3
W1[244:300:6, 6] = 4
W1[304:360:6, 6] = 5
W1[364:420:6, 6] = 6
W1[424:480:6, 6] = 7
W1[484:540:6, 6] = 8

# unit 7: down flag (ball row - trail row)
W1[63:120:6, 7] = 1
W1[123:180:6, 7] = 2
W1[183:240:6, 7] = 3
W1[243:300:6, 7] = 4
W1[303:360:6, 7] = 5
W1[363:420:6, 7] = 6
W1[423:480:6, 7] = 7
W1[483:540:6, 7] = 8
W1[64:120:6, 7] = -1
W1[124:180:6, 7] = -2
W1[184:240:6, 7] = -3
W1[244:300:6, 7] = -4
W1[304:360:6, 7] = -5
W1[364:420:6, 7] = -6
W1[424:480:6, 7] = -7
W1[484:540:6, 7] = -8

# gains; left comparators are negated
W1[:, 0] *= 10
W1[:, 1] *= -10
W1[:, 2] *= 10
W1[:, 3] *= -10
W1[:, 4] *= 10
W1[:, 5] *= -10
W1[:, 6] *= 10
W1[:, 7] *= 10
b1[0:8] = -5

# second layer: comparator AND mode
W2[0, 0] = 5
W2[7, 0] = 5
b2[0] = -5
W2[2, 1] = 5
W2[6, 1] = 5
b2[1] = -5
W2[4, 2] = 5
W2[6, 2] = -5
W2[7, 2] = -5
b2[2] = -10
W2[1, 3] = 5
W2[7, 3] = 5
b2[3] = -5
W2[3, 4] = 5
W2[6, 4] = 5
b2[4] = -5
W2[5, 5] = 5
W2[6, 5] = -5
W2[7, 5] = -5
b2[5] = -10

# outputs: right = action 3, left = action 1, no-op = action 0
W3[0:3, 3] = 20
W3[3:6, 1] = 20
b3[3] = 40
b3[1] = 40
b3[0] = 0
b3[2] = -40
b3[4] = -40
b3[5] = -40
```