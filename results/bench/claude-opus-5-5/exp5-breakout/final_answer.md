**Design**

The policy keeps the paddle "locked" under the ball. Each step it moves the paddle to the column the ball will occupy on its next step. The paddle moves first and at the same horizontal speed as the ball, so once locked it stays locked. Any catch is guaranteed, because both catch cases (`ball_x == pos` or `new_x == pos`) are covered.

**Layer 1 (3 units used)**

- **u0 = tanh(5·s)**, where s = B[bx] − tx − px.
  - bx is the ball column, tx the trail column and px the paddle column.
  - B = [1, 2, 4, 6, 8, 10, 12, 14, 16, 17].
  - This gives the ball's next column, bx + (bx − tx), including the wall clamps. At column 9 the ball stays one extra step and then turns back; column 0 is the mirror case.
  - s > 0 means move right, s < 0 means move left, and s = 0 means stay.
- **u1 = tanh(6·g)**, where g = B'[bx] + tx − 2·px.
  - B' = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 8.5].
  - This handles the step right after a paddle hit, when ball and trail are both in row 8. The trail then shows the old direction, which may be wrong.
  - If the paddle is under the ball, it was a reversing hit, and g says "move toward the trail".
  - If the paddle is under the trail, it was a straight bounce, and g says "move toward the ball". The wall corners are also correct.
- **u2 = Q = tanh(8·(ball in row 8 + trail in row 8 − 1.5))**. This is +1 only right after a paddle hit.

**Layer 2: gating**

- R_a = [u0 > 0 and not Q], L_a = [u0 < 0 and not Q].
- R_b = [u1 > 0 and Q], L_b = [u1 < 0 and Q].

**Output**

- The chosen move gets logit +10, the other move −10, and the four no-op actions 0. This makes the policy nearly deterministic, and it stays still when the paddle is already in place.

```edits
# ---- Layer 1, unit 0: s = B[bx] - tx - px, gain 5 ----
W1[3:600:60, 0] = 5
W1[9:600:60, 0] = 10
W1[15:600:60, 0] = 20
W1[21:600:60, 0] = 30
W1[27:600:60, 0] = 40
W1[33:600:60, 0] = 50
W1[39:600:60, 0] = 60
W1[45:600:60, 0] = 70
W1[51:600:60, 0] = 80
W1[57:600:60, 0] = 85
W1[4:600:60, 0] = 0
W1[10:600:60, 0] = -5
W1[16:600:60, 0] = -10
W1[22:600:60, 0] = -15
W1[28:600:60, 0] = -20
W1[34:600:60, 0] = -25
W1[40:600:60, 0] = -30
W1[46:600:60, 0] = -35
W1[52:600:60, 0] = -40
W1[58:600:60, 0] = -45
W1[540:600:6, 0] = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]
b1[0] = 0
# ---- Layer 1, unit 1: g = B'[bx] + tx - 2 px, gain 6 ----
W1[3:600:60, 1] = 3
W1[9:600:60, 1] = 6
W1[15:600:60, 1] = 12
W1[21:600:60, 1] = 18
W1[27:600:60, 1] = 24
W1[33:600:60, 1] = 30
W1[39:600:60, 1] = 36
W1[45:600:60, 1] = 42
W1[51:600:60, 1] = 48
W1[57:600:60, 1] = 51
W1[4:600:60, 1] = 0
W1[10:600:60, 1] = 6
W1[16:600:60, 1] = 12
W1[22:600:60, 1] = 18
W1[28:600:60, 1] = 24
W1[34:600:60, 1] = 30
W1[40:600:60, 1] = 36
W1[46:600:60, 1] = 42
W1[52:600:60, 1] = 48
W1[58:600:60, 1] = 54
W1[540:600:6, 1] = [0, -12, -24, -36, -48, -60, -72, -84, -96, -108]
b1[1] = 0
# ---- Layer 1, unit 2: Q = ball in row 8 AND trail in row 8 ----
W1[483:540:6, 2] = 8
W1[484:540:6, 2] = 8
b1[2] = -12
# ---- Layer 2: gated direction units ----
W2[0, 0] = 6
W2[2, 0] = -6
b2[0] = -9
W2[0, 1] = -6
W2[2, 1] = -6
b2[1] = -9
W2[1, 2] = 6
W2[2, 2] = 6
b2[2] = -9
W2[1, 3] = -6
W2[2, 3] = 6
b2[3] = -9
# ---- Output: right (3) from R_a,R_b ; left (1) from L_a,L_b ----
W3[0, 3] = 10
W3[2, 3] = 10
b3[3] = 10
W3[1, 1] = 10
W3[3, 1] = 10
b3[1] = 10
```