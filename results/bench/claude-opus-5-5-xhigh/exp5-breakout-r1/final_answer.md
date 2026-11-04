## Design

**Strategy:** The paddle moves to the column where the ball will land and waits there. It uses no learned values, and I haven't run it, so its behaviour comes only from hand-tracing the game code.

**Ball direction:** The ball and trail cells give the direction, with two special cases:
- **Wall clamp:** if the ball's column didn't change, it is moving away from that wall.
- **Vertical bounce:** if its row didn't change, it is moving down, unless it is on row 8, where it has just left the paddle. A bounce off the top of a brick is misread for one step, which is harmless.

**Landing column:** A lookup table on the ball's cell gives the target for each of the four directions, including wall reflections.
- **Ball moving down:** the landing column is exact unless the ball hits another brick on the way down.
- **Ball moving up:** the table assumes the ball bounces at the next possible brick, which is the case with the least time to react. The target shifts by 2 columns per row if the ball climbs higher, and the extra flight time more than covers this.

**Avoiding stuck loops:** Depending on where the paddle catches the ball, it either keeps its horizontal direction or reverses it. The ball's motion falls into two parity classes, and the targets steer it into the safer one:
- **Right-moving ball:** the paddle aims for an even column.
- **Left-moving ball:** the paddle aims for an odd column.
- **Safer class:** corner landings here have two catchable cells, so a sticky action costs less. After 2 lost moves the longest crossing (9 columns in 10 moves) can still be caught, against 1 in the other class.
- **Recovery:** if a sticky action knocks the ball into the other class, the next catch away from a corner moves it back.
- **No dead loops:** without brick hits, this path repeats about every 180 steps and passes through every cell, so every remaining brick eventually gets hit.

**Hidden units (only the listed ones are used):**

| Layer | Units | What they compute |
|---|---|---|
| h1 | 0 | Ball is moving right (+1) or left (−1) |
| h1 | 1 | Ball is moving up (+1) or down (−1) |
| h1 | 2–9 | For each of the 4 directions, one unit for target > paddle and one for target < paddle |
| h2 | 0–7 | AND of the horizontal flag, the vertical flag and the matching comparison; even units mean "go right", odd units mean "go left" |

**Outputs:** Exactly one "go" unit is active, giving a logit of +12 for that move and −12 for the opposite one. If none is active, no-op wins with logit 0. Up, down and fire are set to −30.

```edits
# h1 unit 0: ball moving right (ball column + wall bonus - trail column), times 8
W1[3:600:60, 0] = 4
W1[9:600:60, 0] = 8
W1[15:600:60, 0] = 16
W1[21:600:60, 0] = 24
W1[27:600:60, 0] = 32
W1[33:600:60, 0] = 40
W1[39:600:60, 0] = 48
W1[45:600:60, 0] = 56
W1[51:600:60, 0] = 64
W1[57:600:60, 0] = 68
W1[10:600:60, 0] = -8
W1[16:600:60, 0] = -16
W1[22:600:60, 0] = -24
W1[28:600:60, 0] = -32
W1[34:600:60, 0] = -40
W1[40:600:60, 0] = -48
W1[46:600:60, 0] = -56
W1[52:600:60, 0] = -64
W1[58:600:60, 0] = -72

# h1 unit 1: ball moving up (trail row - ball row, row 8 bonus), times 8
W1[63:120:6, 1] = -8
W1[123:180:6, 1] = -16
W1[183:240:6, 1] = -24
W1[243:300:6, 1] = -32
W1[303:360:6, 1] = -40
W1[363:420:6, 1] = -48
W1[423:480:6, 1] = -56
W1[483:540:6, 1] = -56
W1[64:120:6, 1] = 8
W1[124:180:6, 1] = 16
W1[184:240:6, 1] = 24
W1[244:300:6, 1] = 32
W1[304:360:6, 1] = 40
W1[364:420:6, 1] = 48
W1[424:480:6, 1] = 56
W1[484:540:6, 1] = 64
b1[1] = -4

# h1 unit 2: moving down-right, target > paddle
W1[3:540:6, 2] = [8,9,9,7,7,5,5,3,3,1,
8,8,9,9,7,7,5,5,3,3,
6,8,8,9,9,7,7,5,5,3,
6,6,8,8,9,9,7,7,5,5,
4,6,6,8,8,9,9,7,7,5,
4,4,6,6,8,8,9,9,7,7,
2,4,4,6,6,8,8,9,9,7,
2,2,4,4,6,6,8,8,9,9,
0,2,2,4,4,6,6,8,8,9]
W1[540:600:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 2] *= 8
b1[2] = -4

# h1 unit 3: moving down-right, target < paddle
W1[3:540:6, 3] = [8,9,9,7,7,5,5,3,3,1,
8,8,9,9,7,7,5,5,3,3,
6,8,8,9,9,7,7,5,5,3,
6,6,8,8,9,9,7,7,5,5,
4,6,6,8,8,9,9,7,7,5,
4,4,6,6,8,8,9,9,7,7,
2,4,4,6,6,8,8,9,9,7,
2,2,4,4,6,6,8,8,9,9,
0,2,2,4,4,6,6,8,8,9]
W1[540:600:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 3] *= -8
b1[3] = -4

# h1 unit 4: moving down-left, target > paddle
W1[3:540:6, 4] = [8,6,6,4,4,2,2,0,0,1,
6,6,4,4,2,2,0,0,1,1,
6,4,4,2,2,0,0,1,1,3,
4,4,2,2,0,0,1,1,3,3,
4,2,2,0,0,1,1,3,3,5,
2,2,0,0,1,1,3,3,5,5,
2,0,0,1,1,3,3,5,5,7,
0,0,1,1,3,3,5,5,7,7,
0,1,1,3,3,5,5,7,7,9]
W1[540:600:6, 4] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 4] *= 8
b1[4] = -4

# h1 unit 5: moving down-left, target < paddle
W1[3:540:6, 5] = [8,6,6,4,4,2,2,0,0,1,
6,6,4,4,2,2,0,0,1,1,
6,4,4,2,2,0,0,1,1,3,
4,4,2,2,0,0,1,1,3,3,
4,2,2,0,0,1,1,3,3,5,
2,2,0,0,1,1,3,3,5,5,
2,0,0,1,1,3,3,5,5,7,
0,0,1,1,3,3,5,5,7,7,
0,1,1,3,3,5,5,7,7,9]
W1[540:600:6, 5] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 5] *= -8
b1[5] = -4

# h1 unit 6: moving up-right, target > paddle
W1[3:540:6, 6] = [9,9,7,7,5,5,3,3,1,1,
9,7,7,5,5,3,3,1,1,0,
8,8,9,9,7,7,5,5,3,3,
6,8,8,9,9,7,7,5,5,3,
6,6,8,8,9,9,7,7,5,5,
6,8,8,9,9,7,7,5,5,3,
8,8,9,9,7,7,5,5,3,3,
8,9,9,7,7,5,5,3,3,1,
9,9,7,7,5,5,3,3,1,1]
W1[540:600:6, 6] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 6] *= 8
b1[6] = -4

# h1 unit 7: moving up-right, target < paddle
W1[3:540:6, 7] = [9,9,7,7,5,5,3,3,1,1,
9,7,7,5,5,3,3,1,1,0,
8,8,9,9,7,7,5,5,3,3,
6,8,8,9,9,7,7,5,5,3,
6,6,8,8,9,9,7,7,5,5,
6,8,8,9,9,7,7,5,5,3,
8,8,9,9,7,7,5,5,3,3,
8,9,9,7,7,5,5,3,3,1,
9,9,7,7,5,5,3,3,1,1]
W1[540:600:6, 7] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 7] *= -8
b1[7] = -4

# h1 unit 8: moving up-left, target > paddle
W1[3:540:6, 8] = [8,8,6,6,4,4,2,2,0,0,
9,8,8,6,6,4,4,2,2,0,
6,6,4,4,2,2,0,0,1,1,
6,4,4,2,2,0,0,1,1,3,
4,4,2,2,0,0,1,1,3,3,
6,4,4,2,2,0,0,1,1,3,
6,6,4,4,2,2,0,0,1,1,
8,6,6,4,4,2,2,0,0,1,
8,8,6,6,4,4,2,2,0,0]
W1[540:600:6, 8] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 8] *= 8
b1[8] = -4

# h1 unit 9: moving up-left, target < paddle
W1[3:540:6, 9] = [8,8,6,6,4,4,2,2,0,0,
9,8,8,6,6,4,4,2,2,0,
6,6,4,4,2,2,0,0,1,1,
6,4,4,2,2,0,0,1,1,3,
4,4,2,2,0,0,1,1,3,3,
6,4,4,2,2,0,0,1,1,3,
6,6,4,4,2,2,0,0,1,1,
8,6,6,4,4,2,2,0,0,1,
8,8,6,6,4,4,2,2,0,0]
W1[540:600:6, 9] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 9] *= -8
b1[9] = -4

# h2 units 0-7: AND of comparison, horizontal flag and vertical flag
W2[2, 0] = 4
W2[0, 0] = 4
W2[1, 0] = -4
W2[3, 1] = 4
W2[0, 1] = 4
W2[1, 1] = -4
W2[4, 2] = 4
W2[0, 2] = -4
W2[1, 2] = -4
W2[5, 3] = 4
W2[0, 3] = -4
W2[1, 3] = -4
W2[6, 4] = 4
W2[0, 4] = 4
W2[1, 4] = 4
W2[7, 5] = 4
W2[0, 5] = 4
W2[1, 5] = 4
W2[8, 6] = 4
W2[0, 6] = -4
W2[1, 6] = 4
W2[9, 7] = 4
W2[0, 7] = -4
W2[1, 7] = 4
b2[0:8] = -8

# outputs: even h2 units drive right (3), odd h2 units drive left (1)
W3[0, 3] = 12
W3[2, 3] = 12
W3[4, 3] = 12
W3[6, 3] = 12
W3[1, 1] = 12
W3[3, 1] = 12
W3[5, 1] = 12
W3[7, 1] = 12
b3[3] = 36
b3[1] = 36
b3[2] = -30
b3[4] = -30
b3[5] = -30
```