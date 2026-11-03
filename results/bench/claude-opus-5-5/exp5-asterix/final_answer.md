# Hand-Written Policy Network for MinAtar Asterix: Hold Column 5, Dodge and Collect Vertically

## Strategy

The player never moves left or right. It stays in **column 5**, where it starts, and only chooses among stay, up and down.

Every enemy and every treasure crosses every column, so all treasure passes through column 5. Holding one column also turns the problem into a set of row-local tests. Each row holds at most one entity, because entity slot equals row. A row that contains treasure therefore has no enemy in it and is safe while the treasure is there. The left, right and fire logits are set to −200, so the player never leaves column 5. Sticky actions only ever repeat stay, up or down.

## Layer 1 (27 units used): per-row facts about column 5, plus the player's row

- **T_r (units 0–7), immediate threat in row r.** True if an enemy is at column 5, at column 4 moving right (trail at 3), or at column 6 moving left (trail at 7).
  - Score = 2·E5 + E4 + E6 − Trail5, with threshold 0.5.
  - The trail at 5 cancels a neighbour that is moving away.
- **ST_r (units 8–15), threat 2 cells away.** True if an enemy is at column 3 moving right or at column 7 moving left.
  - Score = E3 + E7 − Trail4 − Trail6.
- **GA_r (units 16–23), treasure in row r still approaching or at column 5.**
  - Score = ψ(gold column) + φ(trail column), where φ(t) = −min(t, 9−t) and ψ is tuned so approaching treasure scores +0.5 to +2.
  - Receding treasure, enemy trails and empty rows all score ≤ 0. This also handles a freshly spawned treasure, which has no trail yet.
- **Units 24–26:** a 3-bit binary code of the player's row.

## Layer 2 (24 units used): one unit per (player row k, action)

Each unit is gated by the row bits with weight ±20 and bias −60. It is only "live" when the player is in row k; otherwise it outputs a constant −1, which the output biases cancel. When live, it outputs tanh(0.3·D), where D is a hand-set desirability:

- **Stay:** D = 0.5 − 10·T_k − 3·ST_k + 2.5·GA_k
- **Up / down:**
  - −14 · T(target row)
  - −3 · ST(target row)
  - −1 · T and −0.5 · ST two rows away (risk of a sticky repeat)
  - plus Σ treasure weights by vertical distance (2.0, 1.6, 1.3, 1.1, 0.9, 0.8, 0.7)
  - plus a small pull toward the centre rows from rows 1, 2, 7 and 8
- Up from row 1 and down from row 8 are constant −1.

The logits are 60 × (live unit) for no-op, up and down. The resulting priorities, in order:

1. Never stay in or enter a row that is under immediate threat.
2. Leave early when an enemy is 2 cells away.
3. Move toward approaching treasure, and wait in its row.
4. Otherwise stay still.

```edits
# ---------- Layer 1: immediate threat T_r (units 0..7) ----------
W1[91, 0] = 24
W1[85, 0] = 12
W1[97, 0] = 12
W1[94, 0] = -12
W1[151, 1] = 24
W1[145, 1] = 12
W1[157, 1] = 12
W1[154, 1] = -12
W1[211, 2] = 24
W1[205, 2] = 12
W1[217, 2] = 12
W1[214, 2] = -12
W1[271, 3] = 24
W1[265, 3] = 12
W1[277, 3] = 12
W1[274, 3] = -12
W1[331, 4] = 24
W1[325, 4] = 12
W1[337, 4] = 12
W1[334, 4] = -12
W1[391, 5] = 24
W1[385, 5] = 12
W1[397, 5] = 12
W1[394, 5] = -12
W1[451, 6] = 24
W1[445, 6] = 12
W1[457, 6] = 12
W1[454, 6] = -12
W1[511, 7] = 24
W1[505, 7] = 12
W1[517, 7] = 12
W1[514, 7] = -12
b1[0:8] = -6
# ---------- Layer 1: threat at distance 2, ST_r (units 8..15) ----------
W1[79, 8] = 12
W1[103, 8] = 12
W1[88, 8] = -12
W1[100, 8] = -12
W1[139, 9] = 12
W1[163, 9] = 12
W1[148, 9] = -12
W1[160, 9] = -12
W1[199, 10] = 12
W1[223, 10] = 12
W1[208, 10] = -12
W1[220, 10] = -12
W1[259, 11] = 12
W1[283, 11] = 12
W1[268, 11] = -12
W1[280, 11] = -12
W1[319, 12] = 12
W1[343, 12] = 12
W1[328, 12] = -12
W1[340, 12] = -12
W1[379, 13] = 12
W1[403, 13] = 12
W1[388, 13] = -12
W1[400, 13] = -12
W1[439, 14] = 12
W1[463, 14] = 12
W1[448, 14] = -12
W1[460, 14] = -12
W1[499, 15] = 12
W1[523, 15] = 12
W1[508, 15] = -12
W1[520, 15] = -12
b1[8:16] = -6
# ---------- Layer 1: approaching treasure GA_r (units 16..23) ----------
W1[62:120:6, 16] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[64:120:6, 16] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[122:180:6, 17] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[124:180:6, 17] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[182:240:6, 18] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[184:240:6, 18] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[242:300:6, 19] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[244:300:6, 19] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[302:360:6, 20] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[304:360:6, 20] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[362:420:6, 21] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[364:420:6, 21] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[422:480:6, 22] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[424:480:6, 22] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
W1[482:540:6, 23] = [12, 24, 48, 72, 84, 120, 72, 48, 24, 12]
W1[484:540:6, 23] = [0, -24, -48, -72, -96, -96, -72, -48, -24, 0]
b1[16:24] = -6
# ---------- Layer 1: player-row bits (units 24..26), code = row-1 ----------
W1[60:120:6, 24] = -10
W1[120:180:6, 24] = 10
W1[180:240:6, 24] = -10
W1[240:300:6, 24] = 10
W1[300:360:6, 24] = -10
W1[360:420:6, 24] = 10
W1[420:480:6, 24] = -10
W1[480:540:6, 24] = 10
W1[60:180:6, 25] = -10
W1[180:300:6, 25] = 10
W1[300:420:6, 25] = -10
W1[420:540:6, 25] = 10
W1[60:300:6, 26] = -10
W1[300:540:6, 26] = 10
# ---------- Layer 2: row gating (unit 3(k-1)+{0 stay,1 up,2 down}) ----------
W2[24:27, 0] = [-20, -20, -20]
W2[24:27, 2] = [-20, -20, -20]
W2[24:27, 3:6] = [[20, 20, 20], [-20, -20, -20], [-20, -20, -20]]
W2[24:27, 6:9] = [[-20, -20, -20], [20, 20, 20], [-20, -20, -20]]
W2[24:27, 9:12] = [[20, 20, 20], [20, 20, 20], [-20, -20, -20]]
W2[24:27, 12:15] = [[-20, -20, -20], [-20, -20, -20], [20, 20, 20]]
W2[24:27, 15:18] = [[20, 20, 20], [-20, -20, -20], [20, 20, 20]]
W2[24:27, 18:21] = [[-20, -20, -20], [20, 20, 20], [20, 20, 20]]
W2[24:27, 21:23] = [[20, 20], [20, 20], [20, 20]]
# ---------- Layer 2: stay units N_k ----------
W2[0, 0] = -1.5
W2[8, 0] = -0.45
W2[16, 0] = 0.375
W2[1, 3] = -1.5
W2[9, 3] = -0.45
W2[17, 3] = 0.375
W2[2, 6] = -1.5
W2[10, 6] = -0.45
W2[18, 6] = 0.375
W2[3, 9] = -1.5
W2[11, 9] = -0.45
W2[19, 9] = 0.375
W2[4, 12] = -1.5
W2[12, 12] = -0.45
W2[20, 12] = 0.375
W2[5, 15] = -1.5
W2[13, 15] = -0.45
W2[21, 15] = 0.375
W2[6, 18] = -1.5
W2[14, 18] = -0.45
W2[22, 18] = 0.375
W2[7, 21] = -1.5
W2[15, 21] = -0.45
W2[23, 21] = 0.375
# ---------- Layer 2: up units U_k ----------
W2[0, 4] = -2.1
W2[8, 4] = -0.45
W2[16, 4] = 0.3
W2[1, 7] = -2.1
W2[9, 7] = -0.45
W2[0, 7] = -0.15
W2[8, 7] = -0.075
W2[16:18, 7] = [0.24, 0.3]
W2[2, 10] = -2.1
W2[10, 10] = -0.45
W2[1, 10] = -0.15
W2[9, 10] = -0.075
W2[16:19, 10] = [0.195, 0.24, 0.3]
W2[3, 13] = -2.1
W2[11, 13] = -0.45
W2[2, 13] = -0.15
W2[10, 13] = -0.075
W2[16:20, 13] = [0.165, 0.195, 0.24, 0.3]
W2[4, 16] = -2.1
W2[12, 16] = -0.45
W2[3, 16] = -0.15
W2[11, 16] = -0.075
W2[16:21, 16] = [0.135, 0.165, 0.195, 0.24, 0.3]
W2[5, 19] = -2.1
W2[13, 19] = -0.45
W2[4, 19] = -0.15
W2[12, 19] = -0.075
W2[16:22, 19] = [0.12, 0.135, 0.165, 0.195, 0.24, 0.3]
W2[6, 22] = -2.1
W2[14, 22] = -0.45
W2[5, 22] = -0.15
W2[13, 22] = -0.075
W2[16:23, 22] = [0.105, 0.12, 0.135, 0.165, 0.195, 0.24, 0.3]
# ---------- Layer 2: down units D_k ----------
W2[1, 2] = -2.1
W2[9, 2] = -0.45
W2[2, 2] = -0.15
W2[10, 2] = -0.075
W2[17:24, 2] = [0.3, 0.24, 0.195, 0.165, 0.135, 0.12, 0.105]
W2[2, 5] = -2.1
W2[10, 5] = -0.45
W2[3, 5] = -0.15
W2[11, 5] = -0.075
W2[18:24, 5] = [0.3, 0.24, 0.195, 0.165, 0.135, 0.12]
W2[3, 8] = -2.1
W2[11, 8] = -0.45
W2[4, 8] = -0.15
W2[12, 8] = -0.075
W2[19:24, 8] = [0.3, 0.24, 0.195, 0.165, 0.135]
W2[4, 11] = -2.1
W2[12, 11] = -0.45
W2[5, 11] = -0.15
W2[13, 11] = -0.075
W2[20:24, 11] = [0.3, 0.24, 0.195, 0.165]
W2[5, 14] = -2.1
W2[13, 14] = -0.45
W2[6, 14] = -0.15
W2[14, 14] = -0.075
W2[21:24, 14] = [0.3, 0.24, 0.195]
W2[6, 17] = -2.1
W2[14, 17] = -0.45
W2[7, 17] = -0.15
W2[15, 17] = -0.075
W2[22:24, 17] = [0.3, 0.24]
W2[7, 20] = -2.1
W2[15, 20] = -0.45
W2[23, 20] = 0.3
# ---------- Layer 2 biases (gating -60 + feature offsets; U_1, D_8 constant -1) ----------
b2[0:24] = [-61.425, -10, -61.275, -61.425, -62.25, -61.44, -61.425, -62.235, -61.74, -61.425, -62.04, -61.875, -61.425, -61.875, -62.04, -61.425, -61.74, -62.235, -61.425, -61.44, -62.25, -61.425, -61.275, -10]
# ---------- Output ----------
W3[0:24:3, 0] = 60
W3[1:24:3, 2] = 60
W3[2:24:3, 4] = 60
b3[0:6] = [420, -200, 420, -200, 420, -200]
```