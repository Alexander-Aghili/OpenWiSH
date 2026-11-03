**Design: a "column-1 sentinel"**

**Strategy**
- The submarine parks at column 1, facing right, with its back cell on column 0. From there every target in its row is in front of it, so it never needs to turn.
- **Firing** is the default action. When the shot is on cooldown, "fire" acts as a no-op.
- **Seeking:** it moves up or down toward rows that hold a target in columns 2–9, and stops when its own row has one.
  - Divers count as targets only while it carries 1 diver or fewer.
  - It never moves up into row 1 to seek. A sticky repeated "up" from there would surface it, which is fatal with no diver.
- **Dodging:** it moves vertically when there is danger near its own position. It only enters a row whose cells near column 1 are clear.
- **Surfacing:** it surfaces when oxygen is below 40 and it carries at least one diver.
- **Row 0:** at the surface it lines up first. It dives only when a target exists, since oxygen does not drain at row 0.

**Layer 1 (28 units used)**
- u0–u8 S_r: the submarine is in row r.
- u9–u16 T_r: row r has a target in columns 2–9.
  - The reward channel has weight 32 and the danger channel weight 8.
  - Gauge cell (9,7) adds −8: that cell is on when 2 or more divers are carried or oxygen is at least 160. So a lone diver only counts when divers are needed.
- u17–u24 N_r: danger in row r near column 1. This is the danger channel at columns 0–3, minus the reward channel at column 3, so only bullets count at column 3.
- u25 H_L: the submarine has a cell at column 2 or further right, so it should press left.
- u26 H_R: the submarine is at x = 0 facing left (back cell at column 1), so it should press right.
  - Trails at column 1 from other objects are cancelled by subtracting the reward channel at columns 0 and 2.
- u27 Surf: gauge cell (9,8) is on and (9,1) is off, meaning oxygen is below 40 and at least one diver is carried.

**Layer 2: per-row conjunctions**
- v0 Down₀: at row 0, aligned, row 1 safe, some target exists, not surfacing.
- v1–v7 Down_s: in row s and row s+1 is safe, and either:
  - there is danger here, or
  - there is a target below, none in this row, and no need to surface.
- v8 Up₁ and v9 Up₂: these fire only when surfacing.
- v10–v15 Up_s (rows 3–8): in row s and row s−1 is safe, and either:
  - there is danger here, or
  - it needs to surface, or
  - there is a target above (rows 2 and lower), none in this row.
- v16 and v17 pass H_L and H_R through.

**Outputs**
- Up, down, left and right get a logit of 8 when active and −14 otherwise.
- Fire is fixed at 3 and no-op at −14.

```edits
# ===== Layer 1 =====
# S_r: sub in row r (units 0..8)
W1[0:60:6, 0] = 5
W1[60:120:6, 1] = 5
W1[120:180:6, 2] = 5
W1[180:240:6, 3] = 5
W1[240:300:6, 4] = 5
W1[300:360:6, 5] = 5
W1[360:420:6, 6] = 5
W1[420:480:6, 7] = 5
W1[480:540:6, 8] = 5
b1[0:9] = -5
# T_r: target in row r cols 2..9 (units 9..16)
W1[74:120:6, 9] = 32
W1[73:120:6, 9] = 8
W1[134:180:6, 10] = 32
W1[133:180:6, 10] = 8
W1[194:240:6, 11] = 32
W1[193:240:6, 11] = 8
W1[254:300:6, 12] = 32
W1[253:300:6, 12] = 8
W1[314:360:6, 13] = 32
W1[313:360:6, 13] = 8
W1[374:420:6, 14] = 32
W1[373:420:6, 14] = 8
W1[434:480:6, 15] = 32
W1[433:480:6, 15] = 8
W1[494:540:6, 16] = 32
W1[493:540:6, 16] = 8
W1[587, 9:17] = -8
b1[9:17] = -28
# N_r: danger near col 1 in row r (units 17..24)
W1[61:84:6, 17] = 10
W1[80, 17] = -10
W1[121:144:6, 18] = 10
W1[140, 18] = -10
W1[181:204:6, 19] = 10
W1[200, 19] = -10
W1[241:264:6, 20] = 10
W1[260, 20] = -10
W1[301:324:6, 21] = 10
W1[320, 21] = -10
W1[361:384:6, 22] = 10
W1[380, 22] = -10
W1[421:444:6, 23] = 10
W1[440, 23] = -10
W1[481:504:6, 24] = 10
W1[500, 24] = -10
b1[17:25] = -5
# H_L (unit 25): sub extends to col >= 2
W1[0:600:6, 25] = 10
W1[0:600:60, 25] = 0
W1[6:600:60, 25] = 0
b1[25] = -5
# H_R (unit 26): sub at x=0 facing left
W1[0:600:6, 26] = -20
W1[0:600:60, 26] = 5
W1[6:600:60, 26] = 0
W1[10:600:60, 26] = 10
W1[2:600:60, 26] = -10
W1[14:600:60, 26] = -10
b1[26] = -10
# Surf (unit 27): oxygen < 40 and >= 1 diver
W1[593, 27] = 10
W1[551, 27] = -10
b1[27] = -5

# ===== Layer 2 =====
# Down_0 (v0)
W2[0, 0] = 100
W2[17, 0] = -100
W2[25, 0] = -100
W2[26, 0] = -100
W2[27, 0] = -36
W2[9:17, 0] = 4
b2[0] = -408
# Down_1..Down_7 (v1..v7)
W2[1, 1] = 100
W2[18, 1] = -100
W2[17, 1] = 36
W2[10:17, 1] = 4
W2[9, 1] = -32
W2[27, 1] = -36
b2[1] = -208
W2[2, 2] = 100
W2[19, 2] = -100
W2[18, 2] = 36
W2[11:17, 2] = 4
W2[10, 2] = -32
W2[27, 2] = -36
b2[2] = -212
W2[3, 3] = 100
W2[20, 3] = -100
W2[19, 3] = 36
W2[12:17, 3] = 4
W2[11, 3] = -32
W2[27, 3] = -36
b2[3] = -216
W2[4, 4] = 100
W2[21, 4] = -100
W2[20, 4] = 36
W2[13:17, 4] = 4
W2[12, 4] = -32
W2[27, 4] = -36
b2[4] = -220
W2[5, 5] = 100
W2[22, 5] = -100
W2[21, 5] = 36
W2[14:17, 5] = 4
W2[13, 5] = -32
W2[27, 5] = -36
b2[5] = -224
W2[6, 6] = 100
W2[23, 6] = -100
W2[22, 6] = 36
W2[15:17, 6] = 4
W2[14, 6] = -32
W2[27, 6] = -36
b2[6] = -228
W2[7, 7] = 100
W2[24, 7] = -100
W2[23, 7] = 36
W2[16, 7] = 4
W2[15, 7] = -32
W2[27, 7] = -36
b2[7] = -232
# Up_1 (v8), Up_2 (v9): surfacing only
W2[1, 8] = 10
W2[27, 8] = 10
b2[8] = -10
W2[2, 9] = 10
W2[27, 9] = 10
W2[17, 9] = -10
b2[9] = -20
# Up_3..Up_8 (v10..v15)
W2[3, 10] = 100
W2[18, 10] = -100
W2[19, 10] = 36
W2[27, 10] = 36
W2[10, 10] = 4
W2[11, 10] = -32
b2[10] = -160
W2[4, 11] = 100
W2[19, 11] = -100
W2[20, 11] = 36
W2[27, 11] = 36
W2[10:12, 11] = 4
W2[12, 11] = -32
b2[11] = -156
W2[5, 12] = 100
W2[20, 12] = -100
W2[21, 12] = 36
W2[27, 12] = 36
W2[10:13, 12] = 4
W2[13, 12] = -32
b2[12] = -152
W2[6, 13] = 100
W2[21, 13] = -100
W2[22, 13] = 36
W2[27, 13] = 36
W2[10:14, 13] = 4
W2[14, 13] = -32
b2[13] = -148
W2[7, 14] = 100
W2[22, 14] = -100
W2[23, 14] = 36
W2[27, 14] = 36
W2[10:15, 14] = 4
W2[15, 14] = -32
b2[14] = -144
W2[8, 15] = 100
W2[23, 15] = -100
W2[24, 15] = 36
W2[27, 15] = 36
W2[10:16, 15] = 4
W2[16, 15] = -32
b2[15] = -140
# horizontal pass-through
W2[25, 16] = 4
W2[26, 17] = 4

# ===== Output =====
W3[8:16, 2] = 11
W3[0:8, 4] = 11
W3[16, 1] = 11
W3[17, 3] = 11
b3[0:6] = [-14, -3, 74, -3, 74, 3]
```