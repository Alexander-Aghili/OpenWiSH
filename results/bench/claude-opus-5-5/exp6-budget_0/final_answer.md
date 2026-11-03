The network is saved. All 14 tensors match a fresh rebuild from `/work/net3.py`, with the parameters recorded in `/work/current.json`. Because the pass budget was 0, it has never played a game, so everything below is by construction and reasoning, not measurement.

## Design

All weights are written by code as small-integer logic. Each hidden unit computes a 0/1 (or small-count) test on the board. No weight is fit to data.

**Carried board.** Layers 1–5 pass the own-stone and opponent-stone planes forward unchanged, so every layer can still read the board.

**Group-safety detector (layers 1–6), run for both colours.**
- **Layer 1** counts each stone's empty neighbours (its liberties).
- **Layer 2** flags proof that a group has at least two liberties. A flag is raised when:
  - a stone has two or more liberties; or
  - two adjacent stones both have a liberty (adjacent points share no neighbour, so those liberties must differ); or
  - three or more liberty contacts touch one stone's neighbours; or
  - (opponent only) the stones on opposite sides of a surrounded stone both have liberties.
- **Layers 3–6** spread a "clean" bit through each group, four steps deep: a stone stays clean only if no flag exists within four steps along its group.
- Every flag really means at least two liberties, so a group truly in atari is always seen as clean, whatever its size. The error runs one way: some groups with two liberties also look clean, so the network then plays atari on them or reinforces them.

**Move score (one logit per point):**
- **Capture:** +10 for each neighbouring clean opponent stone, plus a little extra per nearby clean stone so larger captures score higher.
- **Save:** +5 for extending a clean (endangered) own group.
- **Self-atari:** a penalty of 6.5 per missing liberty. It estimates liberties after the move from empty neighbours, links to healthy own groups, and captures. A doomed ladder extension therefore scores net negative.
- **Eyes:** −15 for a point with three own neighbours and one empty one, −30 for a true eye. This keeps groups from filling their own eyes or shrinking a two-point eye to one.
- **Cutting points:** +4.5 for connecting where an opponent stone sits on the diagonal.
- **Atari on lone stones:** a small bonus for playing next to a single opponent stone with at most two liberties.
- **Strategy:**
  - a position prior that favours the centre and third/fourth lines and penalises the edge;
  - an influence term, taken over Manhattan distance up to 3, that favours contested boundary points;
  - a penalty for moves deep inside our own area, and a smaller one for moves deep inside the opponent's area.
- **Pass:** fixed at −6, so it is rarely chosen while any reasonable move exists. Under area scoring, filling dame, removing invaders and throwing stones into enemy space still count.

**Numerical safety.** Logic units use integers of at most 34, so they stay exact even if evaluation uses reduced-precision (TF32 or bf16) matrix multiplies. Running-score accumulators carry a +100 offset that the output bias removes, and that keeps them positive. Each accumulator's worst case is checked analytically.

## Why I expect it to win

The opponents are weak sampled policies: they beat random only 79–98% of the time, and they sample at temperature 1, so they often leave groups in atari, fill their own liberties and make self-ataris. This network always sees and punishes any real atari, of any group size. It reliably escapes atari when escape is possible, rarely gives away stones, never fills its own eyes, and fights over boundary points. It also keeps playing until the board is settled, which matters under area scoring with no dead-stone removal.

## Known limitations

- There is no ladder reading.
- Very large groups can produce false "capture" signals.
- The strategic weights are hand-set guesses.