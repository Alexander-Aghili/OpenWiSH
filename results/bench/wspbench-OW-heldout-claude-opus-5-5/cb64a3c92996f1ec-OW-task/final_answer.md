I wrote the submission to `/work/submission.json` and it passed `check_submission`. That check covers format only, so correctness is not graded. I also did not run the network on any inputs, so the network's behaviour has not been tested.

**Logic.** Exact minimisation of the public truth table (124 of 1024 inputs are true) gives a 7-term sum of products:
f = (x0·x3·x4·¬x1·¬x2) | (x3·x4·x8·¬x1·¬x2) | (x4·x7·x9·¬x5·¬x8) | (x4·x7·¬x0·¬x5·¬x8) | (x4·x7·¬x2·¬x5·¬x8) | (x0·x2·x4·x8·¬x5·¬x9) | (x0·x2·x4·¬x5·¬x7·¬x9)

**Construction.** The patch has 74 `set` operations:
- **Layer 0:** Each of the 7 terms gets its own channel (0–6). The weights are +1 for plain inputs and −1 for negated inputs, with b0 = −(number of plain inputs − 1). For 0/1 inputs the ReLU output is exactly 1 when every condition in the term holds, and 0 otherwise.
- **Layers 1–3:** Channels 0–6 are copied forward unchanged (diagonal weight 1, bias 0). The values are never negative, so the ReLU leaves them alone.
- **Output:** Output 1's score is the sum of the 7 term channels. Output 0's score is a constant 0.5 from its bias, and output 2's is −1. Output 1 therefore wins when at least one term is true; otherwise output 0 wins, and output 2 never does.

The one check I did run is symbolic: the terms encoded in the patch are logically equivalent to the minimised formula.