I wrote the submission to `/work/submission.json` and it passed `check_submission`. That check covers format only, so correctness rests on the construction argument below, not on a test.

**Target function.** Minimising the public truth table gives

`f = ~x0 | (x4 & x6) | (~x3 & ~x9) | (x2 & x7 & ~x6)`

I confirmed symbolically that the expression I built is identical to this minimised form.

**Network.** The patch has 71 `set` operations:

- **Layer 0:**
  - Channels 0–9 pass the inputs through unchanged.
  - Channels 10–13 each compute one of the four terms with a single ReLU. Each is exactly 0 or 1 on Boolean inputs:
    - `relu(1 - x0)`
    - `relu(x4 + x6 - 1)`
    - `relu(1 - x3 - x9)`
    - `relu(x2 + x7 - x6 - 1)`
- **Layer 1:**
  - Channels 0–9 pass through again.
  - Let s be the sum of the four terms. Channel 10 holds `relu(s)`, channel 11 is zero, and channel 12 holds `relu(s - 1)`. Read in the task's triple layout (`h[j] + h[j+1] - h[j+2]`), this triple equals min(1, s), which is the OR of the terms.
- **Layers 2 and 3:** identity on channels 0–12.
- **Output head:**
  - Logit 1 is `h10 - h12 - 0.5`, so it is +0.5 when f = 1 and -0.5 when f = 0.
  - Logit 0 is fixed at 0 and logit 2 at -1.
  - Argmax therefore picks class 1 exactly when f = 1, and class 0 otherwise.