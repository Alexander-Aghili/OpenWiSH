I wrote the submission to `/work/submission.json`. `check_submission` returned "ok", but that only checks the format. I did not run the network on any inputs, because the rules forbid it, so correctness rests on the construction below.

**Decoding the truth table.** I re-entered the 1,024 outputs by hand and my first copy had 2 extra values. Exactly one way of dropping two zeros made every block of four outputs constant, and that version also showed the expected symmetry, so I used it. The output depends only on x0 and x2 through x7; x1, x8 and x9 have no effect. Exact two-level minimisation gives 7 product terms:

f = x6·¬x2·¬x7 + x0·x5·¬x2·¬x7 + x0·x5·¬x3·¬x6 + x3·x4·¬x2·¬x7 + x3·¬x2·¬x5·¬x7 + x2·x3·x5·¬x0·¬x4·¬x6 + x3·x5·x7·¬x0·¬x4·¬x6

I evaluated this expression on all 1,024 inputs and it matches the table exactly.

**Construction.**
- **Stage 0:** channels 0–9 copy the inputs. Channels 10–16 compute the 7 terms, each as relu(sum of positive literals − sum of negated literals − (number of positive literals − 1)). For 0/1 inputs this is exactly 1 when the term is true and 0 otherwise.
- **Stage 1:** channel 11 = relu(S) and channel 12 = relu(S − 1), where S is the sum of the terms. The signal triple h10 + h11 − h12 then equals min(1, S), which is exactly the OR of the terms.
- **Stages 2–3:** these pass channels 0–12 through unchanged.
- **Readout:** the class-1 logit is h11 − h12 and the class-0 logit is the bias 0.5, so the network picks class 1 exactly when f = 1. The third logit has bias −10, so it is never chosen.

The patch is 101 `set` operations, one for each nonzero entry, applied to the all-zero network.