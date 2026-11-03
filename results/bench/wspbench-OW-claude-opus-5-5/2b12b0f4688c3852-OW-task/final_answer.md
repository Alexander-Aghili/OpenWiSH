The submission is written to `/work/submission.json` and passed `check_submission`, which checks format only and does not grade. The before and after programs are the same.

**The edit changes nothing.** The before, after and target checkpoints are identical in every tensor. So the network computes the same Boolean function before and after the edit.

**The function, decoded from the weights:**

f = R ⊕ T ⊕ (P ∧ Q), where:

- R = (x3 ∨ x4) ∧ ¬(x0 ∧ x7)
- T = ¬x8 ∧ ¬x4 ∧ ¬x5
- P = NOR(x3∧x4, x5∨x6)
- Q = (x5⊕x8) ⊕ (x6⊕x8), which simplifies to x5⊕x6

The P ∧ Q term is always 0, because P needs x5 = x6 = 0 and Q needs x5 ≠ x6. So in practice f = R ⊕ T. I kept the term in the submitted expression anyway, to match the circuit as built.

**How the output is read:** the logits are (1−S, S, −1), so the argmax output equals the final signal S, which is f.

**Checks:** I evaluated only my own written expression over all 1024 inputs, not the network. P ∧ Q was never 1, and f is 1 on 608 of the inputs.