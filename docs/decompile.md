# Decompile environments: networks back to programs

## The question

You give an agent only the weights of a small trained network. Does the agent write a short deterministic program
which is *the same system*: the same outputs on every input, and the same internal causal structure? This is
mechanistic interpretability with a score.

## Networks: secret programs on exhaustive input spaces

`scripts/prepare_decompile.py` runs outside the agent's sandbox, where training is allowed. The script draws a
random program over 10 input bits from a small DSL: 2 or 3 intermediate variables (parity, AND or OR of literals,
popcount, 2-bit comparison), combined by a random lookup table into 2 or 3 classes. The script trains a ReLU MLP
`10 -> h1 -> h2 -> C` (with L1, then prunes weights under 0.02) on a random half of the 1,024 inputs, with 4 of
those training labels changed to another class.

Why the flipped labels: on clean labels these networks generalise perfectly (no quirks at any training fraction we
tried), so a program which only guesses the task would score full behaviour. The network memorises the flipped
labels as exceptions, and the exceptions spill over to nearby inputs: 5 to 14 inputs where the network disagrees
with the secret program. A correct decompile reproduces those, and the exception circuit is one more thing to find.
The agent never sees the program, the split, the flips or the quirks.

| Network | Widths | Classes | Secret variables | Quirks | Nonzero weights |
|---|---|---|---|---|---|
| d1-a, d1-b | 12, 6 | 2 | 2 (AND, OR, parity of 2) | 5, 8 | 83, 54 |
| d2-a, d2-b | 16, 8 | 3 | 2 (any op) | 6, 8 | 120, 101 |
| d3-a, d3-b | 24, 12 | 3 | 3 (any op) | 9, 14 | 117, 142 |

## What the agent submits

A ```` ```program ```` block in the final reply. Pure Python, no imports, deterministic, with a fixed list of
builtins (the prompt names them):

```python
VARIABLES = {"a": ("h1", [3, 7]), "b": ("h2", {"basis": [[0.7, 0, -0.7, 0, 0, 0]]})}  # units or a subspace
def program(x, fix={}):                                                             # x: tuple of 10 ints (0/1)
    a = fix["a"] if "a" in fix else (x[0] ^ x[4])
    b = fix["b"] if "b" in fix else ...
    return cls, {"a": a, "b": b}
```

- `VARIABLES` is a plain literal. Its numbers are free, and the program never reads `VARIABLES`.
- Places of variables in one layer never overlap: disjoint units, orthogonal subspaces.
- Each variable is a discrete causal variable: an int or bool with at most 16 values over the inputs.
- The program is much smaller than the network: all literals together (every character of every number and string)
  hold at most one character per nonzero weight (at least 64), and the program at most 3,000 bytes. A copy of the
  weights does not fit.
- The `read` and `probe` sandboxes define `check_program(source)`, which applies the scorer's own size and format
  rules. The check never runs the network and never grades.

## Scoring

Exhaustive over the 1,024 inputs, or over fixed-seed pairs with 8 sources per base input.

- **Behaviour**: argmax agreement with the network on all 1,024 inputs. **Quirks** (reported): agreement on the
  inputs where the network disagrees with its secret program.
- **IIA** (interchange intervention accuracy, Geiger et al.): for each variable, the scorer sets
  `h = h_base + (h_source - h_base) P` in the variable's layer (P projects onto its units or subspace) and compares
  the network's class with `program(base, fix={v: value on source})`.
- **Completeness**: in every layer holding variables, the scorer resamples the complement of their places from
  another input with the same variable values. The network's answer must not change (causal-scrubbing style).
- **Not counted** toward the mechanism score, but reported: inert variables (patching changes neither answer on 1 %
  of pairs) and trivial ones (fixing the variable alone fixes the program's answer on 99 % of pairs, for example
  "the whole layer = the class").
- **Score** = behaviour x mechanism, where mechanism = mean IIA over counted variables x mean completeness. The run is
  exact when every part is 1.0 with at least one counted variable.
- **Diagnostics**: how well each declared variable predicts each secret variable, and the calibration below.

The submitted program runs in a separate `python -I -S` process with restricted builtins, an AST check (no imports,
no dunder names or attributes, no `open`, `eval` or `getattr`), 1 GB of memory and 300 s of CPU.

## Calibration

`scripts/calibrate_decompile.py` runs outside the agent's sandbox. For each *secret* variable, the script finds the
best IIA over every set of at most 3 units of one layer:

| Network | Secret variables, best IIA |
|---|---|
| d1-a | 0.965, 0.955 |
| d1-b | 0.989, 0.988 |
| d2-a | 0.893, 0.845 |
| d2-b | 0.870, 0.809 |
| d3-a | 0.888, 0.913, 0.964 |
| d3-b | 0.767, 0.820, 0.899 |

The planted variables sit only partly in single units, and less so in the wider networks. Trained networks spread
variables across units, as the interpretability literature reports. So a variable's place is also allowed to be a
subspace (distributed alignment search, Geiger, Wu et al. 2023), which an agent works out from the weights by linear
algebra, without fitting. The numbers are a reference, not a ceiling: an agent sometimes finds variables of its own
which the network represents more cleanly than the secret ones.

## Tiers

A ladder over the same six networks, so you measure what each channel adds (see [Access levels](access-levels.md)):

- `native`: every nonzero weight as text in the prompt, no code, one reply.
- `read`: weights through Python only, with no forward passes.
- `probe`: Python plus 3 `forward` passes of up to 16 inputs each (48 inputs, under 5 % of the input space),
  returning every layer's activations, with optional patches by units or subspace. The budget stays small, so the
  rule never comes from reading outputs alone.

The no-optimisation rule applies unchanged. No fitting a program or weights to network outputs. The network runs
only through the declared probe.

## Tools for people

`interp/` holds tools for people, kept apart from OpenWiSH runs: an MLP explorer page (after neuron-explorer's
method, with an overlay of a decompile run) and a pinned fetch of neuron-explorer. No OpenWiSH code imports them.

## Results

The ladder results, Claude Opus 5.5 on d2-a and d3-b at all three tiers, are in
[Benchmarks](benchmarks.md#decompile-ladder-mechanistic-interpretability).
