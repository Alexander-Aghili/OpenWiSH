Looking at the examples, the task is: for each token value `t`, output `(t % 10) + (t // 10)` (sum of ones digit and tens digit). For example, token 19 → 9 + 1 = 10, token 28 → 8 + 2 = 10, token 14 → 4 + 1 = 5.

**Analysis of each component:**

**L0_MLP**: Receives the residual stream at each position (token embedding + positional embedding) and applies a position-wise nonlinear transformation. Its role is to extract/represent digit-related features (ones digit, tens digit, or intermediate combinations) from the token's embedding — a single-position-to-single-position, non-binary transformation. → **MAPPER**

**L1_MLP**: Receives the residual stream at each position (now enriched by L0 attention + L0_MLP outputs + L1 attention) and applies another position-wise transformation to produce the final digit sum output. Again, it operates independently per position, transforming a single stream into a non-binary scalar output (the digit sum, range 0–11). → **MAPPER**

Neither MLP aggregates across positions, routes content between positions, emits binary flags, or fuses two separately-sourced upstream signals — they each independently remap the residual stream value at their position.

```json
{"tags": {"L0_MLP": "MAPPER", "L1_MLP": "MAPPER"}}
```