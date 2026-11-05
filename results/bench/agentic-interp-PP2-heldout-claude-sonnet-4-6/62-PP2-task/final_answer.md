```json
{
  "tags": {
    "L0_MLP": "COMBINER"
  },
  "evidence": {
    "L0_MLP": {
      "hypothesis": "L0_MLP fuses the current token's embedding with L0 attention context from prior positions: for the same token=5 at position 5, L0_MLP writes differ by up to 77 units (~25%) depending on what tokens occupied positions 1-4, proving it reads multiple distinct upstream signals (own-position token embedding plus attention-aggregated prior context).",
      "test": "r13",
      "supports": true
    }
  }
}
```