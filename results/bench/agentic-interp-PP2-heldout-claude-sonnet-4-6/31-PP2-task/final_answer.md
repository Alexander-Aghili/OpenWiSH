```json
{
  "tags": {
    "L0_MLP": "INDICATOR"
  },
  "evidence": {
    "L0_MLP": {
      "hypothesis": "L0_MLP detects whether the token at each position is an anagram of 'listen' (i.e., token id 7) from a single input source and emits a binary directional flag: all non-listen tokens give the same direction (pairwise cosine ≥ 0.9999) while 'listen' gives the opposite direction (cosine ≈ −0.996), confirmed by zero-ablation causing all-True predictions.",
      "test": "r7",
      "supports": true
    }
  }
}
```