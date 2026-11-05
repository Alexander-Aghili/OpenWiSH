```json
{
  "tags": {
    "L0_MLP": "MAPPER"
  },
  "evidence": {
    "L0_MLP": {
      "hypothesis": "L0_MLP reads the single residual stream at each position independently and transforms it into a non-binary d_model=4 vector that maps token n to its incremented representation n+1, operating position-wise without cross-position communication.",
      "test": "r3",
      "supports": true
    }
  }
}
```