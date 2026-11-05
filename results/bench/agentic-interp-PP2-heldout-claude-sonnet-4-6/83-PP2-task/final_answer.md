The evidence is conclusive:
- **L0_MLP normal writes** range from -1695 to -670 (large values)
- **L0_MLP writes with attention ablated**: dramatically different (max |diff| = 1385.7)
- L0 attention heads route BOS information to every position, and L0_MLP fuses this with the current token embedding → **COMBINER**

The submission has been written to `/work/submission.json`:

```json
{
  "tags": {
    "L0_MLP": "COMBINER"
  },
  "evidence": {
    "L0_MLP": {
      "hypothesis": "L0_MLP fuses two distinct upstream signals at each position: the current token's embedding and the BOS-routed information written by L0 attention heads (all four L0 heads attend ~100% to BOS), producing a non-binary residual stream write.",
      "test": "r1",
      "supports": true
    }
  }
}
```