```json
{
  "tags": {
    "L0_MLP": "COMBINER"
  },
  "evidence": {
    "L0_MLP": {
      "hypothesis": "L0_MLP combines two distinct upstream signals: the token embedding and L0 attention head outputs that carry context from prior positions; ablating L0_MLP (r5) causes incorrect predictions, and ablating L0 attention heads (r3) removes the context-dependence of L0_MLP's writes (diff drops from 5.79 to 0.0 for same token+position with different prev tokens).",
      "test": "r5",
      "supports": true
    }
  }
}
```