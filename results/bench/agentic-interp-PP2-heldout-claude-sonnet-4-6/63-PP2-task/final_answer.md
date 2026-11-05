```json
{
  "tags": {
    "L0H0_ATTN": "AGGREGATOR",
    "L0_MLP": "COMBINER"
  },
  "evidence": {
    "L0H0_ATTN": {
      "hypothesis": "L0H0 spreads attention across positions containing values smaller than the current token's value, accumulating a count of such positions; ablating it collapses all predictions to 8, confirming it is the critical aggregation source.",
      "test": "r4",
      "supports": true
    },
    "L0_MLP": {
      "hypothesis": "L0_MLP fuses two distinct upstream signals: the aggregated count from L0H0 and the token embedding at its own position, since ablating L0H0 drops the MLP write norm from ~20K to ~7.7K (>100% change) and ablating L0_MLP itself also collapses predictions to all-8.",
      "test": "r11",
      "supports": true
    }
  }
}
```