```json
{
  "tags": {
    "L0H0_ATTN": "AGGREGATOR",
    "L0_MLP": "MAPPER",
    "L1_MLP": "MAPPER",
    "L2H0_ATTN": "ROUTER"
  },
  "evidence": {
    "L0H0_ATTN": {
      "hypothesis": "L0H0 reads from multiple sequence positions via soft content-based attention (mean entropy 1.47, not one-hot), producing a non-binary summary write per position that is not critical for the main swap task (0/36 errors when ablated).",
      "test": "r7",
      "supports": true
    },
    "L0_MLP": {
      "hypothesis": "L0_MLP transforms each position's token+position embedding independently into a form suitable for L2H0 to copy as values; ablating it breaks all cross-routing positions (2-9) but leaves self-routing (pos1) correct.",
      "test": "r8",
      "supports": true
    },
    "L1_MLP": {
      "hypothesis": "L1_MLP transforms each position's representation independently into the final logit-compatible form; ablating it breaks all positions including self-routing pos1, confirming it is a per-position non-binary transformation downstream of routing.",
      "test": "r9",
      "supports": true
    },
    "L2H0_ATTN": {
      "hypothesis": "L2H0 routes content from adjacent positions via purely positional index-based selection (even positions read from pos+1, odd from pos-1, pos0/1 from themselves); ablating it causes 27/36 errors and breaks all swap positions.",
      "test": "r10",
      "supports": true
    }
  }
}
```