```json
{
  "tags": {
    "L0H0_ATTN": "MAPPER",
    "L0_MLP": "MAPPER"
  },
  "evidence": {
    "L0H0_ATTN": {
      "hypothesis": "L0H0 transforms each position's token value independently into a scaled direction vector in the residual stream — magnitude increases monotonically with token value (correlation 0.99), same token at different positions gives identical write direction (cosine=1.0), and ablation causes all predictions to collapse to 8 uniformly.",
      "test": "r5",
      "supports": true
    },
    "L0_MLP": {
      "hypothesis": "L0_MLP amplifies the per-position token value encoding (~1000x) into a large-magnitude direction proportional to token value — same token at any position gives cosine=1.000, and ablation causes all predictions to collapse to 3-4 (median), losing all rank discrimination.",
      "test": "r6",
      "supports": true
    }
  }
}
```