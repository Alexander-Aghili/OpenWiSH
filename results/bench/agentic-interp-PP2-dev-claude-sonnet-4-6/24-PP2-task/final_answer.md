```json
{
  "tags": {
    "L0H0_ATTN": "ROUTER"
  },
  "evidence": {
    "L0H0_ATTN": {
      "hypothesis": "L0H0_ATTN concentrates attention on one specific position per query (itself or the previous occurrence of the same token), routing that position's content to the current position — evidenced by one-hot attention patterns and by zero-ablation that eliminates all 'b' predictions while leaving 'a' and 'c' predictions unchanged.",
      "test": "r6",
      "supports": true
    }
  }
}
```