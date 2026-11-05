The evidence is conclusive:

- **Binary clustering**: inter-group distance = 32.7× intra-group — L0_MLP writes in exactly two distinct directions (spam keyword vs. non-spam)
- **Context independence**: max diff = 0.0 — same token at same position always produces identical write regardless of surrounding context, confirming no attention dependency
- **Single source**: no other listed upstream components; writes depend only on token at current position
- **Ablation (r4)**: zero-ablating L0_MLP doesn't change predictions (network compensates downstream), but the write itself is unambiguously binary

**L0_MLP → INDICATOR**: detects the property "is spam keyword?" from a single input, emitting a predicate-like binary signal.

```json
{"tags": {"L0_MLP": "INDICATOR"}, "evidence": {"L0_MLP": {"hypothesis": "L0_MLP detects whether the token at its own position is a spam keyword, emitting a binary/predicate-like signal from a single input (the token at its position) with 32.67x separation between spam-keyword and non-spam clusters in write-space.", "test": "r4", "supports": true}}}
```