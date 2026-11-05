The evidence is conclusive:

- **r2**: Position variation in L0_MLP writes is 0.47% relative to absolute values — the write is dominated by the token identity, not position.
- **r4**: Zeroing L0_MLP collapses all predictions to class 0 (completely wrong), confirming it is the primary computation.
- Output is non-binary (11 classes) → not INDICATOR.
- MLP cannot move information across positions → not ROUTER or AGGREGATOR.
- One source (token identity), non-binary output → **MAPPER**.

```json
{"tags": {"L0_MLP": "MAPPER"}, "evidence": {"L0_MLP": {"hypothesis": "L0_MLP independently maps the token at each position to a non-binary increment direction; position variation in writes is <0.5% relative (r2), confirming it reads only the local token identity, not position.", "test": "r4", "supports": true}}}
```