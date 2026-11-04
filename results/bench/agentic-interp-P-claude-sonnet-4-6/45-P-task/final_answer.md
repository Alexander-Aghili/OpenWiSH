Pattern confirmed: pos0 predictions [3, 6, 9] = 3 × avg(first-half tokens). L0H0_ATTN aggregates first-half values across positions 1-5 at position 0 — classic AGGREGATOR behavior (reduces content from multiple positions into a summary quantity). The ablation confirmed it doesn't affect positions 1-9 at all.

Final classification summary:
- **L0H0_ATTN → AGGREGATOR**: aggregates first-half token values at position 0 (BOS), producing a summary proportional to the first-half average
- **L0_MLP → COMBINER**: fuses the token-value signal (from W_E) and position signal (from W_pos) to decide whether to double — combining two distinct upstream signals
- **L1_MLP → MAPPER**: per-position arithmetic refinement of the already-combined representation
- **L2_MLP → MAPPER**: per-position final transformation to output class space

{"tags": {"L0H0_ATTN": "AGGREGATOR", "L0_MLP": "COMBINER", "L1_MLP": "MAPPER", "L2_MLP": "MAPPER"}}