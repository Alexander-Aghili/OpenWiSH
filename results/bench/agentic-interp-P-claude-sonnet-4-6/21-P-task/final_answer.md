The submission is finalized. Here's my reasoning summary:

**L0_MLP → MAPPER**: Block 0 MLP runs before any circuit attention, transforming each position's (token+positional) embedding independently. Cos similarity ~0.84-0.91 for same-token first vs dup pairs confirms position-wise behavior.

**L1H0_ATTN → AGGREGATOR**: At pos5 (dup 'b'), attention spreads across *multiple* previous 'b' positions [pos1:0.046, pos2:0.624, self:0.291], accumulating a "seen before" signal. Seeds |first−dup| = 5.984.

**L1_MLP → COMBINER**: First component that sees *both* token identity (from embedding+L0_MLP) and the aggregated seen-before signal (from L1H0_ATTN) as distinct streams in the residual. Amplifies to |first−dup| = 11.559.

**L2_MLP → INDICATOR**: Strongest angular discrimination (cos 0.47 for 'a' first vs dup in input 0 — lowest of all components). Thresholds the combined signal into a binary first-occurrence flag. Amplifies to |first−dup| = 21.357.

**L3H0_ATTN → ROUTER**: Routes content between positions (dup 'a' positions attend with prob ~1.0 to specific dup 'c' positions). Write direction nearly identical for first vs dup of same token (cos~0.994), consistent with routing token-class content rather than discriminating.

{"tags": {"L0_MLP": "MAPPER", "L1H0_ATTN": "AGGREGATOR", "L1_MLP": "COMBINER", "L2_MLP": "INDICATOR", "L3H0_ATTN": "ROUTER"}}