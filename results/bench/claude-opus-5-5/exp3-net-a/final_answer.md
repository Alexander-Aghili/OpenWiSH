Observations:
- **Layer 0, two broad heads.**
  - L0H4 has an essentially zero QK matrix (‖QK‖_F = 0.01, all singular values below 0.005), so it attends uniformly. Its OV is huge (‖OV‖ = 17.2, singular values 13.88, 9.65, 2.51). It reads mainly a 2-D input plane, which turns out to be positional.
  - L0H0 is similar. Its QK is small (0.11; singular values 0.09, 0.07), its OV (3.43, 2.47) reads the same plane, and it writes to the same output plane as L0H4 (overlap 0.99/0.94, OV cosine 0.8).
- **Layer 0, positional heads.**
  - L0H1 and L0H5 have QK energy that is 84% and 70% inside a positional subspace.
  - Their QK eigenvalues are complex, with phases around ±58° (H1) and ±53° (H5) in the main plane, which is a rotation, i.e. relative-position attention.
  - In other planes they differ: the generalized eigenphase between them is ±124°.
  - Their OVs are flat and high-rank (singular values about 1.0 down to 0.4 over roughly 22 dimensions). They read token identity, not the QK plane (overlap ≤ 0.3).
  - In the main plane, L0H2 is about +identity (attends at or next to the current position, with a token preference on the key side). L0H3 is about −identity (opposite phase, so a few characters back), and its OV copies the positional phase.
- **Composition.**
  - L0H4 and L0H0 K-compose into every L1 head at 0.34–0.50, against a random baseline of 0.05.
  - In each L1 head, 76–93% of the query-independent key bias (from ln1_b) lies in that plane. This looks like an attention sink at the start of the window.
  - L0H5's output is read by L1 queries and values (0.17–0.34) and by later keys.
- **Induction test.** I computed the normalised trace of Tᵀ·QK·OV_L0H5·T over the token subspace, compared with random subspaces:

  | Head | Score | Null (mean ± sd) | z |
  |---|---|---|---|
  | L2H1 | 0.90 | 0.42 ± 0.06 | 7.5 |
  | L3H1 | 0.88 | — | 6.4 |
  | L3H5 | 0.90 | — | 8.0 |
  | L3H0 | 0.49 | — | 5.6 |
  | L3H4 | 0.32 | — | 3.2 |

  The per-token diagonal is positive (z 2–4) for about 20 characters.
  - L3H5 and L3H1 put 45% and 39% of their OV energy into the output subspace of W_proj.3 (random would be 8%). They map each token to consistent output directions (cosine 0.6–0.85 between the two heads).
- **L1 heads.**
  - The six L1 heads are alike (QK cosine 0.3–0.6) and have negative OV traces (−0.72 to −0.83).
  - They avoid keys holding the same character as the query (same-token z between −4 and −6).
  - They do not read the positional subspace.
- **Vocabulary size.**
  - The W_proj.3 spectrum is steep for about 28–30 directions (28.2, 20.3, …, 7.9 at #28, then 6.5, 5.3, 4.0, followed by a plateau near 2.7).
  - ICA on the MLP0 input weights finds 26 very sparse single-character detectors (kurtosis 39–297) plus two special characters, so at least 27 characters are in active use.
- **Bigram structure in MLP0.**
  - MLP0 neurons jointly detect (previous, current) character pairs, with strong specific pairs (z up to 6) and several mutual-best pairs.
  - One whitespace-like character is never preceded by space but is strongly followed by space. I read it as newline, and newline→space is its strongest association.
  - L0H2 copies this newline (3.3× the average token). L2H1, L2H5, L3H0, L3H1 and L3H4 match "current = newline" against L0H2's newline flag (z 4.1–4.6).
  - L0H0, L0H3 and L0H4 key on or average the space character.

Heads:
- L0H0: broad — near-uniform (QK singular value 0.09), mild preference for space keys; OV copies the positional plane like H4.
- L0H1: local — purely positional rotation QK (84%); its phases differ from H5 outside the main plane, so it reaches about 1–2 characters back; its token-copying OV feeds L1 queries and values.
- L0H2: local — QK about +identity in the positional plane (current or adjacent position) plus a newline preference; acts as the newline flag.
- L0H3: local — QK about −identity (a few characters back), keys on space, OV copies the position (distance to the last space).
- L0H4: broad — QK ≈ 0, so uniform over the whole prefix; huge OV gives a window-start/sink feature and space density.
- L0H5: previous — positional rotation QK (±53°) with token-copying OV; it is the key source of every induction head.
- L1H0: broad — content-based, no positional input, sink at window start, avoids same character.
- L1H1: broad — same structure.
- L1H2: broad — same structure.
- L1H3: broad — same structure.
- L1H4: broad — same structure.
- L1H5: broad — same structure.
- L2H0: unclear — content features only, sink key bias.
- L2H1: copy — induction z 7.5 via L0H5, plus newline matching.
- L2H2: unclear — anti-induction (z −4), avoids same character.
- L2H3: unclear.
- L2H4: unclear.
- L2H5: unclear — weak induction (z 2.1) and newline matching.
- L3H0: copy — induction z 5.6.
- L3H1: copy — induction z 6.4, writes into the logit space.
- L3H2: unclear.
- L3H3: unclear — smallest QK, largest OV.
- L3H4: copy — weaker, z 3.2.
- L3H5: copy — induction z 8.0, strong direct write to logits.

Reasoning:
- **Order: as is.** The network has a positional previous-token head and induction heads whose keys read it, with token-specific positive matching. Its copy heads write token identity straight into the logits, and MLP0 contains strongly structured (previous, current) conjunctions. With characters shuffled inside each line, previous-character and copy circuits would carry almost no information, so this points firmly to unshuffled text.
- **Romaji is unlikely.** Romaji uses about 24 characters, but the model has at least 26 dedicated character detectors and about 28 output directions.
- **Python over English, with modest confidence.**
  - A lot of machinery is devoted to newline: a layer-0 newline flag, five later heads retrieving what followed earlier newlines, and newline→space as the dominant newline bigram (indentation).
  - The space-density and space-position heads, and the four to five copy heads, also fit short, indented, repetitive code.
  - English Wikipedia lines are long and newlines are rare, though lines that start with lists or numbers also produce newline→space. My attempted letter decodings were ambiguous, so the margin is modest.

corpus: python
order: as is
probabilities: python=0.55 romaji=0.07 english=0.38; as is=0.94 shuffled=0.06