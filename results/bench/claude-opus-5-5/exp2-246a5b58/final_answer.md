**Observations:**

- **Which pixels the network reads (W1 pixel rows).** Rows of W1 for pixels on row 0, column 0 and column 11 are all tiny, about |w| ≤ 0.03. The pixels that carry large weights form a region that is narrow at the top and wide at the bottom:
  - row 1: only columns 3–8 (pixel 17 → unit 19 = 0.389, pixel 20 → unit 14 = 0.176)
  - row 2: columns 2–9 (pixel 29 → unit 19 = 0.629, pixel 32 → unit 14 = 0.396)
  - row 3: columns 2–10
  - rows 4–11: columns 1–10
- **Edge detectors in layer 1.**
  - Unit 0 reads a down-left diagonal: pixel 29 (row 2, column 5) = 0.305, pixel 41 = 0.376, pixel 52 = 0.244, pixels 86/87 = 0.267/0.278, pixel 98 = 0.474, pixel 109 = 0.149. That traces a line from about (2,5) to (9,1).
  - Unit 20 reads the mirror-image diagonal: pixel 31 = 0.238, pixel 56 = 0.400, pixel 68 = 0.355, pixel 81 = 0.514, pixel 93 = 0.366, pixel 94 = 0.395. That traces (2,7) to (7,10).
  - Other units read whole horizontal rows near the bottom:
    - unit 44 is positive on row 9 (about 0.25–0.36);
    - unit 25 is positive on row 10 (0.37, 0.56, 0.41, …);
    - units 17 and 31 are negative on rows 9–10 (−0.21 to −0.36);
    - unit 38 is positive on all of row 11 (0.13–0.33);
    - units 12 and 13 are negative on row 8 (about −0.3).
- **What the output units write (W3).**
  - h2 unit 13 writes a line of white from (1,5) down-left to (8,2): pixel 17 = 0.129, pixel 40 = 0.328, pixel 63 = 0.368, pixel 74 = 0.450, pixel 87 = 0.457, pixel 98 = 0.351.
  - Unit 21 writes the mirror line from (1,6) down-right to (8,10): pixel 18 = 0.350, pixel 43 = 0.464, pixel 56 = 0.387, pixel 81 = 0.419, pixel 106 = 0.301.
  - Units 8, 34 and 47 write similar slanted edges with a slope of about ½ to ¾ column per row. Units 39 and 56 write the same edges with negative sign, i.e. "this edge pixel is black".
  - Units 11, 12, 20, 35, 36, 38, 43 and 61 write horizontal bands. For example, unit 11 writes row 8 positive (up to 0.273) and row 9 negative (about −0.36). Unit 43 writes row 10 at 0.31–0.49 in columns 3–8. Unit 38 writes row 7 positive.
  - Unit 1 writes a small white wedge at the top: rows 2–4, columns 5–9, values 0.04–0.20. It also writes rows 6–7 negative (−0.16 to −0.39).
- **Output biases (b3).** Row 0 and the left and right columns sit at about −0.4 to −0.5. Row 1 is brighter only at columns 5–6 (−0.143, −0.105). Row 2 peaks at columns 5–7 (−0.013, −0.054, −0.055). The interior from row 3 to row 9 is near 0. Rows 10–11 are about −0.15 to −0.25.

**Reasoning:**

- The background is always black, especially row 0 and the side columns.
- The variable region is a wedge. It is only columns 3–8 wide at row 1 and spreads to columns 1–10 by row 4.
- The dominant edge features are two mirror-image diagonals that meet at the top centre (about row 1, columns 5–6), and they diverge at roughly ½ column per row.
- The bottom of the shape is closed by a horizontal edge whose row varies, somewhere between row 7 and row 11.
- Several units write solid white across the area between the diagonals, so the shape is filled rather than an outline.
- Together this is an upward-pointing, filled, isosceles triangle with its apex at the top centre and a flat base.
- What varies between images is the size (where the base sits and how wide it is) and, slightly, the apex column (5–7). Unit 1, for instance, writes a small triangle that stops at row 4.
- Ruled out:
  - circles or squares, which would produce horizontal top edges and column-wise side edges rather than converging diagonals;
  - downward-pointing triangles, since row 0 is never used and the region is widest at the bottom;
  - outlines, because the network writes filled interiors;
  - letters or digits, since there is no stroke structure inside the shape.

**Images:**
- Each image is a single solid white triangle on a black background, pointing up.
- The apex is near the top centre, at row 1–2, columns 5–6.
- The two slanted sides widen by about one pixel every two rows.
- The flat base lies on one of rows 7–11. It is typically about 8–10 pixels wide (columns 1–10) and about 8–9 rows tall.
- Some samples are smaller triangles nearer the top.

**Example:**
```
............
.....##.....
.....##.....
....####....
....####....
...######...
...######...
..########..
..########..
.##########.
............
............
```

**Confidence:** 45%