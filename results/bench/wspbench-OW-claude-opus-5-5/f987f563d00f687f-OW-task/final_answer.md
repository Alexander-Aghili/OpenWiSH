I read the function directly from the weights and wrote it to `/work/submission.json`. `check_submission` returned `ok`, which only confirms the format. I did not evaluate the expression against the network.

**How I decoded it.** Each triple of channels (j, j+1, j+2) holds one Boolean signal, computed as h[j]+h[j+1]−h[j+2]. I read the gate for each triple from its biases and weights:

- **Layer 0:**
  - xor(x0,x7)
  - nand(x0,x7)
  - or(x0,x2), which appears twice
  - xor(x1,x9)
  - not x6
  - nor(x0,x3)
  - nor(x2,x6)
- **Layer 1:**
  - A = xor(xor(x0,x7), nand(x0,x7))
  - B = nor(or(x0,x2), xor(x1,x9))
  - C = and(or(x0,x2), not x6)
  - D = or(nor(x0,x3), nor(x2,x6))
- **Layer 2:** and(A,B), and nand(C,D).
- **Layer 3:** the XOR of the two layer-2 signals.

**Output head.** The head computes logits [1−s, s, −1], so argmax gives exactly s.

**Submission:**

```json
{"program":{"op":"xor","left":{"op":"and","left":{"op":"xor","left":{"op":"xor","left":{"op":"input","index":0},"right":{"op":"input","index":7}},"right":{"op":"nand","left":{"op":"input","index":0},"right":{"op":"input","index":7}}},"right":{"op":"nor","left":{"op":"or","left":{"op":"input","index":0},"right":{"op":"input","index":2}},"right":{"op":"xor","left":{"op":"input","index":1},"right":{"op":"input","index":9}}}},"right":{"op":"nand","left":{"op":"and","left":{"op":"or","left":{"op":"input","index":0},"right":{"op":"input","index":2}},"right":{"op":"not","arg":{"op":"input","index":6}}},"right":{"op":"or","left":{"op":"nor","left":{"op":"input","index":0},"right":{"op":"input","index":3}},"right":{"op":"nor","left":{"op":"input","index":2},"right":{"op":"input","index":6}}}}}}
```