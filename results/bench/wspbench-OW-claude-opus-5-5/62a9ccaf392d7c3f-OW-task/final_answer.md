I've written the recovered Boolean function to `/work/submission.json`. `check_submission` returned "ok", but that checks format only and does not grade the answer.

The function is:

f = mux(x9, xor(nand(x0∧x5, x6∧x9), nor(x0∧x5, x6∨x8)), or(nor(x2∧x5, x3∨x9), or(nand(x5,x7), x5∧x6)))

Each signal is stored in three consecutive channels and read as h[j] + h[j+1] − h[j+2]. Read that way, the layers decode as follows:

- **Layer 0** builds gates on pairs of inputs:
  - x0∧x5, computed twice
  - x6∧x9
  - x6∨x8
  - x2∧x5
  - x3∨x9
  - nand(x5,x7)
  - x5∧x6
- **Layer 1** combines those:
  - **P** = nand(x0∧x5, x6∧x9)
  - **Q** = nor(x0∧x5, x6∨x8)
  - **R** = nor(x2∧x5, x3∨x9)
  - **S** = or(nand(x5,x7), x5∧x6)
- **Layer 2** forms T = xor(P,Q) and U = or(R,S).
- **Layer 3** is a mux on x9: relu(x9 + T − 1) + relu(U − x9) gives T when x9 = 1 and U otherwise.
- **Output:** the logits are 1 − V, V and −1, so the argmax equals V.

The submitted JSON:

{"program":{"op":"mux","condition":{"op":"input","index":9},"then":{"op":"xor","left":{"op":"nand","left":{"op":"and","left":{"op":"input","index":0},"right":{"op":"input","index":5}},"right":{"op":"and","left":{"op":"input","index":6},"right":{"op":"input","index":9}}},"right":{"op":"nor","left":{"op":"and","left":{"op":"input","index":0},"right":{"op":"input","index":5}},"right":{"op":"or","left":{"op":"input","index":6},"right":{"op":"input","index":8}}}},"else":{"op":"or","left":{"op":"nor","left":{"op":"and","left":{"op":"input","index":2},"right":{"op":"input","index":5}},"right":{"op":"or","left":{"op":"input","index":3},"right":{"op":"input","index":9}}},"right":{"op":"or","left":{"op":"nand","left":{"op":"input","index":5},"right":{"op":"input","index":7}},"right":{"op":"and","left":{"op":"input","index":5},"right":{"op":"input","index":6}}}}}}