I’ll trace the encoded Boolean signals through the four layers and return the expression.


The final layer selects between two branches using input 9. I’m checking the earlier signal triples to resolve each branch.


{"program":{"op":"mux","condition":{"op":"input","index":9},"then":{"op":"and","left":{"op":"or","left":{"op":"input","index":4},"right":{"op":"input","index":6}},"right":{"op":"nand","left":{"op":"and","left":{"op":"input","index":0},"right":{"op":"input","index":5}},"right":{"op":"input","index":6}}},"else":{"op":"or","left":{"op":"not","arg":{"op":"input","index":3}},"right":{"op":"or","left":{"op":"nand","left":{"op":"input","index":4},"right":{"op":"input","index":6}},"right":{"op":"input","index":5}}}}}