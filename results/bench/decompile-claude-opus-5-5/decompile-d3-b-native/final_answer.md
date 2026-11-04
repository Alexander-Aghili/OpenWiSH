The program below reproduces the network's class map as I derived it by hand. I could not run it against the network, so the behaviour figure is an estimate. I expect most of the 1,024 inputs to match, with the misses in borderline cases I did not check one by one.

**What the network computes.** Hidden units h2[0]–h2[5] are dead, and h2[9] is nearly constant. The answer depends on h2[6], h2[7], h2[8], h2[10] and h2[11]. These are driven by three input features:

- **s = x1 + x5 + x6 + x9.** h1[6] fires when s = 0, h1[15] when s ≥ 2, and h1[20] grows with s.
- **A = x3 and not x2.** This is what h1[22] detects.
- **B = x4 and not x7 and not x8.** h1[14] and h1[17] carry x7, x8 and x4.

The network's class for each combination:

| (A, B) | s=0 | s=1 | s=2 | s=3 | s=4 |
|---|---|---|---|---|---|
| (0, 0) | 1 | 0 | 2 | 1 | 1 |
| (0, 1) | 0 | 0 | 1 | 0 | 0 |
| (1, 0) | 0 | 0 | 1 | 2 | 1 |
| (1, 1) | 1 | 2 | 0 | 0 | 0 |

There are a few exceptions near the decision boundary, all involving x0. The program handles them explicitly: three cases at s = 0, and one at s = 1 where x2 = 1, x0 = 1 and x4, x7, x8 are all 0.

**How the program is built.** Each live h2 unit gets one discrete variable, set to that unit's typical activation level in each cell of the table. The class is the argmax of approximate logits computed from those levels. I checked by hand that these logits give the right class in all 20 cells, each with a margin of at least 1.

- **lowcount (h2[7]):** large when s < 2, or when s = 2 and A holds. It blocks class 2.
- **highcount (h2[8]):** rises with s from s = 3 upward.
- **pull0 (h2[10]):** large when B holds or s ≥ 3, reduced by A.
- **push0 (h2[11]):** large for small s.
- **stay1 (h2[6]):** large unless A or B holds.

```program
VARIABLES = {
    "stay1": ("h2", [6]),
    "lowcount": ("h2", [7]),
    "highcount": ("h2", [8]),
    "pull0": ("h2", [10]),
    "push0": ("h2", [11]),
}

def program(x, fix={}):
    c, x1, a, b, p, x5, x6, q, r, x9 = x
    s = x1 + x5 + x6 + x9
    A = b * (1 - a)
    B = p * (1 - q) * (1 - r)
    if s == 0:
        if A:
            B = max(B, c * (1 - r) * int(p == q))
        else:
            B = p * (1 - q) * (r * (1 - c) * (1 - a * b) + (1 - r) * (1 - a * b * c))
    e4 = int(s == 1) * a * (1 - b) * (1 - p) * (1 - q) * (1 - r) * c * (1 - x5)

    def get(n, v):
        return fix[n] if n in fix else v

    low = int(s < 2 or (s == 2 and A == 1))
    v7 = get("lowcount", low * (6 + 2 * int(s == 0) - 3 * B))
    v8 = get("highcount", max(0, 3 * s - 7 + 2 * (a - b)))
    if s < 2:
        t10 = (3 - s) * B * (1 - A)
    else:
        t10 = max(0, 2 * s - 4 + 3 * B - A * (2 - B))
    v10 = get("pull0", t10)
    v6 = get("stay1", max(0, 3 + 2 * int(s > 1) - 2 * B - 3 * A))
    if s == 0:
        t11 = 1 - B
    elif s == 1:
        t11 = 3 - A - B - 2 * e4
    elif s == 2:
        t11 = (1 - A) * (1 - B)
    else:
        t11 = 0
    v11 = get("push0", t11)

    L0 = 5 * v10 + 6 * v11 - 3 * v6 - 6
    L1 = 2.5 * v6 - 5 * v11
    L2 = 14 - 3 * v7 - 2 * v8 - 3 * v10
    cls = 0
    if L1 > L0:
        cls = 1
    if L2 > max(L0, L1):
        cls = 2
    return cls, {"stay1": v6, "lowcount": v7, "highcount": v8,
                 "pull0": v10, "push0": v11}
```