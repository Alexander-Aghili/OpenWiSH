The program is an approximate re-derivation of the network from its weights, not a hand-found clean rule. I could not run it, so expect roughly 90% agreement with the network rather than 100%. The answer reaches 118 of the 120 allowed literal characters.

**How the network works:**
- **Dead units.** In layer 1, units 4, 6, 8, 11, 12, 13 and 15 are constants: unit 13 is a fixed 0.05 and the rest are 0. In layer 2, units 0, 3, 5 and 6 are always 0.
- **Linear units.** h1[0] and h1[1] are always positive, so they reduce to linear terms in x. I fold them into the biases of the next layer.
- **h2[4] (variable `c`).** It is always positive and is linear in x3, x5, x7, h1[5] and h1[10].
- **h2[7] (variable `d`).** It is nonzero only when x0=0, x2=1 and x8=1. In that case it is about 5.9 − 1.47·h1[10].
- **Region where x0, x2 and x8 are all 0.** Here h1[3] is large, which makes h2[2] (variable `b`) at least about 4. In the cases I checked the class was always 0, and I expect this for the whole region.
- **h2[1] (variable `a`).** It sets class 1 against class 2. It is roughly 2.8 + 1.2·x0 + 2.5·h1[2] − 1.77·(h1[7] + h1[10] + h1[14]).

**How the program reproduces it:**
- **Layer-1 units.** Each relevant unit is recomputed with integer weights, using a per-unit scale chosen so the weights come out close to integers.
- **Layer-2 units.** Each of the four live units becomes a discrete variable with at most 16 values. `a` and `b` use steps of 0.3, capped at 15. `c` uses steps of 0.7 and `d` steps of 0.4.
- **Class.** It is the largest of three values computed from those discrete variables: L0−L2, L1−L2 and 0. The two differences are rescaled by a common factor of about 1.29 so their coefficients are short.
- **Completeness.** The only parts of h2 outside the variables are the dead units, so swapping them should not change the answer.

```program
VARIABLES = {
    "a": ("h2", [1]),
    "b": ("h2", [2]),
    "c": ("h2", [4]),
    "d": ("h2", [7]),
}

def program(x, fix={}):
    A, B, C, D, E, F, G, H, I, J = x
    R = lambda v: max(v, 0)
    Q = lambda v: min(15, round(R(v) / 3))
    # layer-1 units (rescaled integer weights)
    p = R(11 * (1 + G) - 8 * A - 10 * J + 5 * (F - E) + 4 * D - 3 * C + 2 * (I - H))   # h1[2]
    s = R(11 * (1 - A - C - I) - 3 * A - 1 - B + 4 * E + 2 * (J - G))               # h1[3]
    t = R(21 * (C + I) - 17 * A - 5)                                                # h1[5]
    u = R(20 * (1 - B - J) - 5 * D)                                                 # h1[7]
    v = R(13 * C - 1 - 6 * E + 4 * F + 10 * G + 9 * (I - J))                        # h1[9]
    w = R(5 + 2 * A + 3 * C + D - 8 * B + 26 * E + 10 * F - 20 * G + H + 13 * J)    # h1[10]
    y = R(1 + 14 * I - 10 * C - 4 * D + E + J - G)                                  # h1[14]
    # layer-2 units as discrete variables
    a = fix["a"] if "a" in fix else Q(28 + 12 * A - 2 * D + 1.3 * p - 1.2 * (u + w + y))
    b = fix["b"] if "b" in fix else Q(29 + D - H - 2 * A + 3.6 * s - 1.6 * t + 1.2 * v - .8 * u - .7 * w)
    c = fix["c"] if "c" in fix else round((89 + 15 * (D - H) - 6 * F - .9 * t - .5 * w) / 7)
    d = fix["d"] if "d" in fix else round(R(3 * t - 52 - 9 * A - w) / 4)
    # output: logit differences (L0-L2, L1-L2, 0), common scale
    e = a + 2 * b - 1.6 * c + .7 * d - 3.7
    f = 3 + 2 * a + .35 * b - 1.7 * c - 1.2 * d
    l = [e, f, e - e]
    return l.index(max(l)), {"a": a, "b": b, "c": c, "d": d}
```