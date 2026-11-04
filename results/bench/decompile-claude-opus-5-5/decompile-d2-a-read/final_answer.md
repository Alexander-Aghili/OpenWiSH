**How I read it.** Everything here comes from the weights alone. I never ran the network, so none of the scores below were measured against it.

- **Dead and constant units.** Many units never vary. h1 units 4, 6, 8, 11, 12 and 15 are always 0, and h1_13 is a constant. h2 units 0, 3, 5 and 6 are always 0.
- **Always-linear units.** h1_0, h1_1 and h2_4 are always positive, so they act as plain linear terms.
- **Near-boolean h1 units:**
  - h1_3 ≈ none of x0, x2, x8 is set.
  - h1_5 ≈ relu(2.1·(x2+x8) − 1.7·x0 − 0.5).
  - h1_7 ≈ ¬x1 ∧ ¬x9.
  - h1_14 ≈ x8 ∧ ¬x2.
- **Graded h1 units.** h1_2, h1_9 and h1_10 combine x4, x5, x6 and x9 in a graded way.
- **The four live h2 units:**
  - **P (h2_1)** is class-1 evidence. It is roughly a detector for the condition C: 2·x1 + 3·x6 − 3·x4 − x9 ≥ 2.
  - **Q (h2_2)** is class-0 evidence. It is large when x0 = x2 = x8 = 0, or when C holds and x0+x2+x8 is even with x0 = 1.
  - **R (h2_7)** ≈ ¬x0 ∧ x2 ∧ x8. It suppresses class 1.
  - **D (h2_4)** is always-on evidence for class 2.

The behaviour these give (via my own integer approximation) is a clean rule:
- If x0 = x2 = x8 = 0, the class is 0.
- Otherwise, if C holds, the class is 1 when x0+x2+x8 is odd and 0 when it is even.
- Otherwise the class is 2.

**The program.** Each h1 unit is written with small integer coefficients, about the weights ×10 with tiny terms dropped. Rounding moves each unit by at most about 0.3, and usually about 0.1.

The four h2 units P, Q, D and R are the causal variables. Each is rounded to at most 16 levels (P, Q and R in steps of 0.5, D in steps of 0.75). The class is computed only from those levels, using the logit differences.

Every varying h2 unit is a variable, so the completeness test has nothing left to swap. On my own unrounded integer approximation, standing in for the network, the program:
- agrees on 99.9% of inputs;
- has simulated intervention accuracy of about 0.98–0.99 per variable.

It uses exactly 120 literal characters, and `check_program` passes.

```program
VARIABLES = {"P": ("h2", [1]), "Q": ("h2", [2]), "D": ("h2", [4]), "R": ("h2", [7])}
def r(v):
    return v*(v>0)
def program(x, fix={}):
    a,b,c,d,e,f,g,h,i,j = x
    # layer-1 features (tenths)
    u0 = 46 + 8*d - 3*f - 8*h
    u1 = 26 + 15*a - 2*d
    u2 = r(6 - 4*a - 2*c + 2*d - 3*e + 3*f + 6*g - h + i - 5*j)
    u3 = r(11*(1-c-i) - 14*a - b + 4*e + f - 2*g + 2*j)   # none of x0,x2,x8
    u5 = r(21*(c+i) - 17*a - 5)                          # count x2,x8 vs x0
    u7 = (14 - 4*d)*(1-(b|j))                            # not x1 and not x9
    u9 = r(7*c - 3*e + 2*f + 5*g + 5*i - 5*j - 1)
    u10 = r(4 + a - 5*b + 2*c + 18*e + 7*f - 14*g + h + 9*j)
    u14 = r(10*i - 7*c - 2*d)                            # x8 and not x2
    # layer-2 causal variables (rounded levels)
    v = {}
    v["P"] = r(90 + 8*u1 + 25*u2 - 18*(u7+u10+u14)) // 50   # class-1 evidence (condition C)
    v["Q"] = r(278 + u0 - u1 + 36*u3 - 16*u5 - 11*(u7+u10) + 24*u9) // 50   # class-0 evidence
    v["D"] = (66 + 19*u0 - 9*u5 - 7*u10) // 75               # class-2 evidence
    v["R"] = r(29*u5 - 9*u1 - 15*u10 - u14 - 244) // 50      # not x0, x2 and x8
    v.update(fix)
    P, Q, D, R = v["P"], v["Q"], v["D"], v["R"]
    d1 = 46*P - 73*Q - 60*R - 4*D + 173
    d2 = -42*P - 88*Q - 22*R + 43*D + 97
    cls = 0 if max(d1,d2) < 0 else (1 if d1 > d2 else 2)
    return cls, v
```