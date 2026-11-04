The network reduces to four discrete variables, one per live second-layer unit. The program gets 46 of the 48 inputs I ran through the forward tool right, and those inputs were mostly chosen for their small margins. All 1,024 answers can't be checked within the rules, so overall behaviour accuracy is an estimate.

**Structure read from the weights**
- **First-layer units.** Units 0 and 1 are always on, so they act as linear functions of the inputs. Six units are dead. The rest are thresholded features:
  - unit 5 (`P`) roughly counts x2 and x8, pushed down by x0.
  - unit 3 (`Q`) fires when none of x0, x2, x8 is set.
  - unit 7 (`G`) means neither x1 nor x9 is set.
  - unit 14 (`T`) is roughly "x8 and not x2".
  - units 9, 10 and 2 (`V`, `U`, `R`) are weighted sums.
- **Second-layer units.** Only units 1, 2, 4 and 7 are live (`a`, `b`, `c`, `d`); units 0, 3, 5 and 6 are always 0. So the completeness test should cost nothing: swapping those dead units can't change the answer.
- **Class.**
  - Class 0 wins when `b` is large and `a` is not.
  - Class 1 wins when `a` is large compared with `c`.
  - Class 2 is the default, because `c` is usually large.
  - `d` is only active when x2 and x8 are set and x0 is not.

**How the program was built.** It approximates each unit with small integer coefficients, dropping weights that barely matter. Each second-layer value is then rounded to a step of 0.5 (step 1 for `c`), so it takes at most 16 values. The class is computed from those rounded values using rescaled output weights. I chose which small terms to drop by comparing simpler versions against a more detailed draft of my own program, not the network. Rounding to the 0.5 and 1 steps causes both of the two misses I saw.

**Limits.** The literals total 119 characters against the 120 limit. That count leaves out the variable-name strings, which the rules say are free. The program is 744 bytes.

```program
VARIABLES = {"a": ("h2", [1]), "b": ("h2", [2]), "c": ("h2", [4]), "d": ("h2", [7])}

def program(x, fix={}):
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9=x
    def r(v): return v if v>0 else 0
    # h1 features (tenths): P~u5, Q~u3, G~u7, T~u14, V~u9, U~u10, R~u2
    P=r(21*(x2+x8)-17*x0-4)
    Q=r(11-14*x0-11*(x2+x8)+4*x4-x1)
    G=(not(x1 or x9))*(14-4*x3)
    T=r(10*x8-6*x2-2*x3)
    V=r(7*x2+5*(x8+x6)+2*x5-3*x4-5*x9-1)
    U=r(4+18*x4+9*x9+7*x5-14*x6-5*x1+x0+x2)
    R=r(6-4*x0-2*x2+2*x3-3*x4+3*x5+6*x6-x7+x8-5*x9)
    # h2 units, quantised: a,b,d in half-units, c in units
    a=int(r(305+120*x0-20*x3+25*R-18*(G+U+T))//50)
    b=min(int(r(305-20*x0+36*Q+24*V-16*P-11*(G+U))//50),15)
    c=int(r(945+150*(x3-x7)-60*x5-9*P-7*U)//100)
    d=int(r(29*P-15*U-485-130*x0)//50)
    a=fix.get("a",a); b=fix.get("b",b); c=fix.get("c",c); d=fix.get("d",d)
    l=[73*b+22*d+2*a+6*c-98, 48*a-38*d+76, 64*c-40*a-15*b]
    return l.index(max(l)), {"a":a,"b":b,"c":c,"d":d}
```