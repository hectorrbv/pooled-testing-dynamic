"""Extended exact sweep of the Proposition (AI support, 2026-10-05): under counts +
posterior_zero, with B>=2, G>=2, n>=B+1 the laminar optimum is >= swap bound >
best individual-only value; both boundaries (B=1, n<=B) give equality.
Usage: PYTHONPATH=. python augmented/barrido_proposicion_individuales.py SEED SECONDS NMAX BMAX GMAX
Run 2026-10-05: 8 seeds x 150 s, nmax 5 and 7, Bmax 4, Gmax 4 -> 32,195 instances, 0 failures."""
import random, sys, time
from fractions import Fraction
from itertools import combinations
from math import prod
from augmented.bm17_toy_solver import SolverLaminar

def top(q, u, B):
    o = sorted(range(len(q)), key=lambda i: q[i]*u[i], reverse=True)
    S = o[:min(B, len(q))]
    return S, sum(q[i]*u[i] for i in S)

def bound(q, u, B):
    S, T = top(q, u, B)
    out = [i for i in range(len(q)) if i not in S]
    return T + max((q[a]*q[b]+(1-q[a])*(1-q[b]))*q[c]*u[c]
                   for a, b in combinations(S, 2) for c in out)

def check(q, u, G, B):
    n = len(q)
    v = SolverLaminar({i: 1-x for i, x in enumerate(q)}, dict(enumerate(u)), G,
                      "posterior_zero").V(frozenset(range(n)), (), B)
    _, T = top(q, u, B)
    if n <= B:
        return "boundary_n<=B", v == sum(q[i]*u[i] for i in range(n)), v, T
    if B == 1:
        best = max(prod(q[i] for i in pool)*sum(u[i] for i in pool)
                   for m in range(1, G+1) for pool in combinations(range(n), m))
        return "boundary_B=1", v == best, v, T
    return "interior", (v >= bound(q, u, B) > T), v, T

if __name__ == "__main__":
    seed, budget = int(sys.argv[1]), float(sys.argv[2])
    nmax, Bmax, Gmax = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rng = random.Random(seed)
    t0 = time.time(); counts = {}; fails = []; minratio = None
    while time.time() - t0 < budget:
        n = rng.randint(2, nmax); G = rng.randint(2, Gmax); B = rng.randint(1, Bmax)
        kind = rng.random()
        if kind < 0.2:
            q = (Fraction(rng.randint(1, 99), 100),)*n; u = (Fraction(1),)*n
        elif kind < 0.4:
            q = tuple(rng.choice([Fraction(1, 100), Fraction(99, 100), Fraction(1, 2)]) for _ in range(n))
            u = tuple(Fraction(rng.randint(1, 40), 4) for _ in range(n))
        else:
            q = tuple(Fraction(rng.randint(1, 99), 100) for _ in range(n))
            u = tuple(Fraction(rng.randint(1, 40), 4) for _ in range(n))
        lab, ok, v, T = check(q, u, G, B)
        counts[lab] = counts.get(lab, 0) + 1
        if not ok: fails.append((q, u, G, B, v, T))
        if lab == "interior":
            r = v / T
            if minratio is None or r < minratio[0]: minratio = (r, n, G, B)
    print("seed", seed, "counts", counts, "fails", len(fails))
    print("min optimum/individual ratio in interior:", float(minratio[0]), "at n,G,B =", minratio[1:])
    for f in fails[:5]: print("FAIL", f)
