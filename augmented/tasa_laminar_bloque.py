"""Exact long-run laminar rate c*(G, q) for a homogeneous population (soft clearing, exact counts).

Claim. With independent people, each healthy with probability q, utility 1, pools of size <= G and
pathwise-laminar tests, every laminar test is equivalent to a virgin test (fresh people, size <= G)
or to a split of one known-count atom (atom normal form). Then

    V_lam(n, B, G) <= c*(G, q) * B   for every n, B,
    V_lam(n, B, G) / B -> c*(G, q)   when n >= B*G and B -> infinity,

where c*(G, q) is the best ratio E[cleared] / E[tests] of a *single-block* strategy: test a virgin
pool of size M <= G, then adaptively split the resulting atom (m, r) or abandon it.

Why (sketch). Let W(m, r) = max(stop value, E[W(children)] - c) with stop value m if r = 0 else 0.
At c = c*, W is a potential: a virgin test raises the expected sum of W over atoms by at most c*,
a split raises it by at most c*, and W(m, 0) = m bounds the utility (this is Francisco's potential
method with the optimal potential; his phi = (m-2r)^2/m is a feasible potential with constant 2 at
q = 1/2). The lower bound repeats the optimal block strategy on fresh people (renewal).

c* is computed by Dinkelbach iteration on the ratio, in exact rationals (`rate`) or floats
(`rate_float`, for large G). Status: derivation by the AI support, numerically checked against the
exact laminar optimum (bellman by types); proof to be written and validated by A and B.
"""
from fractions import Fraction as F
from math import comb


def rate(G, q=F(1, 2), max_iter=200):
    """Exact c*(G, q). Returns (c, M*, split_rule, UT) with split_rule[m, r] = k (None = abandon)."""
    q = F(q)
    p = 1 - q
    c = q  # individual tests
    for _ in range(max_iter):
        UT, rule = {}, {}
        for m in range(1, G + 1):
            for r in range(m + 1):
                if r == 0:
                    UT[m, r] = (F(m), F(0)); continue
                if r == m:
                    UT[m, r] = (F(0), F(0)); continue
                best, arg, best_w = (F(0), F(0)), None, F(0)
                for k in range(1, m):
                    U = T = F(0)
                    for x in range(max(0, k - (m - r)), min(k, r) + 1):
                        h = F(comb(r, x) * comb(m - r, k - x), comb(m, k))
                        u1, t1 = UT[k, x]
                        u2, t2 = UT[m - k, r - x]
                        U += h * (u1 + u2)
                        T += h * (t1 + t2)
                    T += 1
                    if U - c * T > best_w:
                        best_w, best, arg = U - c * T, (U, T), k
                UT[m, r], rule[m, r] = best, arg
        candidates = []
        for M in range(1, G + 1):
            U = T = F(0)
            for R in range(M + 1):
                b = comb(M, R) * q ** (M - R) * p ** R
                U += b * UT[M, R][0]
                T += b * UT[M, R][1]
            candidates.append(((U / (T + 1)), M))
        new, M = max(candidates)
        if new == c:
            return c, M, rule, UT
        c = new
    raise RuntimeError('Dinkelbach did not converge')


def rate_float(G, q=0.5, tol=1e-13, max_iter=500):
    """Float c*(G, q) for large G (exploration; exact version for G <= 20)."""
    import numpy as np
    from scipy.stats import binom, hypergeom
    p = 1 - q
    c = q
    H = {(m, r, k): (np.arange(max(0, k - (m - r)), min(k, r) + 1),)
         for m in range(2, G + 1) for r in range(1, m) for k in range(1, m)}
    H = {key: (xs[0], hypergeom.pmf(xs[0], key[0], key[1], key[2])) for key, xs in H.items()}
    for _ in range(max_iter):
        U = np.zeros((G + 1, G + 1)); T = np.zeros((G + 1, G + 1))
        for m in range(1, G + 1):
            U[m, 0] = m
            for r in range(1, m):
                best = (0.0, 0.0, 0.0)
                for k in range(1, m):
                    xs, h = H[m, r, k]
                    u = h @ (U[k, xs] + U[m - k, r - xs])
                    t = 1 + h @ (T[k, xs] + T[m - k, r - xs])
                    if u - c * t > best[0] + 1e-15:
                        best = (u - c * t, u, t)
                U[m, r], T[m, r] = best[1], best[2]
        ratios = []
        for M in range(1, G + 1):
            b = binom.pmf(np.arange(M + 1), M, p)
            ratios.append(((b @ U[M, :M + 1]) / (1 + b @ T[M, :M + 1]), M))
        new, M = max(ratios)
        if abs(new - c) < tol:
            return float(new), M
        c = new
    raise RuntimeError('Dinkelbach did not converge')


if __name__ == '__main__':
    for G in (1, 2, 3, 4, 5, 6, 7, 8):
        c, M, _, _ = rate(G)
        print(f'G={G}  c* = {c} = {float(c):.6f}  first pool {M}')
