"""Exacto: estatico / dinamico binario / dinamico aumentado. n personas iid, prob q de estar SANO.
Utilidad = numero esperado de personas certificadas SANAS con certeza."""
from math import comb
from itertools import product, permutations
from functools import lru_cache
import sys

Q = 0.2
N = 10

# ---------- representacion por atomos ----------
# atoms: tupla de tamanos de celdas vivas
# dist: tupla ordenada de (countvec, prob) ; countvec[i] = # sanos en atomo i

def bank_prune(atoms, dist):
    """banca atomos totalmente determinados; devuelve (ganancia, atoms', dist')"""
    m = len(atoms)
    gain = 0
    keep = []
    for i in range(m):
        vals = {cv[i] for cv, _ in dist}
        if vals == {atoms[i]}:
            gain += atoms[i]
        elif vals == {0}:
            pass
        else:
            keep.append(i)
    agg = {}
    for cv, p in dist:
        k = tuple(cv[i] for i in keep)
        agg[k] = agg.get(k, 0.0) + p
    return gain, tuple(atoms[i] for i in keep), tuple(sorted(agg.items()))

def canon(atoms, dist):
    m = len(atoms)
    if m == 0:
        return ((), ())
    if m <= 6:
        best = None
        for perm in permutations(range(m)):
            a2 = tuple(atoms[i] for i in perm)
            d2 = tuple(sorted((tuple(cv[i] for i in perm), round(p, 12)) for cv, p in dist))
            cand = (a2, d2)
            if best is None or cand < best:
                best = cand
        return best
    order = sorted(range(m), key=lambda i: (atoms[i], sum(cv[i] * p for cv, p in dist)))
    a2 = tuple(atoms[i] for i in order)
    d2 = tuple(sorted((tuple(cv[i] for i in order), round(p, 12)) for cv, p in dist))
    return (a2, d2)

def outcomes(atoms, dist, t, augmented):
    """aplica el pool que toma t[i] del atomo i; devuelve lista de (prob, gain, atoms', dist')"""
    buckets = {}
    m = len(atoms)
    for cv, p in dist:
        ranges = []
        for i in range(m):
            lo = max(0, cv[i] - (atoms[i] - t[i]))
            hi = min(t[i], cv[i])
            ranges.append(range(lo, hi + 1))
        for ys in product(*ranges):
            w = p
            for i in range(m):
                w *= comb(t[i], ys[i]) * comb(atoms[i] - t[i], cv[i] - ys[i]) / comb(atoms[i], cv[i])
            if w == 0.0:
                continue
            R = sum(ys)
            key = R if augmented else (1 if R >= 1 else 0)
            nv = []
            for i in range(m):
                if t[i] > 0:
                    nv.append(ys[i])
                if atoms[i] - t[i] > 0:
                    nv.append(cv[i] - ys[i])
            buckets.setdefault(key, {})
            nv = tuple(nv)
            buckets[key][nv] = buckets[key].get(nv, 0.0) + w
    natoms = []
    for i in range(m):
        if t[i] > 0:
            natoms.append(t[i])
        if atoms[i] - t[i] > 0:
            natoms.append(atoms[i] - t[i])
    natoms = tuple(natoms)
    out = []
    for key, d in buckets.items():
        tot = sum(d.values())
        nd = tuple(sorted((k, v / tot) for k, v in d.items()))
        g, a2, d2 = bank_prune(natoms, nd)
        out.append((tot, g, a2, d2))
    return out

MEMO = {}

def V(atoms, dist, b, augmented):
    if b == 0 or not atoms:
        return 0.0
    key = (canon(atoms, dist), b, augmented)
    if key in MEMO:
        return MEMO[key]
    best = 0.0
    for t in product(*[range(a + 1) for a in atoms]):
        if sum(t) == 0:
            continue
        val = 0.0
        for p, g, a2, d2 in outcomes(atoms, dist, t, augmented):
            val += p * (g + V(a2, d2, b - 1, augmented))
        if val > best:
            best = val
    MEMO[key] = best
    return best

def dynamic(n, b, augmented):
    dist = tuple((( k,), comb(n, k) * Q**k * (1 - Q)**(n - k)) for k in range(n + 1))
    g, a, d = bank_prune((n,), dist)
    return g + V(a, d, b, augmented)

# ---------- estatico ----------
def static_value(counts, B, augmented):
    sigs = [(s, c) for s, c in enumerate(counts) if c > 0]
    groups = {}
    for xs in product(*[range(c + 1) for _, c in sigs]):
        p = 1.0
        for (s, c), x in zip(sigs, xs):
            p *= comb(c, x) * Q**x * (1 - Q)**(c - x)
        obs = []
        for j in range(B):
            R = sum(x for (s, c), x in zip(sigs, xs) if (s >> j) & 1)
            obs.append(R if augmented else (1 if R >= 1 else 0))
        groups.setdefault(tuple(obs), []).append((p, xs))
    tot = 0.0
    for obs, lst in groups.items():
        pg = sum(p for p, _ in lst)
        u = 0
        for idx, (s, c) in enumerate(sigs):
            if s == 0:
                continue
            if all(xs[idx] == c for _, xs in lst):
                u += c
        tot += pg * u
    return tot

def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest

def static_opt(n, B, augmented):
    best, arg = 0.0, None
    for counts in compositions(n, 2**B):
        v = static_value(counts, B, augmented)
        if v > best + 1e-12:
            best, arg = v, counts
    return best, arg

if __name__ == "__main__":
    Bmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"n={N}  q={Q}   (utilidad = personas certificadas sanas, u=1)")
    print(f"{'B':>2} {'B*q':>7} {'est.bin':>9} {'est.aum':>9} {'din.bin':>9} {'din.aum':>9}  {'dinaum/Bq':>10}  disenio-est-aum")
    for B in range(1, Bmax + 1):
        se, arge = static_opt(N, B, True)
        sb, _ = static_opt(N, B, False)
        db = dynamic(N, B, False)
        da = dynamic(N, B, True)
        sig = {f"{s:0{B}b}": c for s, c in enumerate(arge) if c > 0 and s > 0}
        print(f"{B:>2} {B*Q:>7.4f} {sb:>9.4f} {se:>9.4f} {db:>9.4f} {da:>9.4f}  {da/(B*Q):>9.2f}x  {sig}")
