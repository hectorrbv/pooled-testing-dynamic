"""EDITABLE FILE — the only file the agent changes.

sample(rng) -> (n, B, G, q, u): one instance. q = healthy probabilities (Fractions
or floats in (0,1)), u = utilities (> 0). Keep n <= NMAX given to the judge.
Think in FAMILIES, not points: structure (two types, blocks, one very healthy
person, skewed utilities, ...) is what the agent should explore. The judge
draws as many instances as fit in the time budget and keeps the minimum ratio.

Baseline: uniform random, the same family that found ratio 0.978 (binario) and
0.833 (precio, n=6) on 2026-10-12."""
from fractions import Fraction as F

def sample(rng):
    n = rng.randint(3, 6)
    B = rng.randint(2, 4)
    G = rng.randint(2, min(4, n))
    q = [F(rng.randint(1, 19), 20) for _ in range(n)]
    u = [F(1)] * n if rng.random() < 0.5 else [F(rng.randint(1, 8), 4) for _ in range(n)]
    return n, B, G, q, u
