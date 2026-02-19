"""
Expected utility computations for a DAPTS.

Two methods:
  1) exact_expected_utility  — enumerate all 2^n infection profiles
  2) mc_expected_utility     — Monte Carlo sampling
"""

from __future__ import annotations

import random
from typing import List

from augmented.core import indices_from_mask
from augmented.simulator import apply_dapts
from augmented.strategy import DAPTS


def _z_weight(z_mask: int, p: List[float], q: List[float], n: int) -> float:
    """Pr(Z = z_mask) = prod_i p_i^{Z_i} * q_i^{1-Z_i}."""
    w = 1.0
    for i in range(n):
        if z_mask >> i & 1:
            w *= p[i]
        else:
            w *= q[i]
    return w


def exact_expected_utility(
    F: DAPTS,
    p: List[float],
    u: List[float],
    n: int,
) -> float:
    """Compute u(F) = E_Z[u(F,Z)] exactly by summing over all 2^n profiles.

    Parameters
    ----------
    F : DAPTS
        Testing strategy.
    p : list[float]
        Infection probabilities, length n.
    u : list[float]
        Utilities, length n.
    n : int
        Population size.  Should satisfy n <= ~20 for tractability.

    Returns
    -------
    float
        Exact expected utility.
    """
    q = [1.0 - pi for pi in p]
    total = 0.0
    for z_mask in range(1 << n):
        w = _z_weight(z_mask, p, q, n)
        _, _, u_val = apply_dapts(F, z_mask, n, u)
        total += w * u_val
    return total


def mc_expected_utility(
    F: DAPTS,
    p: List[float],
    u: List[float],
    n: int,
    trials: int = 10_000,
    seed: int = 42,
) -> float:
    """Estimate u(F) via Monte Carlo.

    Parameters
    ----------
    F : DAPTS
        Testing strategy.
    p : list[float]
        Infection probabilities, length n.
    u : list[float]
        Utilities, length n.
    n : int
        Population size.
    trials : int
        Number of MC samples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    float
        Monte Carlo estimate of expected utility.
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        z_mask = 0
        for i in range(n):
            if rng.random() < p[i]:
                z_mask |= 1 << i
        _, _, u_val = apply_dapts(F, z_mask, n, u)
        total += u_val
    return total / trials
