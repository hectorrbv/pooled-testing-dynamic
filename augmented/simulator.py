"""
Simulator: run a DAPTS on a fixed infection profile Z.

Given F and z_mask, compute the terminal history h_B(F,Z), the set of
cleared (proven healthy) individuals, and the realized utility u(F,Z).
"""

from __future__ import annotations

from typing import List, Tuple

from augmented.core import indices_from_mask, test_result
from augmented.strategy import DAPTS, History


def apply_dapts(
    F: DAPTS,
    z_mask: int,
    n: int,
    u: List[float],
) -> Tuple[History, int, float]:
    """Simulate DAPTS *F* on infection profile *z_mask*.

    Parameters
    ----------
    F : DAPTS
        The testing strategy.
    z_mask : int
        Infection profile bitmask (bit i set <=> individual i is infected).
    n : int
        Population size.
    u : list[float]
        Utility vector, length n.

    Returns
    -------
    terminal_history : History
        Full sequence of (pool_mask, result) pairs of length B.
    cleared_mask : int
        Bitmask of individuals appearing in at least one pool with result 0.
    u_realized : float
        Sum of u[i] for every individual in *cleared_mask*.
    """
    history: History = ()
    cleared_mask = 0

    for k in range(1, F.B + 1):
        pool = F.choose(k, history)
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool

    u_val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, u_val


def u_realized(
    F: DAPTS,
    z_mask: int,
    u: List[float],
    n: int,
) -> float:
    """Return only the realized utility u(F, Z)."""
    _, _, val = apply_dapts(F, z_mask, n, u)
    return val
