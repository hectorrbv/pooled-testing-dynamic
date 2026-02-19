"""
Bitmask helpers and core primitives for DAPTS.

Individuals are indexed 0..n-1.  A "mask" is a Python int whose bit i
is set iff individual i is in the set (pool, infection profile, etc.).
"""

from __future__ import annotations

from itertools import combinations
from typing import List


# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------

def mask_from_indices(indices: List[int]) -> int:
    """Return a bitmask with the given bit positions set."""
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def indices_from_mask(mask: int, n: int) -> List[int]:
    """Return sorted list of set-bit positions in *mask* (only checks bits 0..n-1)."""
    return [i for i in range(n) if mask >> i & 1]


def popcount(mask: int) -> int:
    """Number of set bits."""
    return bin(mask).count("1")


# ---------------------------------------------------------------------------
# Pool enumeration
# ---------------------------------------------------------------------------

def all_pools(n: int, G: int, *, include_empty: bool = True) -> List[int]:
    """Return every pool mask t with |t| <= G over population 0..n-1.

    Parameters
    ----------
    n : int
        Population size.
    G : int
        Maximum pool size (biological limit).
    include_empty : bool
        If True (default) the empty pool (mask 0) is included.

    Returns
    -------
    list[int]
        Sorted list of bitmasks representing each valid pool.
    """
    pools: List[int] = []
    indices = list(range(n))
    start = 0 if include_empty else 1
    for size in range(start, min(G, n) + 1):
        for combo in combinations(indices, size):
            pools.append(mask_from_indices(combo))
    return pools


# ---------------------------------------------------------------------------
# Augmented test result
# ---------------------------------------------------------------------------

def test_result(pool_mask: int, z_mask: int) -> int:
    """Return the augmented test result r(t, Z) = |t ∩ Z| = popcount(t & Z).

    This is the number of infected individuals in the pool — the
    "idealized augmented test" from the paper.
    """
    return popcount(pool_mask & z_mask)
