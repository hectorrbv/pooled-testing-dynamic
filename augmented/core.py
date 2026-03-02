"""
Bitmask helpers and core primitives for DAPTS.

Individuals are indexed 0..n-1.  A "mask" is a Python int whose bit i
is set iff individual i is in the set (pool, infection profile, etc.).
"""

from itertools import combinations


def mask_from_indices(indices):
    """Bitmask with the given bit positions set. e.g. [0,2] -> 0b101."""
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def indices_from_mask(mask, n=None):
    """Sorted list of set-bit positions. If n given, ignore bits >= n."""
    if n is not None:
        mask &= (1 << n) - 1
    out = []
    while mask:
        lsb = mask & -mask
        out.append(lsb.bit_length() - 1)
        mask ^= lsb
    return out


def popcount(mask):
    """Number of set bits."""
    return mask.bit_count()


def mask_str(mask, n=None):
    """Human-readable string for a bitmask. e.g. 0b101 -> '{0,2}'."""
    idxs = indices_from_mask(mask, n)
    return "{" + ",".join(str(i) for i in idxs) + "}" if idxs else "{}"


def all_pools(n, G, include_empty=True):
    """Every pool mask t with |t| <= G over population 0..n-1."""
    pools = []
    start = 0 if include_empty else 1
    for size in range(start, min(G, n) + 1):
        for combo in combinations(range(n), size):
            pools.append(mask_from_indices(combo))
    return pools


def test_result(pool_mask, z_mask):
    """Augmented test result r(t, Z) = |t ∩ Z| (count of infected in pool)."""
    return popcount(pool_mask & z_mask)
