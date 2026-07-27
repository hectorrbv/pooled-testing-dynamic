"""Subset-count tables for one pool, and their reuse after a split.

The session of 27 July asked for a single object per test: given a pool ``T``
tested with exact count ``R``, the probability ``P(R(T')=r' | R(T)=R)`` for
*every* subset ``T' subseteq T`` and every count ``r'``.  It also asked whether,
when a later test splits ``T`` into two atoms, the children's tables can be
derived from the parent's instead of being recomputed from scratch.

Both answers live in one identity.  Because the prior is a product measure,
for any ``S subseteq T``

    P(R(S)=r | R(T)=R) = PB_S(r) * PB_{T\\S}(R-r) / PB_T(R),

where ``PB_S`` is the unconditional Poisson-binomial pmf of the block ``S``.
Conditioning therefore never touches the blocks themselves: it reweights a
family of pmfs that does not depend on any observation.  The reusable object
is that family --- the *subset pmf cache* --- and not the conditional table.

The consequence for the split is exact rather than approximate.  Testing
``T' subseteq T`` and observing ``r'`` creates the atoms ``T'`` (count ``r'``)
and ``D = T \\ T'`` (count ``R - r'``).  Every block appearing in a child's
table is a subset of ``T``, so it is already cached: the children cost zero
new convolutions.  What the cache buys is not one split but all of them ---
every candidate action and every hypothetical outcome a rollout enumerates
reuses the same precomputation.

Masks are integer bitmasks over the full population, as everywhere else in
:mod:`augmented`.  Inside a cache, subsets are addressed by a *local* index
over the pool members, which is what keeps the storage at ``2^g`` rows.
"""

from numbers import Integral
from typing import NamedTuple

import numpy as np

from augmented.core import indices_from_mask


class PoolSubsetCache(NamedTuple):
    """Unconditional block pmfs for every subset of one pool.

    ``members`` lists the absolute individual indices in ascending order, so
    local subset index ``k`` means ``{members[j] : bit j of k is set}``.
    ``pmfs[k]`` is the Poisson-binomial pmf of that block, of length
    ``popcount(k) + 1``.  ``convolutions`` records how many convolution steps
    were spent building the cache; a cache restricted from a parent reports
    zero, which is the claim this module exists to make checkable.
    """

    pool_mask: int
    members: tuple
    pmfs: tuple
    convolutions: int


class AtomTable(NamedTuple):
    """One atom after a split: its cache, its count and its subset table."""

    cache: PoolSubsetCache
    count: int
    table: np.ndarray


def _as_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validated_probabilities(p):
    probabilities = np.asarray(p, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("p must be a one-dimensional sequence")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("every prior probability must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("every prior probability must lie in [0, 1]")
    return probabilities


def subset_pmf_cache(p, pool_mask):
    """Poisson-binomial pmf of every subset of ``pool_mask``.

    Built by a subset dynamic program: each block is one convolution away
    from the block without its lowest member, so the whole family costs
    ``2^g - 1`` convolution steps rather than ``2^g`` independent builds.
    """

    priors = _validated_probabilities(p)
    n = len(priors)
    pool = _as_integer(pool_mask, "pool mask")
    if pool <= 0 or pool >= (1 << n):
        raise ValueError("pool mask is empty or outside the prior universe")

    members = tuple(indices_from_mask(pool, n))
    size = 1 << len(members)
    pmfs = [None] * size
    pmfs[0] = np.array([1.0], dtype=float)
    convolutions = 0
    for subset in range(1, size):
        lowest = subset & -subset
        probability = priors[members[lowest.bit_length() - 1]]
        pmfs[subset] = np.convolve(
            pmfs[subset ^ lowest],
            np.array([1.0 - probability, probability], dtype=float),
        )
        convolutions += 1
    return PoolSubsetCache(pool, members, tuple(pmfs), convolutions)


def local_index(cache, subset_mask):
    """Local index of an absolute ``subset_mask`` inside ``cache``."""

    subset = _as_integer(subset_mask, "subset mask")
    if subset & ~cache.pool_mask:
        raise ValueError("the subset must be contained in the cached pool")
    index = 0
    for position, member in enumerate(cache.members):
        if subset & (1 << member):
            index |= 1 << position
    return index


def absolute_mask(cache, index):
    """Absolute bitmask of a local subset ``index`` inside ``cache``."""

    index = _as_integer(index, "local index")
    if not 0 <= index < (1 << len(cache.members)):
        raise ValueError("local index is outside the cached pool")
    mask = 0
    for position, member in enumerate(cache.members):
        if index & (1 << position):
            mask |= 1 << member
    return mask


def restrict_cache(cache, sub_mask):
    """View of ``cache`` for a sub-pool, borrowing the parent's pmfs.

    Every block of the sub-pool is a block of the parent, so this performs no
    convolution at all; the returned cache reports ``convolutions == 0``.
    """

    sub = _as_integer(sub_mask, "sub-pool mask")
    if sub <= 0:
        raise ValueError("the sub-pool must be non-empty")
    if sub & ~cache.pool_mask:
        raise ValueError("the sub-pool must be contained in the cached pool")

    positions = {member: index for index, member in enumerate(cache.members)}
    members = tuple(member for member in cache.members if sub & (1 << member))
    parent_bits = [1 << positions[member] for member in members]

    size = 1 << len(members)
    pmfs = [None] * size
    for subset in range(size):
        parent_subset = 0
        remaining = subset
        while remaining:
            lowest = remaining & -remaining
            parent_subset |= parent_bits[lowest.bit_length() - 1]
            remaining ^= lowest
        pmfs[subset] = cache.pmfs[parent_subset]
    return PoolSubsetCache(sub, members, tuple(pmfs), 0)


def conditional_subset_table(cache, count):
    """``P(R(S)=r | R(pool)=count)`` for every subset ``S`` of the pool.

    Returns a ``(2^g, g+1)`` array whose row ``k`` is the distribution of the
    block with local index ``k``; entries beyond that block's size stay zero.
    Every row sums to one, and rows for blocks that cannot reach ``count``
    together with their complement are exactly zero-filled by the identity
    rather than by clipping.
    """

    total = len(cache.members)
    full = (1 << total) - 1
    count = _as_integer(count, "count")
    if not 0 <= count <= total:
        raise ValueError("count is outside the pool size")

    denominator = float(cache.pmfs[full][count])
    if denominator <= 1e-300:
        raise ValueError("the conditioned count has zero probability")

    table = np.zeros((1 << total, total + 1), dtype=float)
    for subset in range(1 << total):
        inside = cache.pmfs[subset]
        outside = cache.pmfs[full ^ subset]
        counts = np.arange(len(inside))
        rest = count - counts
        usable = (rest >= 0) & (rest < len(outside))
        if not usable.any():
            raise ValueError("a block has no mass compatible with the count")
        table[subset, counts[usable]] = (
            inside[usable] * outside[rest[usable]] / denominator
        )
    return table


def split_subset_tables(cache, tested_mask, tested_count, pool_count):
    """Tables of the two atoms created by testing ``tested_mask`` in the pool.

    ``tested_mask`` is a strict non-empty subset of the cached pool, observed
    with exact count ``tested_count``; ``pool_count`` is the count already
    observed for the pool itself.  The atoms are the tested block and the
    residual ``pool \\ tested``, whose count is ``pool_count - tested_count``
    by the residual-count subtraction.

    Both tables are built from the parent cache, so no Poisson-binomial work
    is repeated.  Conditioning each atom on its own count only --- ignoring
    the sibling's --- is exactly the factorization across atoms: the two
    blocks are disjoint and the prior is a product measure.
    """

    tested = _as_integer(tested_mask, "tested mask")
    if tested <= 0 or tested & ~cache.pool_mask:
        raise ValueError("the tested pool must be a non-empty subset")
    if tested == cache.pool_mask:
        raise ValueError("the tested pool must be a strict subset")

    tested_count = _as_integer(tested_count, "tested count")
    pool_count = _as_integer(pool_count, "pool count")
    residual = cache.pool_mask & ~tested
    residual_count = pool_count - tested_count
    if not 0 <= tested_count <= tested.bit_count():
        raise ValueError("tested count is outside the tested pool size")
    if not 0 <= residual_count <= residual.bit_count():
        raise ValueError(
            "the residual count implied by the split is not attainable"
        )

    tested_cache = restrict_cache(cache, tested)
    residual_cache = restrict_cache(cache, residual)
    return (
        AtomTable(
            tested_cache,
            tested_count,
            conditional_subset_table(tested_cache, tested_count),
        ),
        AtomTable(
            residual_cache,
            residual_count,
            conditional_subset_table(residual_cache, residual_count),
        ),
    )


def table_row(cache, table, subset_mask):
    """Distribution row of an absolute ``subset_mask``, trimmed to its size."""

    index = local_index(cache, subset_mask)
    return table[index, : _as_integer(subset_mask, "subset mask").bit_count() + 1]
