"""EDITABLE — the only file the agent changes.

policy(n, B, G) must return next_pool(history). history is a tuple of
(pool, count) pairs seen so far on this path; return the next pool as an
iterable of distinct people in range(n), with 1..G members, or None to stop.
The judge calls next_pool once per distinct history, so adaptive designs are
free: look at the counts and decide.

Experiment 2: sequential blocks solved by an exact dynamic program.
The policy opens a block of at most MMAX fresh people (MMAX <= G), keeps the set
S of block profiles compatible with the counts seen (uniform posterior, since
q = 1/2), and at every step chooses, by backward induction, one of: test a pool
inside the block; abandon the block (its certain-healthy people are accredited)
and open a new block from fresh people; stop. Pools never cross blocks, so the
DP is exact for this class. The state is (block size, S, tests left, fresh
people left); S is a bitmask over the 2^m profiles of the block. An untested
block may not be abandoned (it would be the same as not opening it, and the tie
would hide the test actions). Values are floats; ties keep the first option.
"""
from functools import lru_cache

MMAX = 6
VCAP = 1000000
PMAX6 = 5
_CACHE = {}


def _popcount(x):
    return bin(x).count('1')


def _build(n, B, G):
    mcap = min(MMAX, n)
    sizes = list(range(1, mcap + 1))
    pools, cmask, full = {}, {}, {}
    for m in sizes:
        P = 1 << m
        full[m] = (1 << P) - 1
        pools[m] = [p for p in range(1, 1 << m) if _popcount(p) <= min(G, (PMAX6 if B <= 4 else 5) if m == 6 else m)]
        cmask[m] = {}
        for pool in pools[m]:
            masks = [0] * (m + 1)
            for prof in range(P):
                masks[_popcount(prof & pool)] |= 1 << prof
            cmask[m][pool] = masks

    def accredited(m, S):
        union, s = 0, S
        while s:
            low = s & -s
            union |= low.bit_length() - 1
            s ^= low
        return m - _popcount(union)

    @lru_cache(maxsize=None)
    def fresh(b, r):
        """Best (value, block size) with b tests, r fresh people and no open block."""
        best, choice = 0.0, None
        if b == 0 or r == 0:
            return best, choice
        for m in sizes:
            if m > r:
                continue
            v, act = value(m, full[m], b, r - m)
            if act is not None and v > best + 1e-12:
                best, choice = v, m
        return best, choice

    vcache = {}

    def value(m, S, b, r):
        key = (m, S, b, r)
        hit = vcache.get(key)
        if hit is None:
            if len(vcache) > VCAP:
                vcache.clear()
            hit = vcache[key] = _value(m, S, b, r)
        return hit

    def _value(m, S, b, r):
        """Best (value, pool bits or None = abandon) with an open block of m people in state S."""
        if S == full[m]:
            best, act = 0.0, None
        else:
            best, act = accredited(m, S) + fresh(b, r)[0], None
        if b > 0 and S & (S - 1):
            size = _popcount(S)
            for pool in pools[m]:
                parts = [p for p in (S & cm for cm in cmask[m][pool]) if p]
                if len(parts) < 2:
                    continue
                total = 0.0
                for p in parts:
                    total += _popcount(p) / size * value(m, p, b - 1, r)[0]
                if total > best + 1e-12:
                    best, act = total, pool
        return best, act

    def next_pool(history):
        b, r, nxt = B, n, 0
        block, m, S, i = None, 0, 0, 0
        while True:
            if block is None:
                if b == 0 or r == 0:
                    return None
                _, size = fresh(b, r)
                if size is None:
                    return None
                block = tuple(range(nxt, nxt + size))
                nxt += size
                r -= size
                m, S = size, full[size]
            _, act = value(m, S, b, r)
            if act is None:
                block = None
                continue
            pool = tuple(block[j] for j in range(m) if act >> j & 1)
            if i == len(history):
                return pool
            hpool, count = history[i]
            i += 1
            if tuple(hpool) != pool:
                raise RuntimeError(f'history does not follow the plan: {hpool} vs {pool}')
            S &= cmask[m][act][count]
            b -= 1

    return next_pool


def policy(n, B, G):
    key = (n, B, min(G, MMAX))
    if key not in _CACHE:
        _CACHE[key] = _build(n, B, min(G, MMAX))
    return _CACHE[key]
