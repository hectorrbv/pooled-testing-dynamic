"""Exact small-instance Bellman under posterior-zero, with optional laminarity.

Independent of bm17_toy_solver: enumerate infection profiles, retain all profiles
consistent with observations, and partition them by exact count. Probabilities
and utilities use scaled integers, so maximization and ties are exact.

Laminar mode checks every candidate against every earlier pool through a mask
of still-compatible actions. It allows ancestors as well as refinements, rather
than assuming the companion's atom-normal-form theorem.

This is an exponential reference solver, not a scalable production algorithm.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import lcm, prod
from time import perf_counter


def fraction(x):
    return x if isinstance(x, Fraction) else Fraction(str(x))


def members(mask):
    return tuple(i for i in range(mask.bit_length()) if mask & (1 << i))


def compatible(a, b):
    return not (a & b) or (a & b) == a or (a & b) == b


@dataclass(frozen=True)
class Solution:
    value: Fraction
    additional_value: Fraction
    first_pool: tuple | None
    seconds: float
    states: int


class BellmanPerfiles:
    def __init__(self, p, u, G, *, laminar=False, state_limit=1_000_000):
        self.p = tuple(map(fraction, p))
        self.u = tuple(map(fraction, u))
        self.n = len(self.p)
        if not 1 <= self.n <= 10 or len(self.u) != self.n:
            raise ValueError('Reference solver requires 1 <= n <= 10 and n utilities')
        if not 1 <= G <= self.n:
            raise ValueError('G must be between 1 and n')
        if any(x < 0 or x > 1 for x in self.p) or any(x < 0 for x in self.u):
            raise ValueError('Invalid probabilities or negative utilities')
        self.G, self.laminar, self.state_limit = G, laminar, state_limit
        self.utility_scale = lcm(*(x.denominator for x in self.u))
        self.ui = tuple(int(x * self.utility_scale) for x in self.u)
        self.weights = tuple(prod(
            x.numerator if z & (1 << i) else x.denominator - x.numerator
            for i, x in enumerate(self.p)) for z in range(1 << self.n))
        self.root_support = sum(1 << z for z, w in enumerate(self.weights) if w)
        self.pools = tuple(sum(1 << i for i in S)
                           for k in range(1, G + 1)
                           for S in combinations(range(self.n), k))
        self.pool_index = {a: j for j, a in enumerate(self.pools)}
        self.all_actions = (1 << len(self.pools)) - 1
        self.outcomes = []
        for a in self.pools:
            masks = [0] * (a.bit_count() + 1)
            for z, w in enumerate(self.weights):
                if w:
                    masks[(z & a).bit_count()] |= 1 << z
            self.outcomes.append(tuple(masks))
        self.compat = tuple(sum(1 << j for j, b in enumerate(self.pools)
                                if compatible(a, b)) for a in self.pools)
        self.memo, self.argmax, self.info_cache = {}, {}, {}

    def _info(self, support):
        """Prior integer mass, terminal credited utility, healthy-person mask."""
        if support not in self.info_cache:
            mass, infection_union, todo = 0, 0, support
            while todo:
                bit = todo & -todo
                z = bit.bit_length() - 1
                mass += self.weights[z]
                infection_union |= z
                todo ^= bit
            healthy = ((1 << self.n) - 1) ^ infection_union
            utility = sum(self.ui[i] for i in members(healthy))
            self.info_cache[support] = mass, utility, healthy
        return self.info_cache[support]

    def _state(self, history):
        support, allowed = self.root_support, self.all_actions
        seen = []
        for pool, r in history:
            pool = tuple(pool)
            if len(set(pool)) != len(pool) or any(i < 0 or i >= self.n for i in pool):
                raise ValueError('Invalid history pool')
            a = sum(1 << i for i in pool)
            if a not in self.pool_index or not isinstance(r, int) or not 0 <= r <= len(pool):
                raise ValueError('History violates count or G')
            if self.laminar and any(not compatible(a, prev) for prev in seen):
                raise ValueError('Non-laminar history in laminar mode')
            j = self.pool_index[a]
            support &= self.outcomes[j][r]
            if self.laminar:
                allowed &= self.compat[j]
            seen.append(a)
        if not support:
            raise ValueError('History has zero prior probability')
        return support, allowed

    def _dp(self, b, support, allowed):
        key = (b, support, allowed)
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) >= self.state_limit:
            raise RuntimeError('Exact-state limit reached; no optimum certified')
        mass, utility, _ = self._info(support)
        best, action = mass * utility, None  # stop, reward credited at terminal
        if b:
            choices = allowed
            while choices:
                bit = choices & -choices
                choices ^= bit
                j = bit.bit_length() - 1
                children = tuple(support & m for m in self.outcomes[j] if support & m)
                # A deterministic count adds no knowledge under posterior-zero;
                # in laminar mode it only further restricts the action menu.
                if len(children) == 1:
                    continue
                nxt = allowed & self.compat[j] if self.laminar else allowed
                val = sum(self._dp(b - 1, s, nxt) for s in children)
                if val > best:
                    best, action = val, j
        self.memo[key], self.argmax[key] = best, action
        return best

    def solve(self, B, history=()):
        """B is remaining budget; value includes health already proved by history."""
        if not isinstance(B, int) or B < 0:
            raise ValueError('B must be a nonnegative integer')
        start = perf_counter()
        support, allowed = self._state(history)
        mass, already, _ = self._info(support)
        value = Fraction(self._dp(B, support, allowed), mass * self.utility_scale)
        j = self.argmax[B, support, allowed]
        return Solution(value, value - Fraction(already, self.utility_scale),
                        members(self.pools[j]) if j is not None else None,
                        perf_counter() - start, len(self.memo))

    def tree(self, B, history=()):
        """Compact decision tree, conditional probabilities and crossing witnesses."""
        root, allowed = self._state(history)
        self.solve(B, history)
        previous = tuple(sum(1 << i for i in pool) for pool, _ in history)

        def visit(b, support, actions, past, old_healthy):
            mass, utility, healthy = self._info(support)
            node = {'remaining_tests': b, 'compatible_profiles': support.bit_count(),
                    'newly_healthy': members(healthy & ~old_healthy),
                    'healthy': members(healthy),
                    'value': str(Fraction(self._dp(b, support, actions),
                                          mass * self.utility_scale))}
            j = self.argmax[b, support, actions]
            if j is None:
                node['stop'] = True
                return node
            a = self.pools[j]
            node['pool'] = members(a)
            node['crosses_previous'] = [members(t) for t in past if not compatible(a, t)]
            node['branches'] = []
            nxt = actions & self.compat[j] if self.laminar else actions
            for r, mask in enumerate(self.outcomes[j]):
                s = support & mask
                if s:
                    child = visit(b - 1, s, nxt, past + (a,), healthy)
                    node['branches'].append({'infected_count': r,
                        'probability': str(Fraction(self._info(s)[0], mass)), 'next': child})
            return node

        return visit(B, root, allowed, previous, self._info(root)[2])

    def replay(self, B, history=()):
        """Evaluate the stored policy on every positive-probability latent profile."""
        sol = self.solve(B, history)
        root, allowed = self._state(history)
        history_masks = tuple(sum(1 << i for i in pool) for pool, _ in history)
        total, max_used, crossing_profiles = 0, 0, 0
        for z, w in enumerate(self.weights):
            if not root & (1 << z):
                continue
            support, actions, b, past, crossed = root, allowed, B, history_masks, False
            while b:
                j = self.argmax[b, support, actions]
                if j is None:
                    break
                a = self.pools[j]
                crosses = any(not compatible(a, t) for t in past)
                assert a.bit_count() <= self.G
                assert not self.laminar or not crosses
                crossed |= crosses
                r = (z & a).bit_count()
                support &= self.outcomes[j][r]
                assert support & (1 << z)
                actions = actions & self.compat[j] if self.laminar else actions
                past += (a,)
                b -= 1
            total += w * self._info(support)[1]
            max_used = max(max_used, B - b)
            crossing_profiles += int(crossed)
        value = Fraction(total, self._info(root)[0] * self.utility_scale)
        assert value == sol.value, (value, sol.value)
        return {'value': str(value), 'positive_profiles': root.bit_count(),
                'max_tests_used': max_used, 'profiles_with_crossing': crossing_profiles}
