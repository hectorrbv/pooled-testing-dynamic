"""Exact small-n benchmarks for the laminar weekly atlas.

The public functions in this module deliberately share the signature
``(p, u, B, G) -> value`` used in the weekly plan:

``dynamic_augmented_value``
    Unrestricted adaptive optimum with exact count outcomes.
``laminar_augmented_value``
    Best adaptive value over *all* fixed laminar action libraries.
``greedy_laminar_value``
    Exact value of myopic greedy in a deterministic balanced hierarchy.
``static_binary_value``
    Best non-adaptive binary design (identical welfare under count results).

The atlas only uses n <= 6.  At that scale a fixed laminar library is a
clique in the pairwise-compatibility graph of feasible pools.  It is enough
to inspect maximal cliques: every laminar family extends to one, and adding
available actions cannot lower an optimal policy's value.  This makes
``laminar_augmented_value`` an exact class optimum rather than the best of a
small heuristic portfolio.
"""

from functools import lru_cache
from itertools import combinations
from numbers import Integral
import time

import numpy as np

from augmented.core import all_pools, indices_from_mask, mask_from_indices


_MAX_EXACT_N = 10


def _validated_instance(p, u, B, G):
    probabilities = np.asarray(p, dtype=float)
    utilities = np.asarray(u, dtype=float)
    if probabilities.ndim != 1 or utilities.ndim != 1:
        raise ValueError("p and u must be one-dimensional")
    if len(probabilities) != len(utilities):
        raise ValueError("p and u must have the same length")
    n = len(probabilities)
    if n == 0 or n > _MAX_EXACT_N:
        raise ValueError(f"exact benchmarks require 1 <= n <= {_MAX_EXACT_N}")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("p must contain finite probabilities in [0, 1]")
    if not np.all(np.isfinite(utilities)) or np.any(utilities < 0.0):
        raise ValueError("u must contain finite nonnegative utilities")
    if isinstance(B, bool) or not isinstance(B, Integral) or int(B) < 0:
        raise ValueError("B must be a nonnegative integer")
    if (
        isinstance(G, bool)
        or not isinstance(G, Integral)
        or not 1 <= int(G) <= n
    ):
        raise ValueError("G must be an integer between 1 and n")
    return probabilities, utilities, int(B), int(G)


def laminar_compatible(first, second):
    """Whether two pool masks are disjoint or nested."""

    intersection = int(first) & int(second)
    return intersection in (0, int(first), int(second))


@lru_cache(maxsize=None)
def maximal_laminar_libraries(n, G):
    """Enumerate every maximal laminar library for small ``(n, G)``.

    A deterministic bit-set Bron--Kerbosch search avoids adding NetworkX as a
    runtime dependency.  The largest atlas case, ``n=6, G=3``, has 41 pools
    but only 105 maximal libraries.
    """

    if not 1 <= n <= _MAX_EXACT_N or not 1 <= G <= n:
        raise ValueError("invalid n or G")
    pools = tuple(all_pools(n, G, include_empty=False))
    count = len(pools)
    neighbors = [0] * count
    for i, first in enumerate(pools):
        for j in range(i + 1, count):
            if laminar_compatible(first, pools[j]):
                neighbors[i] |= 1 << j
                neighbors[j] |= 1 << i

    libraries = []

    def visit(chosen, possible, excluded):
        if possible == 0 and excluded == 0:
            libraries.append(tuple(pools[i] for i in _iter_bits(chosen)))
            return

        union = possible | excluded
        if union:
            pivot = max(
                _iter_bits(union),
                key=lambda i: (possible & neighbors[i]).bit_count(),
            )
            candidates = possible & ~neighbors[pivot]
        else:
            candidates = possible

        while candidates:
            bit = candidates & -candidates
            vertex = bit.bit_length() - 1
            visit(
                chosen | bit,
                possible & neighbors[vertex],
                excluded & neighbors[vertex],
            )
            possible &= ~bit
            excluded |= bit
            candidates &= ~bit

    visit(0, (1 << count) - 1, 0)
    libraries.sort()
    return tuple(libraries)


def _iter_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask &= mask - 1


def balanced_laminar_library(p, u, G):
    """Deterministic hierarchy used by the laminar greedy benchmark.

    Individuals are sorted by immediate singleton value ``(1-p_i)u_i``,
    chunked into disjoint roots of size at most ``G``, and each root is split
    recursively into a balanced binary tree.
    """

    probabilities = np.asarray(p, dtype=float)
    utilities = np.asarray(u, dtype=float)
    if probabilities.ndim != 1 or utilities.shape != probabilities.shape:
        raise ValueError("p and u must be equally sized one-dimensional arrays")
    n = len(probabilities)
    if not 1 <= int(G) <= n:
        raise ValueError("G must lie between 1 and n")
    order = np.lexsort((np.arange(n), -((1.0 - probabilities) * utilities)))
    pools = set()

    def add_tree(items):
        if not items:
            return
        pools.add(mask_from_indices(items))
        if len(items) > 1:
            middle = len(items) // 2
            add_tree(items[:middle])
            add_tree(items[middle:])

    for start in range(0, n, int(G)):
        add_tree([int(i) for i in order[start : start + int(G)]])
    return tuple(sorted(pools))


class ExactPolicyEvaluator:
    """Exact scenario DP shared by all four small-n quantities."""

    def __init__(self, p, u, B, G):
        p, u, B, G = _validated_instance(p, u, B, G)
        self.p = p
        self.u = u
        self.B = B
        self.G = G
        self.n = len(p)
        self.pools = tuple(all_pools(self.n, G, include_empty=False))
        self.world_count = 1 << self.n
        self.all_worlds = (1 << self.world_count) - 1

        ids = np.arange(self.world_count, dtype=np.uint64)[:, None]
        bits = np.arange(self.n, dtype=np.uint64)[None, :]
        self.scenarios = ((ids >> bits) & 1).astype(np.int8)
        self.weights = np.prod(
            np.where(self.scenarios == 1, p, 1.0 - p), axis=1
        )

        self.utility = np.zeros(1 << self.n, dtype=float)
        for mask in range(1, 1 << self.n):
            bit = mask & -mask
            self.utility[mask] = (
                self.utility[mask ^ bit] + u[bit.bit_length() - 1]
            )

        self.outcome_worlds = {}
        for pool in self.pools:
            members = indices_from_mask(pool, self.n)
            counts = self.scenarios[:, members].sum(axis=1)
            self.outcome_worlds[pool] = tuple(
                sum(1 << int(z) for z in np.flatnonzero(counts == result))
                for result in range(len(members) + 1)
            )

        self._mass_cache = {0: 0.0, self.all_worlds: float(self.weights.sum())}

    def mass(self, worlds):
        cached = self._mass_cache.get(worlds)
        if cached is not None:
            return cached
        total = 0.0
        rest = worlds
        while rest:
            bit = rest & -rest
            total += self.weights[bit.bit_length() - 1]
            rest &= rest - 1
        self._mass_cache[worlds] = total
        return total

    def branches(self, worlds, cleared, pool):
        total_mass = self.mass(worlds)
        if total_mass <= 0.0:
            return ()
        reward = float(self.utility[pool & ~cleared])
        branches = []
        for result, compatible_worlds in enumerate(self.outcome_worlds[pool]):
            child = worlds & compatible_worlds
            child_mass = self.mass(child)
            if child_mass <= 0.0:
                continue
            new_cleared = cleared | pool if result == 0 else cleared
            branches.append(
                (child_mass / total_mass, child, new_cleared,
                 reward if result == 0 else 0.0)
            )
        return tuple(branches)

    def optimal_value(self, action_pools=None):
        actions = tuple(self.pools if action_pools is None else action_pools)
        if self.B == 0 or not actions:
            return 0.0

        @lru_cache(maxsize=None)
        def solve(step, worlds, cleared):
            if step == self.B:
                return 0.0
            best = 0.0
            for pool in actions:
                value = sum(
                    probability * (reward + solve(step + 1, child, new_cleared))
                    for probability, child, new_cleared, reward
                    in self.branches(worlds, cleared, pool)
                )
                if value > best:
                    best = value
            return best

        return float(solve(0, self.all_worlds, 0))

    def greedy_and_rollout_values(self, action_pools):
        """Exact values of myopic greedy and one-step rollout in one library."""

        actions = tuple(action_pools)
        if self.B == 0 or not actions:
            return 0.0, 0.0

        def greedy_action(worlds, cleared):
            scored = []
            for pool in actions:
                branches = self.branches(worlds, cleared, pool)
                immediate = sum(probability * reward for probability, _, _, reward in branches)
                scored.append((immediate, -pool, pool, branches))
            return max(scored)[2:]

        @lru_cache(maxsize=None)
        def base(step, worlds, cleared):
            if step == self.B:
                return 0.0
            _, branches = greedy_action(worlds, cleared)
            return sum(
                probability * (reward + base(step + 1, child, new_cleared))
                for probability, child, new_cleared, reward in branches
            )

        @lru_cache(maxsize=None)
        def rollout(step, worlds, cleared):
            if step == self.B:
                return 0.0
            candidates = []
            for pool in actions:
                branches = self.branches(worlds, cleared, pool)
                q_base = sum(
                    probability * (reward + base(step + 1, child, new_cleared))
                    for probability, child, new_cleared, reward in branches
                )
                candidates.append((q_base, -pool, branches))
            _, _, chosen = max(candidates)
            return sum(
                probability * (reward + rollout(step + 1, child, new_cleared))
                for probability, child, new_cleared, reward in chosen
            )

        return (
            float(base(0, self.all_worlds, 0)),
            float(rollout(0, self.all_worlds, 0)),
        )

    def static_value(self):
        """Exact best non-adaptive design with at most ``B`` distinct pools."""

        if self.B == 0:
            return 0.0
        design_size = min(self.B, len(self.pools))
        best = 0.0
        for design in combinations(self.pools, design_size):
            value = 0.0
            for world in range(self.world_count):
                cleared = 0
                for pool in design:
                    if pool & world == 0:
                        cleared |= pool
                value += self.weights[world] * self.utility[cleared]
            if value > best:
                best = value
        return float(best)


def dynamic_augmented_value(p, u, B, G):
    return ExactPolicyEvaluator(p, u, B, G).optimal_value()


def laminar_augmented_value(p, u, B, G):
    evaluator = ExactPolicyEvaluator(p, u, B, G)
    unrestricted = evaluator.optimal_value()
    best = 0.0
    for library in maximal_laminar_libraries(evaluator.n, evaluator.G):
        value = evaluator.optimal_value(library)
        if value > best:
            best = value
        if best >= unrestricted - 1e-11 * max(1.0, unrestricted):
            break
    return float(best)


def laminar_ratio(p, u, B, G):
    """Return ``V^L / V*`` without computing the static/greedy controls."""

    evaluator = ExactPolicyEvaluator(p, u, B, G)
    unrestricted = evaluator.optimal_value()
    if unrestricted <= 0.0:
        return float("nan")
    best = 0.0
    for library in maximal_laminar_libraries(evaluator.n, evaluator.G):
        best = max(best, evaluator.optimal_value(library))
        if best >= unrestricted - 1e-11 * max(1.0, unrestricted):
            break
    return float(best / unrestricted)


def greedy_laminar_value(p, u, B, G):
    evaluator = ExactPolicyEvaluator(p, u, B, G)
    library = balanced_laminar_library(p, u, G)
    return evaluator.greedy_and_rollout_values(library)[0]


def rollout_laminar_value(p, u, B, G):
    evaluator = ExactPolicyEvaluator(p, u, B, G)
    library = balanced_laminar_library(p, u, G)
    return evaluator.greedy_and_rollout_values(library)[1]


def static_binary_value(p, u, B, G):
    return ExactPolicyEvaluator(p, u, B, G).static_value()


def four_quantities(p, u, B, G):
    """Compute the weekly plan's four quantities and four declared ratios."""

    evaluator = ExactPolicyEvaluator(p, u, B, G)
    started = time.perf_counter()
    v_star = evaluator.optimal_value()
    full_seconds = time.perf_counter() - started

    started = time.perf_counter()
    v_laminar = 0.0
    best_library = ()
    libraries_checked = 0
    for library in maximal_laminar_libraries(evaluator.n, evaluator.G):
        libraries_checked += 1
        value = evaluator.optimal_value(library)
        if value > v_laminar:
            v_laminar = value
            best_library = library
        if v_laminar >= v_star - 1e-11 * max(1.0, v_star):
            break
    laminar_seconds = time.perf_counter() - started

    practical_library = balanced_laminar_library(p, u, G)
    v_greedy, v_rollout = evaluator.greedy_and_rollout_values(practical_library)
    v_static = evaluator.static_value()

    tolerance = 2e-9 * max(1.0, v_star)
    if not v_greedy <= v_laminar + tolerance <= v_star + 2 * tolerance:
        raise AssertionError("expected greedy <= laminar <= unrestricted")
    if not v_static <= v_star + tolerance:
        raise AssertionError("expected static binary <= dynamic augmented")
    if not v_greedy <= v_rollout + tolerance:
        raise AssertionError("policy improvement must dominate its base policy")

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0.0 else float("nan")

    return {
        "V_star": float(v_star),
        "V_laminar": float(v_laminar),
        "V_greedy_laminar": float(v_greedy),
        "V_rollout_laminar": float(v_rollout),
        "V_static_binary": float(v_static),
        "ratio_laminar_opt": ratio(v_laminar, v_star),
        "ratio_greedy_laminar": ratio(v_greedy, v_laminar),
        "ratio_static_opt": ratio(v_static, v_star),
        "ratio_greedy_static": ratio(v_greedy, v_static),
        "ratio_rollout_greedy": ratio(v_rollout, v_greedy),
        "full_seconds": full_seconds,
        "laminar_seconds": laminar_seconds,
        "libraries_total": len(maximal_laminar_libraries(evaluator.n, evaluator.G)),
        "libraries_checked": libraries_checked,
        "best_library": best_library,
        "practical_library": practical_library,
    }


__all__ = [
    "ExactPolicyEvaluator",
    "balanced_laminar_library",
    "dynamic_augmented_value",
    "four_quantities",
    "greedy_laminar_value",
    "laminar_augmented_value",
    "laminar_ratio",
    "laminar_compatible",
    "maximal_laminar_libraries",
    "rollout_laminar_value",
    "static_binary_value",
]
