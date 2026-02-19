"""
Brute-force DP solver for optimal DAPTS on tiny instances.

State = (step k, remaining_set, cleared_mask)
  - k: number of tests already used (0..B)
  - remaining_set: frozenset of z_masks consistent with observations
  - cleared_mask: bitmask of individuals proven healthy

The solver is exact but exponential.  Guard-railed to n <= 14.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

from augmented.core import all_pools, indices_from_mask, popcount, test_result
from augmented.strategy import DAPTS, History


# Type alias for DP states
_State = Tuple[int, frozenset, int]  # (k, remaining_set, cleared_mask)

# Maximum n allowed for brute force (2^n profiles, combinatorial explosion)
_MAX_N_BRUTEFORCE = 14


def _cleared_utility(cleared_mask: int, u: List[float], n: int) -> float:
    """Sum of u[i] for bits set in cleared_mask."""
    total = 0.0
    m = cleared_mask
    while m:
        i = (m & -m).bit_length() - 1  # lowest set bit
        total += u[i]
        m &= m - 1
    return total


def solve_optimal_dapts(
    p: List[float],
    u: List[float],
    B: int,
    G: int,
) -> Tuple[float, DAPTS]:
    """Solve for the optimal DAPTS via brute-force DP.

    Parameters
    ----------
    p : list[float]
        Infection probabilities, length n.
    u : list[float]
        Utilities, length n.
    B : int
        Budget (number of tests).
    G : int
        Maximum pool size.

    Returns
    -------
    optimal_value : float
        Expected utility of the optimal DAPTS, u(F*).
    optimal_policy : DAPTS
        The optimal strategy object.

    Raises
    ------
    ValueError
        If n > _MAX_N_BRUTEFORCE.
    """
    n = len(p)
    if n > _MAX_N_BRUTEFORCE:
        raise ValueError(
            f"Brute-force solver requires n <= {_MAX_N_BRUTEFORCE}, got n={n}. "
            f"State space is O(2^(2^n)) which is intractable for large n."
        )
    if n == 0:
        return 0.0, DAPTS(B)

    q = [1.0 - pi for pi in p]

    # Precompute weight w[z] = Pr(Z = z) for every infection profile
    num_profiles = 1 << n
    w: List[float] = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz

    # Enumerate candidate pools (exclude empty pool — it wastes a test)
    pools = all_pools(n, G, include_empty=False)

    # DP memoization: state -> (value, best_pool)
    memo: Dict[_State, Tuple[float, int]] = {}

    def dp(k: int, remaining: frozenset, cleared_mask: int) -> Tuple[float, int]:
        """Return (expected_utility, best_pool) from state (k, remaining, cleared).

        k = number of tests already used.  We have B - k tests remaining.
        remaining = frozenset of z_masks still consistent with observations.
        cleared_mask = individuals proven healthy so far.
        """
        state: _State = (k, remaining, cleared_mask)
        if state in memo:
            return memo[state]

        # Terminal: no more tests
        if k == B:
            # Utility depends only on cleared individuals.
            # Expected value = (1/total_mass) * sum_{z in remaining} w[z] * cleared_utility
            # Since cleared_utility is constant w.r.t. z:
            val = _cleared_utility(cleared_mask, u, n)
            # But we need to weight by the conditional distribution.
            # Actually, the DP returns the *weighted* value (not conditional),
            # so terminal value = total_mass * cleared_utility.
            total_mass = sum(w[z] for z in remaining)
            result = (total_mass * val, 0)
            memo[state] = result
            return result

        total_mass = sum(w[z] for z in remaining)

        # If remaining is empty (shouldn't happen in practice), value is 0
        if total_mass == 0.0:
            result = (0.0, 0)
            memo[state] = result
            return result

        best_value = -1.0
        best_pool = 0

        for pool in pools:
            # Partition remaining profiles by outcome r = test_result(pool, z)
            buckets: Dict[int, List[int]] = {}
            for z in remaining:
                r = test_result(pool, z)
                if r not in buckets:
                    buckets[r] = []
                buckets[r].append(z)

            ev = 0.0
            for r, z_list in buckets.items():
                new_remaining = frozenset(z_list)
                new_cleared = cleared_mask | pool if r == 0 else cleared_mask
                sub_val, _ = dp(k + 1, new_remaining, new_cleared)
                ev += sub_val

            if ev > best_value:
                best_value = ev
                best_pool = pool

        # Also consider the empty pool (doing nothing / wasting a test).
        # This is equivalent to not using the test — all outcomes are the same.
        # Value of empty pool: dp(k+1, remaining, cleared_mask | 0) = dp(k+1, remaining, cleared_mask)
        # Since empty pool always has result 0 and cleared |= 0 = cleared.
        # Actually empty pool with result 0 means cleared |= 0 = cleared (no change).
        # We already excluded empty pool from candidates, but let's check if
        # wasting a test could somehow be better (it can't, but for correctness):
        waste_val, _ = dp(k + 1, remaining, cleared_mask)
        if waste_val > best_value:
            best_value = waste_val
            best_pool = 0  # empty pool

        result = (best_value, best_pool)
        memo[state] = result
        return result

    # Initial state: k=0, all profiles possible, nothing cleared
    all_z = frozenset(range(num_profiles))
    optimal_value, _ = dp(0, all_z, 0)

    # --- Policy reconstruction ---
    # Build a DAPTS object by tracing the DP argmax decisions.
    # We reconstruct by simulating all reachable histories.
    policy_obj = DAPTS(B)

    def reconstruct(k: int, remaining: frozenset, cleared_mask: int, history: History):
        if k == B:
            return

        state: _State = (k, remaining, cleared_mask)
        _, best_pool = memo[state]

        # Record this decision
        policy_obj.set_action(k + 1, history, best_pool)

        # Partition by outcome and recurse
        buckets: Dict[int, List[int]] = {}
        for z in remaining:
            r = test_result(best_pool, z)
            if r not in buckets:
                buckets[r] = []
            buckets[r].append(z)

        for r, z_list in buckets.items():
            new_remaining = frozenset(z_list)
            new_cleared = cleared_mask | best_pool if r == 0 else cleared_mask
            new_history = history + ((best_pool, r),)
            reconstruct(k + 1, new_remaining, new_cleared, new_history)

    reconstruct(0, all_z, 0, ())

    return optimal_value, policy_obj
