"""
Dynamic Augmented Pooled Testing Strategies (DAPTS).

Implements the brute-force warm-up machinery from:
  "Dynamic-Augmented Pooled Testing" (Vladimir, Hector, Francisco, Jan 2026)
  Section 2: Warm-up: Brute Force for Small Instances

An augmented pooled test returns the exact COUNT of infected individuals
in the pool, rather than just a binary positive/negative result.
"""

from augmented.core import (
    mask_from_indices,
    indices_from_mask,
    popcount,
    all_pools,
    test_result,
)
from augmented.strategy import DAPTS, History
from augmented.simulator import apply_dapts, u_realized
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts

__all__ = [
    "mask_from_indices",
    "indices_from_mask",
    "popcount",
    "all_pools",
    "test_result",
    "DAPTS",
    "History",
    "apply_dapts",
    "u_realized",
    "exact_expected_utility",
    "mc_expected_utility",
    "u_max",
    "u_single",
    "solve_optimal_dapts",
]
