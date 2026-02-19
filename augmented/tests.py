"""
Unit tests for the DAPTS brute-force machinery.

Run with:
    python -m pytest dapts/tests.py -v
or:
    python dapts/tests.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (
    all_pools,
    indices_from_mask,
    mask_from_indices,
    popcount,
    test_result,
)
from augmented.strategy import DAPTS
from augmented.simulator import apply_dapts
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts


# ===================================================================
# 1) Bitmask helpers
# ===================================================================

def test_mask_from_indices():
    assert mask_from_indices([]) == 0
    assert mask_from_indices([0]) == 1
    assert mask_from_indices([1]) == 2
    assert mask_from_indices([0, 1]) == 3
    assert mask_from_indices([0, 2]) == 5
    assert mask_from_indices([3]) == 8


def test_indices_from_mask():
    assert indices_from_mask(0, 4) == []
    assert indices_from_mask(1, 4) == [0]
    assert indices_from_mask(0b1010, 4) == [1, 3]
    assert indices_from_mask(0b111, 3) == [0, 1, 2]


def test_popcount():
    assert popcount(0) == 0
    assert popcount(1) == 1
    assert popcount(0b111) == 3
    assert popcount(0b10101) == 3


# ===================================================================
# 2) test_result correctness
# ===================================================================

def test_test_result_all_healthy():
    # Pool {0,1,2}, nobody infected
    pool = mask_from_indices([0, 1, 2])
    z = 0
    assert test_result(pool, z) == 0


def test_test_result_one_infected():
    pool = mask_from_indices([0, 1, 2])
    z = mask_from_indices([1])  # only individual 1 infected
    assert test_result(pool, z) == 1


def test_test_result_all_infected():
    pool = mask_from_indices([0, 1, 2])
    z = mask_from_indices([0, 1, 2])
    assert test_result(pool, z) == 3


def test_test_result_partial_overlap():
    pool = mask_from_indices([0, 2])
    z = mask_from_indices([1, 2, 3])
    # pool & z = {2}, count = 1
    assert test_result(pool, z) == 1


# ===================================================================
# 3) all_pools enumeration
# ===================================================================

def test_all_pools_counts():
    # n=3, G=2: pools of size 0,1,2
    # size 0: 1 (empty), size 1: 3, size 2: 3 => total 7
    pools = all_pools(3, 2, include_empty=True)
    assert len(pools) == 7
    pools_no_empty = all_pools(3, 2, include_empty=False)
    assert len(pools_no_empty) == 6

    # n=3, G=3: all subsets = 2^3 = 8
    pools_full = all_pools(3, 3, include_empty=True)
    assert len(pools_full) == 8


# ===================================================================
# 4) apply_dapts correctness
# ===================================================================

def test_apply_dapts_simple():
    # n=3, B=1, G=3. Strategy: test everyone.
    # If nobody is infected, all are cleared.
    n, B = 3, 1
    u_vec = [1.0, 2.0, 3.0]
    F = DAPTS(B)
    pool_all = mask_from_indices([0, 1, 2])
    F.set_action(1, (), pool_all)

    # Z = 0 (nobody infected)
    hist, cleared, u_val = apply_dapts(F, 0, n, u_vec)
    assert len(hist) == 1
    assert hist[0] == (pool_all, 0)
    assert cleared == pool_all
    assert u_val == 6.0

    # Z = {1} (individual 1 infected)
    z = mask_from_indices([1])
    hist, cleared, u_val = apply_dapts(F, z, n, u_vec)
    assert hist[0] == (pool_all, 1)
    assert cleared == 0  # pool was positive, nobody cleared
    assert u_val == 0.0


# ===================================================================
# 5) exact vs MC expected utility
# ===================================================================

def test_exact_vs_mc():
    # n=3, B=1, G=3: test everyone in one pool
    n = 3
    p = [0.1, 0.2, 0.15]
    u_vec = [1.0, 2.0, 3.0]
    B = 1

    F = DAPTS(B)
    pool_all = mask_from_indices([0, 1, 2])
    F.set_action(1, (), pool_all)

    exact = exact_expected_utility(F, p, u_vec, n)
    mc = mc_expected_utility(F, p, u_vec, n, trials=200_000, seed=123)

    # They should agree within a reasonable tolerance
    assert abs(exact - mc) < 0.05, f"exact={exact:.4f}, mc={mc:.4f}"

    # Also check that exact = sum(u) * prod(q)
    # because with one pool of everyone, utility is earned only if ALL are healthy
    prod_q = 1.0
    for pi in p:
        prod_q *= (1.0 - pi)
    expected = sum(u_vec) * prod_q
    assert abs(exact - expected) < 1e-10


# ===================================================================
# 6) Baselines
# ===================================================================

def test_u_max():
    p = [0.1, 0.2, 0.3]
    u_vec = [10.0, 20.0, 30.0]
    val = u_max(p, u_vec)
    expected = 10 * 0.9 + 20 * 0.8 + 30 * 0.7  # 9 + 16 + 21 = 46
    assert abs(val - expected) < 1e-10


def test_u_single():
    p = [0.1, 0.2, 0.3]
    u_vec = [10.0, 20.0, 30.0]
    # scores: 10*0.9=9, 20*0.8=16, 30*0.7=21
    # B=2: top 2 are indices 2 (21) and 1 (16) => 37
    val, selected = u_single(p, u_vec, B=2)
    assert abs(val - 37.0) < 1e-10
    assert set(selected) == {1, 2}


# ===================================================================
# 7) Sanity: U_single <= U_optimal <= U_max
# ===================================================================

def test_inequality_chain():
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    B, G = 2, 2

    u_s, _ = u_single(p, u_vec, B)
    u_opt, F_opt = solve_optimal_dapts(p, u_vec, B, G)
    u_m = u_max(p, u_vec)

    assert u_s <= u_opt + 1e-10, f"U_single={u_s} > U_optimal={u_opt}"
    assert u_opt <= u_m + 1e-10, f"U_optimal={u_opt} > U_max={u_m}"


# ===================================================================
# 8) Hand-checkable case: n=2, G=2, B=1
# ===================================================================

def test_hand_checkable_n2_B1():
    # n=2, B=1, G=2
    # Individuals: 0 (u=10, p=0.1) and 1 (u=1, p=0.5)
    # Possible pools (non-empty): {0}, {1}, {0,1}
    #
    # Pool {0}: clears 0 if healthy => E = 10*0.9 = 9.0
    # Pool {1}: clears 1 if healthy => E = 1*0.5 = 0.5
    # Pool {0,1}: clears both if both healthy => E = (10+1)*0.9*0.5 = 4.95
    #
    # Optimal: test {0} alone => E = 9.0
    p = [0.1, 0.5]
    u_vec = [10.0, 1.0]
    B, G = 1, 2

    opt_val, F_opt = solve_optimal_dapts(p, u_vec, B, G)
    assert abs(opt_val - 9.0) < 1e-10, f"Expected 9.0, got {opt_val}"

    # Verify policy chooses pool {0}
    pool_chosen = F_opt.choose(1, ())
    assert pool_chosen == mask_from_indices([0])


# ===================================================================
# 9) Hand-checkable: n=3, G=2, B=1 — intuitive decision
# ===================================================================

def test_n3_G2_B1():
    # n=3, B=1, G=2
    # u = [1, 1, 1], p = [0.01, 0.01, 0.5]
    # Individuals 0,1 are almost certainly healthy. Individual 2 is 50/50.
    #
    # Best single pool of size <=2:
    #   {0,1}: E = 2 * 0.99 * 0.99 = 1.9602
    #   {0,2}: E = 2 * 0.99 * 0.5  = 0.99
    #   {0}:   E = 1 * 0.99         = 0.99
    #   {1}:   E = 1 * 0.99         = 0.99
    #   etc.
    # Optimal: pool {0,1} => E ≈ 1.9602
    p = [0.01, 0.01, 0.5]
    u_vec = [1.0, 1.0, 1.0]
    B, G = 1, 2

    opt_val, F_opt = solve_optimal_dapts(p, u_vec, B, G)
    expected = 2.0 * 0.99 * 0.99
    assert abs(opt_val - expected) < 1e-10, f"Expected {expected}, got {opt_val}"

    pool_chosen = F_opt.choose(1, ())
    assert pool_chosen == mask_from_indices([0, 1])


# ===================================================================
# 10) Solver result matches exact_expected_utility
# ===================================================================

def test_solver_matches_exact_eu():
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    B, G = 2, 2

    opt_val, F_opt = solve_optimal_dapts(p, u_vec, B, G)
    exact_eu = exact_expected_utility(F_opt, p, u_vec, len(p))

    assert abs(opt_val - exact_eu) < 1e-9, (
        f"Solver value {opt_val} != exact EU {exact_eu}"
    )


# ===================================================================
# 11) Multi-round dynamic benefit: B=2 should beat B=1
# ===================================================================

def test_more_budget_helps():
    p = [0.1, 0.2, 0.3]
    u_vec = [3.0, 3.0, 3.0]
    G = 3

    opt1, _ = solve_optimal_dapts(p, u_vec, B=1, G=G)
    _, F2 = solve_optimal_dapts(p, u_vec, B=2, G=G)

    opt2 = exact_expected_utility(F2, p, u_vec, len(p))
    assert opt2 >= opt1 - 1e-10, f"B=2 ({opt2}) should be >= B=1 ({opt1})"


# ===================================================================
# Run all tests
# ===================================================================

def _run_all():
    """Run all test functions and report results."""
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and getattr(v, "__module__", None) == __name__]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
