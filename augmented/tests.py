"""
Unit tests for the DAPTS machinery.

Run with:  python augmented/tests.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (
    all_pools, indices_from_mask, mask_from_indices, mask_str,
    popcount, test_result,
)
from augmented.strategy import DAPTS
from augmented.simulator import apply_dapts
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.bayesian import (
    bayesian_update_single_test, bayesian_update, _poisson_binomial_pmf,
)
from augmented.greedy import (
    greedy_myopic_simulate, greedy_myopic_expected_utility,
    greedy_lookahead_simulate,
)
from augmented.static_solver import solve_static_non_overlapping, solve_static_overlapping
from augmented.classical_solver import solve_classical_dynamic


# ===================================================================
# 1) Bitmask helpers
# ===================================================================

def test_mask_from_indices():
    assert mask_from_indices([]) == 0
    assert mask_from_indices([0]) == 1
    assert mask_from_indices([0, 2]) == 5
    assert mask_from_indices([3]) == 8

def test_indices_from_mask():
    assert indices_from_mask(0, 4) == []
    assert indices_from_mask(1, 4) == [0]
    assert indices_from_mask(0b1010, 4) == [1, 3]
    assert indices_from_mask(0b111, 3) == [0, 1, 2]

def test_popcount():
    assert popcount(0) == 0
    assert popcount(0b111) == 3
    assert popcount(0b10101) == 3

def test_mask_str():
    assert mask_str(0) == "{}"
    assert mask_str(0b10101) == "{0,2,4}"
    assert mask_str(0b1111, n=2) == "{0,1}"


# ===================================================================
# 2) test_result
# ===================================================================

def test_test_result():
    pool = mask_from_indices([0, 1, 2])
    assert test_result(pool, 0) == 0                          # nobody infected
    assert test_result(pool, mask_from_indices([1])) == 1     # one infected
    assert test_result(pool, pool) == 3                       # all infected
    assert test_result(mask_from_indices([0, 2]),
                       mask_from_indices([1, 2, 3])) == 1     # partial overlap


# ===================================================================
# 3) all_pools
# ===================================================================

def test_all_pools_counts():
    assert len(all_pools(3, 2, include_empty=True)) == 7   # C(3,0)+C(3,1)+C(3,2)
    assert len(all_pools(3, 2, include_empty=False)) == 6
    assert len(all_pools(3, 3, include_empty=True)) == 8    # 2^3


# ===================================================================
# 4) apply_dapts
# ===================================================================

def test_apply_dapts_simple():
    n, B = 3, 1
    u_vec = [1.0, 2.0, 3.0]
    F = DAPTS(B)
    pool_all = mask_from_indices([0, 1, 2])
    F.set_action(1, (), pool_all)

    # Z = 0: all cleared
    hist, cleared, u_val = apply_dapts(F, 0, n, u_vec)
    assert cleared == pool_all and u_val == 6.0

    # Z = {1}: pool positive, nobody cleared
    hist, cleared, u_val = apply_dapts(F, mask_from_indices([1]), n, u_vec)
    assert cleared == 0 and u_val == 0.0


# ===================================================================
# 5) exact vs MC expected utility
# ===================================================================

def test_exact_vs_mc():
    n, p, u_vec, B = 3, [0.1, 0.2, 0.15], [1.0, 2.0, 3.0], 1
    F = DAPTS(B)
    F.set_action(1, (), mask_from_indices([0, 1, 2]))

    exact = exact_expected_utility(F, p, u_vec, n)
    mc = mc_expected_utility(F, p, u_vec, n, trials=200_000, seed=123)
    assert abs(exact - mc) < 0.05

    # One pool of everyone: utility earned only if ALL healthy
    expected = sum(u_vec) * 0.9 * 0.8 * 0.85
    assert abs(exact - expected) < 1e-10


# ===================================================================
# 6) Baselines
# ===================================================================

def test_u_max():
    val = u_max([0.1, 0.2, 0.3], [10.0, 20.0, 30.0])
    assert abs(val - 46.0) < 1e-10  # 9 + 16 + 21

def test_u_single():
    val, selected = u_single([0.1, 0.2, 0.3], [10.0, 20.0, 30.0], B=2)
    assert abs(val - 37.0) < 1e-10  # top 2: 21 + 16
    assert set(selected) == {1, 2}


# ===================================================================
# 7) U_single <= U_optimal <= U_max
# ===================================================================

def test_inequality_chain():
    p, u_vec, B, G = [0.1, 0.2, 0.15], [5.0, 3.0, 4.0], 2, 2
    u_s, _ = u_single(p, u_vec, B)
    u_opt, _ = solve_optimal_dapts(p, u_vec, B, G)
    u_m = u_max(p, u_vec)
    assert u_s <= u_opt + 1e-10
    assert u_opt <= u_m + 1e-10


# ===================================================================
# 8) Hand-checkable: n=2, B=1, G=2
# ===================================================================

def test_hand_checkable_n2_B1():
    # Pool {0}: E = 10*0.9 = 9.0 (best)
    p, u_vec, B, G = [0.1, 0.5], [10.0, 1.0], 1, 2
    opt_val, F = solve_optimal_dapts(p, u_vec, B, G)
    assert abs(opt_val - 9.0) < 1e-10
    assert F.choose(1, ()) == mask_from_indices([0])


# ===================================================================
# 9) Hand-checkable: n=3, G=2, B=1
# ===================================================================

def test_n3_G2_B1():
    # Best pool {0,1}: E = 2 * 0.99 * 0.99 = 1.9602
    p, u_vec, B, G = [0.01, 0.01, 0.5], [1.0, 1.0, 1.0], 1, 2
    opt_val, F = solve_optimal_dapts(p, u_vec, B, G)
    assert abs(opt_val - 2.0 * 0.99 * 0.99) < 1e-10
    assert F.choose(1, ()) == mask_from_indices([0, 1])


# ===================================================================
# 10) Solver matches exact EU
# ===================================================================

def test_solver_matches_exact_eu():
    p, u_vec, B, G = [0.1, 0.2, 0.15], [5.0, 3.0, 4.0], 2, 2
    opt_val, F = solve_optimal_dapts(p, u_vec, B, G)
    exact_eu = exact_expected_utility(F, p, u_vec, len(p))
    assert abs(opt_val - exact_eu) < 1e-9


# ===================================================================
# 11) More budget helps
# ===================================================================

def test_more_budget_helps():
    p, u_vec, G = [0.1, 0.2, 0.3], [3.0, 3.0, 3.0], 3
    opt1, _ = solve_optimal_dapts(p, u_vec, B=1, G=G)
    _, F2 = solve_optimal_dapts(p, u_vec, B=2, G=G)
    opt2 = exact_expected_utility(F2, p, u_vec, len(p))
    assert opt2 >= opt1 - 1e-10


# ===================================================================
# 12) Poisson-Binomial PMF
# ===================================================================

def test_poisson_binomial_pmf():
    # Single coin with p=0.3
    pmf = _poisson_binomial_pmf([0.3])
    assert abs(pmf[0] - 0.7) < 1e-10
    assert abs(pmf[1] - 0.3) < 1e-10

    # Two fair coins: P(0)=0.25, P(1)=0.5, P(2)=0.25
    pmf = _poisson_binomial_pmf([0.5, 0.5])
    assert abs(pmf[0] - 0.25) < 1e-10
    assert abs(pmf[1] - 0.50) < 1e-10
    assert abs(pmf[2] - 0.25) < 1e-10

    # Empty: P(0) = 1
    pmf = _poisson_binomial_pmf([])
    assert abs(pmf[0] - 1.0) < 1e-10

    # PMF should sum to 1
    pmf = _poisson_binomial_pmf([0.1, 0.4, 0.7])
    assert abs(sum(pmf) - 1.0) < 1e-10


# ===================================================================
# 13) Bayesian update — single test
# ===================================================================

def test_bayesian_r0_clears_pool():
    # r=0 means nobody in pool is infected => posterior p_i = 0 for i in pool
    p = [0.3, 0.5, 0.2]
    pool = mask_from_indices([0, 1])
    post = bayesian_update_single_test(p, pool, r=0, n=3)
    assert abs(post[0]) < 1e-10  # individual 0 proven healthy
    assert abs(post[1]) < 1e-10  # individual 1 proven healthy
    assert abs(post[2] - 0.2) < 1e-10  # individual 2 untested, unchanged


def test_bayesian_r_equals_pool_size():
    # r = |pool| means everyone in pool is infected => posterior p_i = 1
    p = [0.3, 0.5, 0.2]
    pool = mask_from_indices([0, 1])
    post = bayesian_update_single_test(p, pool, r=2, n=3)
    assert abs(post[0] - 1.0) < 1e-10
    assert abs(post[1] - 1.0) < 1e-10
    assert abs(post[2] - 0.2) < 1e-10


def test_bayesian_single_individual():
    # Pool of size 1: r=0 => healthy, r=1 => infected
    p = [0.4, 0.6]
    post0 = bayesian_update_single_test(p, mask_from_indices([0]), r=0, n=2)
    assert abs(post0[0]) < 1e-10

    post1 = bayesian_update_single_test(p, mask_from_indices([0]), r=1, n=2)
    assert abs(post1[0] - 1.0) < 1e-10


def test_bayesian_partial_result():
    # Pool {0,1}, r=1: exactly one is infected. Who is more likely?
    # p = [0.1, 0.9] — individual 1 is much more likely to be infected.
    p = [0.1, 0.9]
    pool = mask_from_indices([0, 1])
    post = bayesian_update_single_test(p, pool, r=1, n=2)

    # P(Z_0=1 | r=1): P(r=1|Z_0=1)*p_0 / P(r=1)
    #   P(r=1|Z_0=1) = P(Z_1=0) = 0.1,  so numerator = 0.1 * 0.1 = 0.01
    #   P(r=1|Z_0=0) = P(Z_1=1) = 0.9,  so denom term = 0.9 * 0.9 = 0.81
    #   P(r=1) = 0.01 + 0.81 = 0.82
    #   post[0] = 0.01 / 0.82
    assert abs(post[0] - 0.01 / 0.82) < 1e-10
    assert abs(post[1] - 0.81 / 0.82) < 1e-10


def test_bayesian_untested_unchanged():
    # Individuals not in the pool should keep their prior
    p = [0.1, 0.2, 0.3, 0.4]
    pool = mask_from_indices([1, 2])
    post = bayesian_update_single_test(p, pool, r=1, n=4)
    assert abs(post[0] - 0.1) < 1e-10
    assert abs(post[3] - 0.4) < 1e-10


# ===================================================================
# 14) Bayesian update — full history
# ===================================================================

def test_bayesian_full_history():
    # Two sequential tests
    p = [0.3, 0.3, 0.3]
    n = 3
    # Test 1: pool {0,1}, r=0 => both healthy
    # Test 2: pool {2}, r=1 => individual 2 infected
    history = (
        (mask_from_indices([0, 1]), 0),
        (mask_from_indices([2]), 1),
    )
    post = bayesian_update(p, history, n)
    assert abs(post[0]) < 1e-10      # proven healthy
    assert abs(post[1]) < 1e-10      # proven healthy
    assert abs(post[2] - 1.0) < 1e-10  # proven infected


def test_bayesian_empty_history():
    p = [0.1, 0.5]
    post = bayesian_update(p, (), 2)
    assert abs(post[0] - 0.1) < 1e-10
    assert abs(post[1] - 0.5) < 1e-10


# ===================================================================
# 15) Greedy: U_single <= U_greedy <= U_optimal
# ===================================================================

def test_greedy_inequality_chain():
    p, u_vec, B, G = [0.1, 0.2, 0.15], [5.0, 3.0, 4.0], 2, 2
    u_s, _ = u_single(p, u_vec, B)
    u_greedy = greedy_myopic_expected_utility(p, u_vec, B, G)
    u_opt, _ = solve_optimal_dapts(p, u_vec, B, G)
    u_m = u_max(p, u_vec)
    assert u_s <= u_greedy + 1e-10, f"U_single={u_s} > U_greedy={u_greedy}"
    assert u_greedy <= u_opt + 1e-10, f"U_greedy={u_greedy} > U_opt={u_opt}"
    assert u_opt <= u_m + 1e-10


def test_greedy_myopic_B1():
    # With B=1, myopic greedy IS optimal
    p, u_vec, G = [0.1, 0.5], [10.0, 1.0], 2
    u_greedy = greedy_myopic_expected_utility(p, u_vec, B=1, G=G)
    u_opt, _ = solve_optimal_dapts(p, u_vec, B=1, G=G)
    assert abs(u_greedy - u_opt) < 1e-10


def test_greedy_simulate_all_healthy():
    # Nobody infected: greedy should clear everyone it tests
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    _, cleared, utility = greedy_myopic_simulate(p, u_vec, B=2, G=3, z_mask=0)
    assert utility > 0  # should clear at least some people


def test_greedy_lookahead_simulate():
    # Lookahead should also work correctly
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    _, _, util_la = greedy_lookahead_simulate(p, u_vec, B=2, G=2, z_mask=0)
    _, _, util_my = greedy_myopic_simulate(p, u_vec, B=2, G=2, z_mask=0)
    # Both should clear people when nobody is infected
    assert util_la > 0
    assert util_my > 0


# ===================================================================
# 16) Full inequality chain: U_single <= U_s_NO <= U_s_O <= U_D <= U_D_A <= U_max
# ===================================================================

def test_full_inequality_chain():
    p, u_vec, B, G = [0.1, 0.2, 0.15], [5.0, 3.0, 4.0], 2, 2
    u_s, _ = u_single(p, u_vec, B)
    u_sno, _ = solve_static_non_overlapping(p, u_vec, B, G)
    u_so, _ = solve_static_overlapping(p, u_vec, B, G)
    u_d, _ = solve_classical_dynamic(p, u_vec, B, G)
    u_da, _ = solve_optimal_dapts(p, u_vec, B, G)
    u_m = u_max(p, u_vec)

    assert u_s <= u_sno + 1e-9, f"U_single > U_s_NO"
    assert u_sno <= u_so + 1e-9, f"U_s_NO > U_s_O"
    assert u_so <= u_d + 1e-9, f"U_s_O > U_D"
    assert u_d <= u_da + 1e-9, f"U_D > U_D_A"
    assert u_da <= u_m + 1e-9, f"U_D_A > U_max"


def test_classical_vs_augmented_dynamic():
    # Augmented should be >= classical (strictly more information)
    p, u_vec, B, G = [0.3, 0.4, 0.35, 0.25], [5.0, 3.0, 4.0, 6.0], 2, 2
    u_d, _ = solve_classical_dynamic(p, u_vec, B, G)
    u_da, _ = solve_optimal_dapts(p, u_vec, B, G)
    assert u_da >= u_d - 1e-9


def test_static_non_overlapping_B1():
    # With B=1, static non-overlapping = best single pool
    p, u_vec, G = [0.1, 0.2], [10.0, 1.0], 2
    u_sno, pools = solve_static_non_overlapping(p, u_vec, B=1, G=G)
    # Best pool {0}: 10*0.9 = 9.0
    assert abs(u_sno - 9.0) < 1e-10


def test_overlapping_beats_non_overlapping():
    # With the right instance, overlapping can beat non-overlapping
    p, u_vec, B, G = [0.1, 0.2, 0.3], [5.0, 3.0, 4.0], 2, 2
    u_sno, _ = solve_static_non_overlapping(p, u_vec, B, G)
    u_so, _ = solve_static_overlapping(p, u_vec, B, G)
    assert u_so >= u_sno - 1e-9


# ===================================================================
# Run all tests
# ===================================================================

def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and getattr(v, "__module__", None) == __name__]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
