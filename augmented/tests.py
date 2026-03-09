"""
Unit tests for the DAPTS machinery.

Run with:  python augmented/tests.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (
    all_pools, all_pools_from_mask, compute_active_mask,
    indices_from_mask, mask_from_indices, mask_str,
    popcount, test_result,
)
from augmented.strategy import DAPTS
from augmented.simulator import apply_dapts
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.bayesian import (
    bayesian_update_single_test, bayesian_update,
    bayesian_update_by_counting, estimate_p_from_history,
    _poisson_binomial_pmf,
)
from augmented.greedy import (
    greedy_myopic_simulate, greedy_myopic_expected_utility,
    greedy_lookahead_simulate,
    greedy_myopic_counting_simulate, greedy_myopic_counting_expected_utility,
)
from augmented.tree_extractor import extract_tree, print_tree, tree_to_string, export_tree_dot
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
# 17) Preprocessing: compute_active_mask and all_pools_from_mask
# ===================================================================

def test_compute_active_mask_basic():
    # p = [0.0, 0.5, 1.0, 0.3]: individual 0 is healthy, 2 is infected
    active, confirmed = compute_active_mask([0.0, 0.5, 1.0, 0.3], 0, 4)
    assert active == mask_from_indices([1, 3])  # only uncertain
    assert confirmed == mask_from_indices([2])  # confirmed infected


def test_compute_active_mask_with_cleared():
    # Individual 1 already cleared
    cleared = mask_from_indices([1])
    active, confirmed = compute_active_mask([0.3, 0.5, 0.2], cleared, 3)
    assert active == mask_from_indices([0, 2])  # 1 is cleared


def test_compute_active_mask_all_certain():
    active, confirmed = compute_active_mask([0.0, 1.0], 0, 2)
    assert active == 0
    assert confirmed == mask_from_indices([1])


def test_all_pools_from_mask():
    active = mask_from_indices([0, 2, 3])  # only individuals 0, 2, 3
    pools = all_pools_from_mask(active, G=2, include_empty=False)
    # Should have C(3,1)+C(3,2) = 3+3 = 6 pools from {0,2,3}
    assert len(pools) == 6
    # Verify no pool contains individual 1
    for pool in pools:
        assert not (pool >> 1 & 1), f"Pool {mask_str(pool, 4)} contains individual 1"


def test_all_pools_from_mask_empty():
    pools = all_pools_from_mask(0, G=3, include_empty=False)
    assert pools == []


# ===================================================================
# 18) Bayesian update by counting
# ===================================================================

def test_counting_captures_cross_test_info():
    """Counting-based update captures cross-test information that sequential misses.

    Test 1: pool {0,1}, r=1 → exactly one of {0,1} is infected.
    Test 2: pool {1,2}, r=0 → 1 and 2 are healthy.
    Combined: since 1 is healthy (test 2) and exactly one of {0,1} infected (test 1),
    individual 0 must be infected.

    The counting approach correctly deduces P(Z_0=1) = 1.0.
    The sequential approach only updates individuals IN the pool at each step,
    so it misses this cross-test deduction (leaves P(Z_0=1) = 0.3).
    """
    p = [0.3, 0.5, 0.2]
    n = 3
    history = (
        (mask_from_indices([0, 1]), 1),
        (mask_from_indices([1, 2]), 0),
    )
    counting = bayesian_update_by_counting(p, history, n)
    # Counting correctly deduces: 0 infected, 1 healthy, 2 healthy
    assert abs(counting[0] - 1.0) < 1e-10, f"Expected P(Z_0=1)=1, got {counting[0]}"
    assert abs(counting[1] - 0.0) < 1e-10, f"Expected P(Z_1=1)=0, got {counting[1]}"
    assert abs(counting[2] - 0.0) < 1e-10, f"Expected P(Z_2=1)=0, got {counting[2]}"

    # Sequential misses the cross-test deduction for individual 0
    seq = bayesian_update(p, history, n)
    assert abs(seq[0] - 0.3) < 1e-10  # sequential doesn't update 0 from test 2
    assert abs(seq[1] - 0.0) < 1e-10  # but correctly clears 1
    assert abs(seq[2] - 0.0) < 1e-10  # and 2


def test_counting_r0_clears():
    p = [0.3, 0.5, 0.2]
    pool = mask_from_indices([0, 1])
    post = bayesian_update_by_counting(p, ((pool, 0),), 3)
    assert abs(post[0]) < 1e-10
    assert abs(post[1]) < 1e-10
    assert abs(post[2] - 0.2) < 1e-10


def test_counting_r_equals_pool_size():
    p = [0.3, 0.5, 0.2]
    pool = mask_from_indices([0, 1])
    post = bayesian_update_by_counting(p, ((pool, 2),), 3)
    assert abs(post[0] - 1.0) < 1e-10
    assert abs(post[1] - 1.0) < 1e-10


def test_counting_empty_history():
    p = [0.1, 0.5]
    post = bayesian_update_by_counting(p, (), 2)
    assert abs(post[0] - 0.1) < 1e-10
    assert abs(post[1] - 0.5) < 1e-10


def test_counting_full_history_cross_test():
    """Counting correctly propagates information across overlapping pools.

    Test 1: pool {0,1,2}, r=1 → exactly 1 infected in {0,1,2}
    Test 2: pool {2,3}, r=0   → 2 and 3 are healthy
    Test 3: pool {0}, r=1     → 0 is infected

    Combined: 0 infected, 2 healthy, 3 healthy. Since exactly 1 in {0,1,2}
    is infected (test 1) and that's individual 0, individual 1 must be healthy.

    Counting gets this right. Sequential misses the cross-test deduction for 1.
    """
    p = [0.1, 0.2, 0.3, 0.4]
    n = 4
    history = (
        (mask_from_indices([0, 1, 2]), 1),
        (mask_from_indices([2, 3]), 0),
        (mask_from_indices([0]), 1),
    )
    counting = bayesian_update_by_counting(p, history, n)
    assert abs(counting[0] - 1.0) < 1e-10  # confirmed infected
    assert abs(counting[1] - 0.0) < 1e-10  # deduced healthy via cross-test
    assert abs(counting[2] - 0.0) < 1e-10  # proven healthy
    assert abs(counting[3] - 0.0) < 1e-10  # proven healthy

    # Sequential does NOT deduce individual 1 is healthy
    seq = bayesian_update(p, history, n)
    assert abs(seq[0] - 1.0) < 1e-10  # test 3 confirms
    assert seq[1] > 0.1  # sequential doesn't fully update 1
    assert abs(seq[2] - 0.0) < 1e-10
    assert abs(seq[3] - 0.0) < 1e-10


# ===================================================================
# 19) Edge cases for p = 0 or p = 1
# ===================================================================

def test_bayesian_p_zero():
    """Individual with p=0 should stay at 0 after any update."""
    p = [0.0, 0.5, 0.3]
    pool = mask_from_indices([0, 1, 2])
    post = bayesian_update_single_test(p, pool, r=1, n=3)
    assert abs(post[0]) < 1e-10  # stays at 0


def test_bayesian_p_one():
    """Individual with p=1 should stay at 1 after any update."""
    p = [1.0, 0.5, 0.3]
    pool = mask_from_indices([0, 1, 2])
    post = bayesian_update_single_test(p, pool, r=2, n=3)
    assert abs(post[0] - 1.0) < 1e-10  # stays at 1


# ===================================================================
# 20) Decision tree extraction
# ===================================================================

def test_extract_tree_simple():
    """Extract tree from a simple B=1 policy."""
    p = [0.1, 0.2]
    u_vec = [5.0, 3.0]
    _, F = solve_optimal_dapts(p, u_vec, B=1, G=2)
    tree = extract_tree(F, p, u_vec, n=2)
    assert not tree['terminal']
    assert tree['step'] == 1
    assert 'children' in tree
    # All children should be terminal (B=1)
    for r, child in tree['children'].items():
        assert child['terminal']


def test_tree_to_string():
    """Tree string output should not be empty."""
    p = [0.1, 0.2]
    u_vec = [5.0, 3.0]
    _, F = solve_optimal_dapts(p, u_vec, B=1, G=2)
    tree = extract_tree(F, p, u_vec, n=2)
    s = tree_to_string(tree, n=2)
    assert len(s) > 0
    assert "Step 1" in s


def test_export_tree_dot():
    """DOT export should produce valid DOT syntax."""
    p = [0.1, 0.2]
    u_vec = [5.0, 3.0]
    _, F = solve_optimal_dapts(p, u_vec, B=1, G=2)
    tree = extract_tree(F, p, u_vec, n=2)
    dot = export_tree_dot(tree, n=2)
    assert "digraph" in dot
    assert "n0" in dot


def test_extract_tree_B2():
    """Extract tree from a B=2 policy — should have depth 2."""
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    _, F = solve_optimal_dapts(p, u_vec, B=2, G=2)
    tree = extract_tree(F, p, u_vec, n=3)
    assert not tree['terminal']
    # At least one child should be non-terminal (depth 2)
    has_depth2 = False
    for r, child in tree['children'].items():
        if not child.get('terminal') and 'children' in child:
            has_depth2 = True
    assert has_depth2


# ===================================================================
# 21) Counting-based greedy
# ===================================================================

def test_counting_greedy_simulate():
    """Counting greedy should produce valid results."""
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    hist, cleared, util = greedy_myopic_counting_simulate(
        p, u_vec, B=2, G=2, z_mask=0)
    assert util > 0  # all healthy, should clear some


def test_counting_greedy_matches_sequential_z0():
    """For z=0 (nobody infected), both greedy variants should agree."""
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    _, _, util_seq = greedy_myopic_simulate(p, u_vec, B=2, G=2, z_mask=0)
    _, _, util_cnt = greedy_myopic_counting_simulate(p, u_vec, B=2, G=2, z_mask=0)
    # Both should find the same utility when nobody is infected
    assert abs(util_seq - util_cnt) < 1e-10


def test_counting_greedy_expected_utility():
    """Counting greedy EU should be between U_single and U_max."""
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    u_s, _ = u_single(p, u_vec, B=2)
    u_cnt = greedy_myopic_counting_expected_utility(p, u_vec, B=2, G=2)
    u_m = u_max(p, u_vec)
    assert u_s <= u_cnt + 1e-10, f"U_single={u_s} > U_counting={u_cnt}"
    assert u_cnt <= u_m + 1e-10, f"U_counting={u_cnt} > U_max={u_m}"


def test_counting_vs_sequential_greedy_eu():
    """Sequential and counting greedy EUs should be very close for independent priors."""
    p = [0.1, 0.2, 0.15]
    u_vec = [5.0, 3.0, 4.0]
    eu_seq = greedy_myopic_expected_utility(p, u_vec, B=2, G=2)
    eu_cnt = greedy_myopic_counting_expected_utility(p, u_vec, B=2, G=2)
    # For independent priors, both approaches should give very similar results
    assert abs(eu_seq - eu_cnt) < 0.1, \
        f"Sequential EU={eu_seq:.4f} vs Counting EU={eu_cnt:.4f}"


# ===================================================================
# 22) estimate_p_from_history
# ===================================================================

def test_estimate_p_no_history():
    """With no history, estimate should return the prior."""
    est = estimate_p_from_history((), 3, prior_p=[0.1, 0.2, 0.3])
    assert abs(est[0] - 0.1) < 1e-10
    assert abs(est[1] - 0.2) < 1e-10
    assert abs(est[2] - 0.3) < 1e-10


def test_estimate_p_with_history():
    """With history, estimate should reflect the observations."""
    history = (
        (mask_from_indices([0, 1]), 0),  # both healthy
    )
    est = estimate_p_from_history(history, 3, prior_p=[0.3, 0.3, 0.3])
    assert est[0] < 0.05  # should be near 0 (proven healthy)
    assert est[1] < 0.05
    assert abs(est[2] - 0.3) < 1e-10  # unchanged


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
