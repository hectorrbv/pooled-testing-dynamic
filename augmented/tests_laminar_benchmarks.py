"""Tests for the exact atlas quantities and atom-preserving predictions."""

import numpy as np

from augmented.bayesian import exact_pool_pmf
from augmented.core import mask_from_indices
from augmented.laminar_benchmarks import (
    ExactPolicyEvaluator,
    balanced_laminar_library,
    four_quantities,
    maximal_laminar_libraries,
)
from augmented.laminar_inference import (
    laminar_forest_marginals,
    laminar_pool_pmf,
)
from augmented.static_solver import solve_static_overlapping


def test_maximal_library_counts_small_grid():
    expected = {
        (4, 2): 3,
        (4, 3): 15,
        (5, 2): 15,
        (5, 3): 30,
        (6, 2): 15,
        (6, 3): 105,
    }
    assert {
        key: len(maximal_laminar_libraries(*key)) for key in expected
    } == expected


def test_four_quantities_have_expected_order_and_static_identity():
    p = [0.18, 0.37, 0.51, 0.24, 0.66]
    u = [1.0, 2.2, 0.7, 3.1, 1.4]
    values = four_quantities(p, u, 3, 3)
    reference_static, _ = solve_static_overlapping(p, u, 3, 3)
    assert abs(values["V_static_binary"] - reference_static) < 1e-10
    assert values["V_greedy_laminar"] <= values["V_rollout_laminar"] + 1e-10
    assert values["V_greedy_laminar"] <= values["V_laminar"] + 1e-10
    assert values["V_laminar"] <= values["V_star"] + 1e-10
    assert values["V_static_binary"] <= values["V_star"] + 1e-10


def test_b_one_laminar_class_contains_an_unrestricted_optimum():
    p = [0.13, 0.29, 0.61, 0.42, 0.74, 0.20]
    u = [0.9, 1.7, 2.1, 0.4, 3.2, 1.1]
    values = four_quantities(p, u, 1, 3)
    assert abs(values["V_laminar"] - values["V_star"]) < 1e-10


def test_atom_pmf_matches_world_enumeration_for_any_next_pool():
    p = np.array([0.12, 0.31, 0.46, 0.68, 0.23, 0.57, 0.39, 0.76])
    root = mask_from_indices([0, 1, 2, 3])
    child = mask_from_indices([0, 1])
    other = mask_from_indices([4, 5])
    history = ((root, 2), (child, 1), (other, 1))
    hierarchy = {root: (child,), child: (), other: ()}
    _, atoms = laminar_forest_marginals(p, history, hierarchy)
    candidates = (
        mask_from_indices([6, 7]),       # disjoint
        root,                            # observed node
        mask_from_indices([2, 3]),       # compatible descendant
        mask_from_indices([1, 2]),       # non-laminar crossing
        mask_from_indices([0, 4, 7]),    # intersects two atoms + outside
    )
    for pool in candidates:
        exact = exact_pool_pmf(p, history, pool, len(p))
        atom = laminar_pool_pmf(p, atoms, pool)
        np.testing.assert_allclose(atom, exact, atol=2e-12, rtol=2e-12)


def test_rollout_dominates_greedy_in_practical_library():
    p = [0.19, 0.33, 0.47, 0.58, 0.71]
    u = [1.0, 0.8, 2.4, 1.6, 0.5]
    evaluator = ExactPolicyEvaluator(p, u, 3, 3)
    library = balanced_laminar_library(p, u, 3)
    greedy, rollout = evaluator.greedy_and_rollout_values(library)
    optimum = evaluator.optimal_value(library)
    assert greedy <= rollout + 1e-10 <= optimum + 1e-10
