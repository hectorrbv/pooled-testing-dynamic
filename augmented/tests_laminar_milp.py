"""Tests for the reusable modules extracted from notebook 22.

Run with::

    PYTHONPATH=. python augmented/tests_laminar_milp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.bayesian import bayesian_update_by_counting
from augmented.core import mask_from_indices, test_result as observed_count
from augmented.laminar_inference import (
    conditional_bernoulli_marginals,
    laminar_atoms,
    laminar_forest_marginals,
)
from augmented.scenario_milp import (
    brute_best_pool_scenarios,
    condition_on_count,
    exact_prior_scenarios,
    milp_best_pool_scenarios,
    score_pool_scenarios,
)


def _assert_raises(fragment, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as exc:
        assert fragment in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def _random_branched_history(rng, n, latent_state):
    """A random forest with a root that has two children and optional depth."""

    order = rng.permutation(n).tolist()
    covered_size = int(rng.integers(max(3, n - 2), n + 1))
    covered = order[:covered_size]
    first_size = int(rng.integers(1, covered_size - 1))
    second_size = int(rng.integers(1, covered_size - first_size + 1))

    root = mask_from_indices(covered)
    first_members = covered[:first_size]
    second_members = covered[first_size:first_size + second_size]
    first = mask_from_indices(first_members)
    second = mask_from_indices(second_members)

    hierarchy = {root: (first, second), first: (), second: ()}
    if len(first_members) >= 3:
        grandchildren = (
            mask_from_indices(first_members[:1]),
            mask_from_indices(first_members[1:2]),
        )
        hierarchy[first] = grandchildren
        hierarchy[grandchildren[0]] = ()
        hierarchy[grandchildren[1]] = ()
    if len(second_members) >= 4:
        cut = len(second_members) // 2
        grandchildren = (
            mask_from_indices(second_members[:cut]),
            mask_from_indices(second_members[cut:]),
        )
        hierarchy[second] = grandchildren
        hierarchy[grandchildren[0]] = ()
        hierarchy[grandchildren[1]] = ()

    pools = list(hierarchy)
    rng.shuffle(pools)
    history = tuple((pool, observed_count(pool, latent_state)) for pool in pools)
    return history, hierarchy


def test_laminar_marginals_match_counting_on_random_branched_families():
    rng = np.random.default_rng(2201)
    for n in (4, 6, 8, 10, 12):
        for _ in range(5):
            p = rng.uniform(0.08, 0.92, size=n)
            latent_bits = rng.random(n) < p
            latent_state = mask_from_indices(np.flatnonzero(latent_bits).tolist())
            history, hierarchy = _random_branched_history(rng, n, latent_state)

            exact = np.asarray(
                bayesian_update_by_counting(p.tolist(), history, n)
            )
            laminar, atoms = laminar_forest_marginals(p, history, hierarchy)

            assert np.max(np.abs(exact - laminar)) < 1e-10
            atom_union = 0
            for atom in atoms:
                assert atom_union & atom.mask == 0
                atom_union |= atom.mask
                assert abs(laminar[
                    [i for i in range(n) if (atom.mask >> i) & 1]
                ].sum() - atom.count) < 1e-9


def test_empty_laminar_history_leaves_the_prior_unchanged():
    p = np.array([0.1, 0.4, 0.9])
    posterior, atoms = laminar_forest_marginals(p, (), {})
    assert np.array_equal(posterior, p)
    assert atoms == ()


def test_laminar_history_rejects_crossing_pools():
    first = mask_from_indices([0, 1])
    second = mask_from_indices([1, 2])
    history = ((first, 1), (second, 1))
    hierarchy = {first: (), second: ()}
    _assert_raises("not laminar", laminar_atoms, history, hierarchy, 3)


def test_laminar_history_rejects_duplicate_incompatible_counts():
    pool = mask_from_indices([0, 1, 2])
    history = ((pool, 1), (pool, 2))
    hierarchy = {pool: ()}
    _assert_raises("incompatible counts", laminar_atoms, history, hierarchy, 3)


def test_laminar_history_rejects_count_outside_pool_size():
    pool = mask_from_indices([0, 1])
    _assert_raises(
        "outside the pool size",
        laminar_atoms,
        ((pool, 3),),
        {pool: ()},
        2,
    )


def test_laminar_history_rejects_incompatible_parent_child_counts():
    root = mask_from_indices([0, 1, 2, 3])
    child = mask_from_indices([0, 1])
    hierarchy = {root: (child,), child: ()}

    _assert_raises(
        "parent and child counts",
        laminar_atoms,
        ((root, 0), (child, 1)),
        hierarchy,
        4,
    )
    _assert_raises(
        "parent and child counts",
        laminar_atoms,
        ((root, 4), (child, 1)),
        hierarchy,
        4,
    )


def test_laminar_history_rejects_an_incorrect_supplied_hierarchy():
    root = mask_from_indices([0, 1, 2, 3])
    middle = mask_from_indices([0, 1, 2])
    leaf = mask_from_indices([0])
    history = ((root, 2), (middle, 1), (leaf, 0))
    hierarchy_that_skips_middle = {root: (middle, leaf), middle: (), leaf: ()}
    _assert_raises(
        "does not match",
        laminar_atoms,
        history,
        hierarchy_that_skips_middle,
        4,
    )


def test_conditional_marginals_reject_zero_mass_count():
    _assert_raises(
        "zero probability",
        conditional_bernoulli_marginals,
        [0.0, 0.4, 1.0],
        0,
    )


def test_scenario_milp_matches_brute_force_on_exact_priors_through_n10():
    rng = np.random.default_rng(2202)
    for n in (2, 3, 5, 7, 10):
        p = rng.uniform(0.05, 0.8, size=n)
        u = rng.uniform(0.2, 5.0, size=n)
        G = min(n, 1 + n // 2)
        cleared = mask_from_indices([0]) if n % 2 else 0
        scenarios, weights = exact_prior_scenarios(p)

        brute_value, brute_pool = brute_best_pool_scenarios(
            scenarios, weights, u, G, cleared
        )
        milp_value, milp_pool, result = milp_best_pool_scenarios(
            scenarios, weights, u, G, cleared, time_limit=30
        )

        assert result.success
        assert 1 <= milp_pool.bit_count() <= G
        assert abs(milp_value - brute_value) < 1e-8, (
            f"n={n}: MILP {milp_value:.12f} on {milp_pool:b} != "
            f"brute force {brute_value:.12f} on {brute_pool:b}"
        )


def test_scenario_conditioning_preserves_correlated_posterior_score():
    p = [0.2, 0.35, 0.55, 0.15]
    u = [1.0, 2.5, 1.2, 3.0]
    scenarios, weights = exact_prior_scenarios(p)
    observed_pool = mask_from_indices([0, 1, 2])
    posterior_scenarios, posterior_weights = condition_on_count(
        scenarios, weights, observed_pool, 1
    )

    assert abs(posterior_weights.sum() - 1.0) < 1e-12
    assert np.all(posterior_scenarios[:, [0, 1, 2]].sum(axis=1) == 1)
    assert score_pool_scenarios(
        observed_pool, posterior_scenarios, posterior_weights, u
    ) == 0.0


def test_scenario_conditioning_rejects_unsupported_count():
    scenarios = np.zeros((2, 3), dtype=np.int8)
    weights = np.array([0.25, 0.75])
    pool = mask_from_indices([0, 1])
    _assert_raises(
        "no support", condition_on_count, scenarios, weights, pool, 1
    )


def _run_all():
    import traceback

    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
        and getattr(value, "__module__", None) == __name__
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
