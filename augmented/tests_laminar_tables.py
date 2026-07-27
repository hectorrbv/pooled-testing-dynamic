"""Tests for the subset-count tables and their reuse after a split.

Run with::

    PYTHONPATH=. python augmented/tests_laminar_tables.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask, mask_from_indices
from augmented.laminar_inference import laminar_forest_marginals
from augmented.laminar_tables import (
    absolute_mask,
    conditional_subset_table,
    local_index,
    restrict_cache,
    split_subset_tables,
    subset_pmf_cache,
    table_row,
)


def _assert_raises(fragment, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as exc:
        assert fragment in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError(f"expected ValueError containing {fragment!r}")


def _enumerated_table(p, pool_mask, observed, subset_mask):
    """``P(R(subset)=r | R(pool)=observed)`` by enumerating every world."""

    n = len(p)
    subset_size = subset_mask.bit_count()
    weights = np.zeros(subset_size + 1, dtype=float)
    for world in range(1 << n):
        if (world & pool_mask).bit_count() != observed:
            continue
        probability = 1.0
        for index in range(n):
            probability *= p[index] if world & (1 << index) else 1.0 - p[index]
        weights[(world & subset_mask).bit_count()] += probability
    total = weights.sum()
    assert total > 0.0, "the conditioning event must have positive mass"
    return weights / total


def test_conditional_table_matches_world_enumeration():
    rng = np.random.default_rng(20260727)
    for _ in range(6):
        n = int(rng.integers(4, 8))
        p = rng.uniform(0.1, 0.9, size=n)
        pool = int(mask_from_indices(
            rng.choice(n, size=int(rng.integers(3, n + 1)), replace=False)
        ))
        pool_size = pool.bit_count()
        cache = subset_pmf_cache(p, pool)
        for observed in range(pool_size + 1):
            table = conditional_subset_table(cache, observed)
            for index in range(1 << pool_size):
                subset = absolute_mask(cache, index)
                expected = _enumerated_table(p, pool, observed, subset)
                got = table[index, : subset.bit_count() + 1]
                assert np.allclose(got, expected, atol=1e-12), (
                    f"subset {subset:b} under count {observed}"
                )


def test_every_table_row_is_a_distribution():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, size=9)
    pool = mask_from_indices([0, 2, 3, 5, 7, 8])
    cache = subset_pmf_cache(p, pool)
    for observed in range(pool.bit_count() + 1):
        table = conditional_subset_table(cache, observed)
        assert np.allclose(table.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(table >= -1e-15)
        full = local_index(cache, pool)
        assert abs(table[full, observed] - 1.0) < 1e-12


def test_split_tables_equal_tables_built_from_scratch():
    rng = np.random.default_rng(4242)
    for _ in range(8):
        n = int(rng.integers(6, 10))
        p = rng.uniform(0.1, 0.9, size=n)
        pool = int(mask_from_indices(
            rng.choice(n, size=int(rng.integers(4, min(n, 7) + 1)), replace=False)
        ))
        members = indices_from_mask(pool, n)
        cache = subset_pmf_cache(p, pool)

        chosen = rng.choice(
            members, size=int(rng.integers(1, len(members))), replace=False
        )
        tested = int(mask_from_indices(chosen))
        residual = pool & ~tested
        for pool_count in range(pool.bit_count() + 1):
            for tested_count in range(tested.bit_count() + 1):
                residual_count = pool_count - tested_count
                if not 0 <= residual_count <= residual.bit_count():
                    continue
                tested_atom, residual_atom = split_subset_tables(
                    cache, tested, tested_count, pool_count
                )
                assert tested_atom.cache.convolutions == 0
                assert residual_atom.cache.convolutions == 0

                fresh_tested = conditional_subset_table(
                    subset_pmf_cache(p, tested), tested_count
                )
                fresh_residual = conditional_subset_table(
                    subset_pmf_cache(p, residual), residual_count
                )
                assert np.allclose(tested_atom.table, fresh_tested, atol=1e-12)
                assert np.allclose(
                    residual_atom.table, fresh_residual, atol=1e-12
                )


def test_split_tables_match_enumeration_of_the_full_history():
    """The sibling's count is irrelevant inside an atom, as Lemma A claims."""

    rng = np.random.default_rng(777)
    n = 8
    p = rng.uniform(0.15, 0.85, size=n)
    pool = mask_from_indices([0, 1, 2, 3, 4, 5])
    tested = mask_from_indices([0, 1, 2])
    residual = pool & ~tested
    cache = subset_pmf_cache(p, pool)

    for pool_count in range(1, pool.bit_count()):
        for tested_count in range(tested.bit_count() + 1):
            if not 0 <= pool_count - tested_count <= residual.bit_count():
                continue
            tested_atom, _ = split_subset_tables(
                cache, tested, tested_count, pool_count
            )
            for index in range(1 << tested.bit_count()):
                subset = absolute_mask(tested_atom.cache, index)
                if not subset:
                    continue
                weights = np.zeros(subset.bit_count() + 1, dtype=float)
                for world in range(1 << n):
                    if (world & pool).bit_count() != pool_count:
                        continue
                    if (world & tested).bit_count() != tested_count:
                        continue
                    probability = 1.0
                    for position in range(n):
                        probability *= (
                            p[position] if world & (1 << position)
                            else 1.0 - p[position]
                        )
                    weights[(world & subset).bit_count()] += probability
                if weights.sum() <= 0.0:
                    continue
                expected = weights / weights.sum()
                got = table_row(tested_atom.cache, tested_atom.table, subset)
                assert np.allclose(got, expected, atol=1e-12)


def test_singleton_rows_match_the_laminar_forest_marginals():
    rng = np.random.default_rng(31337)
    n = 7
    p = rng.uniform(0.1, 0.9, size=n)
    pool = mask_from_indices([0, 1, 2, 3, 4])
    tested = mask_from_indices([0, 1])
    cache = subset_pmf_cache(p, pool)
    pool_count, tested_count = 3, 1

    tested_atom, residual_atom = split_subset_tables(
        cache, tested, tested_count, pool_count
    )
    history = ((pool, pool_count), (tested, tested_count))
    hierarchy = {pool: (tested,), tested: ()}
    posterior, _ = laminar_forest_marginals(p, history, hierarchy)

    for atom in (tested_atom, residual_atom):
        for member in atom.cache.members:
            row = table_row(atom.cache, atom.table, 1 << member)
            assert abs(row[1] - posterior[member]) < 1e-12


def test_cache_cost_is_one_convolution_per_non_empty_subset():
    p = np.full(10, 0.3)
    pool = (1 << 10) - 1
    cache = subset_pmf_cache(p, pool)
    assert cache.convolutions == (1 << 10) - 1
    assert restrict_cache(cache, mask_from_indices([0, 1, 2])).convolutions == 0


def test_tables_reject_degenerate_requests():
    p = np.array([0.0, 0.0, 0.5, 0.5])
    cache = subset_pmf_cache(p, 0b1111)
    _assert_raises("zero probability", conditional_subset_table, cache, 4)
    _assert_raises("outside the pool size", conditional_subset_table, cache, 5)

    healthy = subset_pmf_cache(np.full(4, 0.4), 0b1111)
    _assert_raises(
        "strict subset", split_subset_tables, healthy, 0b1111, 2, 2
    )
    _assert_raises(
        "non-empty subset", split_subset_tables, healthy, 0, 0, 2
    )
    _assert_raises(
        "not attainable", split_subset_tables, healthy, 0b0011, 0, 3
    )
    _assert_raises(
        "contained in the cached pool", restrict_cache, healthy, 0b10000
    )
    _assert_raises("empty or outside", subset_pmf_cache, np.full(4, 0.4), 0)


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
