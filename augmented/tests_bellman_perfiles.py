"""Independent reference checks for general/laminar posterior-zero Bellman."""

from fractions import Fraction as F
from itertools import combinations, product
import random

import pytest

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.densidad_companion import PoliticasDensidad


@pytest.mark.parametrize('n,q,B,G,laminar,general', [
    (2, '0.3', 2, 2, '3/5', '3/5'),
    (4, '0.3', 2, 2, '387/500', '387/500'),
    (4, '0.3', 3, 2, '537/500', '1137/1000'),
    (4, '0.5', 3, 3, '7/4', '2'),
    (6, '0.3', 3, 3, '62469/50000', '35487/25000'),
])
def test_exact_examples_and_profile_replay(n, q, B, G, laminar, general):
    p, u = [1-F(q)] * n, [1] * n
    for restricted, expected in [(True, laminar), (False, general)]:
        s = BellmanPerfiles(p, u, G, laminar=restricted)
        assert s.solve(B).value == F(expected)
        report = s.replay(B)
        assert report['max_tests_used'] <= B
        if restricted:
            assert report['profiles_with_crossing'] == 0
    a = SolverLaminar(dict(enumerate(p)), dict(enumerate(u)), G, 'posterior_zero')
    assert a.V(frozenset(range(n)), (), B) == F(laminar)


def test_crossing_advantage_is_localized_after_count_one():
    history = [((0, 1), 1)]
    p, u = [F(7, 10)] * 4, [1] * 4
    l = BellmanPerfiles(p, u, 2, laminar=True)
    g = BellmanPerfiles(p, u, 2)
    assert l.solve(1, history).value == g.solve(1, history).value == 1
    assert l.solve(2, history).value == F(13, 10)
    assert g.solve(2, history).value == F(29, 20)
    assert g.solve(2, history).first_pool == (0, 2)
    assert F(42, 100) * (g.solve(2, history).value - l.solve(2, history).value) == F(63, 1000)


def test_three_crossing_triples_identify_all_four_people():
    """Direct certificate independent of Bellman, for every binary profile."""
    signatures = set()
    for a, b, c, d in product((0, 1), repeat=4):
        x, y, z = a+b+c, a+b+d, a+c+d
        recovered_a = (x+y+z) % 2
        recovered = (recovered_a, (x+y-z-recovered_a)//2,
                     (x+z-y-recovered_a)//2, (y+z-x-recovered_a)//2)
        assert recovered == (a, b, c, d)
        signatures.add((x, y, z))
    assert len(signatures) == 16


def test_heterogeneous_six_person_counterexample():
    p = list(map(F, ['.9', '.825', '.875', '.8', '.95', '.85']))
    u = [2, 1, 1, 1, 4, 2]
    for lam, value in [(True, F(681289, 640000)), (False, F(710697, 640000))]:
        s = BellmanPerfiles(p, u, 4, laminar=lam)
        assert s.solve(3).value == value
        assert s.solve(3).first_pool == (0, 4, 5)
        s.replay(3)


@pytest.mark.parametrize('seed', range(12))
def test_laminar_history_constraint_matches_independent_atom_solver(seed):
    rng = random.Random(seed)
    n = rng.choice([3, 4, 5])
    B, G = rng.randint(1, 3), rng.randint(1, min(3, n))
    p = [F(rng.randint(1, 9), 10) for _ in range(n)]
    u = [F(rng.randint(0, 8), 2) for _ in range(n)]
    l = BellmanPerfiles(p, u, G, laminar=True)
    g = BellmanPerfiles(p, u, G)
    a = SolverLaminar(dict(enumerate(p)), dict(enumerate(u)), G, 'posterior_zero')
    lv, gv = l.solve(B).value, g.solve(B).value
    assert lv == a.V(frozenset(range(n)), (), B)
    assert lv <= gv <= sum((1-pi)*ui for pi, ui in zip(p, u))
    assert g.solve(B+1).value >= gv
    l.replay(B)
    g.replay(B)


@pytest.mark.parametrize('q', ['.2', '.5', '.8'])
def test_general_agrees_with_existing_homogeneous_joint_count_solver(q):
    from augmented.experiments_separacion_n10 import dynamic_value
    s = BellmanPerfiles([1-F(q)] * 4, [1] * 4, 4)
    for B in [1, 2, 3]:
        assert float(s.solve(B).value) == pytest.approx(dynamic_value(4, B, True, float(q)), abs=1e-10)


def test_zero_probability_profiles_and_credit_not_paid_twice():
    s = BellmanPerfiles([0, 1], [2, 3], 2)
    assert s.solve(0).value == s.solve(2).value == 2
    assert s.solve(2).additional_value == 0
    assert s.solve(2).first_pool is None
    s = BellmanPerfiles(['.5', '.5'], [2, 3], 2)
    assert s.solve(1, [((0,), 0)]).value == F(7, 2)
    assert s.solve(1, [((0,), 0)]).additional_value == F(3, 2)
    with pytest.raises(ValueError):
        s.solve(1, [((0,), 0), ((0,), 1)])
    l = BellmanPerfiles(['.5']*3, [1]*3, 2, laminar=True)
    with pytest.raises(ValueError):
        l.solve(1, [((0, 1), 1), ((1, 2), 1)])


def test_immediate_score_complement_is_zero_not_subpool_fully_infected():
    d = PoliticasDensidad(dict.fromkeys(range(3), F(1, 2)), dict.fromkeys(range(3), 1), 3)
    assert d._score_inmediato(('ref', ((0, 1, 2), 2), (0,))) == pytest.approx(1/3)


@pytest.mark.parametrize('r', [1, 2, 3])
def test_immediate_refinement_scores_against_direct_profile_enumeration(r):
    p = {0:F(1, 5), 1:F(2, 5), 2:F(3, 5), 3:F(4, 5)}
    u = {0:1, 1:2, 2:3, 3:4}
    d = PoliticasDensidad(p, u, 4)
    profiles = []
    for z in product((0, 1), repeat=4):
        if sum(z) == r:
            w = F(1)
            for i in range(4):
                w *= p[i] if z[i] else 1-p[i]
            profiles.append((z, w))
    mass = sum(w for _, w in profiles)
    for size in [1, 2, 3]:
        for S in combinations(range(4), size):
            rest = tuple(i for i in range(4) if i not in S)
            expected = sum(w * ((sum(u[i] for i in S) if not sum(z[i] for i in S) else 0)
                                 + (sum(u[i] for i in rest) if not sum(z[i] for i in rest) else 0))
                           for z, w in profiles) / mass
            assert d._score_inmediato(('ref', ((0, 1, 2, 3), r), S)) == pytest.approx(float(expected))
