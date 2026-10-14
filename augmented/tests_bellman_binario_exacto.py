"""Oracle checks against a separate float DP and analytic boundary cases."""
from fractions import Fraction as F
import random
import pytest

from augmented.bellman_binario_exacto import BellmanBinario, replay_binary_tree
from augmented.classical_solver import solve_classical_dynamic


@pytest.mark.parametrize('seed',range(12))
def test_matches_independent_classical_dp_and_raw_profile_replay(seed):
    rng=random.Random(seed)
    n=rng.choice([3,4]); B=rng.choice([1,2,3]); G=rng.randint(1,n)
    q=[F(rng.randint(1,9),10) for _ in range(n)]
    u=[F(rng.randint(1,10),2) for _ in range(n)]
    s=BellmanBinario([1-x for x in q],u,G)
    value=s.solve(B).value
    old,_=solve_classical_dynamic([float(1-x) for x in q],list(map(float,u)),B,G)
    assert float(value)==pytest.approx(old,abs=1e-11)
    assert F(replay_binary_tree(s.tree(B),q,u,B,G)['value'])==value


@pytest.mark.parametrize('q',['.1','.3','.5'])
def test_low_health_regime_equals_best_individuals(q):
    q=F(q); u=[1,2,3,4]
    s=BellmanBinario([1-q]*4,u,4)
    assert s.solve(2).value==7*q


def test_one_test_all_tested_and_explicit_hard_clearing():
    s=BellmanBinario([F(1,10)]*3,[1]*3,3)
    assert s.solve(1).value==3*F(9,10)**3
    assert s.solve(3).value==F(27,10)
    certain=BellmanBinario([0,1],[5,10],2)
    assert certain.solve(0).value==0  # hard clearing needs a negative test
    assert certain.solve(1).value==5
    assert replay_binary_tree(certain.tree(1),[1,0],[5,10],1,2)['value']=='5'


def test_binary_beats_soft_laminar_with_three_people():
    from augmented.experimento_sesion_29 import laminar_two_tests
    q=list(map(F,['.95','.8','.8']))
    binary=BellmanBinario([1-x for x in q],[1]*3,2)
    laminar=laminar_two_tests(q,[F(1)]*3,2)
    assert F(laminar['value'])==F(1183,500)
    assert binary.solve(2).value==F(304,125)>F(laminar['value'])
    replay=replay_binary_tree(binary.tree(2),q,[1]*3,2,2)
    assert replay['profiles']==8 and replay['profiles_with_nonlaminar_pools']>0


def test_state_cap_does_not_return_uncertified_value():
    with pytest.raises(RuntimeError,match='no optimum certified'):
        BellmanBinario([F(1,2)]*4,[1]*4,3,state_limit=1).solve(2)
