from fractions import Fraction as F
import pytest
from augmented.laminar_benchmarks import ExactPolicyEvaluator, balanced_laminar_library
from augmented.malla_precio_laminar_q import measure
from augmented.general_homogeneo import solve_general
from augmented.bellman_perfiles import BellmanPerfiles


def test_soft_clearing_pays_for_a_healthy_complement_after_positive_singleton():
    for clearing, expected in [("hard",0),("soft",1)]:
        evaluator=ExactPolicyEvaluator([.7]*3,[1]*3,2,2,clearing=clearing)
        mixed=evaluator.all_worlds & evaluator.outcome_worlds[3][1]
        positive=next(branch for branch in evaluator.branches(mixed,0,1)
                      if branch[1] & evaluator.outcome_worlds[1][1])
        assert positive[3] == expected


def test_rollout_recovers_the_known_pair_gain_only_under_soft_clearing():
    for clearing, expected in [("hard",.6),("soft",.774)]:
        evaluator=ExactPolicyEvaluator([.7]*3,[1]*3,2,2,clearing=clearing)
        library=balanced_laminar_library(evaluator.p,evaluator.u,2)
        greedy,rollout=evaluator.greedy_and_rollout_values(library)
        assert greedy == pytest.approx(.6)
        assert rollout == pytest.approx(expected)
        assert evaluator.optimal_value(library) == pytest.approx(expected)


def test_soft_clearing_counts_prior_certainty_once_even_without_tests():
    for B, expected in [(0,2),(1,2.5),(2,2.5)]:
        evaluator=ExactPolicyEvaluator([0,.5],[2,1],B,2,clearing="soft")
        library=balanced_laminar_library(evaluator.p,evaluator.u,2)
        assert evaluator.greedy_and_rollout_values(library) == pytest.approx((expected,expected))
        assert evaluator.optimal_value(library) == pytest.approx(expected)


def test_grid_checks_different_representations_and_full_profile_replays():
    row,audit=measure([F(1,2)]*4,[1]*4,3,3,verify=True)
    assert F(row["exact_laminar"]) == F(7,4)
    assert F(row["unrestricted"]) == 2
    assert row["information_check"] == "pass"
    assert audit["checks"]["independent_laminar"] == "7/4"
    assert audit["checks"]["greedy"]["profiles"] == 16
    assert audit["checks"]["rollout"]["profiles"] == 16


@pytest.mark.parametrize("q,u,B,G",[
    ([F(3,10)]*3,[1]*3,2,2),
    ([F(1,2)]*4,[1]*4,3,3),
    ([F(7,10)]*5,[1]*5,3,3),
    ([F(1,2),F(1,3),F(1,4)],[1,3,2],2,2),
])
def test_root_symmetry_matches_full_first_action_enumeration(q,u,B,G):
    value,_,method=solve_general(q,u,B,G)
    full=BellmanPerfiles([1-x for x in q],u,G).solve(B).value
    assert value==full
    assert (method=="root_cardinality_symmetry") == (len(set(q))==1 and len(set(u))==1)
