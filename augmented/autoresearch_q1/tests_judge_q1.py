"""Tests of the fixed Q1 judge: battery, anchors, gates and the baseline scorer."""
from fractions import Fraction as F
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
judge = importlib.import_module('judge')
from q1_tools import Instance, State, expected_gain, pool_mask  # noqa: E402


def test_battery_shape():
    cells = judge.battery()
    assert len(cells) == 125
    assert all(1 <= B <= 4 and 2 <= G <= 5 and 3 <= n <= 7 for _, n, B, G, *_ in cells)


def test_anchors_pass():
    judge.anchors()


def test_expected_gain_matches_hand_count():
    inst = Instance([F(1, 2)] * 4, [F(1)] * 4, 3)
    state = State(inst, 2, inst.root, 0, ())
    assert expected_gain(state, pool_mask((0,))) == F(1, 2)
    assert expected_gain(state, pool_mask((0, 1))) == F(1, 2)          # both clear w.p. 1/4
    assert expected_gain(state, pool_mask((0, 1, 2))) == F(3, 8)       # all three clear w.p. 1/8


def test_candidates_are_laminar():
    inst = Instance([F(1, 2)] * 4, [F(1)] * 4, 3)
    past = (pool_mask((0, 1, 2)),)
    cands = inst.candidates(past)
    assert pool_mask((0, 1)) in cands and pool_mask((3,)) in cands
    assert pool_mask((2, 3)) not in cands


def test_policy_value_never_exceeds_optimum_on_small_cells():
    for label, n, B, G, q, u, opt in judge.battery()[:8]:
        value = judge.policy_value(judge.individuals, n, B, G, q, u)
        assert value <= opt, label


def test_crossing_is_rejected():
    def crossing(state):
        return pool_mask((0, 1)) if not state.past else pool_mask((1, 2))
    with pytest.raises(judge.Invalid):
        judge.policy_value(crossing, 4, 2, 2, [F(1, 2)] * 4, [F(1)] * 4)


def test_baseline_scorer_is_valid_and_bounded():
    scorer = importlib.import_module('scorer')
    label, n, B, G, q, u, opt = next(c for c in judge.battery() if c[0] == 'q05:6,3,3')
    value = judge.policy_value(scorer.choose, n, B, G, q, u)
    assert F(3, 2) <= value <= opt
