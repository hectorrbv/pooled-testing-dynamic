"""Tests of the fixed log-G judge: anchors, gates and the baseline design."""
from fractions import Fraction as F
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
judge = importlib.import_module('judge')


def test_anchors_pass():
    judge.anchors()


def test_individuals_and_pairs():
    assert judge.evaluate(judge.individuals, 16, 4, 2)[0] == 2
    assert judge.evaluate(judge.static([(0, 1), (2, 3)]), 4, 2, 2)[0] == 1


def test_adaptive_policy_is_evaluated_per_history():
    # Test the pair (0,1); if clean, test (2,3); otherwise test person 0 alone.
    def factory(n, B, G):
        def next_pool(history):
            if not history:
                return (0, 1)
            if len(history) == 1:
                return (2, 3) if history[0][1] == 0 else (0,)
            return None
        return next_pool
    # Branch count 0 (prob 1/4): 2 cleared + pair (2,3) clean with prob 1/4 -> +2.
    # Branch count 1 (prob 1/2): test 0 -> if healthy (prob 1/2) clears 0 and proves 1 infected: +1;
    #   if infected, 1 is healthy: +1. Branch count 2: test 0 reveals nothing new: 0.
    expected = F(1, 4) * (2 + F(1, 4) * 2) + F(1, 2) * 1 + 0
    assert judge.evaluate(factory, 4, 2, 2)[0] == expected


def test_detector_blocks_match_hand_count():
    # Two detector blocks of four with six tests clear four people in expectation.
    pools = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (4, 5, 6), (4, 5, 7), (4, 6, 7)]
    assert judge.evaluate(judge.static(pools), 8, 6, 3)[0] == 4


def test_gates():
    with pytest.raises(judge.Invalid):
        judge.evaluate(judge.static([(0, 0)]), 4, 1, 2)
    with pytest.raises(judge.Invalid):
        judge.evaluate(judge.static([(0, 9)]), 4, 1, 2)
    with pytest.raises(judge.Invalid):
        judge.evaluate(judge.static([(0, 1, 2)]), 4, 1, 2)
    with pytest.raises(judge.Invalid):
        judge.evaluate(judge.static([()]), 4, 1, 2)


def test_baseline_design_values():
    design = importlib.import_module('design')
    assert judge.evaluate(design.policy, 16, 4, 2)[0] == 2            # four individuals
    assert judge.evaluate(design.policy, 16, 4, 4)[0] == F(5, 2)      # detector + one individual
    assert judge.evaluate(design.policy, 16, 8, 8)[0] == 5            # two detectors + two individuals
