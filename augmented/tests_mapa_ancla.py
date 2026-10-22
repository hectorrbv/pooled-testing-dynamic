"""Audit the CBS expectation by enumerating labeled infection configurations."""
from fractions import Fraction as F
from itertools import product
import pytest
from augmented.mapa_ancla import cbs_stop_value


@pytest.mark.parametrize('G', [2, 3, 4])
@pytest.mark.parametrize('q', [F(1, 20), F(3, 10)])
def test_cbs_against_complete_realizations(G, q):
    k = 2
    B = k + (G-1).bit_length()
    expected = F(0)
    for world in product([0, 1], repeat=k*G):
        probability = (1-q)**sum(world) * q**(len(world)-sum(world))
        reward, tests = 0, 0
        for start in range(0, len(world), G):
            block = world[start:start+G]
            tests += 1
            if all(block):
                continue
            if not any(block):
                reward = len(block)
                break
            while True:
                half = len(block)//2
                left, right = block[:half], block[half:]
                tests += 1
                reward = (len(left) if not any(left) else 0) + (len(right) if not any(right) else 0)
                if reward:
                    break
                block = left if not all(left) else right
            break
        assert tests <= B
        expected += probability*reward
    assert cbs_stop_value(q, G, B) == expected
    assert expected >= 1-(1-q)**(k*G)


def test_cbs_bound_is_not_policy_value():
    assert cbs_stop_value(F(3, 10), 2, 2) == F(3, 5)
    assert 1-F(7, 10)**2 == F(51, 100)
