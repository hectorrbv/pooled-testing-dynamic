"""FIXED helpers shared by the judge and the editable scorer (loop Q1).

Everything is exact: profile weights are integers, probabilities Fractions.
A profile is an n-bit integer (bit i set = person i infected). A *support* is a
2^n-bit integer whose set bits are the profiles still compatible with the
observed counts. A *pool* is an n-bit mask of people.
"""
from fractions import Fraction as F
from itertools import combinations
from math import lcm, prod


def members(mask):
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)


def pool_mask(people):
    return sum(1 << int(i) for i in set(people))


def laminar_compatible(a, b):
    """Disjoint, or one contains the other."""
    c = a & b
    return not c or c == a or c == b


class Instance:
    """Precomputation for one cell: weights, pools, outcome supports, compatibilities."""

    def __init__(self, q, u, G):
        self.q = tuple(F(x) for x in q)          # healthy probabilities
        self.u = tuple(F(x) for x in u)
        self.n, self.G = len(self.q), G
        self.scale = lcm(*(x.denominator for x in self.u))
        self.ui = tuple(int(x * self.scale) for x in self.u)
        # weight of profile z: product over people of P(state); integer with common denominator
        self.weights = tuple(prod(
            x.denominator - x.numerator if z >> i & 1 else x.numerator
            for i, x in enumerate(self.q)) for z in range(1 << self.n))
        self.root = sum(1 << z for z, w in enumerate(self.weights) if w)
        self.pools = tuple(pool_mask(S) for k in range(1, G + 1)
                           for S in combinations(range(self.n), k))
        self.outcomes = {}
        for a in self.pools:
            masks = [0] * (bin(a).count('1') + 1)
            for z, w in enumerate(self.weights):
                if w:
                    masks[bin(z & a).count('1')] |= 1 << z
            self.outcomes[a] = tuple(masks)
        self._mass, self._healthy = {}, {}

    def mass(self, support):
        if support not in self._mass:
            total, rest = 0, support
            while rest:
                low = rest & -rest
                total += self.weights[low.bit_length() - 1]
                rest ^= low
            self._mass[support] = total
        return self._mass[support]

    def healthy(self, support):
        """People healthy in every compatible profile (posterior zero)."""
        if support not in self._healthy:
            infected, rest = 0, support
            while rest:
                low = rest & -rest
                infected |= low.bit_length() - 1
                rest ^= low
            self._healthy[support] = ((1 << self.n) - 1) & ~infected
        return self._healthy[support]

    def utility(self, people_mask):
        return F(sum(self.ui[i] for i in members(people_mask)), self.scale)

    def candidates(self, past):
        return tuple(a for a in self.pools if all(laminar_compatible(a, b) for b in past))


class State:
    """What the scorer sees at a decision node."""

    def __init__(self, inst, tests_left, support, cleared, past):
        self.inst = inst
        self.n, self.G = inst.n, inst.G
        self.q, self.u = inst.q, inst.u
        self.tests_left = tests_left
        self.support = support
        self.cleared = cleared
        self.past = past
        self.candidates = inst.candidates(past)


def branches(state, pool):
    """(probability, child support, newly accredited mask) for each possible count."""
    inst, out = state.inst, []
    m = inst.mass(state.support)
    for omask in inst.outcomes[pool]:
        child = state.support & omask
        if child:
            new = inst.healthy(child) & ~state.cleared
            out.append((F(inst.mass(child), m), child, new))
    return out


def expected_gain(state, pool):
    """Exact expected utility accredited by this single test."""
    return sum((p * state.inst.utility(new) for p, _, new in branches(state, pool)), F(0))
