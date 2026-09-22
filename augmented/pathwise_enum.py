"""Independent pathwise-laminar oracle for BM17, under both clearing rules.

The profile engine does not import the atom solver or its posterior formulas.
Positive-prior infection profiles encode the whole history. Every future pool
must be disjoint from, contained in, or contain each earlier pool.

``value(B, history)`` returns ADDITIONAL utility after the supplied history;
B is the remaining hard budget. History uses (tuple_of_people, infection_count).
"""

from fractions import Fraction

from augmented.bellman_perfiles import BellmanPerfiles, members


class PathwiseEnumerator(BellmanPerfiles):
    def __init__(self, p, u, G, convencion='posterior_zero'):
        if convencion not in ('strict', 'posterior_zero'):
            raise ValueError('Unknown clearing convention')
        if isinstance(p, dict):
            if sorted(p) != list(range(len(p))):
                raise ValueError('Individuals must be indexed from zero')
            p = [p[i] for i in range(len(p))]
        if isinstance(u, dict):
            u = [u[i] for i in range(len(u))]
        super().__init__(p, u, G, laminar=True)
        self.convencion = convencion
        self.strict_memo = {}

    def _cleared_u(self, cleared):
        return sum(self.ui[i] for i in members(cleared))

    def _strict_dp(self, b, support, allowed, cleared):
        key = b, support, allowed, cleared
        if key in self.strict_memo:
            return self.strict_memo[key]
        mass = self._info(support)[0]
        best = mass * self._cleared_u(cleared)
        if b:
            choices = allowed
            while choices:
                bit = choices & -choices
                choices ^= bit
                j = bit.bit_length() - 1
                pool = self.pools[j]
                branches = [(r, support & m) for r, m in enumerate(self.outcomes[j])
                            if support & m]
                # A known-zero pool CAN collect previously deduced healthy people
                # in strict mode, even though the observation is deterministic.
                if len(branches) == 1 and not (branches[0][0] == 0 and pool & ~cleared):
                    continue
                future = allowed & self.compat[j]
                val = sum(self._strict_dp(b-1, s, future,
                          cleared | pool if r == 0 else cleared) for r, s in branches)
                best = max(best, val)
        self.strict_memo[key] = best
        return best

    def value(self, B, history=()):
        if not isinstance(B, int) or B < 0:
            raise ValueError('B must be a nonnegative integer')
        history = tuple(history)
        if self.convencion == 'posterior_zero':
            return self.solve(B, history).additional_value
        support, allowed = self._state(history)
        cleared = 0
        for pool, r in history:
            if r == 0:
                cleared |= sum(1 << i for i in pool)
        mass = self._info(support)[0]
        total = self._strict_dp(B, support, allowed, cleared)
        return Fraction(total - mass*self._cleared_u(cleared), mass*self.utility_scale)


def valor_optimo(p, u, G, B, convencion='posterior_zero', history=()):
    return PathwiseEnumerator(p, u, G, convencion).value(B, history)
