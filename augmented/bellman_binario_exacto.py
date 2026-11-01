"""Binary dynamic optimum with exact arithmetic and explicit hard clearing.

Negative means no infected member. Positive means at least one infected.
Only membership in an observed negative pool earns utility, counted once.
The action menu contains all pools up to G, without a laminar restriction.
This small-instance reference is independent of the count Bellman recurrence.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations, product
from math import lcm, prod
from time import perf_counter


@dataclass(frozen=True)
class BinarySolution:
    value: F
    first_pool: tuple | None
    states: int
    seconds: float


class BellmanBinario:
    def __init__(self, p, u, G, *, state_limit=500_000):
        self.p, self.u = tuple(map(F, p)), tuple(map(F, u))
        self.n, self.G = len(self.p), G
        if not 1 <= self.n <= 10 or len(self.u) != self.n:
            raise ValueError('Requires 1 <= n <= 10 and one utility per person')
        if not isinstance(G, int) or not 1 <= G <= self.n:
            raise ValueError('Invalid pool cap')
        if any(x < 0 or x > 1 for x in self.p) or any(x < 0 for x in self.u):
            raise ValueError('Invalid prior or utility')
        self.scale = lcm(*(x.denominator for x in self.u))
        self.ui = tuple(int(x*self.scale) for x in self.u)
        self.weights = tuple(prod(x.numerator if z >> i & 1 else x.denominator-x.numerator
                                 for i,x in enumerate(self.p)) for z in range(1 << self.n))
        self.root = sum(1 << z for z,w in enumerate(self.weights) if w)
        self.pools = tuple(sum(1 << i for i in c)
                           for k in range(1,G+1) for c in combinations(range(self.n),k))
        self.negative = tuple(sum(1 << z for z in range(1 << self.n) if not z & a)
                              for a in self.pools)
        self.memo, self.argmax, self.masses, self.utilities = {}, {}, {}, {}
        self.state_limit = state_limit

    def mass(self, support):
        if support not in self.masses:
            todo, total = support, 0
            while todo:
                bit = todo & -todo
                total += self.weights[bit.bit_length()-1]
                todo ^= bit
            self.masses[support] = total
        return self.masses[support]

    def utility(self, cleared):
        if cleared not in self.utilities:
            self.utilities[cleared] = sum(x for i,x in enumerate(self.ui) if cleared >> i & 1)
        return self.utilities[cleared]

    def _dp(self, b, support, cleared):
        key = b, support, cleared
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) >= self.state_limit:
            raise RuntimeError('Binary exact-state limit; no optimum certified')
        best, action = self.mass(support)*self.utility(cleared), None
        if b:
            for j,a in enumerate(self.pools):
                # Removing already cleared members preserves every outcome and
                # gain. The reduced pool is itself in the feasible action menu.
                if a & cleared:
                    continue
                neg = support & self.negative[j]
                if not neg:  # a certain positive has no information or credit
                    continue
                pos = support ^ neg
                value = self._dp(b-1, neg, cleared | a)
                if pos:
                    value += self._dp(b-1, pos, cleared)
                if value > best:
                    best, action = value, j
        self.memo[key], self.argmax[key] = best, action
        return best

    def solve(self, B):
        if not isinstance(B,int) or B < 0:
            raise ValueError('Invalid budget')
        start = perf_counter()
        value = F(self._dp(B,self.root,0), self.mass(self.root)*self.scale)
        j = self.argmax[B,self.root,0]
        pool = None if j is None else tuple(i for i in range(self.n) if self.pools[j] >> i & 1)
        return BinarySolution(value,pool,len(self.memo),perf_counter()-start)

    def tree(self, B):
        self.solve(B)
        def visit(b,support,cleared):
            node = {'remaining_tests':b,'cleared':[i for i in range(self.n) if cleared >> i & 1],
                    'value':str(F(self._dp(b,support,cleared),self.mass(support)*self.scale))}
            j = self.argmax[b,support,cleared]
            if j is None:
                node['stop'] = True
                return node
            a = self.pools[j]
            node['pool'] = [i for i in range(self.n) if a >> i & 1]
            node['branches'] = []
            neg = support & self.negative[j]
            for r,s in [(0,neg),(1,support^neg)]:
                if s:
                    node['branches'].append({'result':r,'probability':str(F(self.mass(s),self.mass(support))),
                                            'next':visit(b-1,s,cleared|a if r==0 else cleared)})
            return node
        return visit(B,self.root,0)


def replay_binary_tree(tree, q, u, B, G):
    """Independent raw-profile evaluation; no solver masks or reward routine."""
    q,u = tuple(map(F,q)),tuple(map(F,u))
    value,max_tests,nonlaminar,positive_profiles = F(0),0,0,0
    for z in product((0,1),repeat=len(q)):
        weight = prod(q[i] if x==0 else 1-q[i] for i,x in enumerate(z))
        if not weight:
            continue
        positive_profiles += 1
        node,cleared,past = tree,set(),[]
        while not node.get('stop'):
            a=set(node['pool'])
            assert len(a)==len(node['pool']) and 1<=len(a)<=G
            assert a<=set(range(len(q))) and len(past)<B
            r=int(any(z[i] for i in a))
            if r==0:
                cleared |= a
            past.append(a)
            node=next(x['next'] for x in node['branches'] if x['result']==r)
        assert cleared==set(node['cleared'])
        value += weight*sum(u[i] for i in cleared)
        max_tests=max(max_tests,len(past))
        nonlaminar += int(any(a&b and not (a<=b or b<=a) for j,a in enumerate(past) for b in past[:j]))
    return {'value':str(value),'profiles':positive_profiles,'max_tests':max_tests,
            'profiles_with_nonlaminar_pools':nonlaminar}
