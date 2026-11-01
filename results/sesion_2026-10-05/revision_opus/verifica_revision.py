"""Independent finite audit for the notebook review; standard library only.
Run from the repository root. Does not import the project's solvers or evaluators.
"""
from functools import lru_cache
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ALL = tuple(range(16))  # A, B, C, D are bits 0,1,2,3; 1 means infected.


def weights(q):
    return tuple((1-q)**z.bit_count()*q**(4-z.bit_count()) for z in ALL)


def cleared(support):
    possible_infections = 0
    for z in support:
        possible_infections |= z
    return 4-possible_infections.bit_count()


def groups(support, pool):
    bins = {}
    for z in support:
        bins.setdefault((z & pool).bit_count(), []).append(z)
    return {r:tuple(zs) for r,zs in bins.items()}


def unrestricted_values(q, cap):
    w = weights(q)
    pools = [a for a in range(1, 16) if a.bit_count() <= cap]

    @lru_cache(None)
    def mass(support):
        return sum(w[z] for z in support)

    @lru_cache(None)
    def best(support, budget):
        value = F(cleared(support))  # Expected total terminal utility, not an increment.
        if budget == 0:
            return value
        for pool in pools:
            branches = groups(support, pool)
            if len(branches) == 1:
                continue
            candidate = sum(mass(child)*best(child, budget-1)
                            for child in branches.values())/mass(support)
            value = max(value, candidate)
        return value

    return {b:best(ALL, b) for b in (2, 3, 4)}


def fixed_value(q, schedule):
    signatures = {}
    for z in ALL:
        sig = tuple((z & a).bit_count() for a in schedule)
        signatures.setdefault(sig, []).append(z)
    w = weights(q)
    return sum(sum(w[z] for z in support)*cleared(support)
               for support in signatures.values())


def evaluate_decisions(q, decide, budget):
    w = weights(q)
    value = F(0)
    for z in ALL:
        support, history = ALL, []
        for _ in range(budget):
            a = decide(history)
            if a is None:
                break
            r = (z & a).bit_count()
            support = groups(support, a)[r]
            history.append((a, r))
        value += w[z]*cleared(support)
    return value


def run():
    grid = json.loads((ROOT/'results/sesion_2026-10-05/mapa_laminar_28.json').read_text())
    cache, checked = {}, 0
    for row in grid['rows']:
        q, cap = F(row['q']), row['G']
        if (q, cap) not in cache:
            cache[q, cap] = unrestricted_values(q, cap)
        assert cache[q, cap][row['B']] == F(row['general']), row
        checked += 1
    q = F(3, 10)
    pools = [a for a in range(1, 16) if a.bit_count() <= 2]
    families = [f for k in (0, 1, 2) for f in combinations(pools, k)
                if all(not (a & b) or (a & b) in (a, b) for a,b in combinations(f,2))]
    static = max(fixed_value(q, f) for f in families)
    pair_a = fixed_value(q, (3, 1))
    pair_c = fixed_value(q, (3, 4))
    sampled = F(42,100)*pair_a+F(58,100)*pair_c
    assert static == pair_a == F(3,5)
    assert pair_c == F(12,25) and sampled == F(663,1250)
    all_families = [f for k in (0, 1, 2) for f in combinations(pools, k)]
    static_general = max(fixed_value(q, f) for f in all_families)
    assert static_general == fixed_value(q, (3, 5)) == F(627,1000)

    def trimmed(h):
        if not h:return 7
        if len(h)==1:return 8
        return 1 if h[0][1] in (1,2) else None

    def replanned(h):
        if not h:return 7
        if h[0][1] in (0,3):return 8 if len(h)==1 else None
        if len(h)==1:return 1
        return 8 if (h[0][1],h[1][1]) in ((1,1),(2,0)) else 2

    trim = evaluate_decisions(F(1,2), trimmed, 3)
    ratio = evaluate_decisions(F(1,2), replanned, 3)
    crossing = fixed_value(F(1,2), (7,11,13))
    assert trim == F(3,2) and ratio == F(7,4) and crossing == 2
    report = {'unrestricted_optimum_cases_checked':checked, 'max_difference':'0',
              'implementation':'standard library; no project solver or evaluator imports',
              'static_laminar_families_checked':len(families),
              'best_static_laminar_q03_B2':str(static),
              'static_general_families_checked':len(all_families),
              'best_static_general_q03_B2':str(static_general),
              'AB_A':str(pair_a),'AB_C':str(pair_c),'sampled_path':str(sampled),
              'trimmed':str(trim),'replanned_as_drawn':str(ratio),'crossing_plan':str(crossing),
              'scope':'Finite numerical verification; no general approximation theorem.'}
    (OUT/'verificacion_independiente.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
