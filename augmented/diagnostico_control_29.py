"""Locate the old q=.95 heuristic gap without rerunning the search.

Enumerate the two remaining tests after the common ABCD opening, and compare
the saved heuristic traces. Uses raw profiles, not Bellman's transitions.
"""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path

from augmented.experimento_sesion_28 import profiles, count, credited, laminar_pair


def best_two_tests(support, first, pools, u):
    best = sum(support.values()) * credited(support, u)
    for second in pools:
        if not laminar_pair(first, second):
            continue
        total = F(0)
        for r in range(len(second)+1):
            child = {z:w for z,w in support.items() if count(z,second)==r}
            if not child:
                continue
            continuation = sum(child.values()) * credited(child,u)
            for third in pools:
                if not all(laminar_pair(third,old) for old in [first,second]):
                    continue
                value = F(0)
                for s in range(len(third)+1):
                    leaf = {z:w for z,w in child.items() if count(z,third)==s}
                    if leaf:
                        value += sum(leaf.values()) * credited(leaf,u)
                continuation = max(continuation,value)
            total += continuation
        best = max(best,total)
    return best


def main():
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root/'results/sesion_2026-10-12/evidencia_29.json').read_text())['homogeneous']
    q,u = tuple(map(F,saved['q'])), tuple(map(F,saved['u']))
    prior = profiles(q)
    first = (0,1,2,3)
    pools = [a for k in range(1,5) for a in combinations(range(5),k)]
    rows = []
    for r in range(5):
        support = {z:w for z,w in prior.items() if count(z,first)==r}
        mass = sum(support.values())
        best = best_two_tests(support,first,pools,u)/mass
        row = dict(r=r,probability=str(mass),best_conditional=str(best),policies={})
        for kind in ['pi_L','pi_ratio']:
            traces = [t for t in saved['policies'][kind]['audit']['traces'] if t['history'][0][1]==r]
            assert all(tuple(t['history'][0][0])==first for t in traces)
            assert sum(F(t['probability']) for t in traces)==mass
            value = sum(F(t['probability'])*F(t['credited_utility']) for t in traces)/mass
            row['policies'][kind] = dict(conditional=str(value),weighted_loss=str(mass*(best-value)),
                next=sorted({tuple(t['history'][1][0]) for t in traces if len(t['history'])>1}))
        rows.append(row)
    forced = sum(F(r['probability'])*F(r['best_conditional']) for r in rows)
    assert forced == F(731063,160000)
    for kind in ['pi_L','pi_ratio']:
        loss = sum(F(r['policies'][kind]['weighted_loss']) for r in rows)
        assert forced-loss == F(saved['policies'][kind]['value'])
    result = dict(instance={k:saved[k] for k in ['n','B','G','q','u']},
        laminar_optimum=saved['policies']['laminar']['value'],
        forced_root_ABCD_best_continuation=str(forced),
        root_loss=str(F(saved['policies']['laminar']['value'])-forced),branches=rows,
        method='Direct enumeration of both remaining tests; saved policies evaluated over all profiles')
    out = root/'results/sesion_2026-10-14/diagnostico_control_29.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['branches','instance']},indent=2))


if __name__=='__main__':
    main()
