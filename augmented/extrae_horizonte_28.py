"""Compare A and ACD after AB=1 with one or three remaining tests.

Run: python -m augmented.extrae_horizonte_28
Frozen evidence is reused by the notebook builder. All probabilities and
credits are also reconstructed from explicit profiles, independently of DP.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.extrae_separacion_28 import audit_nodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sesion_2026-10-06/horizonte_AB1'
HISTORY = (((0, 1), 1),)
STATES = tuple(product((0, 1), repeat=6))
SUPPORT = tuple(z for z in STATES if z[0] + z[1] == 1)


def replay(tree, remaining, *, laminar):
    """Replay a complete forced-action tree on the 32 conditional profiles."""
    total = 0
    for actual in SUPPORT:
        support, history, node, used = list(SUPPORT), list(HISTORY), tree, 0
        while not node.get('stop'):
            pool = tuple(node['pool'])
            assert 0 < len(pool) <= 3 and used < remaining
            if laminar:
                s = set(pool)
                assert all(not s.intersection(old) or s <= set(old) or set(old) <= s
                           for old, _ in history)
            r = sum(actual[i] for i in pool)
            history.append((pool, r))
            support = [z for z in support if sum(z[i] for i in pool) == r]
            node = next(b['next'] for b in node['branches'] if b['infected_count'] == r)
            used += 1
        total += sum(all(z[i] == 0 for z in support) for i in range(6))
    return F(total, len(SUPPORT))


def forced_tree(solver, pool, remaining):
    branches = []
    for r in range(len(pool) + 1):
        branch_support = [z for z in SUPPORT if sum(z[i] for i in pool) == r]
        if not branch_support:
            continue
        history = HISTORY + ((pool, r),)
        child = solver.tree(remaining - 1, history)
        branches.append({'infected_count': r,
                         'probability': str(F(len(branch_support), len(SUPPORT))),
                         'next': child})
    value = sum(F(b['probability']) * F(b['next']['value']) for b in branches)
    tree = {'remaining_tests': remaining, 'compatible_profiles': len(SUPPORT),
            'newly_healthy': [], 'healthy': [], 'value': str(value),
            'pool': list(pool), 'branches': branches}
    nodes = audit_nodes(tree, HISTORY)
    assert replay(tree, remaining, laminar=solver.laminar) == value
    return {'tree': tree, 'value': str(value), 'nodes_audited': nodes,
            'independent_profile_checks': len(SUPPORT)}


def run():
    general = BellmanPerfiles([F(1, 2)] * 6, [1] * 6, 3)
    laminar = BellmanPerfiles([F(1, 2)] * 6, [1] * 6, 3, laminar=True)
    evidence = {'setup': {'n': 6, 'G': 3, 'q': '1/2', 'p': '1/2', 'u': [1]*6,
                          'history': HISTORY, 'remaining_tests': [1, 3],
                          'independent': True, 'clearing': 'soft'}, 'cases': []}
    for b in (1, 3):
        a_general = forced_tree(general, (0,), b)
        a_laminar = forced_tree(laminar, (0,), b)
        acd = forced_tree(general, (0, 2, 3), b)
        general_best, laminar_best = general.solve(b, HISTORY), laminar.solve(b, HISTORY)
        assert a_general['value'] == a_laminar['value'] == str(laminar_best.value)
        assert general_best.value == max(F(a_general['value']), F(acd['value']))
        evidence['cases'].append({'total_budget': b+1, 'remaining_tests': b,
                                  'A_general': a_general, 'A_laminar': a_laminar,
                                  'ACD_general': acd,
                                  'optimal_general': str(general_best.value),
                                  'optimal_laminar': str(laminar_best.value)})
    assert [(c['A_general']['value'], c['ACD_general']['value'])
            for c in evidence['cases']] == [('1', '1/2'), ('9/4', '3')]
    reference_path = ROOT / 'results/sesion_2026-10-06/separacion_3_2_5/evidencia.json'
    reference = json.loads(reference_path.read_text())
    stored = next(b['next'] for b in reference['policies']['general']['tree']['branches']
                  if b['infected_count'] == 1)
    new = evidence['cases'][1]['ACD_general']['tree']
    assert [(b['infected_count'], b['probability'], b['next']['value']) for b in new['branches']] == [
        (b['infected_count'], b['probability'], b['next']['value']) for b in stored['branches']]
    sources = ['augmented/extrae_horizonte_28.py', 'augmented/bellman_perfiles.py',
               'augmented/extrae_separacion_28.py', str(reference_path.relative_to(ROOT))]
    evidence['source_sha256'] = {s: sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    evidence['audit'] = {'conditional_profiles': len(SUPPORT),
                         'all_six_forced_trees_replayed': True,
                         'all_nodes_checked': True, 'matches_stored_B4_branches': True,
                         'scope': 'Two horizons at one fixed history; not a theorem for the whole grid.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'evidencia.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({str(c['remaining_tests']): {
        'A': c['A_general']['value'], 'ACD': c['ACD_general']['value'],
        'laminar_optimum': c['optimal_laminar'], 'general_optimum': c['optimal_general']}
        for c in evidence['cases']}, indent=2))


if __name__ == '__main__':
    run()
