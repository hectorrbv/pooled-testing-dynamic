"""Small exact map for notebook 28: restriction loss, not a policy leaderboard.

Run from the repository root: python -m augmented.experimento_mapa_laminar_28
76 homogeneous cases, n=4, B in {2,3}, G in {2,3}, q=.05,.10,...,.95.
Every chosen policy is independently evaluated over all 16 latent profiles.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from pathlib import Path
import json

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.experimento_sesion_28 import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sesion_2026-10-05'


def policy_from_tree(tree):
    def action(support, history, remaining):
        node = tree
        for pool, result in history:
            assert tuple(node['pool']) == tuple(pool)
            node = next(branch['next'] for branch in node['branches']
                        if branch['infected_count'] == result)
        return node.get('pool')
    return action


def run():
    OUT.mkdir(exist_ok=True)
    rows = []
    independent_checks = atom_checks = 0
    for B, G, numerator in product((2, 3), (2, 3), range(1, 20)):
        q = F(numerator, 20)
        values = {}
        for name, laminar in [('general', False), ('laminar', True)]:
            solver = BellmanPerfiles([1-q]*4, [1]*4, G, laminar=laminar)
            sol = solver.solve(B)
            audited = evaluate(policy_from_tree(solver.tree(B)), [q]*4,
                               [F(1)]*4, B, G, require_laminar=laminar)
            assert F(audited['value']) == sol.value
            independent_checks += 1
            values[name] = sol.value
        atom = SolverLaminar(dict(enumerate([1-q]*4)), dict(enumerate([F(1)]*4)),
                             G, 'posterior_zero')
        assert atom.V(frozenset(range(4)), (), B) == values['laminar']
        atom_checks += 1
        assert 0 < values['laminar'] <= values['general'] <= 4*q
        ratio = values['laminar'] / values['general']
        rows.append(dict(n=4, B=B, G=G, q=str(q), q_decimal=float(q),
                         general=str(values['general']), laminar=str(values['laminar']),
                         retained_ratio=str(ratio), loss_percent=float(100*(1-ratio))))
    assert len(rows) == 76
    worst = max(rows, key=lambda row: row['loss_percent'])
    special = next(row for row in rows if (row['B'],row['G'],F(row['q'])) == (3,3,F(1,2)))
    assert special['general'] == '2' and special['laminar'] == '7/4'
    signatures = {}
    for z in product((0,1), repeat=4):
        sig = tuple(sum(z[i] for i in pool) for pool in [(0,1,2),(0,1,3),(0,2,3)])
        assert sig not in signatures
        signatures[sig] = z
    assert signatures[(1,1,2)] == (0,0,1,1)
    q = F(3,10)
    pairs = [{'first':a,'second':b,'probability':str((q if a==0 else 1-q)*(q if b==0 else 1-q))}
             for a,b in product((0,1),repeat=2)]
    assert pairs[1]['probability'] == pairs[2]['probability'] == '21/100'
    paths = ['augmented/experimento_mapa_laminar_28.py','augmented/bellman_perfiles.py',
             'augmented/bm17_toy_solver.py','augmented/experimento_sesion_28.py']
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  model=dict(observation='exact infection count', clearing='soft',
                             q='probability healthy', independent_priors=True, u=[1]*4,
                             initial_history=[], laminar='nested or disjoint on each trajectory'),
                  scope='76 homogeneous n=4 examples; finite map, not a worst-case bound.',
                  arithmetic='Fraction throughout both exact solvers and independent policy evaluation',
                  source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  audit=dict(independent_policy_checks=independent_checks,
                             independent_atom_solver_checks=atom_checks,
                             profiles_per_policy=16, max_difference='0'),
                  rows=rows,
                  summary=dict(equal_cases=sum(row['general']==row['laminar'] for row in rows),
                               worst_case=worst),
                  count_witness=dict(tests=['ABC','ABD','ACD'],counts=[1,1,2],
                                     unique_profile_infection=[0,0,1,1],healthy=['A','B'],
                                     distinct_signatures=len(signatures)),
                  independent_pair_example=dict(q='3/10',state_zero='healthy',state_one='infected',
                                                pairs=pairs,conditional_orientation='1/2'))
    (OUT/'mapa_laminar_28.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'audit':report['audit'],'summary':report['summary']},indent=2))


if __name__ == '__main__':
    run()
