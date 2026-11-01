"""Extract and independently replay the n=6, B=4, G=3, q=1/2 separation.

Run: python -m augmented.extrae_separacion_28
Only this instance is solved. Notebook builds reuse the frozen evidence.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.experimento_sesion_28 import evaluate, profiles
from augmented.experimento_mapa_laminar_28 import policy_from_tree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sesion_2026-10-06/separacion_3_2_5'
Q = [F(1, 2)]*6
U = [F(1)]*6


def audit_nodes(tree, history=()):
    """Check every node's posterior, action and conditional value from raw profiles."""
    prior = profiles(Q)
    support = {z: w for z, w in prior.items()
               if all(sum(z[i] for i in pool) == r for pool, r in history)}
    assert len(support) == tree['compatible_profiles']
    healthy = [i for i in range(6) if all(z[i] == 0 for z in support)]
    assert healthy == list(tree['healthy'])
    if tree.get('stop'):
        assert F(tree['value']) == len(healthy)
        return 1
    mass = sum(support.values())
    conditional_value, nodes = F(0), 1
    for branch in tree['branches']:
        pool, r = tuple(tree['pool']), branch['infected_count']
        branch_mass = sum(w for z, w in support.items() if sum(z[i] for i in pool) == r)
        assert F(branch['probability']) == branch_mass/mass
        conditional_value += F(branch['probability'])*F(branch['next']['value'])
        nodes += audit_nodes(branch['next'], history+((pool, r),))
    assert conditional_value == F(tree['value'])
    return nodes


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    report = {'setup': {'n':6, 'B':4, 'G':3, 'q':'1/2', 'p':'1/2',
                        'u':[1]*6, 'clearing':'soft', 'independent':True,
                        'history':[], 'labels':'ABCDEF'}, 'policies':{}}
    for name, lam in [('general',False), ('laminar',True)]:
        solver = BellmanPerfiles(Q, U, 3, laminar=lam)
        sol = solver.solve(4)
        tree = solver.tree(4)
        audit = evaluate(policy_from_tree(tree), Q, U, 4, 3, require_laminar=lam)
        assert str(sol.value) == audit['value'] == ('3' if name=='general' else '5/2')
        assert tuple(tree['pool']) == (0,1)
        report['policies'][name] = {'tree':tree, 'audit':audit,
                                    'nodes_audited':audit_nodes(tree)}
    atom = SolverLaminar(dict(enumerate(Q)), dict(enumerate(U)), 3, 'posterior_zero')
    assert atom.V(frozenset(range(6)), (), 4) == F(5,2)
    branches = []
    for r in range(3):
        g, l = [next(x for x in report['policies'][name]['tree']['branches']
                     if x['infected_count']==r) for name in ['general','laminar']]
        assert g['probability']==l['probability']
        contribution = F(g['probability'])*(F(g['next']['value'])-F(l['next']['value']))
        branches.append({'R_AB':r, 'probability':g['probability'],
                         'general':g['next']['value'], 'laminar':l['next']['value'],
                         'general_next':g['next']['pool'], 'laminar_next':l['next']['pool'],
                         'gap_contribution':str(contribution)})
    assert [F(x['gap_contribution']) for x in branches] == [F(1,16), F(3,8), F(1,16)]
    assert sum(F(x['gap_contribution']) for x in branches)==F(1,2)
    report['root_branches']=branches
    chosen=(0,1,0,1,0,0)
    witness={}
    for name in ['general','laminar']:
        trace=next(x for x in report['policies'][name]['audit']['traces']
                   if tuple(x['profile'])==chosen)
        node=report['policies'][name]['tree']; steps=[]
        for pool,r in trace['history']:
            branch=next(x for x in node['branches'] if x['infected_count']==r)
            nxt=branch['next']
            steps.append({'pool':list(pool), 'R':r, 'conditional_probability':branch['probability'],
                          'healthy_after':list(nxt['healthy']),
                          'newly_healthy':list(nxt['newly_healthy'])})
            node=nxt
        assert node.get('stop')
        witness[name]={'steps':steps,'healthy':list(node['healthy']),
                       'utility':trace['credited_utility']}
    assert (witness['general']['utility'],witness['laminar']['utility'])==('4','2')
    assert witness['general']['steps'][1]['pool']==[0,2,3]
    assert witness['laminar']['steps'][1]['pool']==[0]
    report['witness']={'infection_profile':chosen,'infected':['B','D'],
                       'probability':'1/64','policies':witness,
                       'scope':'One illustrative realization, not the expected-value comparison.'}
    report['audit']={'independent_profiles_per_policy':64,'all_nodes_checked':True,
                     'independent_laminar_atom_check':True,
                     'root_gap':'1/2','mixed_branch_contribution':'3/8',
                     'mixed_share_of_gap':'3/4'}
    paths=['augmented/extrae_separacion_28.py','augmented/bellman_perfiles.py',
           'augmented/bm17_toy_solver.py','augmented/experimento_sesion_28.py',
           'augmented/experimento_mapa_laminar_28.py']
    report['source_sha256']={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    (OUT/'evidencia.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'branches':branches, 'witness':report['witness'], 'audit':report['audit']},indent=2))


if __name__=='__main__':
    run()
