"""Exact evidence for the five questions of the October 6 research plan.

Run: python -m augmented.experimento_sesion_29
The presentation generator reads this output; opening the notebook runs no search.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json

from augmented.bellman_binario_exacto import BellmanBinario, replay_binary_tree
from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.classical_solver import solve_classical_dynamic
from augmented.experimento_sesion_28 import profiles, credited, count, evaluate, fixed_plan, laminar_pair
from augmented.experimento_mapa_laminar_28 import policy_from_tree
from augmented.indice_lagrangiano import PoliticaLagrangiana
from augmented.politica_ratio import PoliticaRatio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sesion_2026-10-12'


def laminar_two_tests(q, u, G):
    """Independent exhaustive two-level decision tree: no DP or atom recurrence."""
    prior = profiles(q)
    pools = [a for k in range(1,G+1) for a in combinations(range(len(q)),k)]
    roots = []
    for first in pools:
        value, branches = F(0), []
        for r in range(len(first)+1):
            support = {z:w for z,w in prior.items() if count(z,first)==r and w}
            mass = sum(support.values())
            if not mass:
                continue
            best, second = mass*credited(support,u), None
            for pool in pools:
                if not laminar_pair(first,pool):
                    continue
                candidate = F(0)
                for s in range(len(pool)+1):
                    leaf = {z:w for z,w in support.items() if count(z,pool)==s}
                    if leaf:
                        candidate += sum(leaf.values())*credited(leaf,u)
                if candidate > best:
                    best,second = candidate,pool
            value += best
            branches.append(dict(result=r,probability=str(mass),second=second,
                                 conditional_value=str(best/mass)))
        roots.append(dict(first=first,value=str(value),branches=branches))
    return dict(value=str(max(F(x['value']) for x in roots)),roots=roots)


def heuristic_adapter(policy,n):
    def action(support,history,b):
        untested,atoms = frozenset(range(n)), ()
        for pool,r in history:
            if set(pool)<=untested:
                untested -= set(pool)
                pieces = [(pool,r)]
            else:
                old = next(a for a in atoms if set(pool)<set(a[0]))
                atoms = tuple(a for a in atoms if a!=old)
                pieces = [(pool,r),(tuple(i for i in old[0] if i not in pool),old[1]-r)]
            atoms = tuple(sorted(atoms+tuple((tuple(sorted(a)),k) for a,k in pieces if 0<k<len(a))))
        decision = policy.decide(untested,atoms,b)
        return (decision[1] if decision[0]=='open' else decision[2]) if decision else None
    return action


def measure(q,B,G,u=None, *, heuristics=True):
    q=tuple(map(F,q));u=tuple(map(F,u or [1]*len(q)));p=tuple(1-x for x in q)
    result=dict(n=len(q),B=B,G=G,q=list(map(str,q)),u=list(map(str,u)),policies={})
    binary=BellmanBinario(p,u,G)
    value=binary.solve(B).value;tree=binary.tree(B)
    audit=replay_binary_tree(tree,q,u,B,G)
    assert F(audit['value'])==value
    old,_=solve_classical_dynamic(list(map(float,p)),list(map(float,u)),B,G)
    assert abs(old-float(value))<1e-11
    result['policies']['binary']=dict(value=str(value),tree=tree,audit=audit,
                                    independent_float_value=old)
    for name,laminar in [('laminar',True),('general',False)]:
        solver=BellmanPerfiles(p,u,G,laminar=laminar)
        value=solver.solve(B).value;tree=solver.tree(B)
        audit=evaluate(policy_from_tree(tree),q,u,B,G,require_laminar=laminar)
        assert F(audit['value'])==value
        result['policies'][name]=dict(value=str(value),tree=tree,audit=audit)
    atom=SolverLaminar(dict(enumerate(p)),dict(enumerate(u)),G,'posterior_zero')
    independent=atom.V(frozenset(range(len(q))),(),B)
    assert independent==F(result['policies']['laminar']['value'])
    result['independent_atom_value']=str(independent)
    if B==2:
        result['independent_two_test_enumeration']=laminar_two_tests(q,u,G)
        assert F(result['independent_two_test_enumeration']['value'])==independent
    if heuristics:
        for name,cls,parameters in [('pi_L',PoliticaLagrangiana,dict(lam=.01,horizonte=3)),
                                    ('pi_ratio',PoliticaRatio,dict(alpha=1,horizonte=3))]:
            policy=cls(dict(enumerate(p)),dict(enumerate(u)),G,**parameters)
            action=heuristic_adapter(policy,len(q))
            audit=evaluate(action,q,u,B,G,require_laminar=True)
            assert abs(float(F(audit['value']))-policy.valor(frozenset(range(len(q))),(),B))<1e-10
            result['policies'][name]=dict(value=audit['value'],audit=audit,
                                         first=action(profiles(q),[],B),parameters=parameters)
            assert F(audit['value'])<=independent
            policy.valor.cache_clear()
            if name=='pi_L':
                policy._I_atomos.cache_clear();policy._I_virgen.cache_clear()
    return result


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    pair=measure(['.3']*6,2,2)
    assert pair['policies']['laminar']['value']=='387/500'
    witness=measure(['.95','.8','.8'],2,2)
    assert [witness['policies'][x]['value'] for x in ['laminar','binary','general']]==['1183/500','304/125','306/125']
    # Exact, hand-checkable root comparison independent of any Bellman implementation.
    assert sorted(F(r['value']) for r in witness['independent_two_test_enumeration']['roots'])==sorted(map(F,['2.23','2.32','2.32','2.366','2.366','2.246']))
    matrix=measure(['.5']*4,3,3)
    plan=[(0,1,2),(0,1,3),(0,2,3)]
    signatures={tuple(count(z,a) for a in plan):z for z in profiles([F(1,2)]*4)}
    assert len(signatures)==16
    matrix['detector']=dict(pools=plan,rows=[[int(i in a) for i in range(4)] for a in plan],
                           distinct_signatures=16,audit=evaluate(fixed_plan(plan),[F(1,2)]*4,[F(1)]*4,3,3))
    assert matrix['detector']['audit']['value']==matrix['policies']['general']['value']=='2'
    homogeneous=measure(['.95']*5,3,4)
    # A local, explicitly delimited slice. This is not a prevalence phase diagram.
    slice_rows=[]
    for numerator in range(5,100):
        t=F(numerator,100);q=[F(19,20),t,t];p=[1-x for x in q]
        b=BellmanBinario(p,[1]*3,2).solve(2).value
        l=BellmanPerfiles(p,[1]*3,2,laminar=True).solve(2).value
        a=BellmanPerfiles(p,[1]*3,2).solve(2).value
        slice_rows.append(dict(t=str(t),binary=str(b),laminar=str(l),general=str(a),delta=str(b-l)))
    search=json.loads((OUT/'busqueda/summary.json').read_text())
    paths=['augmented/experimento_sesion_29.py','augmented/bellman_binario_exacto.py',
           'augmented/bellman_perfiles.py','augmented/bm17_toy_solver.py',
           'augmented/classical_solver.py','augmented/experimento_sesion_28.py',
           'augmented/indice_lagrangiano.py','augmented/politica_ratio.py']
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),
              conventions=dict(q='healthy probability',R='number infected',independent_priors=True,
                  binary='dynamic, unrestricted pools, hard clearing',laminar='dynamic exact counts, pathwise laminar, soft clearing',
                  general='dynamic exact counts, unrestricted pools, soft clearing'),
              source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              pair=pair,witness=witness,matrix=matrix,homogeneous=homogeneous,slice=slice_rows,
              search_summary={k:v for k,v in search.items() if k not in ['best','best_cases','source_sha256']})
    # Keep the frozen broad-search provenance separate from the current replay fix.
    data['search_summary'].pop('top',None)
    (OUT/'evidencia_29.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:{name:p['value'] for name,p in case['policies'].items()}
                      for k,case in [('pair',pair),('witness',witness),('matrix',matrix),('homogeneous',homogeneous)]},indent=2))


if __name__=='__main__':
    run()
