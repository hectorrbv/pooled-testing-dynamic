"""Exact first experiment: when and how much grouping improves on individuals.
Run: python -m augmented.experimento_agrupar_individuales
n=6, B=1..6, G=2,3, q=.05,.10,...,.95. General dynamic policies;
counts, soft clearing, independent homogeneous priors, unit utilities.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import signal

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.experimento_sesion_28 import evaluate
from augmented.experimento_mapa_laminar_28 import policy_from_tree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/diseno_mapa_2026-10-05'


def timeout(signum, frame):
    raise TimeoutError('Per-case time limit reached; optimum not certified')


def run():
    OUT.mkdir(exist_ok=True)
    signal.signal(signal.SIGALRM, timeout)
    rows=[]
    start=perf_counter()
    for G in (2,3):
        for numerator in range(1,20):
            q=F(numerator,20)
            solver=BellmanPerfiles([1-q]*6,[1]*6,G,state_limit=500_000)
            for B in range(1,7):
                t=perf_counter();signal.alarm(30)
                row={'n':6,'G':G,'B':B,'q':str(q),'q_decimal':float(q)}
                try:
                    individual=B*q
                    if B==6:
                        value=6*q
                        action=lambda support,history,remaining: (len(history),) if len(history)<6 else None
                        certified_by='all-individual witness attains total healthy utility'
                        first_sizes=None
                    else:
                        sol=solver.solve(B);value=sol.value
                        action=policy_from_tree(solver.tree(B))
                        certified_by='exact Bellman over all feasible pools'
                        root=solver.root_support
                        denominator=solver._info(root)[0]*solver.utility_scale
                        first_sizes=[]
                        for size in range(1,G+1):
                            # All size-k roots are equivalent under this homogeneous prior.
                            j=solver.pool_index[(1<<size)-1]
                            score=sum(solver._dp(B-1,root & mask,solver.all_actions)
                                      for mask in solver.outcomes[j] if root & mask)
                            if F(score,denominator)==value:first_sizes.append(size)
                    audit=evaluate(action,[q]*6,[F(1)]*6,B,G)
                    assert F(audit['value'])==value
                    assert individual<=value<=6*q
                    if B==1:assert value==max(k*q**k for k in range(1,G+1))
                    if 2<=B<6:assert value>individual
                    if B==6:assert value==individual
                    row.update(status='exact',individual=str(individual),grouping=str(value),
                               gain=str(value-individual),gain_percent=float(100*(value/individual-1)),
                               optimal_first_sizes=first_sizes,certified_by=certified_by,
                               independent_profile_check=True,profiles=64,
                               seconds=round(perf_counter()-t,4))
                except (TimeoutError,RuntimeError) as exc:
                    row.update(status='not_certified',reason=str(exc),seconds=round(perf_counter()-t,4))
                finally:signal.alarm(0)
                rows.append(row)
            print(f'G={G}, q={float(q):.2f}: {len(rows)}/228',flush=True)
    assert len(rows)==228
    exact=[r for r in rows if r['status']=='exact']
    bykey={(r['G'],r['B'],r['q']):r for r in exact}
    for (G,B,q),r in bykey.items():
        if (G,B+1,q) in bykey:assert F(r['grouping'])<=F(bykey[G,B+1,q]['grouping'])
        if (G+1,B,q) in bykey:assert F(r['grouping'])<=F(bykey[G+1,B,q]['grouping'])
    paths=['augmented/experimento_agrupar_individuales.py','augmented/bellman_perfiles.py',
           'augmented/experimento_sesion_28.py','augmented/experimento_mapa_laminar_28.py']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'question':'How much does allowing adaptive groups improve on the best individual-only policy?',
            'model':{'n':6,'G':[2,3],'B':[1,2,3,4,5,6],'q':'healthy probability',
                     'u':[1]*6,'independent_priors':True,'homogeneous_q':True,
                     'history':[],'observation':'exact infection count','clearing':'soft',
                     'grouping_policy_class':'unrestricted dynamic; crossing pools allowed'},
            'metric':'100*(grouping_optimum / individual_optimum - 1)',
            'rows':rows,
            'audit':{'exact_cases':len(exact),'independent_profile_evaluations':len(exact),
                     'profiles_per_policy':64,'B1_formula_checked':True,'B_equals_n_control_checked':True,
                     'strict_improvement_for_2_le_B_lt_n_checked':True,
                     'budget_and_group_cap_monotonicity_checked':True},
            'source_sha256':{p:sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'seconds':round(perf_counter()-start,3)}
    (OUT/'agrupar_vs_individuales.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_cases':len(exact),'seconds':result['seconds']},indent=2))


if __name__=='__main__':run()
