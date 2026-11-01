"""Small exact checks for the research discussion after 29 September.

Run: python -m augmented.experimento_sesion_28
All new policy values are enumerated over latent profiles using Fractions.
The crossing-repair rule is a pilot, not an approximation theorem.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations, product
from math import prod
from pathlib import Path
import json

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.politica_ratio import PoliticaRatio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sesion_2026-10-04'


def profiles(q):
    return {z: prod(q[i] if x == 0 else 1-q[i] for i,x in enumerate(z))
            for z in product((0,1), repeat=len(q))}


def count(z, pool):
    return sum(z[i] for i in pool)


def laminar_pair(a, b):
    a, b = set(a), set(b)
    return not a.intersection(b) or a <= b or b <= a


def credited(support, u):
    return sum(u[i] for i in range(len(u)) if all(z[i] == 0 for z in support))


def evaluate(policy, q, u, B, G, *, require_laminar=False):
    """Independent history filtering; does not use Bellman's transition/reward code."""
    prior = {z:w for z,w in profiles(q).items() if w}
    value, max_used, traces = F(0), 0, []
    for actual, weight in prior.items():
        support, history = dict(prior), []
        for used in range(B):
            pool = policy(support, history, B-used)
            if pool is None:
                break
            pool = tuple(pool)
            assert len(pool) == len(set(pool)) and 0 < len(pool) <= G
            assert all(0 <= i < len(q) for i in pool)
            if require_laminar:
                assert all(laminar_pair(pool, old) for old,_ in history)
            r = count(actual, pool)
            history.append((pool,r))
            support = {z:w for z,w in support.items() if count(z,pool) == r}
        reward = credited(support,u)
        value += weight*reward
        max_used = max(max_used,len(history))
        traces.append({'profile':actual,'probability':str(weight),
                       'history':history,'credited_utility':str(reward)})
    return {'value':str(value),'decimal':float(value),'max_tests':max_used,
            'profiles':len(prior),'traces':traces}


def fixed_plan(plan):
    return lambda support,history,b: plan[len(history)] if len(history)<len(plan) else None


def trim_to_disjoint(plan):
    """Keep only previously untouched members; empty pools consume no test."""
    seen, result = set(), []
    for pool in plan:
        new = tuple(i for i in pool if i not in seen)
        if new:
            result.append(new)
        seen.update(pool)
    return result


def immediate_action(support, history, u, G):
    """Choose the greatest expected new soft credit among laminar-legal pools."""
    mass = sum(support.values())
    base = credited(support,u)
    best, gain = None, F(0)
    for k in range(1,G+1):
        for pool in combinations(range(len(u)),k):
            if any(not laminar_pair(pool,old) for old,_ in history):
                continue
            branches = {}
            for z,w in support.items():
                branches.setdefault(count(z,pool),{})[z] = w
            score = sum(sum(s.values())*credited(s,u) for s in branches.values())/mass-base
            if score > gain:
                best,gain = pool,score
    return best


def trimmed_with_reuse(plan, u, G):
    trimmed = trim_to_disjoint(plan)
    def policy(support, history, b):
        if len(history)<len(trimmed):
            return trimmed[len(history)]
        return immediate_action(support,history,u,G)
    return policy


def first_pool_then_ratio(first_pool, q, u, G):
    """Pilot: keep the first source pool, then use the existing laminar ratio rule."""
    ratio = PoliticaRatio(dict(enumerate(1-x for x in q)),dict(enumerate(u)),G)
    def policy(support,history,b):
        if not history:
            return first_pool
        U, atoms = frozenset(range(len(q))), ()
        for pool,r in history:
            if set(pool)<=U:
                U -= set(pool)
                pieces = [(pool,r)]
            else:
                old = next(a for a in atoms if set(pool)<set(a[0]))
                atoms = tuple(a for a in atoms if a!=old)
                pieces = [(pool,r),(tuple(i for i in old[0] if i not in pool),old[1]-r)]
            atoms = tuple(sorted(atoms+tuple((tuple(sorted(a)),k) for a,k in pieces if 0<k<len(a))))
        action = ratio.decide(U,atoms,b)
        return (action[1] if action[0]=='open' else action[2]) if action else None
    return policy


def verify_pair_improvement():
    """Check the pair-replacement identity on six heterogeneous examples."""
    examples = [(['.9','.1','.2','.3'],[1,1,1,1]),
                (['.2']*4,[5,1,2,3]),
                (['.9','.8','.7','.6'],[1,2,3,4])]
    rows = []
    for rawq,rawu in examples:
        q,u = tuple(map(F,rawq)),tuple(map(F,rawu))
        ranked = sorted(range(4),key=lambda i:(-u[i]*q[i],i))
        for B in (2,3):
            a,b = ranked[:2]
            def policy(support,history,remaining):
                if not history:
                    return (a,b)
                mixed = history[0][1] == 1
                if mixed and len(history) == 1:
                    return (a,)
                tail = ranked[2:B] if mixed else ranked[2:B+1]
                offset = len(history)-(2 if mixed else 1)
                return (tail[offset],) if offset<len(tail) else None
            got = evaluate(policy,q,u,B,2,require_laminar=True)
            base = sum(q[i]*u[i] for i in ranked[:B])
            extra = (q[a]*q[b]+(1-q[a])*(1-q[b]))*q[ranked[B]]*u[ranked[B]]
            assert F(got['value']) == base+extra and extra>0
            rows.append({'q':list(map(str,q)),'u':list(map(str,u)),'B':B,
                         'best_all_individual':str(base),'pair_policy':got['value'],
                         'gain':str(extra)})
    return rows


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    q,u = (F(9,10),F(1,10)),(F(1),F(1))
    counter = {name:evaluate(fixed_plan([pool]),q,u,1,2,require_laminar=True)
               for name,pool in [('A',(0,)),('B',(1,)),('AB',(0,1))]}
    assert [counter[k]['value'] for k in ['A','B','AB']] == ['9/10','1/10','9/50']
    exact = BellmanPerfiles([1-x for x in q],u,2).solve(1)
    assert exact.value == F(9,10) and exact.first_pool == (0,)

    q4,u4 = (F(1,2),)*4,(F(1),)*4
    crossing = [(0,1,2),(0,1,3),(0,2,3)]
    signatures = {tuple(count(z,pool) for pool in crossing) for z in profiles(q4)}
    assert len(signatures)==16
    original = evaluate(fixed_plan(crossing),q4,u4,3,3)
    trimmed = evaluate(fixed_plan(trim_to_disjoint(crossing)),q4,u4,3,3,
                       require_laminar=True)
    repaired = evaluate(trimmed_with_reuse(crossing,u4,3),q4,u4,3,3,
                        require_laminar=True)
    replanned = evaluate(first_pool_then_ratio(crossing[0],q4,u4,3),q4,u4,3,3,
                         require_laminar=True)
    assert original['value']=='2' and trimmed['value']=='7/8' and repaired['value']=='3/2'
    assert replanned['value']=='7/4'
    opt = {}
    for name,lam in [('general',False),('laminar',True)]:
        solver = BellmanPerfiles([1-x for x in q4],u4,3,laminar=lam)
        sol = solver.solve(3)
        opt[name] = {'value':str(sol.value),'replay':solver.replay(3),
                     'first_pool':sol.first_pool,'states':sol.states}
    atom = SolverLaminar(dict(enumerate(1-x for x in q4)),dict(enumerate(u4)),3,'posterior_zero')
    assert atom.V(frozenset(range(4)),(),3) == F(opt['laminar']['value']) == F(7,4)
    assert opt['general']['value'] == original['value']

    # Every occurrence, including the two occurrences of A, has its own coin.
    outcomes = []
    for coins in product((0,1),repeat=4):
        pools = [tuple(i for i,keep in zip(pool,bits) if keep)
                 for pool,bits in zip([(0,1),(0,2)],[coins[:2],coins[2:]])]
        pools = [p for p in pools if p]
        legal = all(laminar_pair(a,b) for a,b in combinations(pools,2))
        outcomes.append({'coins':coins,'pools':pools,'laminar':legal})
    failure = F(sum(not x['laminar'] for x in outcomes),len(outcomes))
    assert failure == F(1,16)

    sources = ['augmented/bellman_perfiles.py','augmented/bm17_toy_solver.py','augmented/politica_ratio.py',
               'augmented/experimento_sesion_28.py']
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'model':{'q':'healthy probability','p':'infection probability',
                       'observation':'exact infection count','clearing':'soft',
                       'independent_priors':True,'laminar':'pathwise, every earlier pool'},
              'arithmetic':'exact rational enumeration; two laminar solvers agree',
              'source_sha256':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
              'counterexample':{'q':list(map(str,q)),'u':['1','1'],'B':1,'G':2,'policies':counter},
              'crossing_pilot':{'q':['1/2']*4,'u':['1']*4,'B':3,'G':3,
                  'original_pools':crossing,'trimmed_pools':trim_to_disjoint(crossing),
                  'distinct_signatures':len(signatures),'original':original,
                  'trimmed_without_reuse':trimmed,'trimmed_with_reuse':repaired,
                  'first_pool_then_ratio':replanned,
                  'optima':opt,'ratio_repair_to_general':'3/4',
                  'ratio_laminar_to_general':'7/8',
                  'scope':'One pilot on a fixed crossing plan; no approximation guarantee'},
              'half_thinning':{'path':[(0,1),(0,2)],'nonlaminar_probability':str(failure),
                               'outcomes':outcomes},
              'pair_replacement_checks':verify_pair_improvement()}
    (OUT/'evidencia_28.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print({'B1':{k:v['value'] for k,v in counter.items()},
           'crossing':original['value'],'laminar_opt':opt['laminar']['value'],
           'repair':repaired['value'],'replanned':replanned['value'],'thinning_nonlaminar':str(failure),
           'pair_identity_checks':len(report['pair_replacement_checks'])})
    return report


if __name__=='__main__':
    run()
