"""Reproducible exact search for BIN > count/soft pathwise LAM.

Frozen evaluators, rational instances, per-case limits, append-only measurements.
Run locally or on a server; an unfinished or timed-out case is never an optimum.
This searches instances, not solver code. No language-model calls are needed.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json
import random
import signal
import time

from augmented.bellman_binario_exacto import BellmanBinario, replay_binary_tree
from augmented.bellman_perfiles import BellmanPerfiles

ROOT=Path(__file__).resolve().parents[1]


def timeout(signum, frame):
    raise TimeoutError('Per-case time limit; comparison not certified')


def measure(case, seconds=12, states=700_000):
    t=time.monotonic()
    n,B,G=case['n'],case['B'],case['G']
    p=[1-F(x) for x in case['q']]; u=list(map(F,case['u']))
    row=dict(case)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        bin_s=BellmanBinario(p,u,G,state_limit=states)
        bs=bin_s.solve(B)
        lam_s=BellmanPerfiles(p,u,G,laminar=True,state_limit=states)
        ls=lam_s.solve(B)
        row.update(status='exact',binary=str(bs.value),laminar=str(ls.value),
                   ratio=str(bs.value/ls.value),delta=str(bs.value-ls.value),
                   first_binary=bs.first_pool,first_laminar=ls.first_pool,
                   binary_states=bs.states,laminar_states=ls.states)
        if bs.value>ls.value:
            row['binary_tree']=bin_s.tree(B)
            row['binary_replay']=replay_binary_tree(row['binary_tree'],case['q'],u,B,G)
            row['laminar_tree']=lam_s.tree(B)
            row['laminar_replay']=lam_s.replay(B)
            assert F(row['binary_replay']['value'])==bs.value
    except (TimeoutError,RuntimeError) as exc:
        row.update(status='limited',reason=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    row['seconds']=round(time.monotonic()-t,6)
    return row


def canonical(case):
    pairs=sorted(zip(case['q'],case['u']),key=lambda t:(F(t[0]),F(t[1])))
    return (case['B'],case['G'],tuple(pairs))


def homogeneous():
    for n in [3,4,5,6]:
        for B in range(2,n):
            for G in range(2,n+1):
                for a in [11,13,15,17,19]:
                    yield dict(n=n,B=B,G=G,q=[str(F(a,20))]*n,u=['1']*n,phase='homogeneous')


def random_case(rng):
    n=rng.choice([4,5,5,6])
    q=[F(rng.choice([2,4,8,10,11,13,15,17,18,19]),20) for _ in range(n)]
    if all(x<=F(1,2) for x in q):
        q[0]=F(4,5)
    mode=rng.randrange(3)
    u=[F(1) if mode==0 else F(rng.choice([1,2,3,5,8,13])) if mode==1
       else F(rng.choice([1,2,4,8,16]),rng.choice([1,2,4])) for _ in range(n)]
    return dict(n=n,B=rng.choice([2,3]),G=rng.randint(2,n),q=list(map(str,q)),u=list(map(str,u)),phase='heterogeneous')


def mutate(rng,case):
    c={k:case[k] for k in ['n','B','G','q','u']}
    c['q'],c['u']=list(c['q']),list(c['u'])
    i=rng.randrange(c['n'])
    if rng.random()<.65:
        q=F(c['q'][i])+F(rng.choice([-4,-2,-1,1,2,4]),100)
        c['q'][i]=str(min(F(99,100),max(F(1,100),q)))
    else:
        c['u'][i]=str(F(c['u'][i])*rng.choice([F(1,2),F(3,4),F(5,4),F(2)]))
    c['phase']='adaptive'
    return c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/sesion_2026-10-12/busqueda')
    ap.add_argument('--seed',type=int,default=20261012)
    ap.add_argument('--random',type=int,default=600)
    ap.add_argument('--adaptive',type=int,default=150)
    ap.add_argument('--seconds',type=float,default=600)
    ap.add_argument('--case-seconds',type=float,default=10)
    ap.add_argument('--skip-grid',action='store_true')
    args=ap.parse_args()
    signal.signal(signal.SIGALRM,timeout)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'cases.jsonl').exists():
        raise SystemExit('Choose a fresh --out directory; evidence is never overwritten')
    rng=random.Random(args.seed);seen=set();rows=[];elite=[]
    started=time.monotonic(); exhausted=False
    def submit(c):
        nonlocal exhausted
        if time.monotonic()-started>=args.seconds:
            exhausted=True;return False
        key=canonical(c)
        if key in seen:return True
        seen.add(key)
        row=measure(c,min(args.case_seconds,max(.01,args.seconds-(time.monotonic()-started))))
        rows.append(row)
        with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if row['status']=='exact':
            elite.append(row)
            elite.sort(key=lambda r:(F(r['ratio']),r['n']),reverse=True)
            del elite[30:]
            if F(row['delta'])>0:
                (out/f'witness_{len(rows):04}.json').write_text(json.dumps(row,indent=2)+'\n')
                print('WITNESS '+json.dumps({k:row[k] for k in ['n','B','G','q','u','binary','laminar','ratio']}),flush=True)
        if len(rows)%50==0:
            print(json.dumps({'cases':len(rows),'exact':sum(r['status']=='exact' for r in rows),
                              'best_ratio':elite[0]['ratio'] if elite else None,
                              'seconds':round(time.monotonic()-started,1)}),flush=True)
        return True
    if not args.skip_grid:
        for c in homogeneous():
            if not submit(c):break
    if not exhausted:
        for _ in range(args.random):
            if not submit(random_case(rng)):break
    if not exhausted:
        for _ in range(args.adaptive):
            if not elite or not submit(mutate(rng,rng.choice(elite))):break
    exact=[r for r in rows if r['status']=='exact']
    sources=['augmented/bellman_binario_exacto.py','augmented/bellman_perfiles.py','augmented/busca_binario_vs_laminar.py']
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'parameters':vars(args),
            'scope':'Finite search; no universal dominance inferred from absence of a witness',
            'binary':'optimal dynamic, binary infection observation, hard clearing',
            'laminar':'optimal dynamic exact infection counts, soft clearing, pathwise laminar',
            'independent_priors':True,'cases':len(rows),'exact':len(exact),
            'limited':len(rows)-len(exact),'time_budget_exhausted':exhausted,
            'binary_wins':sum(F(r['delta'])>0 for r in exact),
            'laminar_wins':sum(F(r['delta'])<0 for r in exact),
            'ties':sum(F(r['delta'])==0 for r in exact),
            'best':elite[:5],'seconds':time.monotonic()-started,
            'source_sha256':{s:sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['best','source_sha256']}),flush=True)


if __name__=='__main__':main()
