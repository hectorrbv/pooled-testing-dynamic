"""Bounded exact search across budgets, stdlib-only worker for the VPS."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json,random,signal,time
from augmented.bellman_binario_exacto import BellmanBinario
from augmented.bellman_perfiles import BellmanPerfiles


def timeout(*args):raise TimeoutError('case time limit')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=480)
    ap.add_argument('--seed',type=int,default=20261013);ap.add_argument('--out',required=True)
    ap.add_argument('--states',type=int,default=160000);ap.add_argument('--case-seconds',type=float,default=4)
    ap.add_argument('--warm');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(args.seed);t0=time.monotonic();rows=[];elite=[];seen=set()
    signal.signal(signal.SIGALRM,timeout)
    def submit(c):
        if time.monotonic()-t0>=args.seconds:return False
        q=tuple(map(F,c['q']));u=tuple(map(F,c['u']));key=(c['B'],c['G'],tuple(sorted(zip(q,u))))
        if key in seen:return True
        seen.add(key);row=dict(c);start=time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,min(args.case_seconds,max(.01,args.seconds-(start-t0))))
        try:
            b=BellmanBinario([1-x for x in q],u,c['G'],state_limit=args.states).solve(c['B'])
            l=BellmanPerfiles([1-x for x in q],u,c['G'],laminar=True,state_limit=args.states).solve(c['B'])
            row.update(status='exact',binary=str(b.value),laminar=str(l.value),ratio=str(b.value/l.value),
                       binary_states=b.states,laminar_states=l.states)
        except (TimeoutError,RuntimeError,MemoryError) as e:row.update(status='limited',reason=str(e))
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        row['seconds']=time.monotonic()-start;rows.append(row)
        with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if row['status']=='exact':
            old=float(F(elite[0]['ratio'])) if elite else 0
            elite.append(row);elite.sort(key=lambda r:F(r['ratio']),reverse=True);del elite[30:]
            if float(F(row['ratio']))>old+1e-12:
                (out/'best.json').write_text(json.dumps(row,indent=2)+'\n')
                print(json.dumps(dict(event='best',**row)),flush=True)
        if len(rows)%100==0:print(json.dumps(dict(event='progress',cases=len(rows),seconds=time.monotonic()-t0)),flush=True)
        return True
    if args.warm:
        for c in json.loads(Path(args.warm).read_text()):
            for B in [2,3,4]:
                if B>=len(c['q']):continue
                trial=dict(c,B=B,phase='warm')
                if not submit(trial):break
    # Symmetric carrier family: one nearly healthy agent reused by the binary tree.
    warm=[]
    for n,B,G in [(4,3,2),(5,3,2),(5,3,3),(6,3,2),(6,3,3),(6,4,2),(7,3,2),(7,4,2),(8,3,2)]:
        for q in ['1/4','2/5','1/2','3/5','3/4','9/10']:
            for u in ['1/4','1/2','1','2','4']:
                warm.append(dict(n=n,B=B,G=G,q=['9999/10000']+[q]*(n-1),u=[u]+['1']*(n-1),phase='carrier'))
    rng.shuffle(warm)
    # Alternate structure, fresh heterogeneous instances and mutation throughout.
    attempts=0
    while time.monotonic()-t0<args.seconds:
        attempts+=1
        if attempts%3==1 and warm:c=warm.pop()
        elif attempts%3==0 and elite:
            parent=rng.choice(elite);c={k:parent[k] for k in ['n','B','G','q','u']}
            c['q']=list(c['q']);c['u']=list(c['u']);c['phase']='mutation'
            i=rng.randrange(c['n'])
            if rng.random()<.65:
                x=F(c['q'][i])+rng.choice([F(-1,10),F(-1,50),F(1,50),F(1,10)])
                c['q'][i]=str(min(F(9999,10000),max(F(1,10000),x)))
            else:c['u'][i]=str(F(c['u'][i])*rng.choice([F(1,2),F(3,4),F(5,4),F(2)]))
        else:
            n=rng.choice([4,5,5,6,6,7,8]);B=rng.choice([2,3,3,4]);B=min(B,n-1)
            G=rng.choice([2,2,3,3,4]);G=min(G,n)
            q=[rng.choice(['1/10','1/4','2/5','1/2','3/5','3/4','9/10','19/20','99/100','9999/10000']) for _ in range(n)]
            u=['1']*n if rng.random()<.4 else [rng.choice(['1/4','1/2','1','2','4','8']) for _ in range(n)]
            c=dict(n=n,B=B,G=G,q=q,u=u,phase='random')
        if not submit(c):break
    root=Path(__file__).resolve().parents[1]
    source=['augmented/busca_brecha_presupuestos.py','augmented/bellman_binario_exacto.py','augmented/bellman_perfiles.py']
    report=dict(parameters=vars(args),created_utc=datetime.now(timezone.utc).isoformat(),cases=len(rows),
                counts=dict(Counter(r['status'] for r in rows)),by_phase=dict(Counter(r['phase'] for r in rows)),
                seconds=time.monotonic()-t0,best=elite[:10],
                source_sha256={p:sha256((root/p).read_bytes()).hexdigest() for p in source})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['best','source_sha256']}),flush=True)


if __name__=='__main__':main()
