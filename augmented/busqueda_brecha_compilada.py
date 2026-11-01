"""Optimize positive product priors on a fixed exhaustive Bellman DAG.

The original exact reference solvers generate the complete reachable topology.
Only numerical evaluation is compiled; candidate optima still need Fraction
certification. All q must be strictly between 0 and 1: supports are fixed.
"""
import argparse,json,time
from datetime import datetime,timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution,minimize
from augmented.bellman_binario_exacto import BellmanBinario
from augmented.bellman_perfiles import BellmanPerfiles

ROOT=Path(__file__).resolve().parents[1]


@njit(cache=True)
def evaluate_dag(q,u,support_offsets,profiles,state_support,health,state_actions,action_children,children):
    n=len(q);w=np.ones(1<<n);utilities=np.zeros(1<<n)
    for z in range(1<<n):
        for i in range(n):
            if z&(1<<i):w[z]*=1-q[i];utilities[z]+=u[i]
            else:w[z]*=q[i]
    masses=np.zeros(len(support_offsets)-1)
    for j in range(len(masses)):
        for k in range(support_offsets[j],support_offsets[j+1]):masses[j]+=w[profiles[k]]
    values=np.zeros(len(state_support))
    for j in range(len(values)):
        best=masses[state_support[j]]*utilities[health[j]]
        for a in range(state_actions[j],state_actions[j+1]):
            value=0.
            for k in range(action_children[a],action_children[a+1]):value+=values[children[k]]
            best=max(best,value)
        values[j]=best
    return values[-1]


class FixedDAG:
    def __init__(self,n,B,G,kind):
        self.n,self.B,self.G,self.kind=n,B,G,kind
        if kind=='binary':s=BellmanBinario([F(1,2)]*n,[1]*n,G,state_limit=300000)
        else:s=BellmanPerfiles([F(1,2)]*n,[1]*n,G,laminar=True,state_limit=300000)
        s.solve(B);states=list(s.memo);ids={x:j for j,x in enumerate(states)}
        supports={key[1]:0 for key in states};supports={x:j for j,x in enumerate(supports)}
        state_support=[];health=[];sa=[0];ac=[0];children=[]
        for j,(b,support,extra) in enumerate(states):
            state_support.append(supports[support])
            if kind=='binary':health.append(extra)
            else:health.append(s._info(support)[2])
            if b:
                for k,a in enumerate(s.pools):
                    if kind=='binary':
                        if a&extra:continue
                        neg=support&s.negative[k]
                        if not neg:continue
                        child=[(b-1,neg,extra|a)]
                        if support^neg:child.append((b-1,support^neg,extra))
                    else:
                        if not extra&(1<<k):continue
                        parts=[support&m for m in s.outcomes[k] if support&m]
                        if len(parts)<=1:continue
                        child=[(b-1,c,extra&s.compat[k]) for c in parts]
                    ci=[ids[x] for x in child];assert all(k<j for k in ci)
                    children.extend(ci);ac.append(len(children))
            sa.append(len(ac)-1)
        so=[0];profiles=[]
        for support in supports:
            while support:
                bit=support&-support;profiles.append(bit.bit_length()-1);support^=bit
            so.append(len(profiles))
        self.arrays=tuple(np.asarray(x,dtype=np.int64) for x in [so,profiles,state_support,health,sa,ac,children])
        self.states=len(states);self.actions=len(ac)-1
        assert abs(self.value([.5]*n,[1]*n)-float(s.solve(B).value))<1e-10

    def value(self,q,u):
        return evaluate_dag(np.asarray(q),np.asarray(u,dtype=float),*self.arrays)


def main():
    ap=argparse.ArgumentParser()
    for name,default in [('n',4),('B',3),('G',2),('seed',1),('iterations',250),('restarts',3)]:ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--mode',choices=['weighted','equal','homogeneous'],default='weighted')
    ap.add_argument('--seconds',type=float,default=100);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    try:binary=FixedDAG(args.n,args.B,args.G,'binary');laminar=FixedDAG(args.n,args.B,args.G,'laminar')
    except RuntimeError as exc:
        (out/'limited.json').write_text(json.dumps(dict(parameters=vars(args),reason=str(exc))))
        print(json.dumps(dict(event='limited',parameters=vars(args),reason=str(exc))),flush=True);return
    prepared=time.monotonic()-t0;calls=0;best=0;bestrow=None;start=time.monotonic()
    def unpack(x):
        q=np.repeat(x[0],args.n) if args.mode=='homogeneous' else x[:args.n]
        u=np.exp(np.r_[x[args.n:],0]) if args.mode=='weighted' else np.ones(args.n)
        return q,u/u.max()
    bounds=[(.0001,.9999)]*(1 if args.mode=='homogeneous' else args.n)
    if args.mode=='weighted':bounds += [(-5,5)]*(args.n-1)
    def objective(x):
        nonlocal calls,best,bestrow
        calls+=1;q,u=unpack(x);b=binary.value(q,u);l=laminar.value(q,u);ratio=b/l
        if ratio>best+1e-10:
            best=ratio
            bestrow=dict(n=args.n,B=args.B,G=args.G,mode=args.mode,q=q.tolist(),u=u.tolist(),
                         binary=b,laminar=l,ratio=ratio,calls=calls,seconds=time.monotonic()-start)
            with (out/'improvements.jsonl').open('a') as f:f.write(json.dumps(bestrow)+'\n')
            (out/'best.json').write_text(json.dumps(bestrow,indent=2)+'\n')
        return -ratio
    for r in range(args.restarts):
        if time.monotonic()-start>args.seconds:break
        result=differential_evolution(objective,bounds,seed=args.seed+r*1009,maxiter=args.iterations,
            popsize=8,tol=1e-8,polish=False,callback=lambda x,c:time.monotonic()-start>args.seconds)
        if time.monotonic()-start<args.seconds:
            minimize(objective,result.x,method='Nelder-Mead',bounds=bounds,options={'maxiter':400})
    report=dict(parameters=vars(args),created_utc=datetime.now(timezone.utc).isoformat(),calls=calls,
                seconds=time.monotonic()-start,preparation_seconds=prepared,
                binary_states=binary.states,laminar_states=laminar.states,best=bestrow,
                status='floating search candidate, requires rational certification',
                source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                    'augmented/busqueda_brecha_compilada.py','augmented/bellman_binario_exacto.py','augmented/bellman_perfiles.py']})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
