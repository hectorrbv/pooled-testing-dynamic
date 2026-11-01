"""Instance optimization for BIN/LAM; float proposals, rational certification.

For B=2 use a separate closed-form evaluator. With a first pool A, a laminar
second test can be outside A or inside A. A test containing A is dominated by
testing its new members, since the count of A is known. Thus this enumeration
also covers the full pathwise laminar menu, not only an atom heuristic.
Run independent small exact solvers to certify every retained candidate.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT=Path(__file__).resolve().parents[1]


class TwoTests:
    def __init__(self,n,G):
        self.n,self.G=n,G
        masks=np.arange(1<<n,dtype=np.int64)
        self.bits=((masks[:,None]>>np.arange(n))&1)
        self.groups=[]
        for k in range(1,n+1):
            ids=masks[self.bits.sum(axis=1)==k]
            bit=np.array([(int(a)&-int(a)).bit_length()-1 for a in ids])
            parent=ids^(1<<bit)
            self.groups.append((ids,bit,parent))
        self.pools=masks[(self.bits.sum(axis=1)>0)&(self.bits.sum(axis=1)<=G)]
        self.union=self.pools[:,None]|self.pools[None,:]
        self.disjoint=(self.pools[:,None]&self.pools[None,:])==0
        subs=[];rest=[];valid=[]
        width=max(1,(1<<G)-2)
        for a in self.pools:
            choices=[s for s in range(1,int(a)) if s&a==s]
            subs.append(choices+[0]*(width-len(choices)))
            rest.append([int(a)^s for s in choices]+[0]*(width-len(choices)))
            valid.append([1]*len(choices)+[0]*(width-len(choices)))
        self.subs=np.array(subs);self.rest=np.array(rest);self.valid=np.array(valid,dtype=bool)

    def values(self,q,u):
        q=np.asarray(q);u=np.asarray(u)
        z=np.zeros((1<<self.n,self.n+1));z[0,0]=1
        for ids,bit,parent in self.groups:
            z[ids,:]=q[bit,None]*z[parent,:]
            z[ids,1:]+=(1-q[bit,None])*z[parent,:-1]
        Q=z[:,0];U=self.bits@u;Qp=Q[self.pools];Up=U[self.pools]
        outside=np.max(np.where(self.disjoint,(Qp*Up)[None,:],0),axis=1)
        positive=np.max(Up[None,:]*(Qp[None,:]-Q[self.union]),axis=1)
        binary=float(np.max(Qp*(Up+outside)+positive))
        refinement=(U[self.subs]*Q[self.subs])[...,None]*z[self.rest,:self.G+1]
        refinement+=(U[self.rest]*Q[self.rest])[...,None]*z[self.subs,:self.G+1]
        refinement=np.where(self.valid[...,None],refinement,0)
        best_ref=np.max(refinement,axis=1);best_ref[:,0]=0
        laminar=float(np.max(Qp*Up+np.maximum(z[self.pools,:self.G+1]*outside[:,None],best_ref).sum(axis=1)))
        return binary,laminar


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=3);ap.add_argument('--G',type=int,default=2)
    ap.add_argument('--mode',choices=['equal','weighted','homogeneous'],default='weighted')
    ap.add_argument('--seed',type=int,default=1);ap.add_argument('--iterations',type=int,default=300)
    ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--restarts',type=int,default=3)
    ap.add_argument('--out',required=True)
    args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    judge=TwoTests(args.n,args.G);start=time.monotonic();calls=0;best=0;rows=[]
    def unpack(x):
        if args.mode=='homogeneous':return np.repeat(x[0],args.n),np.ones(args.n)
        q=x[:args.n]
        u=np.exp(np.r_[x[args.n:],0]) if args.mode=='weighted' else np.ones(args.n)
        return q,u/u.max()
    bounds=[(.0001,.9999)]*(1 if args.mode=='homogeneous' else args.n)
    if args.mode=='weighted':bounds += [(-5,5)]*(args.n-1)
    def objective(x):
        nonlocal calls,best
        calls+=1;q,u=unpack(x);b,l=judge.values(q,u);ratio=b/l
        if ratio>best+1e-10:
            best=ratio
            row=dict(n=args.n,B=2,G=args.G,mode=args.mode,seed=args.seed,
                     q=q.tolist(),u=u.tolist(),ratio=ratio,binary=b,laminar=l,calls=calls,seconds=time.monotonic()-start)
            rows.append(row)
            with (out/'improvements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            (out/'best.json').write_text(json.dumps(row,indent=2)+'\n')
        return -ratio
    completed=0
    for repeat in range(args.restarts):
        remaining=args.seconds-(time.monotonic()-start)
        if remaining<=0:break
        result=differential_evolution(objective,bounds,seed=args.seed+repeat*1009,
            maxiter=args.iterations,popsize=10,tol=1e-9,polish=False,
            callback=lambda x,c:time.monotonic()-start>args.seconds,
            updating='immediate')
        completed+=1
        if time.monotonic()-start<args.seconds:
            minimize(objective,result.x,method='Nelder-Mead',bounds=bounds,
                     options=dict(maxiter=500,xatol=1e-9,fatol=1e-10))
    report=dict(parameters=vars(args),created_utc=datetime.now(timezone.utc).isoformat(),
                calls=calls,completed_restarts=completed,seconds=time.monotonic()-start,
                best=rows[-1],status='floating search candidate, requires rational certification',
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
