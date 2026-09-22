"""Compiled exact-state DP for homogeneous finite or saturated populations.

Same transitions as bellman_tipos, no search truncation. A sorted multiset of
atoms is packed in 7-bit slots. Supports G<=16, B<=8. Float64 arithmetic.
Numba is optional and imported only when this backend is explicitly selected.
"""
import numpy as np
from math import comb
from numba import njit, types
from numba.typed import Dict


@njit(cache=True)
def insert(code, item):
    if item == 0:
        return code
    low, shift = 0, 0
    while code and (code & 127) < item:
        low |= (code & 127) << shift
        code >>= 7
        shift += 7
    return low | (item << shift) | (code << (shift+7))


@njit(cache=True)
def remove(code, item):
    low, shift = 0, 0
    while (code & 127) != item:
        low |= (code & 127) << shift
        code >>= 7
        shift += 7
    return low | ((code >> 7) << shift)


@njit(cache=True)
def value(code, virgins, b, G, opening, gain, starts, ends, rewards, first, last,
          probabilities, child1, child2, immediate, memo):
    if b == 0:
        return 0.0
    virgins=min(virgins,b*G)
    if b == 1:
        best = np.max(gain[:min(G,virgins)+1])
        todo = code
        while todo:
            best = max(best, immediate[todo & 127])
            todo >>= 7
        return best
    key = (code << 12) | (virgins << 4) | b
    if key in memo:
        return memo[key]
    if len(memo)>=2500000:
        raise RuntimeError('Exact-state limit 2500000 reached; no optimum certified')
    best = 0.0
    for k in range(1,min(G,virgins)+1):
        base = value(code,virgins-k,b-1,G,opening,gain,starts,ends,rewards,first,last,
                     probabilities,child1,child2,immediate,memo)
        val = gain[k]+(opening[k,0]+opening[k,k])*base
        for r in range(1,k):
            atom=(k-1)*(k-2)//2+r
            val += opening[k,r]*value(insert(code,atom),virgins-k,b-1,G,opening,gain,
                       starts,ends,rewards,first,last,probabilities,child1,child2,immediate,memo)
        best=max(best,val)
    todo, previous = code, 0
    while todo:
        atom=todo & 127; todo >>= 7
        if atom==previous:
            continue
        previous=atom
        rest=remove(code,atom)
        for a in range(starts[atom],ends[atom]):
            val=rewards[a]
            for j in range(first[a],last[a]):
                child=insert(insert(rest,child1[j]),child2[j])
                val += probabilities[j]*value(child,virgins,b-1,G,opening,gain,starts,ends,
                      rewards,first,last,probabilities,child1,child2,immediate,memo)
            best=max(best,val)
    memo[key]=best
    return best


def prepare(q,G):
    if not 0<q<1 or not 1<=G<=16:
        raise ValueError('Compiled saturation backend requires 0<q<1, 1<=G<=16')
    count=G*(G-1)//2+G
    opening=np.zeros((G+1,G+1));gain=np.zeros(G+1)
    starts=np.zeros(count+1,dtype=np.int64);ends=starts.copy();immediate=np.zeros(count+1)
    rewards=[];first=[];last=[];probabilities=[];child1=[];child2=[]
    for k in range(1,G+1):
        for r in range(k+1):
            opening[k,r]=comb(k,r)*(1-q)**r*q**(k-r)
        gain[k]=k*opening[k,0]
    def atom(m,r):
        return (m-1)*(m-2)//2+r if 0<r<m else 0
    for m in range(2,G+1):
        for r in range(1,m):
            a=atom(m,r);starts[a]=len(rewards)
            for size in range(1,m//2+1):
                first.append(len(probabilities));reward=0.
                for s in range(max(0,r-m+size),min(size,r)+1):
                    pr=comb(size,s)*comb(m-size,r-s)/comb(m,r)
                    reward+=pr*((size if s==0 else 0)+(m-size if r-s==0 else 0))
                    probabilities.append(pr);child1.append(atom(size,s));child2.append(atom(m-size,r-s))
                last.append(len(probabilities));rewards.append(reward)
                immediate[a]=max(immediate[a],reward)
            ends[a]=len(rewards)
    return (G,opening,gain,starts,ends,np.array(rewards,dtype=np.float64),
            np.array(first,dtype=np.int64),np.array(last,dtype=np.int64),
            np.array(probabilities,dtype=np.float64),np.array(child1,dtype=np.int64),
            np.array(child2,dtype=np.int64),immediate)


class CompiledSolver:
    def __init__(self,p_inf,G,u=1.):
        self.args=prepare(1-float(p_inf),G)
        self.G,self.u=G,float(u)
        self.memo=Dict.empty(key_type=types.int64,value_type=types.float64)
        self.argmax={}

    def __call__(self,n,atoms,b):
        if n < 0 or not 0<=b<=8 or len(atoms)+b>8:
            raise ValueError('Require n>=0, b<=8 and len(atoms)+b<=8')
        code=0
        for m,r in sorted(atoms):
            if not 0<r<m<=self.G:
                raise ValueError('Invalid atom')
            code=int(insert(code,(m-1)*(m-2)//2+r))
        return self.u*value(code,min(n,b*self.G),b,*self.args,self.memo)

    def action_values(self,n,atoms,b):
        if atoms:
            raise ValueError('This public action menu exposes the virgin root only')
        if b<=0:
            return ()
        self(n,atoms,b)
        _,opening,gain,*_=self.args
        out=[]
        for k in range(1,min(n,self.G)+1):
            v=gain[k]+(opening[k,0]+opening[k,k])*value(0,n-k,b-1,*self.args,self.memo)
            for r in range(1,k):
                a=(k-1)*(k-2)//2+r
                v+=opening[k,r]*value(a,n-k,b-1,*self.args,self.memo)
            out.append((('open',k),v*self.u))
        return tuple(out)

    def optimal_actions(self,n,atoms,b,tol=1e-10):
        options=self.action_values(n,atoms,b)
        best=max((v for _,v in options),default=0.)
        actions=tuple(a for a,v in options if abs(v-best)<=tol*max(1.,abs(best)))
        self.argmax[n,atoms,b]=actions
        return actions
