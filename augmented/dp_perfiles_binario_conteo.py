"""Exact profile-set DP for small instances (IA support, 2026-10-12; NOT yet validated by B — validated against 5 known values: 387/500, 1137/1000, 2 (aug) and 3/5, 9/10 (binary, Thm 3.3(i))). channel='binary' (positive iff any
infected) or 'count'. Unrestricted pools of size <= G, posterior-zero clearing
(healthy in every consistent profile). Exact Fractions. For laminar-augmented we
reuse augmented.bellman_perfiles.BellmanPerfiles(laminar=True)."""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import prod

class DP:
    def __init__(self, q, u, G, channel):
        self.n=len(q); self.q=[Fraction(x) for x in q]; self.u=[Fraction(x) for x in u]
        self.G=G; self.ch=channel
        n=self.n
        # profile z: bit i = infected
        self.prof=list(range(1<<n))
        self.pz=[prod((1-self.q[i]) if z>>i&1 else self.q[i] for i in range(n)) for z in self.prof]
        self.pools=[sum(1<<i for i in c) for k in range(1,min(G,n)+1) for c in combinations(range(n),k)]
    def _healthy_all(self, supp):  # mask of people healthy in every profile of supp
        m=(1<<self.n)-1
        z=supp
        while z:
            low=z&-z; idx=low.bit_length()-1; m&=~idx; z^=low
        return m
    @lru_cache(maxsize=None)
    def V(self, supp, b, cleared):
        if b==0: return Fraction(0)
        best=Fraction(0)
        for T in self.pools:
            groups={}
            z=supp
            while z:
                low=z&-z; idx=low.bit_length()-1; z^=low
                cnt=bin(idx&T).count('1')
                key=(cnt>0) if self.ch=='binary' else cnt
                groups[key]=groups.get(key,0)|low
            val=Fraction(0); ptot=self.P(supp)
            for key,sub in groups.items():
                pr=self.P(sub)/ptot
                if pr==0: continue
                h=self._healthy_all(sub); new=h&~cleared
                r=sum(self.u[i] for i in range(self.n) if new>>i&1)
                val+=pr*(r+self.V(sub,b-1,cleared|new))
            if val>best: best=val
        return best
    @lru_cache(maxsize=None)
    def P(self, supp):
        return sum(self.pz[i] for i in self._bits(supp))
    @staticmethod
    def _bits(z):
        while z:
            low=z&-z; yield low.bit_length()-1; z^=low
    def value(self, B):
        return self.V((1<<(1<<self.n))-1, B, 0)   # total prob = 1

def Lambda(q, G):
    n=len(q); best=Fraction(0)
    for k in range(1,min(G,n)+1):
        for c in combinations(range(n),k):
            s=sum(prod(Fraction(q[j]) for j in c if j!=i) for i in c)
            best=max(best,s)
    return best
