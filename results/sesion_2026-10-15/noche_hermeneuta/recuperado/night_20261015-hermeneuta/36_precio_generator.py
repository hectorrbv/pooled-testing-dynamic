"""EDITABLE generator, seeded from the certified examples of October 12–14.

GROUPCOUNT_MISSION selects binario or precio. Initial draws replay the saved
seeds; later draws explore neighboring families and fresh instances. No solver
or judge changes are permitted during the experiment loop.
"""
from fractions import Fraction as F
import json
import os
from pathlib import Path

_calls = {}
_seen = {}
_seeds = json.loads(Path(__file__).with_name("warm_start.json").read_text())

def sample(rng):
    mission=os.environ.get("GROUPCOUNT_MISSION","binario")
    nmax=int(os.environ.get("GROUPCOUNT_NMAX","6"))
    if mission not in ("binario","precio"):
        raise ValueError("GROUPCOUNT_MISSION must be binario or precio")
    seeds=[s for s in _seeds[mission] if s["n"] <= nmax and s["G"] <= 4]
    call=_calls.get(mission,0);_calls[mission]=call+1
    if call < len(seeds):
        s=seeds[call]
        inst=s["n"],s["B"],s["G"],list(map(F,s["q"])),list(map(F,s["u"]))
        _seen.setdefault(mission,set()).add(_key(inst))
        return inst
    if mission=="precio" and os.environ.get("GROUPCOUNT_PRICE_Q_HALF_ONLY","0")=="1":
        raise ValueError("The homogeneous q=1/2 domain up to n=7, B=4, G=4 is already covered; expand the experiment before searching it again")
    while True:
        inst=_candidate(rng,mission,nmax)
        key=_key(inst)
        seen=_seen.setdefault(mission,set())
        if key not in seen:
            seen.add(key)
            return inst


def _key(inst):
    n,B,G,q,u=inst
    return n,B,G,tuple(q),tuple(u)


def _candidate(rng,mission,nmax):
    if mission=="binario" and nmax>=4 and rng.random()<.75:
        s=rng.choice([F(19,20),F(99,100),F(9999,10000),F(999999,1000000)])
        q=[s,F(1,2),F(1,2),s];u=list(map(F,[1,3,3,2]))
        if rng.random()<.5:
            i=rng.randrange(4);q[i]=min(F(999999,1000000),max(F(1,1000000),q[i]+rng.choice([-1,1])*F(1,rng.choice([20,100,1000]))))
        else:
            i=rng.randrange(4);u[i]*=rng.choice([F(4,5),F(9,10),F(11,10),F(5,4)])
        return 4,2,2,q,u
    n=rng.randint(3,nmax);B=rng.randint(2,min(4,n));G=rng.randint(2,min(4,n))
    if mission=="precio":
        # The homogeneous half-probability grid is complete. Explore its
        # probability neighborhood instead of rediscovering its optimum.
        B=min(B,n-1)
        q0=F(1,2)+rng.choice([-1,1])*F(rng.randint(1,99),1000)
        q=[q0]*n
        if rng.random()<.5:
            q=[F(1,2)]*n
            q[rng.randrange(n)]=q0
        return n,B,G,q,[F(1)]*n
    q=[F(rng.randint(1,19),20) for _ in range(n)]
    u=[F(1)]*n if rng.random()<.5 else [F(rng.randint(1,8),4) for _ in range(n)]
    return n, B, G, q, u

NIGHT_CONFIG = {'mission': 'precio', 'mode': 'heterogeneous', 'denominator': 10000, 'seed': 2026101535, 'best': None}

import random as _night_random
_night_base_sample = sample
_night_rng = _night_random.Random(NIGHT_CONFIG['seed'])
if NIGHT_CONFIG.get('best'):
    _best = NIGHT_CONFIG['best']
    _same_seed = lambda s: all(s[k] == _best[k] for k in ('n','B','G','q','u'))
    _seeds[NIGHT_CONFIG['mission']] = [_best] + [
        s for s in _seeds[NIGHT_CONFIG['mission']] if not _same_seed(s)]

def sample(rng):
    return _night_base_sample(_night_rng)

def _candidate(rng, mission, nmax):
    mode = NIGHT_CONFIG['mode']
    delta = F(1, NIGHT_CONFIG['denominator'])
    clip = lambda x: min(F(999999,1000000), max(F(1,1000000), x))
    if mission == 'binario':
        seed = next(s for s in _seeds[mission] if s['n'] <= nmax)
        n, B, G = seed['n'], seed['B'], seed['G']
        q, u = list(map(F,seed['q'])), list(map(F,seed['u']))
        if mode in ('probabilities', 'joint', 'extra_person'):
            for i in rng.sample(range(n), rng.randint(1,min(3,n))):
                step = delta*(rng.randint(1,4)+F(rng.randint(1,9999),10000))
                q[i] = clip(q[i] + rng.choice([-1,1])*step)
        if mode in ('utilities','joint','extra_person'):
            for i in rng.sample(range(n), rng.randint(1,min(3,n))):
                u[i] *= F(rng.randint(5000,20000),10000)
        if mode == 'extra_person' and n < nmax:
            n += 1; q.append(F(rng.randint(1,19),20)); u.append(F(rng.randint(1,12),4))
            G = rng.choice([2,3])
        return n,B,G,q,u
    n = rng.choice([min(5,nmax),nmax])
    B,G = (3,3) if n <= 5 else (4,3)
    offset = rng.choice([-1,1])*(rng.randint(1,9)+F(rng.randint(1,9999),10000))*delta
    q = [F(1,2)]*n
    if mode == 'homogeneous':
        q = [clip(F(1,2)+offset)]*n
    elif mode == 'one_person':
        q[rng.randrange(n)] = clip(F(1,2)+offset)
    else:
        for i in rng.sample(range(n), rng.randint(2,n)):
            step = (rng.randint(1,9)+F(rng.randint(1,9999),10000))*delta
            q[i] = clip(F(1,2)+rng.choice([-1,1])*step)
    return n,B,G,q,[F(1)]*n
