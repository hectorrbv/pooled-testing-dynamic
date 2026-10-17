"""Bounded remote search: vary only the generator; keep judge and solvers fixed.

Run in an isolated copy, never in an interactive working checkout. This is a
deterministic family-search controller, not an unattended language-model agent.
Every candidate record is provisional until the independent human-side review.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXED = [HERE / 'judge.py', HERE / 'run_bounded.py',
         ROOT / 'augmented/bellman_perfiles.py',
         ROOT / 'augmented/dp_perfiles_binario_conteo.py']

# Appended to the original generator. The original sampler still handles
# seed replay and duplicate suppression; only the sampling family is changed.
VARIANT = '''
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
'''


def stamp():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in FIXED}


def rows(mission):
    path = HERE / f'results_{mission}.tsv'
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=int, default=40)
    ap.add_argument('--seconds', type=int, default=180)
    ap.add_argument('--hard-seconds', type=int, default=360)
    ap.add_argument('--memory-mb', type=int, default=1024)
    ap.add_argument('--run-id', default='20261015-hermeneuta')
    args = ap.parse_args()
    if not 1 <= args.runs <= 40 or not 0 < args.seconds < args.hard_seconds <= 360:
        ap.error('require 1<=runs<=40 and 0<seconds<hard-seconds<=360')
    output = HERE / ('night_' + args.run_id)
    output.mkdir(exist_ok=False)
    base = (HERE / 'generator.py').read_text()
    fixed = hashes()
    seeds = json.loads((HERE / 'warm_start.json').read_text())
    best = {m: min(v, key=lambda s:F(s['ratio'])) for m,v in seeds.items()}
    kept = {m: base for m in seeds}
    state = {'started_utc':stamp(), 'pid':os.getpid(), 'status':'running',
             'configuration':vars(args), 'fixed_sha256':fixed,
             'completed_experiments':0, 'best':best,
             'interpretation':'Finite search records; independent review still required.'}

    def save():
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(output / 'status.json')

    def execute(name,mission,nmax,source,seconds,description):
        if hashes() != fixed:
            raise RuntimeError('Fixed judge or solver hash changed')
        (HERE / 'generator.py').write_text(source)
        (output / (name+'_generator.py')).write_text(source)
        old_rows = len(rows(mission))
        state['current'] = {'name':name,'mission':mission,'nmax':nmax,
                            'description':description,'started_utc':stamp()}
        save()
        print(json.dumps(state['current']),flush=True)
        command = [sys.executable,'-u','-m','augmented.autoresearch_ejemplos.run_bounded',
                   mission,'--seconds',str(seconds),'--nmax',str(nmax),
                   '--hard-seconds',str(args.hard_seconds),'--memory-mb',str(args.memory_mb),
                   '--description',description]
        start = time.monotonic()
        with (output/(name+'.log')).open('w') as log:
            proc = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,PYTHONUNBUFFERED='1'))
        new_rows = rows(mission)[old_rows:]
        row = new_rows[-1] if new_rows else None
        outcome = 'limited' if proc.returncode == 124 else (
            'ok' if proc.returncode == 0 and row and row['status']=='ok' else 'fail')
        event = dict(state['current'], finished_utc=stamp(), status=outcome,
                     exit_code=proc.returncode, elapsed_seconds=round(time.monotonic()-start,2),
                     result=row)
        with (output/'experiments.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        state['last_result'] = event
        if hashes() != fixed:
            raise RuntimeError('Fixed code changed during experiment')
        return event

    try:
        # This is a feasibility control, not a new discovery. If it is too
        # expensive, do not repeat the same n=7 seed in every later run.
        pilot = execute('pilot_n7','precio',7,base,1,
                        'Feasibility only: replay existing 41/52 seed with production limits')
        state['n7_pilot'] = pilot['status']
        if pilot['status'] == 'fail':
            raise RuntimeError('n7 pilot failed a gate; inspect log')
        price_nmax = 7 if pilot['status']=='ok' else 6
        for i in range(args.runs):
            mission = 'binario' if i%2==0 else 'precio'
            j = i//2
            modes = ['probabilities','utilities','joint','extra_person'] if mission=='binario' else [
                'homogeneous','one_person','heterogeneous']
            mode = modes[j%len(modes)]
            denominator = [20,100,1000,10000][j%4] if mission=='binario' else [100,1000,10000][(j//3)%3]
            nmax = 6 if mission=='binario' else price_nmax
            current_best = best[mission] if best[mission]['n'] <= nmax else None
            config = {'mission':mission,'mode':mode,'denominator':denominator,
                      'seed':2026101500+i,'best':current_best}
            source = base+'\nNIGHT_CONFIG = '+repr(config)+'\n'+VARIANT
            desc = f'run {i+1}/40; {mode}; step 1/{denominator}; seed {config["seed"]}; nmax {nmax}'
            event = execute(f'{i+1:02d}_{mission}',mission,nmax,source,args.seconds,desc)
            improved = False
            if event['status']=='fail':
                (HERE/'generator.py').write_text(kept[mission])
                raise RuntimeError('Judge gate or process failed; stop and inspect log')
            if event['status']=='limited' and mission=='precio':
                price_nmax = 6
            if event['status']=='ok':
                row=event['result']; ratio=F(row['laminar'])/F(row['other'])
                if ratio < F(best[mission]['ratio']):
                    improved = True
                    best[mission] = {k:int(row[k]) for k in ('n','B','G')}
                    best[mission].update(q=ast.literal_eval(row['q']),u=ast.literal_eval(row['u']),
                        laminar=row['laminar'],other=row['other'],ratio=str(ratio),
                        source=f'{args.run_id}/{i+1:02d}_{mission}')
                    kept[mission] = source
                    (output/f'generator_best_{mission}.py').write_text(source)
            (HERE/'generator.py').write_text(kept[mission])
            state.update(completed_experiments=i+1,best=best,price_nmax=price_nmax)
            save()
            with (output/'notes.md').open('a') as f:
                f.write(f'- {stamp()} | {desc} | {event["status"]} | improvement={improved}\n')
        state['status']='completed'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (HERE/'generator.py').write_text(base)
        state['finished_utc']=stamp()
        save()


if __name__=='__main__':
    main()
