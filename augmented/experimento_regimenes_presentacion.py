"""Extend the six-policy comparison to q=.05,.10,...,.95 (n=6).

Uses the original notebook's evaluator and score verbatim, without executing
its plotting or export cells. Run: python -m augmented.experimento_regimenes_presentacion
"""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json
import math
import time

import nbformat
import pandas as pd

from augmented.bm17_toy_solver import SolverLaminar
from augmented.densidad_companion import PoliticasDensidad
from augmented.indice_lagrangiano import PoliticaLagrangiana
from augmented.politica_ratio import PoliticaRatio
from augmented.evolucion_scores import PoliticaInducida, compila
from augmented.provenance import write_canonical_csv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'augmented/notebooks/27_mapa4_greedies_con_planeacion.ipynb'
OUT = ROOT / 'results/sesion_2026-09-29'
POLICIES = ['pi_M', 'pi_C', 'pi_R', 'C3', 'pi_L', 'pi_ratio']


def evaluator_source():
    nb = nbformat.read(SOURCE, as_version=4)
    cell = next(c.source for c in nb.cells if 'C3_SRC =' in c.source)
    return cell.split('t0 = time.time()')[0]


def one_probability(q):
    namespace = dict(globals())
    exec(compile(evaluator_source(), str(SOURCE), 'exec'), namespace)
    rows = [namespace['evalua_celda'](q, G, B)
            for B in (2, 3) for G in (2, 3, 4)]
    # Class-level memoization otherwise retains previous policy instances.
    for method in [PoliticaLagrangiana._I_atomos, PoliticaLagrangiana._I_virgen,
                   PoliticaLagrangiana.valor, PoliticaRatio.valor]:
        method.cache_clear()
    return rows


def refined_lambda_scan():
    p = {i: Fraction(4, 5) for i in range(6)}
    u = {i: 1 for i in p}
    values = sorted(set([.001,.01,.05,.1,.2,.4,.8]
                        + [i/100 for i in range(20,41)]))
    rows = []
    for lam in values:
        policy = PoliticaLagrangiana(p, u, 4, lam=lam, horizonte=3)
        action = policy.decide(frozenset(p), (), 3)
        value = policy.valor(frozenset(p), (), 3)
        rows.append({'lambda': lam, 'first_pool_size': len(action[1]),
                     'value': value, 'opt': .91072,
                     'relative_value': value/.91072})
        for method in [policy._I_atomos, policy._I_virgen, policy.valor]:
            method.cache_clear()
    write_canonical_csv(OUT/'lambda_refinado.csv', rows,
        generator='augmented.experimento_regimenes_presentacion.refined_lambda_scan',
        seed=None, params={'n':6, 'B':3, 'G':4, 'q':.2, 'u':1,
            'clearing':'soft', 'planning_limit':3, 'no_paralisis':True,
            'lambda_values':values,
            'motivation':'Fill interval .2-.4 missed by previous seven-point scan',
            'source_sha256':{'augmented/indice_lagrangiano.py':
                sha256((ROOT/'augmented/indice_lagrangiano.py').read_bytes()).hexdigest()}})
    return rows


def main():
    started = time.time()
    qs = [i/20 for i in range(1, 20)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        batches = list(pool.map(one_probability, qs))
    frame = pd.DataFrame([r for batch in batches for r in batch])
    assert len(frame) == 114
    assert frame[POLICIES].min().min() >= 0
    assert frame[POLICIES].max().max() <= 1+1e-9
    frame['decisiones'] = frame[['tam_'+p for p in POLICIES]].nunique(axis=1)
    old = pd.read_csv(ROOT/'results/mapa4_razones.csv')
    joined = old.merge(frame, on=['q', 'B', 'G'], suffixes=('_old', '_new'))
    assert len(joined) == 36
    errors = {p: float((joined[p+'_old']-joined[p+'_new']).abs().max())
              for p in POLICIES+['opt']}
    assert max(errors.values()) < 1e-9, errors
    files = ['augmented/bm17_toy_solver.py', 'augmented/densidad_companion.py',
             'augmented/indice_lagrangiano.py', 'augmented/politica_ratio.py',
             'augmented/evolucion_scores.py']
    source_hashes = {f: sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    write_canonical_csv(OUT/'regimenes_politicas.csv', frame.to_dict('records'),
        generator='augmented.experimento_regimenes_presentacion', seed=None,
        params={'n': 6, 'q': qs, 'B': [2,3], 'G': [2,3,4], 'u': 1,
                'clearing': 'soft', 'history': 'empty', 'independent_priors': True,
                'lambda': .01, 'alpha': 1, 'planning_limit': 3,
                'method': 'exact finite-state evaluation, no sampling',
                'score_and_evaluator_sha256': sha256(evaluator_source().encode()).hexdigest(),
                'source_sha256': source_hashes})
    audit = {'rows': len(frame), 'reproduced_old_rows': len(joined),
             'max_old_grid_error': max(errors.values()),
             'runtime_seconds': time.time()-started,
             'optimal_cases': {p: int((frame[p]>=1-1e-9).sum()) for p in POLICIES},
             'ratio_vs_L': {'ratio_wins': int((frame.pi_ratio>frame.pi_L+1e-9).sum()),
                            'L_wins': int((frame.pi_L>frame.pi_ratio+1e-9).sum()),
                            'ties': int(((frame.pi_ratio-frame.pi_L).abs()<=1e-9).sum())}}
    (OUT/'regimenes_verificacion.json').write_text(json.dumps(audit,indent=2)+'\n')
    refined_lambda_scan()
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
