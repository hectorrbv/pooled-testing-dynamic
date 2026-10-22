"""Map 3: exact finite-population Bellman, with explicit failure reporting.

The published CBS number is a lower bound. We also evaluate a precisely
specified CBS policy: cover until the first root containing a healthy member;
bisect it, following the left child when both contain healthy members; stop
as soon as any healthy members are certified. This is one feasible policy,
not every possible extension of cover-then-bisect.
"""
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import comb
from pathlib import Path
from time import perf_counter
import json
import subprocess
import sys

from augmented.acid_test import k_from_budget
from augmented.bellman_tipos import crear_solver
from augmented.provenance import write_canonical_csv

ROOT = Path(__file__).resolve().parent.parent


def cbs_stop_value(q, G, B):
    """Exact rational expectation of the stop-at-first-certification policy."""
    q = Fraction(str(q))
    p = 1 - q
    k = k_from_budget(B, G)
    depth = (G - 1).bit_length()

    @lru_cache(None)
    def extract(m, r, b):
        if r == 0:
            return Fraction(m)
        if r == m or b == 0:
            return Fraction(0)
        size = m // 2
        val = Fraction(0)
        for s in range(max(0, r - m + size), min(size, r) + 1):
            pr = Fraction(comb(size, s) * comb(m-size, r-s), comb(m, r))
            gain = (size if s == 0 else 0) + (m-size if r-s == 0 else 0)
            if gain:
                continuation = gain
            elif s < size:
                continuation = extract(size, s, b-1)
            else:
                continuation = extract(m-size, r-s, b-1)
            val += pr * continuation
        return val

    first = sum((Fraction(comb(G, r)) * p**r * q**(G-r) * extract(G, r, depth)
                 for r in range(G)), Fraction(0))
    return first * sum((p**(G*t) for t in range(k)), Fraction(0))


def solve_case(n, G, B, q=0.05):
    solver = crear_solver(1-q, G, backend='compiled')
    start = perf_counter()
    try:
        opt = solver(n, (), B)
        sizes = [a[1] for a in solver.optimal_actions(n, (), B)]
        result = dict(status='exact_float64', optimo=opt,
                      tamanos_optimos=';'.join(map(str, sizes)), reason='')
    except RuntimeError as error:
        result = dict(status='state_limit', optimo=None,
                      tamanos_optimos='', reason=str(error))
    return dict(result, seconds=perf_counter()-start, states=len(solver.memo))


def bounded_case(n, G, B, seconds=120):
    """Separate process bounds memory lifetime and time of the anchor attempt."""
    try:
        proc = subprocess.run(
            [sys.executable, '-m', 'augmented.mapa_ancla', '--worker', str(n), str(G), str(B)],
            cwd=ROOT, capture_output=True, text=True, timeout=seconds,
        )
    except subprocess.TimeoutExpired:
        return dict(status='timeout', optimo=None, tamanos_optimos='',
                    reason=f'Exact computation exceeded {seconds}s; no optimum certified',
                    seconds=seconds, states=None)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def map3():
    rows, runtime = [], []
    for label, n, G, B in [('ancla', 48, 16, 7), ('frontera', 24, 8, 6),
                           ('ancla_acciones_G8', 48, 8, 7)]:
        print(f'Mapa 3: {label}, n={n}, G={G}, B={B}', flush=True)
        result = bounded_case(n, G, B)
        runtime.append(dict(case=label, seconds=result.pop('seconds'), states=result.pop('states')))
        q = Fraction(1, 20)
        k = k_from_budget(B, G)
        bound = 1-(1-q)**(k*G)
        policy = cbs_stop_value(q, G, B)
        rows.append(dict(case=label, q_sano=float(q), n=n, G=G, B=B, k=k,
                         convencion='posterior_zero', clase='pathwise_laminar',
                         cota_cbs=float(bound), cbs_stop=float(policy),
                         cbs_stop_fraction=str(policy), baseline_singletons=B*float(q), **result))
        print(result, flush=True)
    hashes = {p: sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ['augmented/acid_test.py', 'augmented/bellman_tipos.py',
               'augmented/bellman_tipos_compilado.py', 'augmented/mapa_ancla.py']}
    write_canonical_csv(ROOT/'results/mapa_homogeneo_3.csv', rows,
                        generator='augmented.mapa_ancla.map3', seed=None,
                        params=dict(source_sha256=hashes, convention='posterior_zero',
                                    n='finite: anchor 48, frontier 24',
                                    state_limit=2500000, timeout_seconds=120,
                                    cbs='cover until first non-all-infected root; bisect; prefer left child with healthy members; stop at first certification',
                                    backend='compiled exact-state float64'))
    (ROOT/'results/mapa_homogeneo_3.runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        print(json.dumps(solve_case(*map(int, sys.argv[2:5]))))
    else:
        map3()
