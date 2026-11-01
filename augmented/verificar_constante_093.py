"""Reproduce the registered 0.930703 candidate, not the entire search corpus.

Inputs and lambda grid are frozen from dapts-autoresearch/instancia_constante.py
and run_constante.py, inspected 2026-09-21. Uses the same BM17 initial state and
policy implementations as that historical measurement.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json

from augmented.bm17_toy_solver import SolverLaminar
from augmented.densidad_companion import PoliticasDensidad
from augmented.evolucion_scores import PoliticaInducida, compila
from augmented.indice_lagrangiano import PoliticaLagrangiana
from augmented.provenance import write_canonical_csv

ROOT = Path(__file__).resolve().parent.parent
P = ['0', '.02722', '0', '.975', '.975', '.975', '.51236']
U = ['1.0457', '.9536', '.9588', '38.6689', '40', '39.8024', '1.0837']
B, G = 3, 4


def run():
    p, u = dict(enumerate(map(F, P))), dict(enumerate(map(F, U)))
    initial = frozenset(p)
    opt = SolverLaminar(p, u, G, 'posterior_zero').V(initial, (), B)
    scale = float(sum((1-p[i])*u[i] for i in p))/B
    factors, factor = [], .02
    while factor <= 2+1e-9:
        factors.append(factor)
        factor *= 1.4
    lambdas = sorted({round(scale*f, 6) for f in factors} | {.001})
    rows = []

    def record(name, value, lam=None):
        rows.append(dict(policy=name, lambda_value=lam, value=float(value),
                         opt_laminar=float(opt), ratio=float(value)/float(opt)))

    density = PoliticasDensidad(p, u, G)
    for name, key in [('pi_M', 'inmediato'), ('pi_C', 'committed'), ('pi_R', 'receding')]:
        record(name, density.valor(key, B))
    note = ROOT/'docs/notes/2026-09-01-mision-vhat-C3.md'
    score = compila(note.read_text().split('```python\n', 1)[1].split('```', 1)[0])
    record('C3', PoliticaInducida(p, u, G, score, B).V(initial, (), B))
    for lam in lambdas:
        policy = PoliticaLagrangiana(p, u, G, lam=lam, horizonte=3, no_paralisis=True)
        record('pi_L', policy.valor(initial, (), B), lam)
    best = max(rows, key=lambda row: row['value'])
    assert abs(best['ratio']-.9307030280342034) < 1e-10
    assert best['lambda_value'] == 1.238469
    sources = ['augmented/verificar_constante_093.py', 'augmented/bm17_toy_solver.py',
               'augmented/densidad_companion.py', 'augmented/indice_lagrangiano.py',
               'augmented/evolucion_scores.py', str(note.relative_to(ROOT))]
    write_canonical_csv(ROOT/'results/verificacion_constante_093.csv', rows,
        generator='augmented.verificar_constante_093.run', seed=None,
        params=dict(p_infection=P, u=U, B=B, G=G, convention='posterior_zero',
                    initial_state='BM17 historical harness: all 7 people virgin, no atoms',
                    lambdas=lambdas, grid_factor=1.4, horizon=3, no_paralisis=True,
                    opt_exact=str(opt), scope='one registered candidate; not a minimum over 7000 cases',
                    selection='maximize policy expected value before execution, independently of OPT',
                    source_sha256={s:sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}))
    print(json.dumps(best, ensure_ascii=False))
    return rows


if __name__ == '__main__':
    run()
