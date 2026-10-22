"""Reproduce the small posterior-zero comparison used in the study notebooks.

Run from the repository root: python -m augmented.experimento_precio_laminar
Writes a CSV, exact fractions, decision trees, and source hashes to results/.
No external services, simulation or parameter search are used.
"""

import csv
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.densidad_companion import PoliticasDensidad
from augmented.evolucion_scores import PoliticaInducida, compila
from augmented.indice_lagrangiano import PoliticaLagrangiana

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'precio_laminar_2026-09-21'


def homogeneous(id, label, n, q, B, G, history=()):
    return dict(id=id, label=label, p=[str(1-F(q))]*n, u=['1']*n,
                B=B, G=G, history=history)


CASES = [
    homogeneous('individuales', 'Control: solo individuales', 3, '.3', 2, 1),
    homogeneous('par_B2', 'Ejemplo conocido: dos pruebas', 4, '.3', 2, 2),
    homogeneous('par_B3', 'Misma población: tres pruebas', 4, '.3', 3, 2),
    homogeneous('AB1_b1', 'AB tiene conteo 1; queda una prueba', 4, '.3', 1, 2, [((0, 1), 1)]),
    homogeneous('AB1_b2', 'AB tiene conteo 1; quedan dos pruebas', 4, '.3', 2, 2, [((0, 1), 1)]),
    homogeneous('cuatro_triples', 'Cuatro personas y grupos de tres', 4, '.5', 3, 3),
    homogeneous('seis_homogeneo', 'Seis personas homogéneas', 6, '.3', 3, 3),
    dict(id='seis_heterogeneo', label='Contraejemplo de la batería de cuatro',
         p=['.9', '.825', '.875', '.8', '.95', '.85'],
         u=['2', '1', '1', '1', '4', '2'], B=3, G=4, history=[]),
]


def atom_value(case):
    p, u = dict(enumerate(map(F, case['p']))), dict(enumerate(map(F, case['u'])))
    a = SolverLaminar(p, u, case['G'], 'posterior_zero')
    U, atoms = frozenset(p), ()
    # The selected conditional examples have one interior-count observation.
    assert len(case['history']) <= 1
    for pool, r in case['history']:
        assert 0 < r < len(pool)
        U -= set(pool)
        atoms += ((tuple(pool), r),)
    start = perf_counter()
    value = a.V(U, atoms, case['B'])
    return value, perf_counter()-start, len(a.memo)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for c in CASES:
        models = {}
        for name, lam in [('laminar', True), ('general', False)]:
            s = BellmanPerfiles(c['p'], c['u'], c['G'], laminar=lam)
            v = s.solve(c['B'], c['history'])
            models[name] = dict(value=str(v.value), first_pool=v.first_pool,
                seconds=v.seconds, states=v.states,
                replay=s.replay(c['B'], c['history']), tree=s.tree(c['B'], c['history']))
        av, at, states = atom_value(c)
        lv, gv = F(models['laminar']['value']), F(models['general']['value'])
        assert lv == av and gv >= lv
        models['atom_solver'] = dict(value=str(av), seconds=at, states=states)
        row = dict(id=c['id'], caso=c['label'], n=len(c['p']), B_restante=c['B'],
            G=c['G'], historia=json.dumps(c['history']), convencion='posterior_zero',
            opt_laminar=float(lv), opt_general=float(gv),
            laminar_exacto=str(lv), general_exacto=str(gv),
            brecha=float(gv-lv), ratio_laminar_general=float(lv/gv),
            perdida_relativa=1-float(lv/gv),
            primera_laminar=''.join('ABCDEFGHIJ'[i] for i in models['laminar']['first_pool'] or ()),
            primera_general=''.join('ABCDEFGHIJ'[i] for i in models['general']['first_pool'] or ()))
        rows.append(row)
        details.append(dict(instance=c, **models))
        print(c['id'], str(lv), str(gv), f'ratio={float(lv/gv):.6f}', flush=True)

    c = CASES[-1]
    p, u = dict(enumerate(map(F, c['p']))), dict(enumerate(map(F, c['u'])))
    lv, gv = F(details[-1]['laminar']['value']), F(details[-1]['general']['value'])
    d = PoliticasDensidad(p, u, c['G'])
    policies = []
    for key, label in [('inmediato', 'π_M'), ('committed', 'π_C'), ('receding', 'π_R')]:
        w = d.valor(key, c['B'])
        policies.append(dict(policy=label, value=w,
                             ratio_laminar=w/float(lv), ratio_general=w/float(gv)))
    note = ROOT/'docs/notes/2026-09-01-mision-vhat-C3.md'
    source = note.read_text().split('```python\n', 1)[1].split('```', 1)[0]
    score = compila(source)
    c3 = PoliticaInducida(p, u, c['G'], score, c['B'])
    w = c3.V(frozenset(p), (), c['B'])
    policies.append(dict(policy='C3', value=float(w), exact=str(w),
                         ratio_laminar=float(w/lv), ratio_general=float(w/gv)))
    lam_values = []
    for lam in [.001, .1, .5, 1., 2.]:
        pol = PoliticaLagrangiana(p, u, c['G'], lam=lam, horizonte=3, no_paralisis=True)
        w = pol.valor(frozenset(p), (), c['B'])
        data = dict(policy='π_L', lambda_value=lam, horizon=3, no_paralisis=True,
                    value=w, ratio_laminar=w/float(lv), ratio_general=w/float(gv),
                    first_action=pol.decide(frozenset(p), (), c['B']))
        lam_values.append(data)
        if lam == .001:
            policies.append(data)

    # Direct, non-Bellman certificate: three fixed crossing triples identify
    # all four bits. This attains the all-healthy-utility upper bound at n=4.
    signatures = []
    for z in range(16):
        a,b,c,d = ((z >> i) & 1 for i in range(4))
        counts = (a+b+c, a+b+d, a+c+d)
        signatures.append(dict(profile=[a,b,c,d], counts=counts))
    assert len({tuple(x['counts']) for x in signatures}) == 16

    files = ['augmented/bellman_perfiles.py', 'augmented/bm17_toy_solver.py',
             'augmented/densidad_companion.py', 'augmented/indice_lagrangiano.py',
             'augmented/evolucion_scores.py', 'augmented/experimento_precio_laminar.py',
             str(note.relative_to(ROOT))]
    artifact = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_modified=True,
        source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        model=dict(p='infection probability', q='health probability',
                   result='exact infection count', clearance='posterior_zero',
                   laminar='pathwise compatibility with every earlier pool',
                   B='remaining tests, hard cap on every branch'),
        arithmetic='Integer-scaled exact Bellman; policy decisions use existing floating-point scores',
        scope='Eight chosen examples, not an exhaustive worst-case search',
        cases=details, policy_six_person=policies, lambda_sensitivity=lam_values,
        four_person_static_certificate=signatures)
    (OUT/'resultados.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False)+'\n')
    with (OUT/'comparacion.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Saved', OUT)
    return artifact, rows


if __name__ == '__main__':
    run()
