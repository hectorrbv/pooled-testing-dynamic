"""Independent value checks for the described policies, and artifact validation.

Run from the repository root. The policy replay imports no project solver.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from pathlib import Path
import base64
import io
import json
import math
import re

import nbformat
from PIL import Image
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def policy_value(n, q, budget, cap, choose):
    states = list(product((0, 1), repeat=n))
    value = F(0)
    for real in states:
        history, compatible = [], states
        while (pool := choose(history)) is not None:
            assert len(history) < budget and 0 < len(pool) <= cap
            count = sum(real[i] for i in pool)
            history.append((pool, count))
            compatible = [s for s in compatible if sum(s[i] for i in pool) == count]
        assert real in compatible
        healthy_known = sum(all(s[i] == 0 for s in compatible) for i in range(n))
        weight = math.prod((1-q if infected else q) for infected in real)
        value += weight * healthy_known
    return value


def pair(history):
    if not history:
        return (0, 1)
    if len(history) == 1:
        return (0,) if history[0][1] == 1 else (2,)


def trim(history):
    if not history:
        return (0, 1, 2)
    if len(history) == 1:
        return (3,)
    if len(history) == 2 and history[0][1] in (1, 2):
        return (0,)


def replan(history):
    if not history:
        return (0, 1, 2)
    if len(history) == 1:
        return (0,) if history[0][1] in (1, 2) else (3,)
    if len(history) == 2 and history[0][1] in (1, 2):
        return (1,) if history[0][1] - history[1][1] == 1 else (3,)


values = {
    'pair_n6': policy_value(6, F(3, 10), 2, 2, pair),
    'simple_conversion': policy_value(4, F(1, 2), 3, 3, trim),
    'replan_after_ABC': policy_value(4, F(1, 2), 3, 3, replan),
    'crossing_three_counts': policy_value(
        4, F(1, 2), 3, 3,
        lambda h: [(0, 1, 2), (0, 1, 3), (0, 2, 3)][len(h)] if len(h) < 3 else None),
}
assert values == {'pair_n6': F(387, 500), 'simple_conversion': F(3, 2),
                  'replan_after_ABC': F(7, 4), 'crossing_three_counts': F(2)}
evidence_path = ROOT/'results/diseno_mapa_2026-10-05/agrupar_vs_individuales.json'
evidence = json.loads(evidence_path.read_text())
for source, digest in evidence['source_sha256'].items():
    assert sha256((ROOT/source).read_bytes()).hexdigest() == digest
rows = evidence['rows']
assert len(rows) == 228
assert sum(F(r['gain']) > 0 for r in rows) == 170
assert sum(F(r['gain']) == 0 for r in rows) == 58
interior = [r for r in rows if 2 <= r['B'] <= 5]
assert len(interior) == 152 and all(F(r['gain']) > 0 for r in interior)
for r in rows:
    q = F(r['q'])
    assert F(r['individual']) == r['B']*q
    assert F(r['gain']) == F(r['grouping']) - F(r['individual'])
    assert math.isclose(r['gain_percent'], float(100*F(r['gain'])/F(r['individual'])))
    if r['B'] == 1:
        assert F(r['grouping']) == max(k*q**k for k in range(1, r['G']+1))
    if r['B'] == 6:
        assert F(r['grouping']) == 6*q
sample = next(r for r in rows if r['G'] == 2 and r['B'] == 2 and r['q'] == '3/10')
assert F(sample['grouping']) == values['pair_n6'] and sample['gain_percent'] == 29

comparison = json.loads((OUT/'tres_referencias_28.json').read_text())
assert comparison['reference_sha256'] == sha256(evidence_path.read_bytes()).hexdigest()
assert comparison['summary']['cases'] == 228
assert comparison['summary']['general_beats_laminar'] == 110
original = {(r['G'], r['B'], r['q']): r for r in rows}
for r in comparison['rows']:
    old = original[r['G'], r['B'], r['q']]
    assert r['individual'] == old['individual'] and r['general'] == old['grouping']
    ind, lam, gen = map(F, [r['individual'], r['laminar'], r['general']])
    assert ind <= lam <= gen
    assert math.isclose(r['laminar_gain_percent'], float(100*(lam/ind-1)))
    assert math.isclose(r['crossing_loss_percent'], float(100*(1-lam/gen)))
worst = comparison['summary']['worst_laminar_case']
assert (worst['n'], worst['B'], worst['G'], worst['q']) == (6,4,3,'1/2')
assert (worst['individual'], worst['laminar'], worst['general']) == ('2','5/2','3')

figures = json.loads((OUT/'datos_figuras_28.json').read_text())
for name, field in [('laminar_vs_individuales', 'laminar_gain_percent'),
                    ('general_vs_laminar', 'crossing_loss_percent')]:
    figure = figures[name]
    for row in comparison['rows']:
        value = figure['values_by_G'][str(row['G'])][row['B']-1][figure['q'].index(row['q_decimal'])]
        assert value == row[field]

artifacts = {}
for name in ['28_pooling_y_precio_laminar', '28_notas_de_estudio']:
    path = ROOT/'augmented/notebooks'/f'{name}.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) == (6 if name == '28_pooling_y_precio_laminar' else 15)
    images = 0
    for cell in nb.cells:
        assert '**' not in cell.source and 'posterior-zero' not in cell.source
        for attachment in re.findall(r'attachment:([^\s)]+)', cell.source):
            png = base64.b64decode(cell.attachments[attachment]['image/png'])
            Image.open(io.BytesIO(png)).verify()
            assert png == (OUT/'figures'/attachment).read_bytes()
            images += 1
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)\)', cell.source):
            if not target.startswith(('https:', 'http:', 'attachment:')):
                assert (path.parent/target).resolve().exists(), target
    for source, digest in nb.metadata.evidence_sha256.items():
        assert sha256((ROOT/source).read_bytes()).hexdigest() == digest
    assert images == (3 if name == '28_pooling_y_precio_laminar' else 9)
    html, _ = HTMLExporter().from_notebook_node(nb)
    (OUT/f'{name}.html').write_text(html)
    artifacts[name] = {'cells': len(nb.cells), 'embedded_images': images,
                       'sha256': sha256(path.read_bytes()).hexdigest()}

report = {'independent_policy_values': {k: str(v) for k, v in values.items()},
          'grouping_cases': 228, 'improvements': 170, 'ties': 58,
          'strict_interior_improvements': 152, 'artifacts': artifacts,
          'matched_three_reference_cases': len(comparison['rows']),
          'general_beats_laminar': 110, 'general_ties_laminar': 118,
          'note': 'Value replay is independent; stored Bellman results establish optimality.'}
(OUT/'verificacion_presentacion.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
