"""Notebook 30 in the format of the trimmed notebook 29: markdown cells, one figure each, short text.

All numbers are computed here, checked with asserts, saved as CSV under results/, and only then
written into the notebook. Run: python3 augmented/notebooks/build_30_pendientes.py
"""
from base64 import b64encode
from fractions import Fraction as F
from math import comb, log2
from pathlib import Path
import csv
import io
import os
import subprocess
import sys
import time

import numpy as np
import nbformat as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figuras_separacion_28 import INK, GRAY, BLUE, WARM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/sesion_2026-10-26/notebook_30'
BOOK = HERE / '30_experimentos_pendientes.ipynb'
TEXT = dict(ha='center', va='center')


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return buf.getvalue()


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)


# ------------------------------------------------------------------ 1. split of the potential
def phi(m, r):
    return F((m - 2 * r) ** 2, m)


def split_gain(m, r, k):
    total = F(0)
    for x in range(max(0, k - (m - r)), min(k, r) + 1):
        total += F(comb(r, x) * comb(m - r, k - x), comb(m, k)) * (phi(k, x) + phi(m - k, r - x) - phi(m, r))
    return total


def potencial():
    rows = []
    for m in range(2, 13):
        for r in range(m + 1):
            for k in range(1, m):
                gain = split_gain(m, r, k)
                assert gain == F(4 * r * (m - r), m * (m - 1)) and gain <= 2
                rows.append((m, r, k, gain))
    assert all(sum(F(comb(M, R), 2 ** M) * phi(M, R) for R in range(M + 1)) == 1 for M in range(1, 17))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    fig.subplots_adjust(left=.1, right=.86, bottom=.18, top=.95)
    for m, color in [(2, WARM), (4, BLUE), (12, INK)]:
        pts = [(r / m, float(g)) for (mm, r, k, g) in rows if mm == m and k == 1]
        ax.plot(*zip(*pts), 'o-', color=color, lw=1.6, markersize=4)
        ax.text(1.03, pts[len(pts) // 2][1] if m == 2 else max(p[1] for p in pts) - (0.12 if m == 12 else 0),
                f'm = {m}', color=color, fontsize=11, va='center')
    ax.axhline(2, color=GRAY, linestyle='--', lw=1)
    ax.text(0.0, 2.05, 'cota 2', color=GRAY, fontsize=10, va='bottom')
    ax.set(xlim=(0, 1), ylim=(0, 2.25), xticks=[0, .5, 1], yticks=[0, 1, 2],
           xlabel='infectados / tamaño del átomo', ylabel='salto esperado')
    clean(ax)
    return png(fig), len(rows)


# ------------------------------------------------------------------ 2. detecting designs
def pools_of(matrix):
    return [sum(int(bit) << i for i, bit in enumerate(row)) for row in matrix]


def identifies(pools, n):
    z = np.arange(1 << n, dtype=np.uint32)
    sig = np.stack([np.bitwise_count(z & np.uint32(p)) for p in pools], axis=1)
    return len(np.unique(sig, axis=0)) == (1 << n)


def search_design(n, B, G, trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        rows = [sum(1 << int(i) for i in rng.choice(n, size=int(rng.integers(1, G + 1)), replace=False))
                for _ in range(B)]
        if identifies(rows, n):
            return rows
    return None


def disenos():
    from augmented.experimento_matriz15_laminar import ExactHomogeneousLaminar, matrix15
    from augmented.autoresearch_logg.judge import evaluate, static
    det4 = pools_of([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1]])
    designs = {'detector de 4': (4, 3, det4),
               'dos detectores de 4': (8, 3, det4 + [p << 4 for p in det4])}
    for n in (5, 6, 7, 8):
        for B in range(3, n + 1):
            found = search_design(n, B, n)
            if found:
                designs[f'búsqueda n={n}'] = (n, max(int(p).bit_count() for p in found), found)
                break
    for G in (4,):
        for B in range(4, 9):
            found = search_design(8, B, G)
            if found:
                designs[f'búsqueda n=8, G≤{G}'] = (8, G, found)
                break
    m15 = pools_of(matrix15())
    designs['Söderberg–Shapiro'] = (15, 7, m15)
    designs['S–S más un individuo'] = (16, 7, m15 + [1 << 15])
    table = []
    for name, (n, G, pools) in designs.items():
        B = len(pools)
        assert identifies(pools, n) and max(int(p).bit_count() for p in pools) <= G
        general = evaluate(static([tuple(i for i in range(n) if p >> i & 1) for p in pools]), n, B, G)[0]
        assert general == F(n, 2)
        lam = ExactHomogeneousLaminar(G).value(n, B)
        assert lam <= general
        table.append(dict(diseno=name, n=n, B=B, G=G, laminar=str(lam), general=str(general),
                          razon=float(lam / general)))
    assert next(r for r in table if r['n'] == 15)['laminar'] == '203479/32768'
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'disenos_detectores.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.12, right=.97, bottom=.18, top=.95)
    for r in table:
        ax.plot(r['G'], r['razon'], 'o', color=BLUE, markersize=6)
        ax.annotate(f"n={r['n']}, B={r['B']}", (r['G'], r['razon']), textcoords='offset points',
                    xytext=(7, 0), fontsize=9, color=GRAY, va='center')
    ax.axhline(1, color=GRAY, linestyle=':', lw=1)
    ax.set(xlim=(2.5, 9.2), ylim=(.78, 1.02), xticks=[3, 4, 5, 6, 7, 8], yticks=[.8, .9, 1.0],
           xlabel='G', ylabel='óptimo laminar / (n/2)')
    clean(ax)
    return png(fig), table


# ------------------------------------------------------------------ 3. information lemma on the grid
def lema():
    from augmented.bellman_perfiles import BellmanPerfiles
    grid = list(csv.DictReader((ROOT / 'results/precio_q05_G/malla_2026-10-14_completa.csv').open()))

    def leaves(node, prob=F(1)):
        if 'branches' not in node:
            yield prob, len(node['healthy']); return
        for b in node['branches']:
            yield from leaves(b['next'], prob * F(b['probability']))

    rows = []
    for r in grid:
        n, B, G = int(r['n']), int(r['B']), int(r['G'])
        if n == 7 and G >= 4:
            continue
        solver = BellmanPerfiles([F(1, 2)] * n, [F(1)] * n, G, laminar=False)
        value = solver.solve(B).value
        assert value == F(r['unrestricted'])
        hojas = list(leaves(solver.tree(B)))
        K = len(hojas)
        assert all(m <= log2(1 / p) + 1e-12 for p, m in hojas)
        assert sum(p * m for p, m in hojas) == value and float(value) <= log2(K) + 1 and K <= (G + 1) ** B
        rows.append(dict(n=n, B=B, G=G, despejados=float(value), K=K, log2K=log2(K), cota=B * log2(G + 1)))
    with (OUT / 'lema_informacion_malla.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    margin = max(r['despejados'] - r['log2K'] for r in rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.12, right=.86, bottom=.18, top=.95)
    x = np.linspace(0, 7, 20)
    ax.plot(x, x + 1, color=WARM, linestyle='--', lw=1.2)
    ax.text(7.1, 8, 'log₂K + 1', color=WARM, fontsize=11, va='center')
    ax.plot(x, x, color=GRAY, linestyle=':', lw=1.2)
    ax.text(7.1, 7, 'log₂K', color=GRAY, fontsize=11, va='center')
    ax.plot([r['log2K'] for r in rows], [r['despejados'] for r in rows], 'o', color=BLUE, markersize=5)
    ax.set(xlim=(0, 7), ylim=(0, 8.3), xticks=[0, 2, 4, 6], yticks=[0, 2, 4, 6, 8],
           xlabel='log₂ del número de historiales K', ylabel='despejados esperados')
    clean(ax)
    return png(fig), rows, margin


# ------------------------------------------------------------------ 4. sweep for the signature
def barrido(seed=20261026, seconds=30):
    out = subprocess.run([sys.executable, 'augmented/barrido_proposicion_individuales.py', str(seed), str(seconds), '6', '4', '4'],
                         cwd=ROOT, env=dict(os.environ, PYTHONPATH='.'), capture_output=True, text=True)
    assert out.returncode == 0 and 'fails 0' in out.stdout, out.stdout[-400:] + out.stderr[-400:]
    (OUT / 'barrido_individuales.txt').write_text(out.stdout)
    counts = eval(out.stdout.split('counts ')[1].split(' fails')[0])
    ratio = out.stdout.split('interior: ')[1].split(' at')[0]
    return sum(counts.values()), ratio


def build():
    t0 = time.time()
    images = {}
    images['potencial'], tripletas = potencial()
    images['disenos'], table = disenos()
    images['lema'], rows, margin = lema()
    instancias, ratio = barrido()
    for name, content in images.items():
        (OUT / f'{name}.png').write_bytes(content)
    ss = next(r for r in table if r['n'] == 15)
    small = [r['razon'] for r in table if r['n'] <= 8]

    def cell(key, source, *figures):
        c = nb.v4.new_markdown_cell(source.strip(), id=key)
        c.attachments = {f'{name}.png': {'image/png': b64encode(images[name]).decode()} for name in figures}
        return c

    cells = [cell('p30-potencial', rf'''
## 1. Partir un átomo sube el potencial a lo sumo 2

$\phi(m,r)=(m-2r)^2/m$ por átomo; $q=1/2$, $u=1$.

![Salto esperado del potencial al partir un átomo de m personas con r infectadas; vale 4r(m−r)/(m(m−1)), no depende del tamaño de la prueba y toca 2 solo en m=2.](attachment:potencial.png)

Enumeración exacta de {tripletas} tripletas $(m,r,k)$ con $m\le12$: el salto es $4r(m-r)/(m(m-1))\le m/(m-1)$. Una prueba virgen sube el potencial exactamente 1. Con eso el telescopio da $V_{{\rm lam}}\le2B$.
''', 'potencial'),
             cell('p30-disenos', rf'''
## 2. Diseños que identifican a todos

$q=1/2$, $u=1$, $n\le16$; general $=n/2$ por construcción.

![Óptimo laminar entre n/2 para cada diseño detector, según G: entre {min(small):.2f} y {max(small):.2f} en los diseños chicos, {ss['razon']:.3f} en Söderberg–Shapiro con n=15, B=9, G=7.](attachment:disenos.png)

Cada diseño distingue los $2^n$ perfiles, comprobado perfil por perfil. Los de búsqueda aleatoria no son mínimos. En $n\le8$ el laminar conserva entre {min(small):.2f} y {max(small):.2f} de $n/2$; en $n=15$ conserva {ss['razon']:.3f}.
''', 'disenos'),
             cell('p30-lema', rf'''
## 3. El lema de información en la malla

{len(rows)} celdas con árbol óptimo general exacto ($n\le7$; las de $n=7$ con $G\ge4$ exceden el solver de perfiles).

![Despejados esperados contra log₂ del número de historiales: todas las celdas quedan bajo log₂K, y la cota del lema es log₂K + 1.](attachment:lema.png)

Hoja por hoja, despejar $m$ personas exige probabilidad $\le2^{{-m}}$. En promedio ninguna celda supera $\log_2K$; el margen frente a $\log_2K+1$ es {1 - margin:.1f} personas.
''', 'lema'),
             cell('p30-barrido', rf'''
## 4. Barrido de la nota "individuales nunca óptimas"

Semilla 20261026, 30 s, $n\le6$, $B\le4$, $G\le4$: {instancias} instancias, `fails 0`. Razón mínima óptimo laminar / mejor individual en el interior: {ratio[:8]}.

Falta la firma: correr los 934 tests y escribir la fila en la nota de encargos.
''')]
    # Section 5 comes verbatim from the protected base of notebook 29 (loop 1 design, n=16).
    base = nb.read(HERE / '29_tres_clases_y_contraejemplo.ipynb', as_version=4)
    loop1 = next(c for c in base.cells if c.id == 'logg-loop1')
    cells.append(nb.v4.new_markdown_cell(loop1.source.strip(), id='p30-loop1', attachments=loop1.attachments))
    book = nb.v4.new_notebook(cells=cells)
    book.metadata.update(language_info={'name': 'python'}, generator='build_30_pendientes.py')
    nb.validate(book)
    nb.write(book, BOOK)
    print(BOOK, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    build()
