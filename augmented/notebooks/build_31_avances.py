"""Notebook 31: the exact laminar rate c*(G, q) and what it changes. Same format as notebooks 29-30.

Every number is computed here, checked with asserts, saved under results/, then written into
markdown cells with one figure each. Run: python3 augmented/notebooks/build_31_avances.py
"""
from base64 import b64encode
from fractions import Fraction as F
from math import log2
from pathlib import Path
import csv
import io
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
from augmented.tasa_laminar_bloque import rate, rate_float  # noqa: E402
from augmented.experimento_matriz15_laminar import ExactHomogeneousLaminar, matrix15  # noqa: E402

OUT = ROOT / 'results/sesion_2026-11-01/notebook_31'
BOOK = HERE / '31_tasa_laminar.ipynb'
OUT.mkdir(parents=True, exist_ok=True)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return buf.getvalue()


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ 1. the rate
def tasa():
    rows = []
    for G in range(1, 21):
        c, M, _, _ = rate(G)
        rows.append(dict(G=G, c=float(c), exacto=f'{c.numerator}/{c.denominator}', M=M))
        assert M == G and c <= 2
    for G in (24, 32, 48, 64, 96):
        c, M = rate_float(G)
        rows.append(dict(G=G, c=c, exacto='', M=M))
        assert M == G and c <= 2
    assert [r['exacto'] for r in rows[:4]] == ['1/2', '2/3', '9/13', '14/19']
    assert all(a['c'] < b['c'] for a, b in zip(rows, rows[1:]))
    write_csv('tasa_laminar_q05.csv', rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.subplots_adjust(left=.1, right=.8, bottom=.17, top=.95)
    Gs = np.array([r['G'] for r in rows]); cs = np.array([r['c'] for r in rows])
    xs = np.arange(1, 97)
    ax.plot(xs, np.log2(xs + 1), color=GRAY, linestyle=':', lw=1.4)
    ax.text(99, log2(97), 'techo log₂(G+1)', color=GRAY, fontsize=10, va='center')
    ax.axhline(2, color=WARM, linestyle='--', lw=1.2)
    ax.text(99, 2, 'potencial de Francisco: 2', color=WARM, fontsize=10, va='center')
    ax.plot(Gs, cs, 'o-', color=BLUE, lw=1.8, markersize=4)
    ax.text(99, cs[-1], 'laminar exacto c*(G)', color=BLUE, fontsize=10, va='center')
    ax.set(xlim=(0, 97), ylim=(0, 7), xticks=[1, 16, 32, 48, 64, 96], yticks=[0, 1, 2, 4, 6],
           xlabel='G', ylabel='sanos acreditados por prueba')
    clean(ax)
    return png(fig), rows


# ------------------------------------------------------------------ 2. second way
def segunda_via():
    rows = []
    for G in (2, 3, 4):
        c = rate(G)[0]
        for B in range(2, 13, 2):
            v = ExactHomogeneousLaminar(G).value(B * G, B)
            assert v <= c * B, (G, B, v)
            rows.append(dict(G=G, B=B, n=B * G, V_lam=str(v), por_prueba=float(v / B), c=float(c)))
    write_csv('segunda_via.csv', rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    fig.subplots_adjust(left=.1, right=.82, bottom=.18, top=.95)
    for G, color in ((2, WARM), (3, GRAY), (4, BLUE)):
        sub = [r for r in rows if r['G'] == G]
        ax.plot([r['B'] for r in sub], [r['por_prueba'] for r in sub], 'o-', color=color, lw=1.6, markersize=4)
        ax.axhline(sub[0]['c'], color=color, linestyle='--', lw=1)
        ax.text(12.4, sub[0]['c'], f'c*({G})', color=color, fontsize=10, va='center')
    ax.set(xlim=(1.5, 12.2), ylim=(.6, .76), xticks=[2, 4, 6, 8, 10, 12], yticks=[.6, .65, .7, .75],
           xlabel='presupuesto B (con n = B·G personas)', ylabel='óptimo laminar exacto / B')
    clean(ax)
    return png(fig), rows


# ------------------------------------------------------------------ 3. the rule
def regla(G=8):
    _, _, rule, _ = rate(G)
    for m in range(2, G + 1):
        for r in range(1, m):
            k = rule[m, r]
            assert (k is None) == (2 * r > m), (m, r, k)        # abandon iff more than half infected
            assert k is None or k <= (m + 1) // 2                 # tested part is at most about half
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=.12, right=.98, bottom=.15, top=.95)
    for m in range(2, G + 1):
        for r in range(1, m):
            k = rule[m, r]
            color = BLUE if k else GRAY
            ax.add_patch(plt.Rectangle((r - .45, m - .45), .9, .9, color=color, alpha=.18 if k else .1, lw=0))
            ax.text(r, m, str(k) if k else '×', ha='center', va='center', fontsize=10, color=color)
    ax.set(xlim=(.4, G - .4), ylim=(1.4, G + .6), xticks=range(1, G), yticks=range(2, G + 1),
           xlabel='infectados conocidos r', ylabel='tamaño del átomo m')
    clean(ax)
    return png(fig)


# ------------------------------------------------------------------ 4. general against laminar
def general():
    def identifies(pools, n):
        z = np.arange(1 << n, dtype=np.uint32)
        sig = np.stack([np.bitwise_count(z & np.uint32(p)) for p in pools], axis=1)
        return len(np.unique(sig, axis=0)) == (1 << n)
    m15 = [sum(int(b) << i for i, b in enumerate(row)) for row in matrix15()]
    det4 = [0b0111, 0b1011, 0b1101]
    assert identifies(m15, 15) and identifies(det4, 4)
    points = [dict(diseno='detector de 4', G=3, n=4, B=3, por_prueba=4 / 2 / 3),
              dict(diseno='Söderberg–Shapiro', G=7, n=15, B=9, por_prueba=15 / 2 / 9)]
    for pt in points:
        pt['c'] = float(rate(pt['G'])[0])
    # loop 1 design against the exact laminar optimum at the same (n, B, G)
    rows = list(csv.DictReader((ROOT / 'augmented/autoresearch_logg/results_logg.tsv').open(), delimiter='\t'))
    best = max((r for r in rows if r['status'] == 'ok'), key=lambda r: float(r['score']))
    loop = F(best['value_n16_B4_G8'])
    lam16 = ExactHomogeneousLaminar(8).value(16, 4)
    assert lam16 < loop
    write_csv('general_vs_laminar.csv', points + [dict(diseno='loop 1, n=16 B=4', G=8, n=16, B=4,
                                                         por_prueba=float(loop) / 4, c=float(lam16) / 4)])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    fig.subplots_adjust(left=.1, right=.8, bottom=.18, top=.95)
    Gs = np.arange(1, 21); cs = [float(rate(G)[0]) for G in Gs]
    ax.plot(Gs, cs, '-', color=BLUE, lw=1.8)
    ax.text(20.4, cs[-1], 'laminar c*(G)', color=BLUE, fontsize=10, va='center')
    ax.plot(Gs, np.log2(Gs) / 4, color=GRAY, linestyle=':', lw=1.3)
    ax.text(20.4, log2(20) / 4, 'Lindström, asintótico\n(por cotejar)', color=GRAY, fontsize=9, va='center')
    for pt in points:
        ax.plot(pt['G'], pt['por_prueba'], 's', color=WARM, markersize=7)
        offset = (8, -14) if pt['G'] == 3 else (7, 6)
        ax.annotate(pt['diseno'], (pt['G'], pt['por_prueba']), textcoords='offset points', xytext=offset,
                    fontsize=9, color=WARM)
    ax.set(xlim=(0, 20.2), ylim=(0, 1.2), xticks=[1, 4, 8, 12, 16, 20], yticks=[0, .5, 1],
           xlabel='G', ylabel='sanos acreditados por prueba')
    clean(ax)
    return png(fig), points, float(loop), float(lam16)


# ------------------------------------------------------------------ 5. other q
def otros_q():
    qs = (.3, .5, .7, .9)
    rows = [dict(q=q, G=G, c=rate_float(G, q)[0]) for q in qs for G in range(1, 17)]
    assert rate(8, F(9, 10))[0] > 2
    write_csv('tasa_laminar_por_q.csv', rows)
    grid = np.round(np.arange(.05, .96, .05), 2)
    bound = []
    for q in grid:
        c = rate_float(8, float(q))[0]
        bound.append(min(8, log2(9) / log2(1 / q)) / c)
    q_max = float(grid[int(np.argmax(bound))])
    q9_48 = rate_float(48, .9)[0]
    assert q9_48 > next(r['c'] for r in rows if r['q'] == .9 and r['G'] == 16)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    fig.subplots_adjust(left=.1, right=.86, bottom=.18, top=.95)
    for q, color in zip(qs, (GRAY, BLUE, INK, WARM)):
        sub = [r for r in rows if r['q'] == q]
        ax.plot([r['G'] for r in sub], [r['c'] for r in sub], 'o-', color=color, lw=1.6, markersize=3)
        ax.text(16.4, sub[-1]['c'], f'q = {q}', color=color, fontsize=10, va='center')
    ax.set(xlim=(0, 16.2), ylim=(0, 5.5), xticks=[1, 4, 8, 12, 16], yticks=[0, 1, 2, 3, 4, 5],
           xlabel='G', ylabel='laminar c*(G, q) por prueba')
    clean(ax)
    return png(fig), rows, q_max, q9_48


def techo():
    rows = list(csv.DictReader((ROOT / 'results/sesion_2026-10-26/notebook_30/lema_informacion_malla.csv').open()))
    assert all(float(r['despejados']) <= float(r['log2K']) + 1e-12 for r in rows)
    return len(rows), min(float(r['log2K']) - float(r['despejados']) for r in rows)


def build():
    t0 = time.time()
    images = {}
    images['tasa'], tasa_rows = tasa()
    images['segunda_via'], sv_rows = segunda_via()
    images['regla'] = regla()
    images['general'], points, loop, lam16 = general()
    images['otros_q'], q_rows, q_max, q9_48 = otros_q()
    n_cells, gap = techo()
    for name, content in images.items():
        (OUT / f'{name}.png').write_bytes(content)
    by_G = {r['G']: r for r in tasa_rows}
    ss = next(p for p in points if p['G'] == 7)
    q9 = next(r for r in q_rows if r['q'] == .9 and r['G'] == 16)['c']
    q5 = next(r for r in q_rows if r['q'] == .5 and r['G'] == 16)['c']

    def cell(key, source, *figures):
        c = nb.v4.new_markdown_cell(source.strip(), id=key)
        c.attachments = {f'{n}.png': {'image/png': b64encode(images[n]).decode()} for n in figures}
        return c

    cells = [cell('p31-tasa', rf'''
## 1. La tasa laminar exacta

$q=1/2$, $u=1$, población grande.

![Sanos acreditados por prueba: el óptimo laminar crece despacio con G, muy por debajo del techo de información y de la cota 2 del potencial.](attachment:tasa.png)

$c^*(G)$ es el mejor cociente $\mathbb E[\text{{acreditados}}]/\mathbb E[\text{{pruebas}}]$ de una estrategia que trabaja un solo bloque. Con $n\ge BG$, $V_{{\rm lam}}\le c^*(G)\,B$ y $V_{{\rm lam}}/B\to c^*(G)$. Vale $1/2,\ 2/3,\ 9/13,\ 14/19$ y llega a {by_G[96]['c']:.3f} en $G=96$. El potencial de Francisco da 2; el techo, $\log_2(G+1)$.
''', 'tasa'),
             cell('p31-segunda-via', rf'''
## 2. El óptimo laminar exacto se acerca a $c^*(G)$

{len(sv_rows)} casos exactos, $G\in\{{2,3,4\}}$, $B\le12$, $n=BG$.

![Óptimo laminar exacto entre B, contra B: sube hacia c*(G) y nunca la cruza.](attachment:segunda_via.png)

Ningún caso supera $c^*(G)\,B$. Con $G=2$ y $B=12$ el óptimo vale {sv_rows[5]['por_prueba']:.3f} por prueba, contra $2/3$.
''', 'segunda_via'),
             cell('p31-regla', r'''
## 3. La regla óptima

$G=8$. Cada casilla dice cuántas personas del átomo se prueban.

![Regla óptima para G=8: si el átomo tiene a lo sumo la mitad infectada se prueba una parte de alrededor de la mitad; si tiene más, se abandona.](attachment:regla.png)

Abrir el grupo más grande posible. Si un átomo tiene más de la mitad infectada, abandonarlo; si no, probar una parte de él, de alrededor de la mitad en átomos chicos. Es una regla de cociente, como la que pidió Francisco el 22 de septiembre, y es óptima con $q$ homogéneo y población grande.
''', 'regla'),
             cell('p31-general', rf'''
## 4. General contra laminar, por prueba

![Tasa laminar c*(G) contra dos diseños generales verificados; la curva punteada es la tasa asintótica de Lindström.](attachment:general.png)

Söderberg–Shapiro acredita {ss['por_prueba']:.3f} por prueba con $G=7$; el laminar, {ss['c']:.3f}. El detector de 4 no le gana al laminar. El diseño del loop 1 ($n=16$, $B=4$, $G=8$) acredita {loop:.3f} contra el óptimo laminar exacto {lam16:.3f}. Si la tasa de Lindström valiera ya en $G\approx10$, la separación empezaría ahí.
''', 'general'),
             cell('p31-otros-q', rf'''
## 5. Otros $q$

![Tasa laminar contra G para cuatro valores de q: el nivel cambia mucho con q, pero en todos la curva se aplana.](attachment:otros_q.png)

Con $q=0.9$ la tasa laminar vale {q9:.2f} por prueba en $G=16$ y {q9_48:.2f} en $G=48$; con $q=1/2$, {q5:.2f} en $G=16$. En todos los $q$ la curva se aplana; lo que cambia es el nivel. La cota del precio por información alcanza su máximo en $q\approx{q_max:.2f}$, donde el techo cambia de régimen: no decide si $q=1/2$ es el peor caso.
''', 'otros_q'),
             cell('p31-techo', rf'''
## 6. El techo sin el $+1$

Hoja por hoja, $m(h)\le\log_2(1/P(h))$. Al promediar, $\mathbb E[m]\le H(\text{{historial}})\le\log_2K\le B\log_2(G+1)$. Con $q$ general: $\mathbb E[m]\le B\log_2(G+1)/\log_2(1/q)$ y $\le BG$.

En las {n_cells} celdas de la malla, los despejados quedan bajo $\log_2K$ con margen mínimo {gap:.2f}. Derivación por validar con A.
''')]
    book = nb.v4.new_notebook(cells=cells)
    book.metadata.update(language_info={'name': 'python'}, generator='build_31_avances.py')
    nb.validate(book)
    nb.write(book, BOOK)
    print(BOOK, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    build()
