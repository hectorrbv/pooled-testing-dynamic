"""Rebuild the presentation notebook 29 with one idea per cell and a figure per idea.

Reads the verified evidence (session 12 JSON and the October 14 grid); never reruns solvers.
The study notebook is not touched.
"""
from base64 import b64encode
from fractions import Fraction as F
from math import log2
from pathlib import Path
import json
import nbformat as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figuras_separacion_28 import INK, GRAY, BLUE, WARM
from figura_detector_29 import contour, dots, panel, detector_png
from enfocar_log_g_29 import evidence

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/sesion_2026-10-16/enfoque_log_G'
BOOK = ROOT / 'augmented/notebooks/29_tres_clases_y_contraejemplo.ipynb'
TEXT = dict(ha='center', va='center')


def png(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return buf.getvalue()


def dot_panels(k, width=9.6, height=3.0):
    fig, axes = plt.subplots(1, k, figsize=(width, height))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98, wspace=.08)
    for ax in axes:
        panel(ax)
        ax.set_ylim(-2.9, 2.9)
    return fig, axes


def laminar_general():
    fig, axes = dot_panels(2, width=6.4, height=3.3)
    for ax, title, pools in [(axes[0], 'Laminar', [('ABC', BLUE, .58), ('AB', WARM, .33), ('D', WARM, .33)]),
                             (axes[1], 'General', [('ABC', BLUE, .58), ('ABD', WARM, .40)])]:
        for pool, color, pad in pools:
            contour(ax, pool, color, pad=pad)
        dots(ax)
        ax.text(0, 2.55, title, fontsize=16, color=INK, **TEXT)
    return png(fig)


def utilities(m):
    labels = ['General', 'Laminar', 'Individuales']
    values = [float(F(m['policies']['general']['value'])),
              float(F(m['policies']['laminar']['value'])), m['B'] / 2]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    fig.subplots_adjust(left=.22, right=.97, bottom=.1, top=.95)
    bars = ax.barh(labels[::-1], values[::-1], color=[GRAY, BLUE, WARM], height=.55)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + .03, bar.get_y() + bar.get_height() / 2, f'{value:g}', va='center',
                fontsize=13, color=INK)
    ax.set(xlim=(0, 2.4), xticks=[0, 1, 2], xlabel='Sanos acreditados en promedio')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY, length=0)
    ax.tick_params(axis='y', labelsize=12, labelcolor=INK)
    return png(fig), values


def control(d):
    fig, axes = dot_panels(3)
    titles = ['Historial: ABC = 1']
    for branch in d['conditional_branches']:
        titles.append(f"Prueba A = {branch['result']}   (prob. {branch['probability']})")
    contour(axes[0], 'ABC', BLUE)
    dots(axes[0])
    axes[0].text(0, -2.55, 'nadie acreditado', fontsize=15, color=GRAY, **TEXT)
    for ax, branch in zip(axes[1:], d['conditional_branches']):
        contour(ax, 'A', BLUE, pad=.33)
        healthy = ''.join('ABCD'[i] for i in branch['healthy'])
        dots(ax, hollow=healthy, infected='A' if branch['result'] else '')
        ax.text(0, -2.55, f"+{branch['additional_utility']}", fontsize=17, color=BLUE, **TEXT)
    for ax, title in zip(axes, titles):
        ax.text(0, 2.55, title, fontsize=13, color=INK, **TEXT)
    return png(fig)


def grid(d):
    gs = [int(r['G']) for r in d['slice']]
    B = int(d['slice'][0]['B']); n = int(d['slice'][0]['n'])
    ceiling = n / 2 / B
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    fig.subplots_adjust(left=.12, right=.78, bottom=.18, top=.93)
    series = {}
    for key, label, color in [('unrestricted', 'General', WARM), ('exact_laminar', 'Laminar', BLUE)]:
        ys = [float(F(r[key]) / B) for r in d['slice']]
        series[key] = ys
        ax.plot(gs, ys, 'o-', color=color, lw=1.8, markersize=5)
        ax.text(5.12, ys[-1], label, va='center', color=color, fontsize=12)
    ax.axhline(ceiling, color=GRAY, linestyle='--', lw=1)
    ax.text(5.12, ceiling, f'Tope {n/2:g}/{B}', va='center', color=GRAY, fontsize=11)
    ax.axhline(.5, color=GRAY, linestyle=':', lw=1)
    ax.text(5.12, .5, 'Individuales', va='center', color=GRAY, fontsize=11)
    ax.set(xlim=(1.9, 5.05), ylim=(.47, .92), xticks=gs, yticks=[.5, .6, .7, .8, .9],
           xlabel='G', ylabel='Utilidad esperada / B')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)
    return png(fig), series, ceiling


def bound(rows):
    gs = sorted({int(r['G']) for r in rows})
    observed = [max(float(F(r['unrestricted']) / F(r['exact_laminar'])) for r in rows if int(r['G']) == g)
                for g in gs]
    cap = [2 * log2(g + 1) for g in gs]
    assert all(o <= c for o, c in zip(observed, cap))
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    fig.subplots_adjust(left=.12, right=.70, bottom=.18, top=.93)
    ax.plot(gs, cap, 'o-', color=GRAY, lw=1.8, markersize=5)
    ax.text(5.12, cap[-1], 'Cota  2·log₂(G+1)', va='center', color=GRAY, fontsize=12)
    ax.plot(gs, observed, 'o-', color=WARM, lw=1.8, markersize=5)
    ax.text(5.12, observed[-1], 'Máximo en la malla', va='center', color=WARM, fontsize=12)
    ax.axhline(1, color=GRAY, linestyle=':', lw=1)
    ax.set(xlim=(1.9, 5.05), ylim=(0.8, 5.6), xticks=gs, yticks=[1, 2, 3, 4, 5],
           xlabel='G', ylabel='Óptimo general / óptimo laminar')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)
    return png(fig), observed, cap


def build():
    d = evidence(ROOT)
    session = json.loads((ROOT / 'results/sesion_2026-10-12/evidencia_29.json').read_text())
    m = session['matrix']
    import csv
    rows = list(csv.DictReader((ROOT / d['grid_source']).open()))
    assert len(rows) == 32

    images = {'detector': detector_png(), 'laminar_general': laminar_general()}
    images['utilidades'], values = utilities(m)
    images['control'] = control(d)
    images['malla'], series, ceiling = grid(d)
    images['cota'], observed, cap = bound(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in images.items():
        (OUT / f'{name}.png').write_bytes(content)

    general, laminar, individual = values
    sep = F(m['policies']['general']['value']) / F(m['policies']['laminar']['value'])
    g3 = next(r for r in d['slice'] if r['G'] == '3')
    loss3 = 100 * (1 - float(F(g3['exact_laminar']) / F(g3['unrestricted'])))
    gain = F(d['conditional_gain'])
    hist = F(d['history_probability'])
    n7 = int(d['slice'][0]['n']); B7 = int(d['slice'][0]['B'])

    def cell(key, source, *figures):
        c = nb.v4.new_markdown_cell(source.strip(), id=key)
        c.attachments = {f'{name}.png': {'image/png': b64encode(images[name]).decode()}
                         for name in figures}
        return c

    cells = [cell('logg-pregunta', r'''
# ¿Cuánto puede costar exigir laminaridad?

Pregunta de la sesión: ¿el cociente entre el óptimo general y el óptimo laminar crece como $\log G$?

Dos piezas por demostrar. Toda política laminar obtiene a lo sumo $CB$. Alguna construcción general alcanza al menos $cB\log_2 G$. Las constantes no dependen del tamaño del problema.

Convenciones: $n$ personas, presupuesto $B$, a lo sumo $G$ personas por prueba. Cada persona está sana con probabilidad $q=1/2$, independiente, y vale $u=1$. Todas las políticas son dinámicas aumentadas: la prueba devuelve el conteo de infectados. Laminar y general difieren solo en qué grupos se permiten.

Cómo leerlo: un ejemplo verificado (1), un control para la prueba (2), lo que dice la malla (3), lo que ya acota la ventaja (4) y lo que llevamos a la discusión (5). Cada número se regenera desde la evidencia del 12 de octubre y la malla del 14 de octubre.
'''),
             cell('logg-regimenes', r'''
## Laminar y general

![Izquierda, grupos anidados o disjuntos. Derecha, dos grupos que se cruzan.](attachment:laminar_general.png)

Laminar: a lo largo de una trayectoria, dos grupos son disjuntos o uno contiene al otro. General: cualquier grupo de hasta $G$ personas, aunque se cruce con los anteriores.
''', 'laminar_general'),
             cell('logg-ejemplo', rf'''
## 1. Tres pruebas identifican a cuatro personas

$n={m['n']}$, $B={m['B']}$, $G={m['G']}$.

El plan general prueba ABC, ABD y ACD. Los tres conteos distinguen los {m['detector']['distinct_signatures']} perfiles. Abajo, el perfil con C y D infectados.

![Tres pruebas: ABC devuelve 1, ABD devuelve 1 y ACD devuelve 2. Eso identifica a A y B como sanos.](attachment:detector.png)

Los grupos se cruzan: el plan no es laminar.
''', 'detector'),
             cell('logg-ejemplo-valor', rf'''
![Sanos acreditados en promedio por el óptimo general, el óptimo laminar y las pruebas individuales.](attachment:utilidades.png)

General acredita {general:g} sanos en promedio; laminar, {laminar:g}; individuales, {individual:g}. La separación general/laminar es ${sep.numerator}/{sep.denominator}$.

Para discutir: ¿cómo crece este diseño sin pasar de $G$ personas por prueba?
''', 'utilidades'),
             cell('logg-control', rf'''
## 2. Una prueba laminar puede cobrar más de uno

$n=4$, $B=2$, $G=3$. ABC devolvió 1. Queda una prueba. Probar A es laminar.

![Tras ABC=1, probar A da cero con probabilidad dos tercios y acredita a A; da uno con probabilidad un tercio y acredita a B y C.](attachment:control.png)

Utilidad adicional esperada: $\frac23\cdot1+\frac13\cdot2=\frac{{{gain.numerator}}}{{{gain.denominator}}}>1$.

El argumento «cada prueba laminar cobra a lo sumo uno» falla en este historial. No refuta la cota $CB$: el historial ocurre con probabilidad ${hist.numerator}/{hist.denominator}$.

Para discutir: ¿qué cantidad acumulada, un potencial, acota a la vez lo cobrado y la información pendiente?
''', 'control'),
             cell('logg-malla', rf'''
## 3. Lo que muestra la malla

$n={n7}$, $B={B7}$, $G\in\{{2,3,4,5\}}$; cuatro de los {len(rows)} casos exactos.

![Utilidad esperada dividida entre B. El general se separa del laminar y ambos se detienen antes del tope que impone la población.](attachment:malla.png)

Con $G=3$ la pérdida laminar es {loss3:.1f}% del óptimo general. De $G=4$ a $G=5$ nada cambia: con {n7/2:g} sanos esperados, $V/B\le{ceiling:g}$ y la población tapa el crecimiento.

Para discutir: ¿qué familia $(n,B,G)$ propone el argumento de matrices? Esa familia fija el siguiente experimento.
''', 'malla'),
             cell('logg-cota', rf'''
## 4. Lo que ya acota la ventaja

![Cociente general/laminar: la cota por información frente al máximo observado en la malla.](attachment:cota.png)

Ninguna política acredita más de $B\log_2(G+1)$ sanos en promedio. Probar individuos es laminar y da $B/2$, así que el cociente es a lo sumo $2\log_2(G+1)$ cuando $B\le n$. La prueba por información está en las notas; falta su revisión.

El máximo observado es {max(observed):.2f}, lejos del tope, y no crece con $G$ a población fija.
''', 'cota'),
             cell('logg-aporte', rf'''
## 5. Qué llevamos a la discusión

1. El ejemplo {general:g} frente a {laminar:g}: cruzar grupos ahorra una prueba.
2. El control de ${gain.numerator}/{gain.denominator}$: la cota laminar necesita un argumento global, no uno por prueba.
3. La petición: la familia $(n,B,G)$ de las matrices, para verificarla.

La búsqueda de Hermeneuta no mejoró los ejemplos de partida. Los encargos binario frente a laminar, par frente a individuales y greedy frente a rollout siguen en las notas de estudio. Ninguno prueba el crecimiento logarítmico.
''')]
    book = nb.read(BOOK, as_version=4)
    book.cells = cells
    book.metadata['research_focus'] = 'log-G-2026-10-16'
    nb.validate(book)
    nb.write(book, BOOK)
    return book


if __name__ == '__main__':
    build()
    print(BOOK)
