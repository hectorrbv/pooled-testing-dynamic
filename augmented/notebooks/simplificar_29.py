"""Rebuild the presentation notebook 29 with one idea per cell and a figure per idea.

Reads the verified evidence (session 12 JSON, October 14 grid and loop audit);
never reruns solvers. Refreshes the loop update in the study notebook too.
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


def contraejemplo_arboles():
    """Restore the earlier decision-tree layout for the current q=(.9,.7,.7)."""
    from figuras_separacion_28 import canvas, text, box, line
    qa, qb, qc = F(9, 10), F(7, 10), F(7, 10)
    p0, p1, p2 = qa*qb, qa*(1-qb)+(1-qa)*qb, (1-qa)*(1-qb)
    negative_value = 2 + qc
    positive_value = 2 * qa * (1-qb) * qc / (1-p0)
    binary_value = p0*negative_value + (1-p0)*positive_value
    laminar_value = p0*negative_value + p1 + p2*qc
    assert binary_value == F(2079, 1000) and laminar_value == F(1031, 500)
    fig, ax = canvas(13.2, 4.2)
    for center, title, color, branches in [
            (.23, f'Binario · {float(binary_value):.3f}', WARM,
             [(.09, f'−  ·  {float(p0):.2f}', 'C', f'{float(negative_value):g}'),
              (.37, f'+  ·  {float(1-p0):.2f}', 'AC', f'{float(positive_value):.4f}')]),
            (.75, f'Laminar · {float(laminar_value):.3f}', BLUE,
             [(.57, f'R = 0  ·  {float(p0):.2f}', 'C', f'{float(negative_value):g}'),
              (.75, f'R = 1  ·  {float(p1):.2f}', 'A', '1'),
              (.93, f'R = 2  ·  {float(p2):.2f}', 'C', f'{float(qc):g}')])]:
        text(ax, center, .94, title, 19, color)
        box(ax, center, .77, .085, .12, 'AB', INK, 16)
        for x, label, action, value in branches:
            line(ax, (center, .70), (x, .39), GRAY)
            text(ax, x, .56, label, 12, GRAY,
                 bbox=dict(facecolor='white', edgecolor='none', pad=3))
            box(ax, x, .32, .085, .12, action, color, 17)
            line(ax, (x, .25), (x, .14), color)
            text(ax, x, .075, value, 20, color)
    return png(fig)


def contraejemplo_valores():
    """Exact values of the witness and the four-person family, recomputed with the reference solvers."""
    import sys
    sys.path.insert(0, str(ROOT))
    from augmented.dp_perfiles_binario_conteo import DP, Lambda
    from augmented.bellman_perfiles import BellmanPerfiles

    def laminar(q, u, G, B):
        return BellmanPerfiles([1 - x for x in q], list(u), G, laminar=True).solve(B).value

    q, u = [F(9, 10), F(7, 10), F(7, 10)], [F(1)] * 3
    witness = {'Binario': DP(q, u, 3, 'binary').value(2), 'Laminar': laminar(q, u, 3, 2),
               'Irrestricto': DP(q, u, 3, 'count').value(2)}
    assert witness == {'Binario': F(2079, 1000), 'Laminar': F(1031, 500), 'Irrestricto': F(2121, 1000)}
    lam = Lambda(q, 3)
    assert lam == F(7, 4)
    family = []
    for s in (F(19, 20), F(99, 100), F(999, 1000), F(999999, 1000000)):
        qs, us = [s, F(1, 2), F(1, 2), s], [F(1), F(3), F(3), F(2)]
        family.append((s, laminar(qs, us, 2, 2) / DP(qs, us, 2, 'binary').value(2)))
    assert family[-1][1] == F(8999994000003, 2000000000000) / F(1999997000001, 400000000000)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={'width_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.13, right=.98, bottom=.2, top=.9, wspace=.45)
    labels = list(witness)
    values = [float(v) for v in witness.values()]
    bars = ax.barh(labels[::-1], values[::-1], color=[GRAY, BLUE, WARM], height=.55)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + .01, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center',
                fontsize=12, color=INK)
    ax.set(xlim=(0, 2.45), xticks=[0, 1, 2], xlabel='Utilidad esperada, testigo n = 3')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY, length=0)
    ax.tick_params(axis='y', labelsize=12, labelcolor=INK)
    xs = [float(1 - s) for s, _ in family]
    ys = [float(r) for _, r in family]
    bx.plot(xs, ys, 'o-', color=WARM, lw=1.8, markersize=5)
    bx.axhline(.9, color=GRAY, linestyle='--', lw=1)
    bx.text(xs[-1], .9 - .004, 'límite 9/10', va='top', ha='left', color=GRAY, fontsize=11)
    bx.set(xscale='log', xlabel='1 − s', ylabel='Laminar / binario', ylim=(.885, max(ys) + .01))
    bx.invert_xaxis()
    bx.spines[['top', 'right']].set_visible(False)
    bx.tick_params(colors=GRAY)
    return png(fig), witness, lam, family


def familia_arboles():
    """Largest known separation: n=4, B=G=2, q=(s,1/2,1/2,s), u=(1,3,3,2). Trees from the certified note,
    values recomputed with the reference solvers at s=0.99 and checked against the closed forms."""
    import sys
    sys.path.insert(0, str(ROOT))
    from augmented.dp_perfiles_binario_conteo import DP
    from augmented.bellman_perfiles import BellmanPerfiles
    s = F(99, 100)
    q, u = [s, F(1, 2), F(1, 2), s], [F(1), F(3), F(3), F(2)]
    vbin = DP(q, u, 2, 'binary').value(2)
    solver = BellmanPerfiles([1 - x for x in q], u, 2, laminar=True)
    vlam = solver.solve(2).value
    assert vbin == F(5, 2) * s * (1 + s) and vlam == 3 + F(3, 2) * s * s
    assert tuple(solver.tree(2)['pool']) == (1, 2)          # laminar opens BC
    pos = {'B': (-1, 1), 'D': (1, 1), 'C': (-1, -1), 'A': (1, -1)}
    labels = {'A': 'A  u=1', 'B': 'B  u=3', 'C': 'C  u=3', 'D': 'D  u=2'}
    fig, axes = dot_panels(2, width=6.4, height=3.3)
    for ax, title, pools, caption in [
            (axes[0], 'Binario: BD, luego AC o AD', [('BD', BLUE, .58), ('AC', WARM, .40), ('AD', WARM, .40)],
             'D se reutiliza en AD'),
            (axes[1], 'Laminar: BC, luego B o AD', [('BC', BLUE, .58), ('B', WARM, .33), ('AD', WARM, .40)],
             'B refina; AD es disjunto')]:
        for pool, color, pad in pools:
            contour(ax, pool, color, pad=pad, positions=pos)
        dots(ax, positions=pos, labels=labels)
        ax.text(0, 2.55, title, fontsize=13, color=INK, **TEXT)
        ax.text(0, -2.55, caption, fontsize=13, color=WARM, **TEXT)
    return png(fig), s, vbin, vlam


def simplify_explanations(book):
    """Short presentation text for sections 1 and 2; keep order and images."""
    import re
    text = {
        'logg-contraejemplo': r'''
$n=3$, $B=2$, $G=3$, $q_A=0.9$, $q_B=q_C=0.7$, $u=1$.

El binario solo dice si hay algún infectado, pero permite cualquier grupo. El conteo da más información, aunque aquí exigimos que sus grupos sean laminares.

Ambos empiezan probando AB. En binario, un positivo significa uno o dos infectados: después se prueba AC y, si sale negativo, se identifican dos sanos. En laminar, si AB devuelve exactamente uno, probar A identifica al sano de AB con certeza. AC no está permitido porque no es laminar con AB.

![Árboles de las políticas binaria y laminar, con probabilidades y utilidad esperada de cada rama.](attachment:contraejemplo_arboles.png)

Las ramas indican resultados y probabilidades. Los números inferiores son la utilidad total esperada al seguir cada rama.

![Utilidad esperada de las tres clases y separación de la familia de cuatro personas.](attachment:contraejemplo_valores.png)

Promediando todas las ramas: binario obtiene 2.079; conteo laminar, 2.062; conteo general, 2.121. La pérdida laminar frente al binario es pequeña, 0.8%, pero demuestra que restringir los grupos puede pesar más que disponer de conteos. La curva de la derecha corresponde a la familia siguiente.
''',
        'logg-familia': r'''
$n=4$, $B=G=2$, $q=(s,1/2,1/2,s)$, $u=(1,3,3,2)$, con $0.95\le s<1$.

A y D tienen alta probabilidad de estar sanos. B y C son más inciertos, pero cada uno aporta utilidad 3.

El binario prueba BD. Si sale negativo, sigue con AC; si sale positivo, reutiliza D en AD. La mejor política laminar empieza con BC: si hay un infectado, prueba B; si hay cero o dos, pasa a AD.

![El binario reutiliza D en un grupo no laminar; el laminar refina BC o prueba el par disjunto AD.](attachment:familia_arboles.png)

Con $s=0.99$, los óptimos son 4.92525 para binario y 4.47015 para laminar. Al acercar $s$ a uno, laminar conserva aproximadamente el 90% de la utilidad binaria. Es una pérdida cercana al 10%; no sabemos si otra familia puede perder más. El límite usa $s<1$: en $s=1$ A y D ya serían sanos conocidos.

Para discutir: ¿podemos encontrar una familia donde la pérdida laminar frente al binario supere el 10%?
''',
        'logg-ejemplo': r'''
$n=4$, $B=3$, $G=3$, $q=1/2$, $u=1$.

Hacemos tres pruebas: ABC, ABD y ACD. Cada una devuelve cuántos infectados hay en ese grupo. Juntos, los tres resultados permiten saber el estado de las cuatro personas.

Por ejemplo, si C y D están infectados, obtenemos los conteos $(1,1,2)$. Solo ese perfil produce esos resultados: A y B están sanos.

![ABC devuelve uno, ABD devuelve uno y ACD devuelve dos: C y D están infectados; A y B están sanos.](attachment:detector.png)

Comprobamos las 16 combinaciones posibles de sanos e infectados: ninguna repite el mismo vector de resultados. El diseño no es laminar, porque ABC y ABD comparten personas sin que uno contenga al otro.
''',
        'logg-ejemplo-valor': r'''
![Sanos identificados en promedio: general 2, laminar 1.75, individuales 1.5.](attachment:utilidades.png)

La matriz identifica a todos. Como cada persona tiene probabilidad $1/2$ de estar sana, identifica $4\times1/2=2$ sanos en promedio.

Con las mismas tres pruebas, el óptimo laminar identifica 1.75 sanos en promedio. Probar tres personas individualmente da 1.5. Exigir laminaridad pierde el 12.5% de la utilidad general, incluso eligiendo la mejor política laminar.

La pregunta siguiente es si esa pérdida aumenta al construir diseños mayores.
''',
    }
    for cell in book.cells:
        # The two counterexample cells keep their own text and figures (restored 2026-10-20).
        if cell.id not in text or cell.id in ('logg-contraejemplo', 'logg-familia'):
            continue
        title = cell.source.splitlines()[0]
        if cell.id == 'logg-contraejemplo':
            title = re.sub(r'(^## \d+\. ).*', r'\1El binario puede superar al conteo laminar', title)
        elif cell.id == 'logg-familia':
            title = '### Una familia donde la pérdida se acerca al 10%'
        cell.source = (title + '\n\n' if cell.id != 'logg-ejemplo-valor' else '') + text[cell.id].strip()


def build():
    d = evidence(ROOT)
    session = json.loads((ROOT / 'results/sesion_2026-10-12/evidencia_29.json').read_text())
    m = session['matrix']
    import csv
    rows = list(csv.DictReader((ROOT / d['grid_source']).open()))
    assert len(rows) == 32

    images = {'detector': detector_png(), 'laminar_general': laminar_general()}
    images['utilidades'], values = utilities(m)
    images['malla'], series, ceiling = grid(d)
    images['cota'], observed, cap = bound(rows)
    images['contraejemplo_arboles'] = contraejemplo_arboles()
    images['contraejemplo_valores'], witness, lam, family = contraejemplo_valores()
    images['familia_arboles'], s99, vbin99, vlam99 = familia_arboles()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in images.items():
        (OUT / f'{name}.png').write_bytes(content)

    general, laminar, individual = values
    sep = F(m['policies']['general']['value']) / F(m['policies']['laminar']['value'])
    g3 = next(r for r in d['slice'] if r['G'] == '3')
    loss3 = 100 * (1 - float(F(g3['exact_laminar']) / F(g3['unrestricted'])))
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

Cómo leerlo: un ejemplo verificado (1), lo que dice la malla (2), lo que ya acota la ventaja (3) y los nuevos experimentos y propuestas. Cada número se regenera desde la evidencia guardada.
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
             cell('logg-malla', rf'''
## 2. Lo que muestra la malla

$n={n7}$, $B={B7}$, $G\in\{{2,3,4,5\}}$; cuatro de los {len(rows)} casos exactos.

![Utilidad esperada dividida entre B. El general se separa del laminar y ambos se detienen antes del tope que impone la población.](attachment:malla.png)

Con $G=3$ la pérdida laminar es {loss3:.1f}% del óptimo general. De $G=4$ a $G=5$ nada cambia: con {n7/2:g} sanos esperados, $V/B\le{ceiling:g}$ y la población tapa el crecimiento.

Para discutir: ¿qué familia $(n,B,G)$ propone el argumento de matrices? Esa familia fija el siguiente experimento.
''', 'malla'),
             cell('logg-cota', rf'''
## 3. Lo que ya acota la ventaja

![Cociente general/laminar: la cota por información frente al máximo observado en la malla.](attachment:cota.png)

Ninguna política acredita más de $B\log_2(G+1)$ sanos en promedio. Probar individuos es laminar y da $B/2$, así que el cociente es a lo sumo $2\log_2(G+1)$ cuando $B\le n$. La prueba por información está en las notas; falta su revisión.

El máximo observado es {max(observed):.2f}, lejos del tope, y no crece con $G$ a población fija.
''', 'cota'),
             cell('logg-aporte', rf'''
## 4. Qué llevamos a la discusión

1. El ejemplo {general:g} frente a {laminar:g}: cruzar grupos ahorra una prueba.
2. La petición: la familia $(n,B,G)$ de las matrices, para verificarla.

La búsqueda de Hermeneuta no mejoró los ejemplos de partida. Los encargos binario frente a laminar, par frente a individuales y greedy frente a rollout siguen en las notas de estudio. Ninguno prueba el crecimiento logarítmico.
''')]
    book = nb.read(BOOK, as_version=4)
    book.cells = cells
    book.metadata['research_focus'] = 'log-G-2026-10-16'
    from actualizar_loops_29 import update
    study_path = BOOK.with_name('29_notas_de_estudio.ipynb')
    study = nb.read(study_path, as_version=4) if study_path.exists() else None
    book, study = update(book, study)
    # The counterexample Francisco asked for on 6 October goes right before the discussion cell.
    s_last = family[-1][0]
    counter = cell('logg-contraejemplo', rf'''
## 6. El contraejemplo que pidió Francisco: el binario gana al laminar

Encargo del 6 de octubre: un caso donde el óptimo dinámico binario supere al óptimo laminar, "lo peor posible". Testigo: $n=3$, $B=2$, $G=3$, $u\equiv1$, $q=(0.9,0.7,0.7)$.

![Ambos prueban AB primero. Tras un positivo, el binario cruza con AC y apuesta por dos personas; el laminar refina con A y cobra una por deducción.](attachment:contraejemplo_arboles.png)

Tras el positivo, A es el sano probable y C está fresco. El cruce AC vale $2\times0.511=1.02$; la deducción segura vale $1$. Con todo $q_i\le1/2$ la deducción siempre gana: el ejemplo necesita $\Lambda>1$ (aquí $\Lambda={lam.numerator}/{lam.denominator}$).

![Izquierda, valores exactos del testigo. Derecha, la familia de cuatro personas: laminar/binario baja hacia 0.9 cuando s tiende a 1.](attachment:contraejemplo_valores.png)

Binario {float(witness['Binario']):.3f}, laminar {float(witness['Laminar']):.3f}, irrestricto {float(witness['Irrestricto']):.3f}. La brecha del testigo es {100 * (1 - float(witness['Laminar'] / witness['Binario'])):.1f}%. A la derecha, la familia de abajo lleva el cociente laminar/binario a {float(family[-1][1]):.6f} con $s=1-10^{{-6}}$.
''', 'contraejemplo_arboles', 'contraejemplo_valores')
    familia = cell('logg-familia', rf'''
### La mayor separación conocida: laminar/binario tiende a $9/10$

$n=4$, $B=G=2$, $q=(s,1/2,1/2,s)$, $u=(1,3,3,2)$, con $19/20\le s<1$. A y D son casi seguro sanos; B y C valen más y son inciertos.

![Binario abre BD y, si sale positivo, reutiliza D en AD. Laminar abre BC y solo puede refinar B o pasar al par disjunto AD.](attachment:familia_arboles.png)

$V_{{\rm bin}}=\tfrac52\,s(1+s)$ y $V_{{\rm lam}}=3+\tfrac32\,s^2$. En $s={float(s99):g}$: binario {float(vbin99):.5f}, laminar {float(vlam99):.5f}. El cociente tiende a $9/10$ cuando $s\to1$; en $s=1$ el problema cambia, porque A y D ya son sanos conocidos. Certificado simbólico: 454 desigualdades en el intervalo (`tests_busqueda_brecha_ampliada.py`). Ni el barrido de 6 800 instancias ni la noche de Hermeneuta ($n\le6$) encontraron nada por debajo.

Para discutir: ¿tu ejemplo debe tener $\Lambda>1$? ¿Cuánto puede bajar laminar/binario? El $1/4$ de tu cota queda lejos en $n\le6$.
''', 'familia_arboles')
    # Order: portada, the counterexample as section 1, then the log-G thread; no closing discussion cell.
    import re
    book.cells = [c for c in book.cells if c.id not in (counter.id, familia.id, 'logg-aporte')]
    book.cells.insert(1, counter)
    book.cells.insert(2, familia)
    numbering = {'logg-contraejemplo': 1, 'logg-ejemplo': 2, 'logg-malla': 3, 'logg-cota': 4,
                 'logg-loop1': 5, 'logg-loop2': 6}
    for c in book.cells:
        if c.id in numbering:
            c.source = re.sub(r'^## \d+\. ', f'## {numbering[c.id]}. ', c.source, count=1, flags=re.M)
        if c.id == 'logg-pregunta':
            c.source = c.source.replace('Primero: separación, malla y cotas. Después:',
                                        'Primero: el contraejemplo pedido el 6 de octubre. Luego: separación, '
                                        'malla y cotas. Después:')
    simplify_explanations(book)
    nb.validate(book)
    from base_29 import write_unless_base
    write_unless_base(book, BOOK)
    if study is not None:
        nb.validate(study)
        nb.write(study, study_path)
    return book


if __name__ == '__main__':
    build()
    print(BOOK)
