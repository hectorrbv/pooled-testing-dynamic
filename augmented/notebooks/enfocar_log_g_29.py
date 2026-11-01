"""Connect the verified small examples to Francisco's proposed log-G gap.

The transformation preserves the earlier main cells in the study notebook.
New numbers are enumerated exactly; the larger grid is read, never rerun here.
"""
from base64 import b64encode
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import nbformat as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figuras_separacion_28 import canvas, text, box, line, png, INK, GRAY, BLUE, WARM


def evidence(root):
    profiles = list(product((0, 1), repeat=4))
    history = [z for z in profiles if sum(z[:3]) == 1]
    branches = []
    for result in (0, 1):
        compatible = [z for z in history if z[0] == result]
        healthy = [i for i in range(4) if all(z[i] == 0 for z in compatible)]
        branches.append(dict(result=result,
                             probability=str(F(len(compatible), len(history))),
                             healthy=healthy, additional_utility=len(healthy)))
    gain = sum(F(b['probability']) * b['additional_utility'] for b in branches)
    assert gain == F(4, 3)
    signatures = [tuple(sum(z[i] for i in pool)
                        for pool in ((0, 1, 2), (0, 1, 3), (0, 2, 3)))
                  for z in profiles]
    assert len(set(signatures)) == 16
    assert signatures[profiles.index((0, 0, 1, 1))] == (1, 1, 2)
    data = root / 'results/precio_q05_G/malla_2026-10-14_completa.csv'
    with data.open() as f:
        rows = [r for r in csv.DictReader(f)
                if int(r['n']) == 7 and int(r['B']) == 4 and F(r['q']) == F(1, 2)]
    rows.sort(key=lambda r: int(r['G']))
    assert [int(r['G']) for r in rows] == [2, 3, 4, 5]
    return dict(history_probability=str(F(len(history), len(profiles))),
                conditional_branches=branches, conditional_gain=str(gain),
                detector_distinct_signatures=len(set(signatures)),
                grid_source=str(data.relative_to(root)),
                grid_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                slice=rows, saturation_per_test='7/8')


def make_figures(root, d):
    fig, ax = canvas(7.8, 3.0)
    box(ax, .5, .84, .12, .14, 'A', INK, 20)
    for x, branch in zip((.24, .76), d['conditional_branches']):
        line(ax, (.5, .76), (x, .38), GRAY)
        text(ax, x, .59, f"R = {branch['result']}  ·  {branch['probability']}",
             15, GRAY, bbox=dict(facecolor='white', edgecolor='none', pad=3))
        names = ', '.join('ABCD'[i] for i in branch['healthy'])
        box(ax, x, .25, .22, .21, f"{names}\n+{branch['additional_utility']}", BLUE, 18)
    images = {'ganancia_condicional': png(fig)}

    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    fig.subplots_adjust(left=.12, right=.80, bottom=.18, top=.93)
    gs = [int(r['G']) for r in d['slice']]
    for key, label, color, offset in [('unrestricted', 'General', WARM, 0),
                                       ('exact_laminar', 'Laminar', BLUE, 0)]:
        ys = [float(F(r[key]) / 4) for r in d['slice']]
        ax.plot(gs, ys, 'o-', color=color, lw=1.8, markersize=5)
        ax.text(5.12, ys[-1] + offset, label, va='center', color=color, fontsize=12)
    ax.axhline(.5, color=GRAY, linestyle=':', lw=1)
    ax.text(5.12, .5, 'Individuales', va='center', color=GRAY, fontsize=11)
    ax.set(xlim=(1.9, 5.05), ylim=(.47, .9), xticks=gs,
           yticks=[.5, .6, .7, .8, .9], xlabel='G', ylabel='Utilidad esperada / B')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)
    images['utilidad_por_prueba'] = png(fig)
    out = root / 'results/sesion_2026-10-16/enfoque_log_G'
    out.mkdir(parents=True, exist_ok=True)
    for name, content in images.items():
        (out / f'{name}.png').write_bytes(content)
    (out / 'evidencia.json').write_text(json.dumps(d, indent=2) + '\n')
    return images


def focus(presentation, study, root):
    d = evidence(root)
    images = make_figures(root, d)
    detector = next(c for c in study.cells if 'detector.png' in c.get('attachments', {}))

    def cell(key, source, *figures):
        c = nb.v4.new_markdown_cell(source.strip(), id=key)
        c.attachments = {f'{name}.png': {'image/png': b64encode(images[name]).decode()}
                         for name in figures}
        return c

    # Archive each previous presentation block once; never discard its figure.
    if presentation.metadata.get('research_focus') != 'log-G-2026-10-16':
        for old in presentation.cells[1:]:
            saved = deepcopy(old)
            saved.id = 'consulta-' + old.id
            saved.source = '## Consulta: encargos y resultados anteriores\n\n' + saved.source
            study.cells.append(saved)

    intro = cell('logg-pregunta', r'''
# ¿Cuánto puede costar exigir laminaridad?

Francisco propone que, en una familia de casos, el cociente entre el óptimo general y el laminar crece como $\log G$.

Dos piezas por demostrar: que toda política laminar obtiene como máximo $CB$, y que una construcción general alcanza al menos $cB\log_2G$. Las constantes $C,c>0$ deben ser independientes del tamaño del problema. Ambas piezas deben usar los mismos recursos.

Aquí $q_i=1/2$ es la probabilidad de estar sano, $u_i=1$ y los estados son independientes. Usamos conteos y soft clearing. $n$ es la población, $B$ el presupuesto y $G$ el tamaño máximo de pool. «Laminar» permite grupos disjuntos o contenidos uno en otro a lo largo de cada trayectoria; «general» permite cualquier pool de tamaño hasta $G$.

Traemos un ejemplo verificado y los límites de nuestra evidencia. El crecimiento logarítmico sigue siendo la dirección propuesta, no un resultado demostrado por estos experimentos.
''')
    matrix = cell('logg-ejemplo', r'''
## 1. Tres pruebas identifican a cuatro personas

$n=4$, $B=3$, $G=3$.

El plan general prueba ABC, ABD y ACD. Cada prueba devuelve un conteo; los tres juntos distinguen los 16 perfiles posibles. La figura muestra uno: C y D infectados, A y B sanos.

![Tres pruebas: ABC devuelve 1, ABD devuelve 1 y ACD devuelve 2. Eso identifica a A y B como sanos.](attachment:detector.png)

Como el plan identifica a todos, acredita 2 sanos en promedio. El óptimo laminar obtiene 1.75; las individuales, 1.5. Son utilidades esperadas desde el inicio, con los mismos recursos.

La separación frente a laminar es $2/1.75=8/7$. El ejemplo enseña cómo combinar conteos ahorra pruebas; todavía no muestra una familia cuya ventaja crezca con $G$.

Para discutir: ¿cómo se amplía este diseño manteniendo el límite de $G$ integrantes por prueba? Lo que buscamos es identificar más personas por prueba a medida que crece $G$.
''')
    matrix.attachments['detector.png'] = deepcopy(detector.attachments['detector.png'])
    grid = cell('logg-malla', r'''
## 2. Lo que muestra la malla, y hasta dónde llega

$n=7$, $B=4$, $G\in\{2,3,4,5\}$. Este corte usa cuatro de los 32 casos ya calculados.

![Utilidad esperada dividida entre B: el general se separa del laminar, y ambos permanecen iguales al pasar de G=4 a G=5.](attachment:utilidad_por_prueba.png)

Dividimos la utilidad por el presupuesto para mirar directamente la propuesta: laminar acotado por una constante y general capaz de crecer como $\log G$ en una familia adecuada.

Con $G=3$, general obtiene 3.25 y laminar 2.5625. La pérdida laminar es 21.15% del óptimo general. Pasar a $G=4$ reduce esa pérdida; $G=5$ no cambia los valores.

Este corte no puede mostrar crecimiento indefinido: solo hay 3.5 sanos en promedio, así que $V/B\le3.5/4=0.875$. Para estudiar crecimiento hay que aumentar también la población y elegir el presupuesto de la construcción.

Para discutir: ¿qué familia $(n,B,G)$ propone el argumento de matrices? Esa familia debería orientar el siguiente experimento, en vez de ampliar $G$ con la misma población.
''', 'utilidad_por_prueba')
    next_steps = cell('logg-aporte', r'''
## 3. Qué podemos aportar a la discusión

Ya tenemos una ruta para la cota en la otra dirección: ninguna política puede acreditar más de $B\log_2(G+1)$ sanos en promedio bajo estas hipótesis. Como probar individuos es laminar y da $B/2$, el cociente general/laminar es a lo sumo $2\log_2(G+1)$ cuando $B\le n$. La prueba por información está en las notas; falta cerrar su redacción y revisión humana según el plan.

Esto limita cuánto puede crecer la ventaja. Para mostrar que sí alcanza ese orden siguen haciendo falta la construcción general y la cota para toda la clase laminar.

Nuestra aportación inmediata: explicar el ejemplo $2$ frente a $1.75$ y pedir la familia de parámetros de las matrices para poder verificarla. Un experimento pequeño no demuestra por sí solo una afirmación asintótica.

La búsqueda de Hermeneuta terminó sin mejorar los ejemplos de partida. Los encargos de binario frente a laminar, el par frente a individuales y la comparación greedy/rollout siguen en las notas de estudio, con sus figuras y respuestas para discutir. Responden a preguntas adicionales del plan; no prueban el crecimiento logarítmico.
''')
    presentation.cells = [intro, matrix, grid, next_steps]
    presentation.metadata['research_focus'] = 'log-G-2026-10-16'
    study.cells[0].source = r'''# Estudio y discusión del notebook 29

## Retomar después de la pausa: 40 minutos

1. Diez minutos: bloque 1. Dibujar ABC, ABD, ACD y reconstruir los conteos si solo C y D están infectados. Explicar qué se ahorra y por qué el valor esperado es 2. Después distinguir este plan del óptimo laminar 1.75.
2. Diez minutos: bloques 2 y 3. Leer la malla y distinguir una separación observada de la cota de información.
3. Diez minutos: bloque 4. Explicar qué mejoró en los diseños y por qué mantener bloques pequeños limita el experimento.
4. Diez minutos: bloques 5 y 6. Separar la comparación de heurísticas de la pregunta general/laminar y defender el siguiente experimento.

No hace falta dominar antes todo el paper de Nic. Para la discusión sí hace falta distinguir un ejemplo, una evidencia finita y una afirmación sobre todos los tamaños.

## Guion de cinco minutos

«El diseño pequeño ya ahorra una prueba y supera al óptimo laminar. Para estudiar cuánto puede crecer esa ventaja, necesitamos una familia de matrices y una cota para toda política laminar. Nuestra malla localiza separaciones, pero mantiene una población tan pequeña que no puede mostrar crecimiento logarítmico.»

El material anterior se conserva abajo como consulta. Los bloques binario/laminar y greedy/rollout siguen siendo encargos del plan semanal, aunque ahora el pase principal se centra en la propuesta de Francisco.
'''
    explanation = cell('logg-estudio-control', r'''
## Cómo explicar el control de 4/3

Antes de cualquier prueba, los 16 perfiles de cuatro personas son equiprobables. El historial ABC=1 deja seis: tres ubicaciones posibles del infectado en ABC, cada una combinada con los dos estados de D. Por eso ocurre con probabilidad $6/16=3/8$.

Condicionados a ese historial, A es el infectado en un tercio de los casos. Si al probar A resulta infectado, B y C son sanos: cobramos 2. En los otros dos tercios A resulta sano: cobramos 1; sigue sin saberse cuál de B y C es el infectado. D sigue desconocido en ambos casos.

El cálculo $\frac23\cdot1+\frac13\cdot2=\frac43$ es utilidad adicional de esa última prueba, condicionada al historial. Su contribución al promedio desde el inicio sería $\frac38\cdot\frac43=\frac12$; faltan las otras ramas de ABC. No es una política completa de una prueba que obtenga 4/3 desde el prior.

Respuesta provisional para discutir: el argumento «como mucho uno por prueba en cada historial» falla. Una prueba global podría asignar crédito tanto a sanos ya identificados como a información todavía pendiente de aprovechar. El ejemplo sugiere qué debe contabilizar; no hemos construido ni demostrado ese potencial.

Pregunta a Francisco: «¿Tu cota laminar controla una cantidad acumulada que incluya la información pendiente? Este historial nos da 4/3 con una sola prueba restante».

La enumeración exacta de los 16 perfiles y el solver por perfiles coinciden. Código reproducible: `augmented/notebooks/enfocar_log_g_29.py`; evidencia en `results/sesion_2026-10-16/enfoque_log_G/evidencia.json`.
''')
    study.cells = [study.cells[0], explanation] + [c for c in study.cells[1:]
                                                       if c.id != explanation.id]
    for book in (presentation, study):
        nb.validate(book)
    return presentation, study


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    paths = [root / 'augmented/notebooks' / name for name in
             ('29_tres_clases_y_contraejemplo.ipynb', '29_notas_de_estudio.ipynb')]
    books = focus(*(nb.read(p, as_version=4) for p in paths), root)
    for path, book in zip(paths, books):
        nb.write(book, path)
        print(path)
