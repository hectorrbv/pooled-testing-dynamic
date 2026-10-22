"""Add the audited loop results to notebook 29 without rerunning experiments.

Preserves existing cells and attachments; replaces only the current discussion
and inserts two presentation cells plus two study cells. Safe to run again.
"""
from base64 import b64encode
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/sesion_2026-10-20/loops_revision'
BOOKS = ROOT / 'augmented/notebooks'
BLUE, GRAY, INK = '#276582', '#7b8187', '#20272c'


def read_csv(path, delimiter=','):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def evidence():
    audit = json.loads((OUT / 'loop1_audit.json').read_text())
    assert audit['complete'] and len(audit['targets']) == 6
    assert all(r['difference'] == '0' for r in audit['targets'])
    design = ROOT / 'augmented/autoresearch_logg/design_best.py'
    assert hashlib.sha256(design.read_bytes()).hexdigest() == audit['design_sha256']
    baseline = read_csv(design.parent / 'results_logg.tsv', '\t')[0]
    scorer = ROOT / 'augmented/autoresearch_q1/scorer_best.py'
    scorer_sha = hashlib.sha256(scorer.read_bytes()).hexdigest()
    selected = [r for r in read_csv(scorer.parent / 'results_q1.tsv', '\t')
                if r['status'] == 'ok' and r['scorer_sha'] == scorer_sha[:10]][-1]
    metrics = []
    for G in (3, 5):
        path = ROOT / f'results/sesion_2026-10-19/q1_n50/q1_n50_G{G}_B5.csv'
        rows = read_csv(path)
        count = len(rows)
        mean = lambda k: statistics.mean(float(r[k]) for r in rows)
        draws = {int(r['draws']) for r in rows}
        assert len(draws) == 1 and all(int(r['G']) == G and int(r['B']) == 5 for r in rows)
        comparisons = {}
        for new, base in [('rollout', 'greedy'), ('lookahead', 'greedy'), ('lookahead', 'rollout')]:
            delta = [float(r[f'w_{new}']) - float(r[f'w_{base}']) for r in rows]
            difference = statistics.mean(delta)
            se = statistics.stdev(delta) / count**.5
            margin = float(t.ppf(.975, count - 1)) * se
            scale = 100 / mean(f'w_{base}')
            comparisons[f'{new}_vs_{base}'] = dict(
                difference=difference, se=se, percent=difference * scale,
                se_percent=se * scale,
                scaled_difference_ci95=[(difference-margin)*scale, (difference+margin)*scale])
        metrics.append(dict(G=G, n=50, B=5, instances=count, draws=draws.pop(),
                            comparisons=comparisons,
                            seconds={p: mean(f's_{p}') for p in ('greedy', 'rollout', 'lookahead')},
                            cost_ratio=mean('s_lookahead') / mean('s_rollout'),
                            source=str(path.relative_to(ROOT)),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(loop1=audit, loop1_baseline=baseline, loop2_selected=selected,
                scorer_sha256=scorer_sha, n50=metrics,
                interval_method='Student-t interval on per-instance paired differences, '
                'scaled by observed mean comparator welfare; approximate, not an exact ratio CI.',
                time_unit='Seconds per instance for all draws in that G setting, with cache reuse.')


def png(fig):
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=180, bbox_inches='tight', pad_inches=.15, facecolor='white')
    plt.close(fig)
    return output.getvalue()


def figures(data):
    rows = [r for r in data['loop1']['targets'] if r['B'] == 4]
    gs = [r['G'] for r in rows]
    values = [float(Fraction(r['value']) / 4) for r in rows]
    baseline = [float(Fraction(data['loop1_baseline'][f'value_n16_B4_G{g}']) / 4) for g in gs]
    fig, ax = plt.subplots(figsize=(7.7, 3.1))
    fig.subplots_adjust(left=.13, right=.74, bottom=.22, top=.94)
    for ys, color, label, style in [(values, BLUE, 'Encontrado', 'o-'),
                                    (baseline, GRAY, 'Inicial', 'o--')]:
        ax.plot(gs, ys, style, color=color, lw=1.8, markersize=5)
        ax.text(16.5, ys[-1], label, color=color, va='center', fontsize=12)
    ax.set(xticks=gs, yticks=[.5, .6, .7, .8, .9], xlim=(1.5, 16.2), ylim=(.47, .91),
           xlabel='G permitido', ylabel='Utilidad esperada / B')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=GRAY)
    images = {'loop1_diseno': png(fig)}

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.9), gridspec_kw={'width_ratios':[1.3, 1]})
    fig.subplots_adjust(left=.08, right=.96, bottom=.25, top=.96, wspace=.48)
    for y, row in zip([1, 0], data['n50']):
        delta = row['comparisons']['lookahead_vs_rollout']
        lo, hi = delta['scaled_difference_ci95']
        axes[0].errorbar(delta['percent'], y,
                        xerr=[[delta['percent'] - lo], [hi - delta['percent']]],
                        fmt='o', color=BLUE, capsize=4, lw=1.8, markersize=6)
        axes[1].barh(y, row['cost_ratio'], color=BLUE, height=.35)
        axes[1].text(row['cost_ratio']+.2, y, f"{row['cost_ratio']:.1f}×", va='center', fontsize=12, color=INK)
    axes[0].axvline(0, color=GRAY, linestyle='--', lw=1)
    axes[0].set(xlim=(-1.3, 1.95), xticks=[-1, 0, 1], xlabel='Mejora sobre rollout (%)')
    axes[1].axvline(1, color=GRAY, linestyle='--', lw=1)
    axes[1].set(xlim=(0, 11.6), xticks=[1, 5, 10], xlabel='Tiempo relativo a rollout')
    for ax in axes:
        ax.set(ylim=(-.6, 1.6), yticks=[1, 0], yticklabels=['G = 3', 'G = 5'])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(colors=GRAY, axis='y', length=0)
    images['loop2_utilidad_costo'] = png(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in images.items():
        (OUT / f'{name}.png').write_bytes(content)
    return images


def markdown(key, text, images=None):
    cell = nb.v4.new_markdown_cell(text.strip(), id=key)
    if images:
        cell.attachments = {f'{name}.png': {'image/png': b64encode(content).decode()}
                            for name, content in images.items()}
    return cell


def add_matrix_result(presentation):
    evidence_dir = ROOT / 'results/sesion_2026-10-20/matriz15_laminar'
    data = json.loads((evidence_dir / 'resultados.json').read_text())
    laminar, general = Fraction(data['laminar']['value']), Fraction(data['general']['value'])
    assert Fraction(data['laminar']['replay']['value']) == laminar
    assert data['matrix']['distinct_count_vectors'] == 2**15
    assert general == Fraction(15, 2) and data['general']['optimal']
    loss = 100 * float(1 - laminar/general)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    fig.subplots_adjust(left=.23, right=.97, bottom=.25, top=.96)
    labels, values = ['Laminar óptimo', 'General (matriz)'], [float(laminar), float(general)]
    bars = ax.barh(labels, values, color=['#34536b', '#986347'], height=.48)
    for bar, label in zip(bars, [f'{float(laminar):.3f}', f'{float(general):g}']):
        ax.text(bar.get_width()+.12, bar.get_y()+bar.get_height()/2,
                label, va='center', fontsize=13, color=INK)
    ax.set(xlim=(0, 8.45), xticks=[0, 2, 4, 6, 8], xlabel='Sanos identificados en promedio')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(colors=GRAY, length=0)
    ax.tick_params(axis='y', labelsize=12, labelcolor=INK)
    figure = png(fig)
    (evidence_dir / 'comparacion.png').write_bytes(figure)
    cell = markdown('logg-matriz15', rf'''
## Una matriz de 15 personas y 9 pruebas

$n=15$, $B=9$, $G=7$, $q=1/2$, $u=1$.

El diseño de [Söderberg y Shapiro (1963, §3)](https://www.jstor.org/stable/2312835) distingue los $32\,768$ perfiles. Su grupo más grande tiene siete personas. Identifica a todos y alcanza el óptimo general: $15/2=7.5$ sanos en promedio.

![Con quince personas, nueve pruebas y grupos de hasta siete: el óptimo laminar identifica 6.210 sanos en promedio y la matriz general identifica 7.5.](attachment:matriz15_comparacion.png)

El óptimo laminar, calculado con Bellman exacto, es $203479/32768\approx{float(laminar):.4f}$. Verificamos su política sobre los $32\,768$ perfiles, respetando laminaridad y $B=9$ en cada trayectoria.

Exigir laminaridad reduce la utilidad un {loss:.1f}% en este caso. Por unidad de presupuesto, $V/B\approx{float(laminar/9):.3f}$ para laminar y ${float(general/9):.3f}$ para la matriz. Esta separación finita todavía no demuestra crecimiento $\log G$.
''', {'matriz15_comparacion': figure})
    cell.metadata['evidence'] = '../../results/sesion_2026-10-20/matriz15_laminar/resultados.json'
    presentation.cells = [c for c in presentation.cells if c.id != cell.id]
    position = next(i for i, c in enumerate(presentation.cells) if c.id == 'logg-cota')
    presentation.cells.insert(position + 1, cell)


def update(presentation, study=None):
    data = evidence()
    images = figures(data)
    small, large = data['n50']
    selected = data['loop2_selected']
    worst_loss = 100 * (1 - float(selected['worst_ratio']))
    presentation.cells = [c for c in presentation.cells if c.id != 'logg-control']
    for c in presentation.cells:
        if c.id == 'logg-pregunta':
            intro = c.source.split('Cómo leerlo:')[0].split('Primero: separación, control y cotas.')[0].split('Primero: separación, malla y cotas.')[0]
            c.source = intro.rstrip() + (
                '\n\nPrimero: separación, malla y cotas. Después: qué aportó la búsqueda de diseños '
                'y qué conviene probar. La comparación final de heurísticas usa otro setup y responde '
                'a una pregunta distinta: cuánto vale planificar más dentro de una familia laminar.')
        elif c.id == 'logg-regimenes':
            c.source = c.source.replace('dos grupos que se cruzan', 'dos grupos no laminares')
            c.source = c.source.replace('aunque se cruce con los anteriores', 'sin exigir esa relación con los anteriores')
        elif c.id == 'logg-ejemplo':
            c.source = c.source.replace('Los grupos se cruzan: el plan no es laminar.',
                                         'ABC y ABD comparten personas y ninguno contiene al otro: el plan no es laminar.')
        elif c.id == 'logg-malla':
            c.source = c.source.replace('## 3. Lo que muestra la malla', '## 2. Lo que muestra la malla')
            c.source = c.source.replace('y la población tapa el crecimiento.',
                'es un techo fijo. Esto impide estudiar crecimiento indefinido, pero no explica por sí solo ese empate.')
        elif c.id == 'logg-cota':
            c.source = c.source.replace('## 4. Lo que ya acota la ventaja', '## 3. Lo que ya acota la ventaja')

    add_matrix_result(presentation)
    loop1 = markdown('logg-loop1', r'''
## 4. Mejoró el diseño; queda limitada su escala

$n=16$, $B=4$. Comparamos el diseño inicial con el encontrado por la búsqueda, no con el óptimo general.

![Utilidad por prueba de los diseños inicial y encontrado, según G.](attachment:loop1_diseno.png)

El diseño encontrado gana en los cuatro casos. Pero trabaja en bloques separados de hasta seis personas y prueba como máximo cinco juntas. Permitir $G=8$ o $G=16$ no cambia sus opciones. La meseta es una restricción de la búsqueda; no refuta el crecimiento $\log G$.

Los seis objetivos del loop, incluidos los de $B=8$, coinciden exactamente con una evaluación independiente de los $2^{16}$ perfiles.

Siguiente experimento: permitir grupos que combinen bloques y revisar la primera decisión que mejore el valor. Después, hacer crecer $n$, $B$ y el tamaño efectivo de los grupos.
''', {'loop1_diseno': images['loop1_diseno']})
    loop2 = markdown('logg-loop2', rf'''
## 5. Heurísticas: ¿compensa planificar más?

$n=50$, $B=5$, $G\in\{{3,5\}}$; $q_i$ uniforme en $[0,1]$ y $u_i$ uniforme en $\{{1,2,3\}}$. Las tres reglas usan la misma partición fija y la misma selección de pruebas permitidas dentro de ella. La utilidad se estima por Monte Carlo sobre los mismos perfiles.

Greedy elige el cobro inmediato. Rollout compara una primera prueba suponiendo que después seguirá greedy. La regla nueva explora varias decisiones hasta cuatro pruebas. Ambas vuelven a decidir tras cada resultado.

![Ganancia de la regla nueva sobre rollout con incertidumbre, y tiempo relativo.](attachment:loop2_utilidad_costo.png)

Rollout mejora a greedy un {small['comparisons']['rollout_vs_greedy']['percent']:.2f}% en $G=3$ y un {large['comparisons']['rollout_vs_greedy']['percent']:.2f}% en $G=5$. La regla nueva añade {small['comparisons']['lookahead_vs_rollout']['percent']:.2f}% y {large['comparisons']['lookahead_vs_rollout']['percent']:.2f}%, pero los intervalos aproximados del 95% incluyen cero. Cuesta {small['cost_ratio']:.1f} y {large['cost_ratio']:.1f} veces más en estas mediciones.

Mantendría rollout como referencia práctica. No hay evidencia clara de que la regla nueva lo mejore aquí; tampoco hemos demostrado que rindan igual. Esta comparación no mide el precio de exigir laminaridad.
''', {'loop2_utilidad_costo': images['loop2_utilidad_costo']})
    closing = markdown('logg-aporte', r'''
## 6. Qué proponemos discutir

1. «La búsqueda mejoró diseños pequeños, pero dejó fijo el tamaño de los bloques». Probar grupos que combinen bloques puede revelar un mecanismo nuevo. Repetir bloques pequeños no basta para mostrar crecimiento con $G$.
2. «Falta limitar a toda política laminar». Una mejora del diseño general da una construcción alcanzable; para la separación $\log G$ sigue haciendo falta la cota laminar $CB$ en la misma familia.
3. «Planificar más ayuda frente a greedy; profundizar más aún no se justifica por costo». Comparar rollout y la regla nueva en casos reservados, con las mismas pruebas permitidas y el mismo límite de tiempo.

Prioridad: un piloto de diseños que combinen bloques, con evaluación independiente del árbol ganador. La nueva búsqueda nocturna queda como propuesta.
''')
    added_ids = {'logg-loop1', 'logg-loop2', 'logg-aporte'}
    presentation.cells = [c for c in presentation.cells if c.id not in added_ids] + [loop1, loop2, closing]
    presentation.metadata.update(research_focus='log-G-2026-10-20',
                                 loop_evidence='../../results/sesion_2026-10-20/loops_revision/interpretacion.json',
                                 generator='simplificar_29.py + actualizar_loops_29.py')

    if study is not None:
        study.cells[0].source = study.cells[0].source.replace(
            '2. Diez minutos: bloque 2. Enumerar los tres lugares posibles del único infectado de ABC. Resolver las dos ramas de probar A. Distinguir el 4/3 condicionado a llegar de la utilidad de toda una política.',
            '2. Diez minutos: bloques 2 y 3. Leer la malla y distinguir una separación observada de la cota de información.')
        study.cells[0].source = study.cells[0].source.replace(
            '3. Diez minutos: bloque 3. Leer un punto de cada curva. Explicar por qué una población fija impide observar crecimiento indefinido, aunque ampliemos G.',
            '3. Diez minutos: bloque 4. Explicar qué mejoró en los diseños y por qué mantener bloques pequeños limita el experimento.')
        study.cells[0].source = study.cells[0].source.replace(
            '4. Diez minutos: bloques inicial y 4. Formular con palabras las dos piezas pendientes y qué establece la cota de información. Dejar su demostración técnica para una segunda pasada.',
            '4. Diez minutos: bloques 5 y 6. Separar la comparación de heurísticas de la pregunta general/laminar y defender el siguiente experimento.')
        study.cells[0].source = study.cells[0].source.replace(
            'Verificamos una dificultad para una prueba simple: en ciertos historiales, una sola prueba laminar puede acreditar más de un sano en promedio. ', '')
        route = ('Para esta actualización, empieza por las dos notas siguientes: qué verificamos '
                 'del diseño y qué permite concluir la comparación con rollout. Después vuelve '
                 'a la ruta de estudio del argumento $\\log G$.')
        if route not in study.cells[0].source:
            title, rest = study.cells[0].source.split('\n', 1)
            study.cells[0].source = title + '\n\n' + route + '\n' + rest
        details1 = markdown('loops-estudio-diseno', r'''
## Actualización: qué confirmó la auditoría del diseño

Frase para presentar: «Verificamos el valor del diseño, y encontramos que aumentar el límite $G$ ya no amplía las pruebas que puede elegir».

Se recorrieron los $65\,536$ perfiles equiprobables por objetivo. En cada hoja se acredita solo a quienes están sanos en todos los perfiles compatibles. El evaluador usa sumas enteras y fracciones; no llama al juez original ni usa el valor interno del optimizador. Se reutiliza el generador del diseño, por lo que esto verifica su valor, no vuelve a optimizar toda la clase general.

Los resultados coinciden en los seis objetivos. Para $B=4$, los valores son $11/4$, $107/32$, $3459/1024$ y $3459/1024$ en $G=2,4,8,16$. Para $B=8$, ambos valores son $6769/1024$. Los árboles completos quedaron guardados junto a `loop1_audit.json`.

¿Por qué la curva plana no decide la conjetura? El diseño abre bloques de hasta seis personas, usa pools de hasta cinco y nunca combina bloques. Al pasar de $G=8$ a $G=16$ se conserva exactamente la familia disponible. Además, con $n=16$ hay ocho sanos en promedio: con $B=8$, cualquier política cumple $V/B\le1$.

¿Es buena dirección permitir grupos entre bloques? Sí, como experimento para saber si esa restricción cuesta utilidad. No sabemos si es la única manera de mejorar: ampliar los bloques también cambia la familia. Un piloto debe separar ambos cambios. Para explorar $\log G$, los bloques o componentes usados deberán crecer con $G$, y también la población disponible.

Piloto propuesto: mantener primero los seis objetivos y el diseño actual como candidato de referencia; añadir por separado pools entre bloques o bloques mayores. Mismo evaluador, límites de tiempo y memoria fijados antes de correr. Guardar el primer árbol que mejore, su decisión distinta y el tamaño real de los pools. Validarlo independientemente y repetir en parámetros nuevos antes de una búsqueda larga. Una mejora finita no prueba la cota laminar ni crecimiento asintótico.
''')
        details2 = markdown('loops-estudio-regla', rf'''
## Actualización: qué compara realmente el experimento de heurísticas

Frase para presentar: «La búsqueda encontró una regla casi exacta en los casos usados para seleccionarla. En $n=50$, todavía no vemos que compense su costo frente a rollout».

La regla conservada tiene SHA `{data['scorer_sha256'][:10]}`. En las 125 celdas de selección alcanza el óptimo en 124 y su mayor pérdida es {worst_loss:.3f}%, en $n=6,B=3,G=4,q=0.7$. La batería se reutilizó durante la búsqueda: ese número no es una garantía para casos nuevos. Tampoco debe compararse directamente con la pérdida anterior del 5.26% de rollout: aquella medición usa otra malla y una jerarquía fija que permite menos grupos. Primero hay que igualar casos y acciones.

En $n=50$, $p_i$ se sortea uniforme en $[0,1]$, $q_i=1-p_i$ y $u_i$ uniforme en $\{{1,2,3\}}$. Se ordena por $q_i u_i$ y se fijan bloques de tamaño $G$. Cada decisión considera los bloques ya tocados y los tres intactos con mayor ganancia inmediata. Las tres reglas comparten esta restricción; no se compara con el óptimo laminar.

Rollout evalúa una primera acción seguida de greedy hasta agotar el presupuesto, y reconsidera tras observar el resultado real. La regla nueva mira hasta cuatro pruebas: en el primer nivel interno conserva los cuatro mejores candidatos inmediatos; en los siguientes, el mejor. Además añade el mejor de cada tamaño cuando quedan varios niveles. La última evaluación es de ganancia inmediata, no una continuación greedy hasta el final. La versión de $n=50$ reproduce esta regla con inferencia exacta por bloque.

La simulación usa los mismos perfiles para las tres políticas: 48 instancias × 24 sorteos en $G=3$ y 24 × 12 en $G=5$. El porcentaje es la diferencia media de utilidad dividida entre la media de la referencia. El error estándar se calcula sobre las diferencias pareadas por instancia, no tratando cada sorteo como una instancia nueva. Los segmentos son intervalos t aproximados del 95% para esa diferencia, escalados por la media observada de rollout; no son intervalos exactos para un cociente.

La ventaja adicional de la regla es {small['comparisons']['lookahead_vs_rollout']['percent']:.2f}% (intervalo aproximado {small['comparisons']['lookahead_vs_rollout']['scaled_difference_ci95'][0]:.2f}% a {small['comparisons']['lookahead_vs_rollout']['scaled_difference_ci95'][1]:.2f}%) y {large['comparisons']['lookahead_vs_rollout']['percent']:.2f}% ({large['comparisons']['lookahead_vs_rollout']['scaled_difference_ci95'][0]:.2f}% a {large['comparisons']['lookahead_vs_rollout']['scaled_difference_ci95'][1]:.2f}%). Que ambos incluyan cero significa que no se distingue una ventaja con esta precisión; no prueba igualdad.

Los tiempos son el total medio por instancia para todos sus sorteos, reutilizando cachés: {small['seconds']['rollout']:.1f} s frente a {small['seconds']['lookahead']:.1f} s en $G=3$, y {large['seconds']['rollout']:.1f} s frente a {large['seconds']['lookahead']:.1f} s en $G=5$. No son segundos por decisión, y los dos valores de $G$ usan distinto número de sorteos. La validación a $10^{{-6}}$ del harness en casos de un bloque es un control de implementación; no demuestra optimalidad ni valida todas las políticas laminares.

¿Qué haría después? Mantener rollout como referencia y reservar instancias nuevas antes de ajustar otra regla. Comparar ambos métodos con las mismas acciones y un límite común de tiempo, reportando utilidad y pérdidas además de la media. Solo después decidir si interesa mayor profundidad, diversidad de tamaños o cambiar la partición. Este frente mejora decisiones dentro de una clase; no resuelve la pregunta general/laminar.
''')
        study.cells = [c for c in study.cells if c.id not in {'loops-estudio-diseno', 'loops-estudio-regla'}]
        study.cells[1:1] = [details1, details2]
        study.metadata['loops_update'] = '2026-10-20'
    (OUT / 'interpretacion.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return presentation, study


def main():
    presentation_path = BOOKS / '29_tres_clases_y_contraejemplo.ipynb'
    study_path = BOOKS / '29_notas_de_estudio.ipynb'
    presentation, study = update(nb.read(presentation_path, 4), nb.read(study_path, 4))
    for path, book in [(presentation_path, presentation), (study_path, study)]:
        nb.validate(book)
        from base_29 import write_unless_base
        write_unless_base(book, path)
        print(path)


if __name__ == '__main__':
    main()
