"""Curate notebook 26 and create the new exact-comparison appendix 27.

Preserves the user's modified notebook 25. Does not run research searches.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())

SETUP = '''from pathlib import Path
import sys, json
from fractions import Fraction as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'augmented' / 'bm17_toy_solver.py').exists())
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results' / 'precio_laminar_2026-09-21'
datos = json.loads((RESULTS / 'resultados.json').read_text())
tabla = pd.read_csv(RESULTS / 'comparacion.csv')
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
'''

# Curate existing notebook by stable cell ids, without rebuilding its history.
p26 = HERE / '26_esencial.ipynb'
nb = nbf.read(p26, as_version=4)
byid = {c.id: c for c in nb.cells}
byid['3fb2ad6e'].source = '''# Contraejemplos para discutir con Francisco

**Guion de 15–20 minutos.** Primero el score que no reentra, después la batería
de cuatro heurísticas y su árbol, finalmente π_L y las dos comparaciones de óptimos.

**Convenciones:** las tablas de costo de §5 conservan la variante estricta del
ejemplo histórico. Desde §7 se usa posterior-zero. En todo el notebook,
q = P(sano), p = P(infectado), R = número de infectados.

Resultados y comentarios revisados el 21-sep. El contraste general/laminar está
en [notebook 27](27_precio_laminar_y_repaso.ipynb). No ejecutar los build-scripts
antiguos para preparar la presentación: regeneran versiones anteriores.
'''
setup_cell = nb.cells[1]
if 'ROOT = next(' not in setup_cell.source:
    setup_cell.source = SETUP + '\n' + setup_cell.source
byid['f233ee24'].source = '''## 5. El contraejemplo de no-reentrada

AB ya fue probado con R=1; C y D son vírgenes. Comparamos reentrar con A,
abrir CD y repetir AB. El viejo score V cuenta utilidad sana en el pool elegido.

**Tabla histórica, acreditación estricta.** C incluye la prueba actual y el
trabajo local posterior. Bajo posterior-zero, probar A cobra 1 con certeza,
en lugar de 0.5 en esperanza. Repetir AB sigue sin aportar información ni cobro.

El retest se conserva como diagnóstico del score antiguo. El menú actual de
Bellman por átomos ya descarta esa acción redundante.

**Pregunta para explicar:** ¿por qué un score puede asignar 1 a repetir una
prueba cuyo resultado ya conocemos? Distingue utilidad existente de utilidad
que esta acción hace cobrable.
'''
byid['b452e8bb'].source = r'''## Del contraejemplo a valor/costo

$$V/C^\alpha.$$

En la tabla histórica, retest y reentrada empatan en
$\alpha=\ln 2/\ln 2.5\approx0.756$. Este umbral compara scores antes del filtro
por presupuesto. La malla de experimentos no encontró un α que dominara en
todos los regímenes. La reparación de un ejemplo no demuestra una garantía.

**Transición oral:** «El costo ayuda a ordenar este caso. Después preguntamos
si las reglas de planificación también sobreviven a poblaciones heterogéneas».'''
byid['bbbbe4d1'].source = '''## 7. Una instancia donde falla la batería de cuatro

Seis personas, B=3, G=4, posterior-zero. Probabilidades de infección
(0.9, 0.825, 0.875, 0.8, 0.95, 0.85), utilidades (2, 1, 1, 1, 4, 2).

π_M, π_C y π_R obtienen 0.700000; C3 obtiene 0.69904125. Frente al óptimo
laminar 1.0645140625, sus ratios son 0.657577 y 0.656676, respectivamente.
Es un contraejemplo para esta batería concreta, no para todas las heurísticas.

El menú compara **primera acción + continuación óptima**. Su Q no es el valor
de ejecutar una heurística completa. Después veremos el árbol que logra el óptimo.
'''
nb.cells[12].source = '''from augmented.bm17_toy_solver import SolverLaminar
p_ce = dict(enumerate(map(F, ['.9', '.825', '.875', '.8', '.95', '.85'])))
u_ce = dict(enumerate(map(F, [2, 1, 1, 1, 4, 2])))
sol = SolverLaminar(p_ce, u_ce, 4, 'posterior_zero')
U0 = frozenset(p_ce)
OPT = float(sol.V(U0, (), 3))
NOM = 'ABCDEF'
menu = sorted(((float(sol._q_accion(U0, (), 3, a)),
                ''.join(NOM[i] for i in a[1]))
               for a in sol._acciones(U0, ())), reverse=True)
assert abs(OPT - 1.0645140625) < 1e-12 and menu[0][1] == 'AEF'
assert {s for _, s in menu[-6:]} == set('ABCDEF')
seleccion = {s for _, s in menu[:5]} | set(NOM) | {'ADEF'}
vista = [(v, s) for v, s in menu if s in seleccion]
fig, ax = plt.subplots(figsize=(9.3, 5.7))
labels, colors = [], []
for _, s in vista:
    label = s
    if s == 'AEF': label += '  óptimo laminar'
    if s == 'F': label += '  primera acción de π_M, π_C, π_R'
    if s == 'ADEF': label += '  primera acción de C3 y π_L (λ=.001)'
    labels.append(label)
    colors.append(AZUL if s == 'AEF' else VERDE if s == 'ADEF' else AMBAR if s == 'F' else '#94a3b8')
bars = ax.barh(labels, [v for v, _ in vista], color=colors)
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
ax.invert_yaxis(); ax.set_xlim(0, 1.18)
ax.set_xlabel('Utilidad esperada con continuación óptima después de esa acción')
ax.set_title('Raíz: cinco mejores opciones, ADEF y seis individuales (de 56)')
fig.tight_layout()
fig.savefig(RESULTS / 'menu_seis_personas.png', bbox_inches='tight')
plt.show()
'''
byid['2b00a5df'].source = '''**Lectura del menú.** El óptimo abre AEF. Las tres políticas del companion
empiezan por F. C3 y π_L empiezan por ADEF, pero toman continuaciones distintas
y obtienen valores distintos. No atribuir a las cuatro heurísticas la misma
primera acción. El gráfico muestra una selección del menú de 56 acciones.
'''
nb.cells[14].source = nb.cells[14].source.replace(
    "f'(el arbol completo sin compartir tendria ~116 mil nodos)')", "f'(estados de la recursion laminar)')")
byid['34548855'].source = '''**Lectura del árbol.** AEF sale completamente sano con probabilidad
0.00075. Su interés está en las continuaciones condicionadas al conteo.

Tras R(AEF)=1, probar E cobra **4 inmediatamente en ambas ramas**: si E está
sana, aporta u_E=4; si está infectada, A y F quedan sanas por deducción y
aportan 2+2. Todavía puede quedar utilidad adicional de la última prueba.

Tras R(AEF)=3, no se refina un grupo ya resuelto: se cambia de grupo.
**Pregunta:** ¿qué compra la prueba inicial además de la posibilidad de cobro inmediato?
'''
nb.cells = [c for c in nb.cells if not c.get('metadata', {}).get('repaso_21sep')]
additions = [md('''## 8. π_L y las dos referencias

π_L usa utilidad esperada menos λ por prueba esperada, con posibilidad de
abandonar un plan local. Aquí λ=0.001, horizonte=3 y no-parálisis activada.
Su valor 1.026265 equivale a 96.41% del óptimo **laminar** y 92.42% del **general**.

La corrección del cobro del complemento en π_M se verificó por enumeración.
Los valores de las tres políticas de densidad en esta instancia no cambiaron.
'''), code('''politicas = pd.DataFrame(datos['policy_six_person'])
display(politicas[['policy', 'value', 'ratio_laminar', 'ratio_general']].round(6))
display(pd.DataFrame(datos['lambda_sensitivity'])[
    ['lambda_value', 'value', 'ratio_laminar', 'ratio_general']].round(6))
'''), md('''## 9. Lo que este ejemplo permite concluir

- Las cuatro reglas estudiadas pueden perder bastante frente al óptimo laminar.
- π_L mejora este caso con los parámetros declarados. No hay garantía general derivada de él.
- El óptimo general vale 1.1104640625; exigir laminaridad conserva aquí 95.8621%.

**Cierre oral:** «Necesitamos distinguir la calidad de la heurística del costo
de la restricción laminar. Ahora podemos medir ambos en ejemplos pequeños».

Abrir [notebook 27](27_precio_laminar_y_repaso.ipynb) para ver exactamente dónde
ayudan los cruces. El caso de cuatro personas es el más sencillo para explicarlo.
''')]
for c in additions:
    c.metadata['repaso_21sep'] = True
nb.cells.extend(additions)
for c in nb.cells:
    c.metadata['slideshow'] = {'slide_type': 'slide' if c.cell_type == 'markdown' and c.source.startswith('#') else 'fragment'}
    if c.cell_type == 'code': c.outputs=[]; c.execution_count=None
nbf.write(nb, p26)

nb27 = nbf.v4.new_notebook()
nb27.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name':'python'}}
nb27.cells = [
md('''# Cuánto cuesta exigir laminaridad

**Anexo de presentación al notebook 26.** Ocho ejemplos elegidos para separar
la pérdida por una heurística de la pérdida por restringir las pruebas.

Modelo común: prior independiente, conteos exactos de infectados, posterior-zero,
utilidades no negativas, a lo sumo G personas por prueba y B pruebas por trayectoria.
Los ejemplos con historia declaran B como presupuesto **restante**.

Fuentes: perfiles y árboles en `results/precio_laminar_2026-09-21/resultados.json`.
Reproducción desde la raíz: `python -m augmented.experimento_precio_laminar`.
'''),
code(SETUP + '''
from hashlib import sha256
for path, digest in datos['source_sha256'].items():
    assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, f'Regenerar resultados: cambió {path}'
display(tabla[['caso','n','B_restante','G','opt_laminar','opt_general','ratio_laminar_general']].round(6))
'''),
md('''## 1. Comparación con la misma población y el mismo presupuesto

Las fracciones de Bellman se calculan exactamente. El óptimo laminar coincide
con dos representaciones independientes: perfiles con legalidad sobre toda la
historia y el solver previo de átomos. Cada árbol se evalúa de nuevo sobre todos
los perfiles de infección de probabilidad positiva.

Los empates de los controles no demuestran que laminar sea siempre óptimo.
Los casos con brecha sí muestran que esa igualdad falla en general.
'''),
code('''ids = ['par_B2', 'par_B3', 'cuatro_triples', 'seis_heterogeneo']
t = tabla.set_index('id').loc[ids]
labels = ['4 personas · q=.3 · B=2, G=2', '4 personas · q=.3 · B=3, G=2',
          '4 personas · q=.5 · B=3, G=3', '6 heterogéneas · B=3, G=4']
y=np.arange(len(ids)); fig, ax=plt.subplots(figsize=(9.4,4.7))
for delta, col, label, color in [(-.17,'opt_laminar','Óptimo laminar','#2563eb'),
                                (.17,'opt_general','Óptimo general','#059669')]:
    bars=ax.barh(y+delta,t[col],height=.31,label=label,color=color)
    ax.bar_label(bars,fmt='%.6f',padding=4,fontsize=9)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,2.32)
ax.set_xlabel('Utilidad esperada');ax.legend(loc='lower right')
ax.set_title('Los cruces ayudan en tres de estos cuatro ejemplos')
fig.tight_layout();fig.savefig(RESULTS/'comparacion_optimos.png',bbox_inches='tight');plt.show()
'''),
md(r'''## 2. El ejemplo conocido cambia cuando quedan tres pruebas

Cuatro personas, q=0.3, u=1, B=3, G=2. Abrimos AB.

Si R(AB)=1, quedan **dos** pruebas. El laminar puede probar A, cobrar al sano
del par y usar la última prueba en C: valor de continuación **1.3**.
El general puede cruzar **AC**:

| R(AC) | Probabilidad condicionada a R(AB)=1 | Última prueba | Utilidad total desde este estado |
|---|---:|---|---:|
| 0 | 0.15 | D | 2.3 |
| 1 | 0.50 | A | 1.3 |
| 2 | 0.35 | D | 1.3 |

Su continuación vale $0.15(2.3)+0.50(1.3)+0.35(1.3)=1.45$.
En la rama intermedia, probar A distingue (A sana, B y C infectadas) de
(A infectada, B y C sanas). D no queda identificada en esa rama.

**Ejercicio:** ¿por qué la mejora inicial no es 0.15?

<details><summary>Respuesta para comprobar después de calcular</summary>

El estado R(AB)=1 se alcanza con probabilidad 0.42. La mejora al inicio es
0.42 × (1.45 − 1.30) = **0.063**. Por eso 1.074 pasa a **1.137**.
Los otros dos resultados de AB aportan lo mismo a ambas estrategias.

</details>
'''),
code('''from augmented.bellman_perfiles import BellmanPerfiles
historia = [((0,1),1)]
lam = BellmanPerfiles(['.7']*4,[1]*4,2,laminar=True)
general = BellmanPerfiles(['.7']*4,[1]*4,2)
assert lam.solve(2,historia).value == F(13,10)
assert general.solve(2,historia).value == F(29,20)
assert general.solve(2,historia).first_pool == (0,2)
assert F(42,100)*(F(29,20)-F(13,10)) == F(63,1000)
print('Con una prueba restante:', lam.solve(1,historia).value, general.solve(1,historia).value)
print('Con dos pruebas restantes:', lam.solve(2,historia).value, general.solve(2,historia).value)
print('Mejora al inicio:', F(63,1000), '= 0.063')
'''),
md(r'''## 3. Tres pruebas cruzadas identifican a cuatro personas

Caso nuevo: n=4, q=0.5, u=1, B=3, G=3. Las pruebas **ABC, ABD y ACD**
se pueden fijar de antemano. Sus tres conteos distinguen los 16 perfiles posibles.
Por eso se identifica a toda persona sana y se alcanza la cota máxima
$\sum_i q_i u_i=2$. Bellman laminar da **1.75**: ratio **0.875**.

Este ejemplo muestra una ventaja de los cruces incluso con ese diseño fijo.
No es necesario atribuir toda la ganancia a la adaptación.

<details><summary>Cómo recuperar los cuatro estados a mano</summary>

Sean x=R(ABC), y=R(ABD), z=R(ACD), y a,b,c,d indicadores de infección.
Como x+y+z=3a+2(b+c+d), su paridad determina a.
Después b=(x+y−z−a)/2, c=(x+z−y−a)/2 y d=(y+z−x−a)/2.

</details>
'''),
code('''certificado = datos['four_person_static_certificate']
assert len({tuple(x['counts']) for x in certificado}) == 16
display(pd.DataFrame([{'ABCD infectados': ''.join(map(str,x['profile'])),
                       'R(ABC), R(ABD), R(ACD)': tuple(x['counts'])} for x in certificado]))
print('Las 16 firmas son distintas. Recuperación total con tres pruebas.')
'''),
md('''## 4. El contraejemplo de seis personas tiene dos comparaciones

Retomar el árbol del notebook 26. Ambas clases óptimas pueden comenzar con AEF.
La clase general mejora las continuaciones permitiendo cruces.

- Óptimo laminar: **1.0645140625**.
- Óptimo general: **1.1104640625**.
- Laminar/general: **0.958621**. Pérdida relativa por laminaridad: **4.14%**.

El árbol guardado localiza toda la mejora en dos ramas:

| Conteo de AEF | Siguiente pool laminar | Siguiente pool general | Ganancia ponderada desde el inicio |
|---|---|---|---:|
| 1 | E | DE, que cruza AEF | 0.004195 |
| 2 | F | ADE, que cruza AEF | 0.041755 |

Las ramas 0 y 3 empatan. Las dos ganancias suman **0.045950**.
El refinamiento de E que cobra 4 es la elección del óptimo laminar en esa
rama; al permitir cruces, otra continuación obtiene más utilidad esperada total.

Esta pérdida es distinta de la de las heurísticas frente al óptimo laminar.
'''),
code('''display(pd.DataFrame(datos['policy_six_person'])[
    ['policy','value','ratio_laminar','ratio_general']].round(6))
display(pd.DataFrame(datos['lambda_sensitivity'])[
    ['lambda_value','horizon','value','ratio_laminar']].round(6))
'''),
md('''## 5. Qué presentar y qué queda abierto

**Resultado comprobado:** hay ejemplos pequeños con una pérdida estricta por
laminaridad bajo el modelo vigente. La mayor pérdida relativa entre estos ocho
casos seleccionados es 12.5%; no es una cota general ni una búsqueda exhaustiva.

**Dos preguntas para Francisco:** ¿qué estructura explica la ganancia de cruzar?
¿Y qué política laminar puede acercarse a su propio óptimo con menor costo de cálculo?

La sensibilidad a λ forma parte de especificar π_L. Los valores se evaluaron
sobre todos los perfiles; no son una simulación. Las decisiones de las
heurísticas usan sus scores numéricos existentes.

Volver a [notebook 26](26_esencial.ipynb) para el ensayo oral. El notebook 24,
Acto 4, queda como apoyo para explicar por qué no basta invocar submodularidad
adaptativa: la misma prueba puede valer más después de obtener información.
'''),
]
for c in nb27.cells:
    c.metadata['slideshow'] = {'slide_type': 'slide' if c.cell_type=='markdown' else 'fragment'}
nbf.write(nb27, HERE/'27_precio_laminar_y_repaso.ipynb')
print('Prepared', p26.name, 'and 27_precio_laminar_y_repaso.ipynb')
