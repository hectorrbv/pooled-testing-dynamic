"""Build a six-screen presentation and a separate paper-study notebook.

Run after python -m augmented.experimento_mapa_laminar_28.
The original evidence from October 4 is retained and reused. Figures are
embedded: opening the notebook never runs experiments.
"""
from pathlib import Path
from hashlib import sha256
import base64
import json

import graphviz
import nbformat as nbf
from figuras_28 import make_figures
from figura_agrupar_individuales import make_grouping_figure
from presentacion_28 import add_presentation
from contexto_28 import add_context_consultations

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sesion_2026-10-06'
EVIDENCE = ROOT/'results/sesion_2026-10-04/evidencia_28.json'
MAP_EVIDENCE = ROOT/'results/sesion_2026-10-05/mapa_laminar_28.json'
GROUP_EVIDENCE = ROOT/'results/diseno_mapa_2026-10-05/agrupar_vs_individuales.json'
COMPARISON_EVIDENCE = OUT/'tres_referencias_28.json'
REPLAY_EVIDENCE = ROOT/'results/sesion_2026-10-04/revision_opus/verificacion_direccion.json'
data = json.loads(EVIDENCE.read_text())
grid = json.loads(MAP_EVIDENCE.read_text())
replay = json.loads(REPLAY_EVIDENCE.read_text())
grouping = json.loads(GROUP_EVIDENCE.read_text())
comparison = json.loads(COMPARISON_EVIDENCE.read_text())
for evidence in [data,grid,grouping,comparison]:
    for path,digest in evidence['source_sha256'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
assert sha256((ROOT/replay['source']).read_bytes()).hexdigest()==replay['source_sha256']
assert grid['audit']['independent_policy_checks']==228
assert grid['audit']['independent_atom_solver_checks']==114
assert grouping['audit']['exact_cases']==228
assert grouping['audit']['independent_profile_evaluations']==228
assert sum(row['gain_percent']>0 for row in grouping['rows'])==170
assert sum(row['gain_percent']==0 for row in grouping['rows'])==58
assert comparison['reference_sha256']==sha256(GROUP_EVIDENCE.read_bytes()).hexdigest()
assert comparison['summary']['cases']==228
assert comparison['summary']['general_beats_laminar']==110
assert comparison['audit']['profile_and_atom_solver_agreements']==190
FIGURES=OUT/'figures';FIGURES.mkdir(parents=True,exist_ok=True)
INK,GRAY,TEAL,ORANGE = '#262626','#777777','#262626','#262626'
images,dots = {},{}


def graph(name):
    g=graphviz.Digraph(name)
    g.attr(rankdir='TB',bgcolor='white',pad='.15',nodesep='.35',ranksep='.5',dpi='140',ordering='out')
    g.attr('node',shape='plaintext',fontname='Helvetica',fontsize='14',
           fontcolor=INK,margin='.08,.08')
    g.attr('edge',fontname='Helvetica',fontsize='12',color=GRAY,fontcolor=GRAY,arrowhead='none',penwidth='.8')
    return g


def keep_graph(name,g):
    images[name]=g.pipe(format='png');dots[name]=g.source


g=graph('una_prueba')
with g.subgraph(name='cluster_a') as c:
    c.attr(label='Probar A',fontname='Helvetica',fontsize='18',fontcolor=TEAL,color='white')
    c.node('A','Individual A\nE[U] = 0.90')
    c.node('A0','A sano\nCobramos 1');c.node('A1','A infectado\nCobramos 0')
    c.edge('A','A0','R = 0\n90%');c.edge('A','A1','R = 1\n10%')
with g.subgraph(name='cluster_ab') as c:
    c.attr(label='Probar AB',fontname='Helvetica',fontsize='18',fontcolor=ORANGE,color='white')
    c.node('AB','Grupo AB\nE[U] = 0.18',color=ORANGE)
    for r,prob,label in [(0,'9%','Ambos sanos\nCobramos 2'),
                         (1,'82%','Un sano, sin saber cuál\nCobramos 0'),
                         (2,'9%','Ambos infectados\nCobramos 0')]:
        c.node('AB'+str(r),label,color=ORANGE)
        c.edge('AB','AB'+str(r),f'R = {r}\n{prob}')
keep_graph('una_prueba',g)

new_images,figure_data=make_figures(grid,data,replay)
images.update(new_images)
images['agrupar_vs_individuales'],figure_data['agrupar_vs_individuales']=make_grouping_figure(grouping)
images['laminar_vs_individuales'],figure_data['laminar_vs_individuales']=make_grouping_figure(
    comparison,field='laminar_gain_percent',
    label='Mejora frente a individuales (%)\n100 × (laminar − individuales) / individuales',
    normalization='100*(optimal_laminar/individual_only - 1)',
    annotations='highlight_only',outline_positive=True,
    highlight={'B':2,'G':2,'q':.3,'label':'+29%'},mark_tie_rows=True)
images['general_vs_laminar'],figure_data['general_vs_laminar']=make_grouping_figure(
    comparison,field='crossing_loss_percent',
    label='Pérdida por exigir laminaridad (%)\n100 × (general − laminar) / general',
    normalization='100*(1 - optimal_laminar/optimal_general)',
    annotations='highlight_only',outline_positive=True,
    highlight={'B':4,'G':3,'q':.5,'label':'16.7% · 3 frente a 2.5'},
    palette='warm',mark_tie_rows=True)

cells=[]
def add(source,seconds=0,figure=None,notes='',consultation=False,cell_id=None):
    c=nbf.v4.new_markdown_cell(source.strip(),metadata={
        'slideshow':{'slide_type':'skip' if consultation else 'slide'},
        'talk_seconds':seconds,'speaker_notes':notes})
    c.id=cell_id or f'sesion28-{len(cells)+1}'
    if figure:
        c.attachments={f'{figure}.png':{'image/png':base64.b64encode(images[figure]).decode()}}
        if figure in dots:c.metadata.graphviz_source=dots[figure]
    cells.append(c)

add_presentation(add)
add_context_consultations(add)

add(r'''
## Consulta. El mapa anterior con cuatro personas

$n=4$, $B\in\{2,3,4\}$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$, $u_i=1$.

![Pérdida relativa por exigir laminaridad en la malla anterior de 114 casos con n=4.](attachment:mapa_precio_laminar.png)

Con $B=2$ hay empate en esta malla; con $B=3$ la pérdida alcanza 12.5%; con $B=4=n$ todos pueden probarse individualmente. El empate con cuatro pruebas se debe aquí al tamaño de población. En la malla nueva de $n=6$, cuatro pruebas sí pueden separar ambos óptimos.
''',figure='mapa_precio_laminar',consultation=True,cell_id='consulta-mapa-n4',
notes='Se conserva la malla anterior de114casos. El porcentaje es100(1−laminar/general). La comparación principal ya usa n6 para los tres valores: individual, laminar y general.')

add(r'''
## Consulta. Tres conteos positivos pueden identificar a los sanos

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $u_i=1$.

![Las pruebas ABC, ABD y ACD devuelven 1, 1 y 2 infectados. Juntas fuerzan que A y B sean sanos y C y D infectados.](attachment:conteos_identifican.png)

El dibujo muestra una realización. Los tres conteos distinguen los 16 perfiles posibles: el plan completo obtiene 2 de utilidad esperada. El óptimo laminar, aun adaptándose, obtiene 1.75.

El aporte de estas pruebas no laminares es combinar información para deducir identidades, incluso sin una prueba negativa.
''',0,'conteos_identifican',consultation=True,cell_id='sesion28-mecanismo',notes='''El caso ya existía, pero ahora mostramos qué información aporta. R=(1,1,2) tiene solución única (A,B,C,D)=(0,0,1,1), con 1=infectado. Por ABC=ABD se tiene C=D. Si C=D=0, ACD no puede valer2; luego C=D=1, A=0 y después B=0. Esto explica una realización, no basta para justificar el valor esperado. La inyectividad se verificó para los16 perfiles: diferencia de dos soluciones con los mismos conteos sería (-2t,t,t,t), que solo es vector de diferencias de bits si t=0. Identificar a todos alcanza E[sanos]=2, una cota absoluta. Laminar vale7/4 por dos solvers. La familia de grupos no es laminar y el plan es fijo. Frase oral: aquí permitir familias no laminares ya ayuda sin adaptación. No confundir este contraste con la brecha dinámica-estática del paper. Para el mismo resultado binario, los tres grupos solo serían positivos: el perfil todos infectados seguiría siendo compatible.''')

add(r'''
## Consulta. Fijar un camino puede perder valor antes de imponer laminaridad

$n=4$, $B=2$, $G=2$, $q_i=0.3$, $u_i=1$.

![Tras probar AB, se prueba C si R es cero o dos, y A si R es uno. Las probabilidades de las ramas son 0.09, 0.42 y 0.49; sus utilidades totales esperadas son 2.30, 1 y 0.30.](attachment:adaptacion_y_muestreo.png)

La política del árbol vale 0.774; el mejor plan fijo laminar, 0.600.

Un camino simulado elige AB → A (42%) o AB → C (58%). Al aplicar esas listas a otra población independiente, valen 0.60 y 0.48:

$$0.42(0.60)+0.58(0.48)=0.5304.$$

Dentro de la clase laminar, fijar las decisiones cuesta 0.174. Este muestreo pierde otros 0.0696.

La política inicial ya es laminar: aquí se pierde valor sin modificar su laminaridad. Es un control con conteos, sin el recorte aleatorio de Nick.
''',0,'adaptacion_y_muestreo',consultation=True,cell_id='sesion28-perdidas',notes='''No atribuir estas cifras al modelo binario de Nick ni a su teorema. La política completa: AB; R=1 -> A, R=0/2 -> C. E[U]=.09*2.3+.42*1+.49*.3=.774. Muestrear un camino virtual da AB,A con prob.42 y AB,C con prob.58. Sobre una población independiente, cada calendario vale.60 y.48, respectivamente; media .5304. No existe una elección posterior de segunda acción según los resultados reales. El mejor fijo laminar vale.6, obtenido con dos individuales: cada átomo de tamaño m aporta m(.3)^m<=.3, y hay a lo sumo dos átomos. La pérdida de adaptación dentro de la clase laminar es .774-.6, no .774-.5304. Si permitimos planes fijos no laminares, AB,AC vale .627 y es el óptimo estático general de este caso, verificado enumerando los 56 planes de hasta dos pruebas. Por eso .174 no es la brecha entre dinámico general y estático general. §5 de Nick ya advierte que muestrear un camino sin el recorte aleatorio puede perder mucho; este cálculo es un control propio con conteos, no una novedad de la técnica. Frase oral: el muestreo introduce una pérdida propia incluso cuando la política ya es laminar. Pregunta: merece la pena conservar esta parte de la construcción si nuestro destino puede seguir siendo dinámico.''')

add(r'''
## Consulta. Replanear cambia dónde gastamos la última prueba

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $u_i=1$.
$\pi_{ratio}$ compara planes locales de hasta tres pruebas, limitados por el presupuesto, por $E[U]/E[T]$. Ejecuta una prueba y vuelve a decidir.

Ambas reglas conservan ABC. Quitar repetidos elimina a quienes ya aparecieron: ABC → ABD → ACD queda en ABC → D. La prueba libre se dedica al cobro inmediato.

Si ABC ya está resuelto, ambas reglas prueban D.

![Tras ABC mixto, quitar repetidos lleva a probar D primero. Replanear prueba A y dedica la última prueba a D solo si ABC ya quedó resuelto; en otro caso refina con B.](attachment:orden_continuacion.png)

Las políticas completas valen 1.50 y 1.75. Frente al plan no laminar de valor 2, la primera pierde 0.50: 0.25 por la clase laminar y 0.25 por la regla. Replanear recupera aquí la segunda parte.

Propuesta: seguir la política general con resultados reales y cambiar antes de ejecutar la primera prueba incompatible con la laminaridad. Comparar continuación laminar óptima, cociente y cociente desde el inicio.
''',0,'orden_continuacion',consultation=True,cell_id='sesion28-direccion',notes='''Se mantienen las dos políticas verificadas del piloto anterior. Quitar repetidos de ABC,ABD,ACD deja ABC,D; usar presupuesto libre en cobro inmediato da1.50. ConservarABC y usar ratio da1.75. La figura resalta la rama ABCmixto, prob.75%; los valores1.50/1.75 corresponden a toda la política, incluyendo R0yR3. Dadas las ramas mixtas, A resuelve ABC con prob.1/3: (R_ABC,R_A)=(1,1) o(2,0); queda una prueba paraD. En otro caso se pruebaB para resolverBC. Ambas políticas usanD cuando ABCyaestáresuelto. El árbol se cotejó con los16 trazos guardados. La continuación óptima antes de la primera prueba incompatible con la laminaridad y el control ratio desdeel inicio son el siguiente experimento, no resultados deesta figura. Para cada grupo nuevo o bloque mixto y cada límite entre 1 y min(3,b), la implementación usa un plan local que maximiza utilidad y después compara sus cocientes; no maximiza el cociente sobre todos los árboles posibles. En este piloto conserva ABC y luego quedan como máximo dos pruebas. El cálculo local es exacto y el ejemplo no demuestra eficiencia a escala ni una garantía general. Fraseoral: antes de buscar una constante, separar pérdida de clase, compromiso con el prefijo y pérdida de la heurística.''')

add(r'''
## Para trabajar. Mis tareas para la sesión del 6 de octubre

Según el [plan del 5–6 de octubre](../../docs/notes/2026-10-05-plan-mon-tue.md), en este orden:

1. §5 del paper: preparar una página con los pasos de la demostración. Para cada paso: sirve tal cual, necesita otra idea o falla con conteos; explicar por qué en una frase.
2. Conversión laminar: trabajar los ocho casos existentes. Fijar qué pruebas respetan la laminaridad y cómo desempatar; cambiar antes de ejecutar la primera prueba incompatible. Comparar continuaciones general, laminar óptima y $\pi_{ratio}$, ponderadas por la probabilidad de llegar, y el óptimo laminar desde el inicio. Separar las pérdidas por laminaridad, por conservar el prefijo y por la heurística.
3. Revisión con Vladimir: leer la proposición sobre individuales, ejecutar `pytest augmented/tests_brecha_convencion.py` y cerrar la revisión pendiente de la Parte 2; registrar acuerdo u objeciones.

Lunes 5: revisión antes de las 20:00. Martes 6: reunir comentarios entre 12:00 y 13:00, ensayo a las 16:00 y sesión a las 19:00. Horas de Praga.
''',consultation=True,cell_id='consulta-tareas-hector',notes='''Agenda de trabajo personal, fuera del pase de presentación. Fuente: plan del 5–6 de octubre, §§5 y5-bis. El memo y el experimento de conversión laminar siguen pendientes; los pilotos del notebook son antecedentes. Los ocho casos se dividen en seis desde historial vacío y dos con historial condicionado. Antes del experimento, fijar la reducción de ancestros y preferir acciones laminares entre óptimas. Conservar además el control de cociente desde el principio de nuestra guía. Una revisión automática o un pytest verde no sustituyen la revisión matemática de Héctor ni su aprobación de la nota de Vladimir. El plan prevé combinar los comentarios con Vladimir y enviarlos antes de la sesión; esta celda solo registra el encargo.''')

add(r'''
## Para estudiar el paper. La simulación pertenece a la demostración

Nick: prueba binaria, hard clearing. Negativo significa que todos están sanos; positivo, que hay al menos un infectado.

![La población ficticia selecciona un camino. Sus grupos se recortan aleatoriamente y se evalúan en otra población independiente. Es una construcción matemática del paper.](attachment:nick_construccion.png)

La meta de [§5](https://arxiv.org/html/2601.22419v2#S5) es demostrar que existe un plan fijo con al menos la mitad del valor de una política dinámica. Si el promedio de los planes construidos alcanza esa mitad, alguno de ellos también la alcanza.

El recorte retiene cada aparición de una persona con probabilidad $1/2$. Esas monedas son independientes entre grupos, incluso para la misma persona. Este proceso no es cómo deciden $\pi_L$ y $\pi_{ratio}$.
''',figure='nick_construccion',consultation=True,cell_id='consulta-nick-construccion',notes='''Consulta de estudio, no pantalla principal de presentación. Empezar con la diferencia regla/lista y ejemplo AB→A frente a AB→CD. La población virtual selecciona el camino; el plan se fija antes de observar la población de evaluación. No prometer que cada realización conserve la mitad, ni que cada plan construido sea bueno. La desigualdad promedia sobre perfiles virtuales, reales y todas las monedas. El destino del paper no exige laminaridad. Para AB,AC hay probabilidad1/16 de que ambas pruebas queden intactas: el recorte no asegura laminaridad. La prueba no se aplica tal cual a conteos/soft clearing y no demuestra una garantía de laminaridad. No se afirma que el factor dos sea falso en otro modelo. Incluso una entrada laminar ABC,AB puede producir AC,AB tras el recorte aleatorio, con probabilidad 1/32: este recorte puede romper la laminaridad. No sustituir la prueba por esta ilustración de sus pasos.''')

add(r'''
## Para estudiar §5.4. Las dos orientaciones pesan lo mismo

Una persona, $q=0.3$ sano y $p=0.7$ infectado. Sorteamos su estado dos veces, independientemente y con el mismo prior.

![Cuatro resultados posibles de dos sorteos: sano-sano 0.09, sano-infectado 0.21, infectado-sano 0.21 e infectado-infectado 0.49. Al condicionar en que difieren, las dos orientaciones tienen probabilidad un medio.](attachment:nick_dos_perfiles.png)

Aunque $q\ne1/2$, si sabemos que los dos estados difieren, cada orientación tiene probabilidad $1/2$. Con varias personas independientes, ocurre lo mismo en cada coordenada donde los perfiles difieren. [§5.4](https://arxiv.org/html/2601.22419v2#S5.SS4).

Podemos reutilizar esta identidad probabilística. Todavía necesitamos otra desigualdad que controle cuánta utilidad pierde nuestra transformación laminar.
''',figure='nick_dos_perfiles',consultation=True,cell_id='consulta-nick-simetria',notes='''Una persona sirve para entender el acoplamiento antes de introducir subcubos. 0 representa sano y1infectado en nuestro código; el dibujo evita imponer notación de bits. La independencia da q(1-q)=(1-q)q. El evento que diferencia las orientaciones tiene probabilidad2q(1-q); el condicional es1/2. Esto no implica que los perfiles sigan siendo independientes tras condicionar: en una coordenada libre son complementarios. Para cada persona se fija el par noordenado: sanoslosdos, infectadoslosdos o diferentes. Las coordenadas diferentes conservan orientaciones independientes entrepersonas por el priorproducto. El lema combinatorio debe demostrarse antes de poder promediarlo sobre estas configuraciones. No confundir el1/2 de la orientación con demostrar el factor2 de utilidad.''')

add(r'''
## Consulta. Con una sola prueba, agrupar puede perder

$n=2$, $B=1$, $G=2$, $q_A=0.9$, $q_B=0.1$, $u_A=u_B=1$.
Conteos y soft clearing; estados independientes.

![Probar A obtiene 0.90 y probar AB obtiene 0.18.](attachment:una_prueba.png)

La individual A obtiene 0.90; el grupo AB, $2(0.9)(0.1)=0.18$.
Refuta que siempre convenga agrupar dos o más personas. No resuelve la afirmación restringida a $B\geq2$.
''',figure='una_prueba',consultation=True,cell_id='consulta-contraejemplo-B1',notes='''Se conserva el contraejemplo solicitado en el acta. A individual es óptima entre las tres acciones legales A,B,AB; Bvale.1. TrasR1 las dos identidades posibles tienenprobabilidadpositiva aunque noseanequiprobables. El mapa principal estudia otra pregunta: costo de exigir laminaridad, con B=2,3,4 y q homogéneo. No confundir ambos contraejemplos.''')

add(r'''
## Consulta. Dos parámetros distintos

$\lambda$ es el precio por prueba que elegimos en $\pi_L$: $E[U]-\lambda E[T]$.

$\Lambda$ es una cantidad determinada por la instancia del paper:

$$\Lambda(J,G)=\max_{1\leq|A|\leq G}\sum_{i\in A}\prod_{j\in A\setminus\{i\}}q_j.$$

Los singletons dan 1, por lo que $\Lambda\geq1$. Si todos los $q_i\leq1/2$, entonces $\Lambda=1$ y su modelo binario admite un óptimo individual. [§3.2](https://arxiv.org/html/2601.22419v2#S3.SS2).

No son el mismo parámetro. En el ejemplo del [notebook 27](27_sesion_francisco.ipynb), con $\lambda=0.30$ y $0.31$ usábamos $q_i=0.20$, $p_i=0.80$, y ambos abrían el mismo trío.
''',consultation=True,cell_id='consulta-lambdas')

add(r'''
## Consulta. Qué sigue abierto después del contraejemplo

Con conteos y soft clearing, $B\geq2$, $G\geq2$, $n>B$, priors independientes $0<q_i<1$ y utilidades positivas, una política con un par mejora la mejor política formada solo por individuales.

Ordena por $u_iq_i$. Sustituye las dos primeras individuales por probar el par y, solo si es mixto, probar a uno de sus miembros. Recuperas la misma información y utilidad de ambos. Si el conteo es cero o dos, ahorras una prueba y la dedicas a la persona $B+1$.

La mejora es

$$[q_1q_2+(1-q_1)(1-q_2)]u_{B+1}q_{B+1}>0.$$

La política construida es laminar y mejora el mejor plan de individuales. No demuestra que abrir cualquier grupo sea óptimo, ni que toda primera acción óptima deba agrupar.

En el plan, el caso general sigue como derivación verificada: falta cerrar la nota escrita y la revisión de Héctor.
''',consultation=True,cell_id='consulta-alcance-B')

add(r'''
## Consulta. Objetivo y materiales

Queremos una transformación $T$ que, dada una política general $\pi$, produzca políticas laminares admisibles y satisfaga

$$E_\omega[V(T(\pi,\omega))]\geq cV(\pi).$$

$\omega$ es azar independiente de la población real; $V$ ya promedia sobre la salud. La constante $c>0$ debería ser independiente de $n,B,G$ para obtener la garantía deseada. Cada salida respeta el presupuesto y el tamaño de grupo. La política transformada puede seguir siendo adaptativa.

Si se demostrara para toda $\pi$, implicaría $V^*_{laminar}\geq cV^*_{general}$. Desde historial vacío, con priors independientes de soporte completo, ya hay una garantía básica de $1/G$; el argumento está en la nota enlazada abajo. No se aplica directamente a un historial condicionado. Una constante independiente también de $G$ sigue abierta para políticas dinámicas laminares; para planes estáticos laminares no existe tal constante universal.

La revisión también separa fijar un camino e imponer laminaridad: una política ya laminar de valor 0.774 baja a 0.5304 al reproducir un camino virtual; la mejor estática laminar vale 0.60. Por eso el próximo control conserva las decisiones tomadas con resultados reales y compara continuación óptima con $\pi_{ratio}$.

[Revisión con Opus y dirección de prueba](../../docs/notes/2026-10-04-revision-y-direccion-prueba.md).

[Guion de la presentación y orden de estudio](../../docs/notes/2026-10-06-guion-notebook-28.md).

[Guía y próximos pasos](../../docs/notes/2026-10-04-preparacion-siguiente-sesion.md) · [Evidencia exacta](../../results/sesion_2026-10-04/evidencia_28.json).

Los mapas se calculan con `python -m augmented.experimento_mapa_laminar_28`, `python -m augmented.experimento_agrupar_individuales` y `python -m augmented.experimento_tres_referencias_28`. Para reconstruir ambos notebooks desde los datos guardados: `python augmented/notebooks/build_28_sesion.py`.
''',consultation=True,cell_id='consulta-meta')

main_cells=[c for c in cells if c.metadata.slideshow.slide_type!='skip']
study_cells=[c for c in cells if c.metadata.slideshow.slide_type=='skip']
study_order=['consulta-contexto-sesion28-agrupar','consulta-contexto-sesion28-par',
             'consulta-contexto-sesion28-cierre',
             'consulta-tareas-hector','consulta-nick-construccion','consulta-nick-simetria',
             'consulta-mapa-n4','sesion28-mecanismo','sesion28-perdidas','sesion28-direccion',
             'consulta-contraejemplo-B1','consulta-lambdas','consulta-alcance-B','consulta-meta']
study_cells.sort(key=lambda c:study_order.index(c.id))
nb=nbf.v4.new_notebook(cells=main_cells,metadata={
    'title':'Agrupar y exigir laminaridad: dos comparaciones',
    'language_info':{'name':'python'},'presentation_seconds':sum(c.metadata.talk_seconds for c in cells),
    'evidence_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                       for p in [EVIDENCE,MAP_EVIDENCE,REPLAY_EVIDENCE,GROUP_EVIDENCE,COMPARISON_EVIDENCE]},
    'research_scope':'Main presentation: a pair-first tree, laminar versus individual-only optima, then general versus laminar optima in the same 228 n=6 cases. Local rules, conversion pilots and paper study remain separate consultation topics.',
    'source_meeting':'User-provided Fathom summary, September 29, 2026; not a verbatim transcript.',
    'discussion_notes':'docs/notes/2026-10-06-guion-notebook-28.md'})
nbf.validate(nb)
assert len(main_cells)==6
assert all('**' not in c.source and 'posterior-zero' not in c.source for c in cells)
nbf.write(nb,Path(__file__).with_name('28_pooling_y_precio_laminar.ipynb'))
intro=nbf.v4.new_markdown_cell('''# Notebook 28 · material de estudio

[Volver a la presentación](28_pooling_y_precio_laminar.ipynb).

La presentación principal separa dos preguntas: cuánto mejora el óptimo laminar frente a individuales y cuánto pierde frente al óptimo general. Conservamos aquí los ejemplos y las consultas para profundizar:

- Agrupar frente a individuales: el mapa y el ejemplo del par.
- Evaluar una regla concreta: el piloto de conversión y su continuación.
- Estudiar §5 de Nick: planes estáticos, políticas dinámicas y la demostración.
- Revisar tareas, cálculos y ejemplos adicionales.

Salvo en las consultas del paper de Nick: conteos exactos, soft clearing, estados independientes, q = probabilidad de estar sano, p = 1−q y utilidad 1 por persona.
''',id='consulta-introduccion')
study=nbf.v4.new_notebook(cells=[intro,*study_cells],metadata={
    'title':'Notebook 28: árboles, tareas y estudio de la sección 5',
    'language_info':{'name':'python'},'evidence_sha256':nb.metadata.evidence_sha256})
nbf.validate(study)
nbf.write(study,Path(__file__).with_name('28_notas_de_estudio.ipynb'))
for name,png in images.items():(FIGURES/f'{name}.png').write_bytes(png)
for name,source in dots.items():(FIGURES/f'{name}.dot').write_text(source)
(OUT/'datos_figuras_28.json').write_text(json.dumps(figure_data,indent=2)+'\n')
print(f'{len(main_cells)} pantallas; {len(study_cells)} consultas en archivo separado; {nb.metadata.presentation_seconds/60:.1f} minutos')
