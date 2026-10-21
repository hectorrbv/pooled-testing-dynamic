"""Build the concise research notebook following the 29 September meeting.

Run after python -m augmented.experimento_sesion_28. Figures are retained with
the evidence and embedded in the notebook; no experiment runs on opening it.
"""
from pathlib import Path
from hashlib import sha256
from fractions import Fraction as F
import base64
import io
import json

import graphviz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sesion_2026-10-04'
EVIDENCE = OUT/'evidencia_28.json'
data = json.loads(EVIDENCE.read_text())
for path,digest in data['source_sha256'].items():
    assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
FIGURES=OUT/'figures';FIGURES.mkdir(parents=True,exist_ok=True)
INK,GRAY,TEAL,ORANGE = '#172033','#64748b','#00856a','#b95716'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
                     'axes.spines.top':False,'axes.spines.right':False})
images,dots = {},{}


def graph(name):
    g=graphviz.Digraph(name)
    g.attr(rankdir='TB',bgcolor='white',pad='.15',nodesep='.35',ranksep='.5',dpi='140',ordering='out')
    g.attr('node',shape='box',style='rounded',fontname='Helvetica',fontsize='16',
           color=TEAL,fontcolor=INK,penwidth='1.5',margin='.15,.12')
    g.attr('edge',fontname='Helvetica',fontsize='13',color=GRAY,fontcolor=GRAY,arrowsize='.65')
    return g


def keep_graph(name,g):
    images[name]=g.pipe(format='png');dots[name]=g.source


def keep_plot(name,fig):
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',pad_inches=.16,facecolor='white')
    images[name]=buf.getvalue();plt.close(fig)


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

g=graph('tres_cruces')
g.attr(nodesep='.7')
with g.subgraph() as row:
    row.attr(rank='same')
    for name in ['ABC','ABD','ACD']:
        row.node(name,f'Probar {name}',color=ORANGE)
g.node('decode','Los tres conteos distinguen\nlos 16 perfiles de salud')
for name in ['ABC','ABD','ACD']:g.edge(name,'decode')
g.node('all','Identificamos a todos\nE[U] = 2');g.edge('decode','all')
keep_graph('tres_cruces',g)

fig,ax=plt.subplots(figsize=(12,4.3));ax.set_xlim(0,12);ax.set_ylim(0,4.3);ax.axis('off')
def box(x,y,w,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,1,boxstyle='round,pad=.04',
                               linewidth=1.6,edgecolor=color,facecolor='white'))
    ax.text(x+w/2,y+.5,text,ha='center',va='center',fontsize=13,color=INK)
def arrow(x0,x1,y,color):
    ax.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.8})
ax.text(.1,4.0,'El paper: pruebas binarias y hard clearing',fontsize=15,color=GRAY)
box(.1,2.65,2.65,'Política dinámica',GRAY)
box(3.5,2.65,3.7,'Camino simulado\ny recorte aleatorio',GRAY)
box(8,2.65,3.7,'Asignación estática\npuede tener cruces',GRAY)
arrow(2.82,3.42,3.15,GRAY);arrow(7.27,7.92,3.15,GRAY)
ax.text(.1,1.95,'Nuestra meta: conteos y soft clearing',fontsize=15,color=TEAL)
box(.1,.55,2.65,'Política general\ncon cruces',TEAL)
box(3.5,.55,3.7,'Transformación\npor construir',ORANGE)
box(8,.55,3.7,'Política laminar\nmisma B y G',TEAL)
arrow(2.82,3.42,1.05,TEAL);arrow(7.27,7.92,1.05,TEAL)
keep_plot('dos_transformaciones',fig)

pilot=data['crossing_pilot']
values=[float(F(pilot[k]['value'])) for k in ['original','trimmed_with_reuse','first_pool_then_ratio']]
assert values==[2,1.5,1.75]
fig,ax=plt.subplots(figsize=(10.3,3.4))
fig.subplots_adjust(left=.30,right=.93,bottom=.23,top=.87)
ax.barh([2,1,0],values,color=[GRAY,ORANGE,TEAL],height=.52)
ax.set_yticks([2,1,0],labels=['Plan con cruces','Recortar y reutilizar','Primera prueba + cociente'])
ax.tick_params(axis='y',length=0,pad=12)
ax.set_xlim(0,2.18);ax.set_xticks([0,.5,1,1.5,2]);ax.set_xlabel('Utilidad esperada',labelpad=8)
ax.axvline(1.75,color=TEAL,ls='--',lw=1)
ax.text(1.75,2.48,'Óptimo laminar: 1.75',ha='center',fontsize=11,color=TEAL)
ax.spines['left'].set_visible(False)
for y,value in zip([2,1,0],values):ax.text(value+.035,y,f'{value:.2f}',va='center',fontsize=13,color=INK)
keep_plot('piloto_conversion',fig)

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

add(r'''
## 1. Con una sola prueba, agrupar puede perder

$n=2$, $B=1$, $G=2$, $q_A=0.9$, $q_B=0.1$, $u_A=u_B=1$.
Estados independientes, conteo de infectados y soft clearing.

![Dos árboles: probar A obtiene 0.90; probar AB obtiene 0.18 porque solo acredita cuando ambos están sanos.](attachment:una_prueba.png)

La individual A obtiene 0.90; el grupo AB, $2(0.9)(0.1)=0.18$.
Esto refuta «agrupar siempre es óptimo». No resuelve qué ocurre con dos o más pruebas.
''',95,'una_prueba',notes='''Encargo recogido en el resumen Fathom: un contraejemplo con poco presupuesto y probabilidades distintas. No hace falta infección cero ni salud cierta. Con R=1 las dos identidades del sano siguen siendo posibles, aunque no sean equiprobables. La individual B vale .1, de modo que A es óptima entre todas las acciones legales. La conclusión se refiere a agrupar dos o más personas; las estrategias de pooled testing suelen permitir también singletons. El contraejemplo no refuta que exista una política óptima dentro de la clase que permite grupos. Tampoco refuta una afirmación restringida a B>=2. Ver la consulta sobre sustituir dos individuales por un par.''')

add(r'''
## 2. Los cruces pueden aportar utilidad

$n=4$, $B=3$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

![Tres pruebas fijas, ABC, ABD y ACD, producen una firma distinta para cada uno de los 16 perfiles y permiten identificar a todos.](attachment:tres_cruces.png)

ABC, ABD y ACD se cruzan. Sus conteos identifican a todos: utilidad esperada 2.
El óptimo laminar es 1.75. Aquí imponer laminaridad cuesta 0.25.

La pregunta es cuánto valor podemos garantizar al eliminar los cruces en general.
''',120,'tres_cruces',notes='''Evidencia previa del 21 de septiembre, recalculada para esta sesión. Una familia es laminar si, en cada trayectoria, dos grupos cualesquiera son disjuntos o uno contiene al otro. Aquí cualquier par de triples comparte dos personas sin contenerse. El mapa perfil -> tres conteos es inyectivo en los 16 perfiles: es un certificado independiente del óptimo general. Identificar a todos alcanza la cota superior E[numero de sanos]=2. Dos solvers laminares concuerdan en 7/4. No es una cota universal del precio de laminaridad. La pérdida de clase aquí es 1.75/2=7/8; la constante universal sigue abierta. Para comprobar la inyectividad a mano: si dos perfiles dan los mismos tres conteos, su diferencia debe ser (-2t,t,t,t); siendo diferencias de bits, t solo puede ser cero. La política con cruces es estática, no necesita adaptar, y sigue siendo admisible en la clase dinámica general.''')

add(r'''
## 3. Tomar la idea de la prueba, construir otra transformación

![El paper convierte una política binaria dinámica en una asignación estática aleatoria; nuestra meta es convertir una política con conteos en una política laminar.](attachment:dos_transformaciones.png)

En [§5 del paper](https://arxiv.org/html/2601.22419v2#S5), la asignación estática conserva al menos la mitad del valor en promedio.
Buscamos una garantía para políticas laminares que sigan respondiendo a los resultados reales. Una constante universal sigue abierta.

Recortar al azar no basta: en el camino AB, AC, ambos grupos pueden quedar intactos y seguir cruzándose.
''',120,'dos_transformaciones',notes='''Leer primero la estrategia de prueba de §5 y el teorema 5.5; después §5.1 con el ejemplo del apéndice B y §5.2-5.4. El perfil simulado selecciona un camino y es independiente del perfil real usado para evaluar. Cada aparición persona-prueba se retiene con probabilidad 1/2; por eso las dos apariciones de A tienen monedas distintas. En AB,AC, las cuatro apariciones sobreviven con probabilidad 1/16, y permanece un cruce. Esto es un obstáculo estructural a copiar la construcción, no un contraejemplo a su teorema. Otra diferencia: con conteos y soft clearing una persona puede acreditarse por inferencia sin participar en una prueba negativa. El caso AB=1 y luego A infectado acredita B; la representación binaria de oportunidades de acreditación requiere revisión. Se conserva B y G en cada política realizada, no solo en promedio. La constante que buscamos no se fija en 1/2 por analogía. El objetivo teórico y las hipótesis están en la consulta.''')

add(r'''
## 4. Un primer piloto: la continuación también importa

Mismo setup: $n=4$, $B=3$, $G=3$, $q_i=0.5$, $u_i=1$.

![El plan con cruces obtiene 2, recortar y reutilizar obtiene 1.50, y conservar ABC y continuar con el cociente obtiene 1.75.](attachment:piloto_conversion.png)

Recortar los grupos deja ABC y D. Reutilizar la tercera prueba da 1.50.
Si conservamos solo ABC y después decide $\pi_{ratio}$, que compara utilidad por pruebas esperadas, obtenemos 1.75: aquí alcanza el óptimo laminar.

Siguiente experimento: seguir la política con resultados reales hasta el primer cruce y comparar continuaciones.
''',125,'piloto_conversion',notes='''Piloto nuevo, no pedido literalmente por Francisco y no garantía general. Primera conversión: mantener de cada grupo solo personas no aparecidas antes; ABC,ABD,ACD -> ABC,D,vacío. Ejecutar los grupos no vacíos y usar cualquier presupuesto sobrante en la acción laminar con mayor cobro inmediato. Sin reutilización valdría .875; con reutilización vale1.5. Segunda conversión: conservar ABC y descartar el resto del calendario, continuando con la política ratio existente (horizonte maximo3, alpha1). Vale1.75. Cuando ABC es mixto, ratio refina dentro de ABC antes de decidir si dedica una prueba a D; la primera regla prueba D demasiado pronto. Como tres cuartos de los perfiles tienen ABC mixto, cambiar el orden importa. La pérdida total .5 de la primera regla se divide en .25 por exigir laminaridad y .25 por la regla de conversión. En este caso la segunda recupera toda la parte evitable. No llamar a este resultado una validación del muestreo de caminos: el plan fuente es fijo, por lo que aquí no se probó la parte aleatoria. Las utilidades están integradas exactamente sobre los mismos16 perfiles y ambas salidas respetan B,G y laminaridad. No afirmamos que la conversión basada en ratio sea computacionalmente eficiente a escala.''')

add(r'''
## Consulta. Dos parámetros distintos

$\lambda$ es el precio por prueba que elegimos en $\pi_L$: $E[U]-\lambda E[T]$.

$\Lambda$ es una cantidad determinada por la instancia del paper:

$$\Lambda(J,G)=\max_{1\leq|A|\leq G}\sum_{i\in A}\prod_{j\in A\setminus\{i\}}q_j.$$

Los singletons dan 1, por lo que $\Lambda\geq1$. Si todos los $q_i\leq1/2$, entonces $\Lambda=1$ y su modelo binario admite un óptimo individual. [§3.2](https://arxiv.org/html/2601.22419v2#S3.SS2).

No son el mismo parámetro. En el ejemplo anterior de $\lambda=0.30$ y $0.31$ usábamos $q_i=0.20$, $p_i=0.80$, y ambos abrían el mismo trío.
''',consultation=True,cell_id='consulta-lambdas')

add(r'''
## Consulta. Qué sigue abierto después del contraejemplo

Con $B\geq2$, $G\geq2$, $n>B$, priors independientes $0<q_i<1$ y utilidades positivas, una política con un par mejora la mejor política formada solo por individuales.

Ordena por $u_iq_i$. Sustituye las dos primeras individuales por probar el par y, solo si es mixto, probar a uno de sus miembros. Recuperas la misma información y utilidad de ambos. Si el conteo es cero o dos, ahorras una prueba y la dedicas a la persona $B+1$.

La mejora es

$$[q_1q_2+(1-q_1)(1-q_2)]u_{B+1}q_{B+1}>0.$$

La política construida es laminar y mejora el mejor plan de individuales. No demuestra que abrir cualquier grupo sea óptimo, ni que toda primera acción óptima deba agrupar.
''',consultation=True,cell_id='consulta-alcance-B')

add(r'''
## Consulta. Objetivo y materiales

Queremos una transformación $T$ que, dada una política general $\pi$, produzca políticas laminares admisibles y satisfaga

$$E_\omega[V(T(\pi,\omega))]\geq cV(\pi).$$

$\omega$ es azar independiente de la población real; $V$ ya promedia sobre la salud. La constante $c>0$ debería ser independiente de $n,B,G$ para obtener la garantía deseada. Cada salida respeta el presupuesto y el tamaño de grupo. La política transformada puede seguir siendo adaptativa.

Si se demostrara para toda $\pi$, implicaría $V^*_{laminar}\geq cV^*_{general}$. Con $G$ fijo ya hay una garantía básica de $1/G$. Una constante independiente también de $G$ sigue abierta para políticas dinámicas laminares; para planes estáticos laminares no existe tal constante universal.

La revisión también separa fijar un camino y eliminar cruces: una política ya laminar de valor 0.774 baja a 0.5304 al reproducir un camino virtual; la mejor estática laminar vale 0.60. Por eso el próximo control conserva las decisiones tomadas con resultados reales y compara continuación óptima con $\pi_{ratio}$.

[Revisión con Opus y dirección de prueba](../../docs/notes/2026-10-04-revision-y-direccion-prueba.md).

[Guía y próximos pasos](../../docs/notes/2026-10-04-preparacion-siguiente-sesion.md) · [Evidencia exacta](../../results/sesion_2026-10-04/evidencia_28.json).

Reproducir desde la raíz del repositorio: `python -m augmented.experimento_sesion_28`, seguido de `python augmented/notebooks/build_28_sesion.py`.
''',consultation=True,cell_id='consulta-meta')

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'title':'Del contraejemplo al precio de imponer laminaridad',
    'language_info':{'name':'python'},'presentation_seconds':sum(c.metadata.talk_seconds for c in cells),
    'evidence_sha256':{str(EVIDENCE.relative_to(ROOT)):sha256(EVIDENCE.read_bytes()).hexdigest()},
    'research_scope':'Counterexample closed; two conversion pilots on one instance; approximation theorem open.',
    'source_meeting':'User-provided Fathom summary, September 29, 2026; not a verbatim transcript.',
    'discussion_notes':'docs/notes/2026-10-04-preparacion-siguiente-sesion.md'})
nbf.validate(nb)
assert all('**' not in c.source and 'posterior-zero' not in c.source for c in cells)
nbf.write(nb,Path(__file__).with_name('28_pooling_y_precio_laminar.ipynb'))
for name,png in images.items():(FIGURES/f'{name}.png').write_bytes(png)
for name,source in dots.items():(FIGURES/f'{name}.dot').write_text(source)
print(f'4 pantallas, 3 consultas; {nb.metadata.presentation_seconds/60:.1f} minutos')
