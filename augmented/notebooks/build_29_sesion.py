"""Five research questions, minimal figures, and a separate study notebook.

Run after augmented.experimento_sesion_29. Never edits notebook 28.
"""
from base64 import b64encode
from fractions import Fraction as F
from pathlib import Path
import json
import nbformat as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figuras_separacion_28 import canvas, text, box, line, png, INK, GRAY, BLUE, WARM
from figura_detector_29 import detector_figure

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/sesion_2026-10-12/evidencia_29.json'
OUT=ROOT/'augmented/notebooks'
FIG=ROOT/'results/sesion_2026-10-12/figuras'
FIG.mkdir(exist_ok=True)
d=json.loads(DATA.read_text())
images={}


def save(name,fig):
    images[name]=png(fig)
    (FIG/f'{name}.png').write_bytes(images[name])


def edge(ax,start,end,label,x,y,color=GRAY):
    line(ax,start,end,color)
    text(ax,x,y,label,12,color,bbox=dict(facecolor='white',edgecolor='none',pad=3))


# One branch, one action, one conditional final utility.
fig,ax=canvas(9.6,3.4)
box(ax,.5,.88,.10,.12,'AB',INK,17)
for x,r,prob,action,value in [(.17,0,'.09','C','2.3'),(.5,1,'.42','A','1'),(.83,2,'.49','C','0.3')]:
    edge(ax,(.5,.81),(x,.47),f'R = {r}  ·  {prob}',x,.65)
    box(ax,x,.40,.11,.12,action,BLUE,17)
    line(ax,(x,.33),(x,.21),GRAY)
    text(ax,x,.14,value,20,BLUE)
save('par',fig)

# The comparison uses exactly the same scale and convention in each panel.
fig,ax=canvas(13.2,4.2)
for center,title,color,branches in [
    (.23,'Binario · 2.432',WARM,[(.09,'−  ·  .76','C','2.8'),(.37,'+  ·  .24','AC','1.2667')]),
    (.75,'Laminar · 2.366',BLUE,[(.57,'R = 0  ·  .76','C','2.8'),(.75,'R = 1  ·  .23','A','1'),(.93,'R = 2  ·  .01','C','0.8')])]:
    text(ax,center,.94,title,19,color)
    box(ax,center,.77,.085,.12,'AB',INK,16)
    for x,label,action,value in branches:
        edge(ax,(center,.70),(x,.39),label,x,.56)
        box(ax,x,.32,.085,.12,action,color,17)
        line(ax,(x,.25),(x,.14),color)
        text(ax,x,.075,value,20,color)
save('binario_laminar',fig)

# A small detector: people are dots, each test is a contour around its pool.
save('detector',detector_figure())

# Absolute utility, direct labels; algorithmic loss and restriction loss are separate.
fig,ax=plt.subplots(figsize=(9.4,3.3))
fig.subplots_adjust(left=.18,right=.79,bottom=.24,top=.90)
case=d['homogeneous']['policies']
labels=[('general','General',3),('laminar','Laminar',2),('pi_L',r'$\pi_L$',1),('pi_ratio',r'$\pi_{\rm ratio}$',0)]
for key,label,y in labels:
    v=float(F(case[key]['value'])); color=INK if key=='general' else BLUE
    ax.plot(v,y,'o',color=color,ms=7)
    ax.text(v+.006,y,f'{v:.4f}',va='center',fontsize=12,color=color)
ax.set_yticks([y for _,_,y in labels],[label for _,label,_ in labels],fontsize=12)
ax.set(xlim=(4.5,4.8),ylim=(-.5,3.5),xlabel='Utilidad esperada')
ax.set_xticks([4.5,4.6,4.7,4.8]);ax.tick_params(axis='y',length=0,pad=10)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#cccccc')
save('heuristicas',fig)

# Supplementary local slice, kept outside the five presentation blocks.
fig,ax=plt.subplots(figsize=(9,3.8))
xs=[float(F(r['t'])) for r in d['slice']]
ys=[float(F(r['delta'])) for r in d['slice']]
ax.plot(xs,ys,color=BLUE,lw=2)
ax.fill_between(xs,ys,0,where=[y>0 for y in ys],color='#dce5ea')
ax.axhline(0,color=GRAY,lw=.8)
ax.plot(.8,.066,'o',color=WARM,ms=6)
ax.set(xlabel=r'$q_B=q_C$',ylabel=r'$V_{\rm bin}-V_{\rm lam}$',xlim=(0,1))
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save('regimen_local',fig)


def cell(source,*names):
    c=nb.v4.new_markdown_cell(source.strip())
    if names:
        c['attachments']={f'{name}.png':{'image/png':b64encode(images[name]).decode()} for name in names}
    return c


intro=r'''
# ¿Cuánto cuesta exigir laminaridad?

Comparamos utilidades esperadas óptimas con los mismos $n,B,G,q_i,u_i$.

- Binario: resultado negativo o positivo; hard clearing, se cobra al pertenecer a un pool negativo.
- Laminar: conteo de infectados; soft clearing, se cobra al identificar a un sano con certeza. En cada trayectoria, los pools son disjuntos o uno contiene al otro.
- General: el mismo conteo y soft clearing, sin la restricción laminar.

$q_i$ es probabilidad de salud, $p_i=1-q_i$. Priors independientes y sin pruebas previas en los ejemplos.
'''

c1=r'''
## 1. El par ya supera a individuales

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $u_i=1$.

Probamos AB. Si $R=1$, probar A identifica al sano de AB. Si $R=0$ o $R=2$, dedicamos la última prueba a C.

![Árbol del par; al pie, utilidad final esperada condicionada a cada rama.](attachment:par.png)

Las cifras al pie son la utilidad final esperada de cada rama, incluido lo ya cobrado.

$$V_{\rm par}=0.09(2.3)+0.42(1)+0.49(0.3)=0.774>0.6=V_{\rm individuales}.$$

La mejora viene de resolver AB con una sola prueba en las ramas extremas y aprovechar C. Es el ejemplo que quedó pendiente de enviar.

El intercambio general ya está escrito en [la nota de la proposición](../paper/proposicion_individuales_construccion.md). Para discutir: ¿qué hipótesis permiten que la ganancia sea estrictamente positiva?
'''

c2=r'''
## 2. Las cotas no ordenan binario y laminar

Con independencia inicial, $0<q_i<1$ y utilidades no negativas:

$$V_{\rm bin}\le V_{\rm aug}\le G\,V_{\rm bin},\qquad
\frac14V_{\rm bin}\le V_{\rm lam}\le V_{\rm aug}.$$

El general puede reproducir una política binaria. La cota $G$ limita cuánto puede ganar al contar. La cota $1/4$ combina los dos factores dos de la comparación binaria dinámica → estática → disjunta; véase [introducción y §5 del paper de Nic](https://arxiv.org/html/2601.22419v2).

Todavía permite que el binario supere al laminar. Encontrar ese ejemplo era una pregunta de la reunión.

Para discutir: ¿cuánto puede crecer esa separación? $1/4$ es una garantía, no un ejemplo que la alcance.
'''

c3=r'''
## 3. El binario puede ganar: basta con tres personas

$n=3$, $B=2$, $G=2$, $(q_A,q_B,q_C)=(0.95,0.80,0.80)$, $u_i=1$.

Ambas políticas óptimas empiezan por AB. Si sale negativo, prueban C. Después de un positivo, el binario puede probar AC; AB y AC no forman una familia laminar.

![Políticas óptimas binaria y laminar, con probabilidades de las ramas y valores finales condicionados.](attachment:binario_laminar.png)

Las cifras al pie son utilidades finales esperadas condicionadas a cada rama. En la rama positiva binaria, AC sale negativo con probabilidad $19/30$: cobra 2 o cobra 0, con media $19/15\approx1.2667$.

$$V_{\rm bin}=0.76(2.8)+0.24(19/15)=2.432,$$
$$V_{\rm lam}=0.76(2.8)+0.23(1)+0.01(0.8)=2.366.$$

A sigue siendo suficientemente probable sano como para volver a incluirlo con C. Esa libertad compensa aquí la menor información del binario. El general conserva ambas ventajas y obtiene $2.448$.

Verificado sobre los 8 perfiles y todas las primeras pruebas laminares. La diferencia es $0.066$; este ejemplo demuestra existencia, no la mayor brecha posible.

La búsqueda posterior encontró una familia de cuatro personas cuya ventaja binaria se aproxima al 11.11%. La construcción y sus límites están en [los resultados actualizados](../../docs/notes/2026-10-12-encargos-6-oct.md#4-actualización-de-héctor-14-oct). Para discutir: ¿qué limita cuánto puede crecer esta ventaja?
'''

c4=r'''
## 4. En $q=1/2$, varias ecuaciones pueden resolverlo todo

$n=4$, $B=3$, $G=3$, $q_i=1/2$, $u_i=1$.

El plan fijo ABC, ABD, ACD produce una firma distinta para cada uno de los 16 perfiles. Por ejemplo:

![Tres conteos 1, 1, 2 identifican A y B sanos.](attachment:detector.png)

Identifica a todos con tres pruebas. Por eso alcanza toda la utilidad disponible en promedio:

$$V_{\rm bin}=1.5,\qquad V_{\rm lam}=1.75,\qquad V_{\rm aug}=n/2=2.$$

Este ejemplo retoma la ruta de matrices detectoras que propuso Francisco. Su valor es orientar una prueba: falta acotar cuánto puede obtener cualquier política dinámica laminar.

Para discutir: ¿se puede demostrar $V_{\rm lam}\le K B$ con $K$ independiente de $n,B,G$? Junto con diseños admisibles de $B=O(n/\log n)$, descartaría una garantía laminar constante independiente de $G$. Ambas piezas deben justificarse para esa conclusión.
'''

c5=r'''
## 5. Separar el costo de la restricción del costo de la regla

$\pi_L$ compara planes locales con $E[U]-0.01E[T]$; $\pi_{\rm ratio}$, con $E[U]/E[T]$. Consideran hasta tres pruebas, o las que queden, hacen la primera y vuelven a decidir.

En los tres ejemplos anteriores, ambas alcanzan el óptimo laminar. En el bloque 3, mejorar la elección dentro de la clase laminar no puede cerrar la diferencia frente al binario.

Como control homogéneo: $n=5$, $B=3$, $G=4$, $q_i=0.95$, $u_i=1$.

![Utilidades del general, laminar y las dos heurísticas en el control homogéneo.](attachment:heuristicas.png)

Aquí sí aparecen dos pérdidas: aproximadamente $0.0945$ por laminaridad y $0.0772$ adicionales al usar cualquiera de las dos reglas. No hay evidencia en estos cuatro casos para declarar una ganadora general entre ellas.

Para discutir: ¿qué decisión de continuación explica la pérdida de las heurísticas en este control?
'''

presentation=nb.v4.new_notebook(cells=[cell(intro),cell(c1,'par'),cell(c2),cell(c3,'binario_laminar'),cell(c4,'detector'),cell(c5,'heuristicas')])
presentation.metadata.update(language_info={'name':'python'},
    research_evidence='../../results/sesion_2026-10-12/evidencia_29.json',
    generator='build_29_sesion.py',presentation=True)

study1=r'''
# Estudio y discusión del notebook 29

El ejemplo central es $V_{\rm bin}>V_{\rm lam}$ con tres personas. El par y la matriz son puntos de partida recuperados del notebook 28. La búsqueda inicial fue local; la ampliación posterior sí incluyó Hermeneuta y produjo la familia que se aproxima a $10/9$, descrita al final de estas notas.

Actualización del 14 de octubre: el [plan para el martes 20](../../docs/notes/2026-10-14-plan-semana.md) añade el argumento de información en $q=1/2$ y la comparación de greedy y rollout con el óptimo laminar. El notebook conserva sus cinco bloques; no contiene todavía esa nueva malla.

## Guion de 10 minutos

1. Par · 1 minuto. «Contar permite resolver un par y gastar la prueba restante fuera de él». Reconstruye por qué las ramas tienen valores 2.3, 1 y 0.3.
2. Cotas · 1 minuto. «Las cotas de Francisco permiten que el binario gane; no dicen si ocurre». Distingue una cota de una separación observada.
3. Contraejemplo · 4 minutos. «Ya encontramos esa separación con tres personas». Explica por qué AC sigue siendo atractivo tras AB positivo, qué impide la laminaridad y cómo se certificó su óptimo.
4. Matriz · 2 minutos. «Varias ecuaciones de conteo identifican más que una jerarquía laminar en este caso». Reconstruye la firma (1,1,2) y pregunta por el argumento dinámico que falta.
5. Reglas · 2 minutos. «Ya distinguimos la pérdida de la restricción de la pérdida de la heurística». En el contraejemplo no se corrige la primera mejorando una regla laminar.

## Tu orden de estudio

Empieza por el bloque 3: dibuja AB y sus continuaciones en el iPad; calcula cada probabilidad antes de mirar las fórmulas. Sigue con el intercambio general del bloque 1. Después trabaja el bloque 4 y la pregunta nueva: ¿cuánta información permiten B pruebas con como máximo G+1 resultados cada una? Termina distinguiendo greedy, rollout y nuestras reglas con planificación. Haz una pasada oral de 10 minutos sin leer el texto.
'''

study2=r'''
## Contraejemplo: comprobación a mano

$q_A=19/20$, $q_B=q_C=4/5$, $B=G=2$, $u_i=1$. Los tres modelos tienen los mismos recursos.

AB negativo tiene probabilidad $0.95\times0.8=0.76$. C sigue sano con probabilidad 0.8. La contribución de esa rama es $0.76(2+0.8)=2.128$ en las dos políticas.

AB positivo y AC negativo significa exactamente A sano, B infectado y C sano. Su probabilidad conjunta es $0.95(0.2)(0.8)=0.152$. Es la multiplicación más sencilla: dos sanos acreditados por $0.152$ de probabilidad. Así, la política binaria vale $2.128+2(0.152)=2.432$.

Para dibujar el árbol con probabilidades condicionales: $P(AC^-\mid AB^+)=0.152/0.24=19/30$. A queda sano con probabilidad $0.19/0.24=19/24$, y C sigue independiente con probabilidad $4/5$. Por eso probar AC sigue teniendo sentido.

El laminar observa $R=1$ con probabilidad $0.95(0.2)+0.05(0.8)=0.23$ y puede asegurar un sano probando A. Con $R=2$, probabilidad $0.01$, prueba C y cobra 0.8 en promedio. Total: $2.128+0.23+0.008=2.366$.

## ¿Cómo sabemos que no hay una política laminar mejor?

Hay seis primeras pruebas. Enumerar todas sus continuaciones legales da: A → 2.23; B o C → 2.32; AB o AC → 2.366; BC → 2.246. Su máximo es 2.366. Este cálculo directo coincide con el Bellman sobre perfiles y con el solver de átomos residuales.

Incluso sin certificar el óptimo binario, una política binaria de valor 2.432 ya separaría las clases del óptimo laminar 2.366. Además, dos solvers binarios coinciden en que esa política es óptima.

No atribuimos esta separación a un beneficio de tener menos información. El general siempre puede imitar al binario y aquí obtiene 2.448. Lo que cuesta es restringir las pruebas permitidas.
'''

study3=r'''
## Una región, además de un punto

$n=3$, $B=G=2$, $u_i=1$, $q_A=0.95$, $q_B=q_C=t$. Se evaluaron exactamente $t=0.05,0.06,\ldots,0.99$.

![Diferencia binario menos laminar al variar la salud de B y C.](attachment:regimen_local.png)

En esta malla, el binario gana en los 40 puntos de $0.55$ a $0.94$. El ejemplo presentado es $t=0.80$. Es una sección de una familia heterogénea: no describe todos los regímenes de probabilidad ni localiza fronteras continuas exactas.

La búsqueda amplia examinó 1.386 instancias exactas: 208 con binario superior, 1.142 con laminar superior y 36 empates, sin casos truncados. Incluye una malla homogénea, propuestas heterogéneas y mutaciones de buenos candidatos. Es una búsqueda dirigida, no una muestra representativa; esos porcentajes no miden cuál clase gana «normalmente».

También se encontró separación homogénea: $n=5$, $B=3$, $G=4$, $q=0.95$, $u_i=1$. Binario 4.65554625; laminar 4.6462125. El ejemplo de tres personas se eligió por su explicación más corta.
'''

study4=r'''
## De las cotas a una posible prueba

Para $V_{\rm aug}\le G V_{\rm bin}$, ordena $m_i=q_i u_i$ y llama $T_k$ a la suma de los $k$ mayores. Antes de probar por primera vez a una persona, su estado no influye en que la política decida incluirla: esa decisión depende de los otros estados y de la aleatoriedad independiente. Si $s_i$ es su probabilidad de aparecer, su utilidad esperada acreditada es a lo sumo $m_i s_i$. Como $\sum_i s_i\le BG$, resulta $V_{\rm aug}\le T_{BG}\le G T_B\le G V_{\rm bin}$. Se requiere independencia y priors interiores para que un sano no probado no quede acreditado por inferencia.

Para el factor cuatro: binario dinámico ≤ 2 × binario estático con pools arbitrarios ≤ 4 × binario estático disjunto ≤ 4 × aumentado laminar. [El paper de Nic, introducción y §5](https://arxiv.org/html/2601.22419v2), prueba el primer factor y atribuye el segundo a Finster et al. El último paso es inclusión de políticas.

Acercarse a $G$ requiere contabilizar las pruebas de localización. Si cubrimos $k$ grupos de tamaño $G$ y reservamos $\ell=\lceil\log_2G\rceil$ pruebas para localizar un sano, entonces $B=k+\ell$, $n=kG$, y la política laminar garantiza al menos $1-(1-q)^{kG}$. Para $q\le1/2$, el óptimo binario homogéneo es $Bq$ cuando $n\ge B$, por [Teorema 3.3](https://arxiv.org/html/2601.22419v2#S3.SS2). El cociente de esa garantía con $Bq$ tiende a $kG/(k+\ell)$ cuando $q\to0$. Después, $k\to\infty$ aproxima $G$. No es $G$ para $k$ fijo.

En la matriz del bloque 4, las ecuaciones son $z_A+z_B+z_C=1$, $z_A+z_B+z_D=1$, $z_A+z_C+z_D=2$, con $z_i\in\{0,1\}$. Las dos primeras dan $z_C=z_D$; la tercera obliga $z_C=z_D=1$, $z_A=0$, y entonces $z_B=0$. Son sumas enteras, no módulo dos.

Para convertir el ejemplo de matrices en una separación asintótica faltan dos piezas: una familia admisible de diseños con pocos tests y una cota para cualquier política dinámica laminar. La propuesta $V_{\rm lam}\le KB$ es una pregunta de investigación. Una cota para planes estáticos o para diagnóstico completo no la sustituye.
'''

study5=r'''
## Próximos pasos para discutir con Francisco

1. Revisar la construcción general del par en la nota de la proposición, con sus hipótesis y fronteras. La revisión computacional no sustituye la lectura y validación de Héctor.
2. Consolidar las comprobaciones independientes del solver y los CSV del plan del 14 de octubre.
3. Explicar el contraejemplo de tres personas y después la familia de cuatro: ¿qué amplifica la ventaja? No sabemos si $10/9$ es una cota superior universal.
4. En $q=1/2$, $u_i=1$, estudiar primero la cota de información y ampliar la malla en G. La cota $V_{\rm lam}\le KB$ y las construcciones de matrices quedan en el frente de Francisco según el plan semanal.
5. Comparar greedy y rollout sobre una jerarquía fija con el óptimo laminar por trayectoria. No son $\pi_L$ y $\pi_{\rm ratio}$; la pérdida puede incluir tanto la jerarquía elegida como la regla de decisión.

Preguntas para abrir la discusión: «¿Este ejemplo responde a la separación que buscabas?», «¿Qué parámetro debería amplificar la ventaja de volver a incluir A?», «¿Nos interesa una garantía con G fijo o independiente de G?» y «¿Qué cantidad asociada a los bloques laminares podría acotar la utilidad ganada por prueba?».

## Reproducibilidad

Desde la raíz del repositorio:

```sh
python -m augmented.experimento_sesion_29
python augmented/notebooks/build_29_sesion.py
python -m pytest augmented/tests_bellman_binario_exacto.py -q
```

Los experimentos están en `results/sesion_2026-10-12/evidencia_29.json`; la búsqueda completa, en `busqueda/cases.jsonl` y `busqueda/summary.json`. Los árboles se evaluaron independientemente sobre cada perfil, con aritmética racional. El binario se contrastó además con el solver clásico; el laminar, con el solver de átomos.

Para continuar buscando, el programa acepta una carpeta nueva y límites de tiempo:

```sh
python -m augmented.busca_binario_vs_laminar --out results/otra_busqueda --seconds 600 --random 1000 --adaptive 200
```

Es búsqueda de instancias con evaluadores fijos, semilla y registro de cada intento. No es una ejecución del agente de Karpathy ni modifica los solvers para mejorar resultados. Ese comando reproduce la primera etapa local; la ampliación posterior sí utilizó Hermeneuta. Sus registros y certificados están en `results/sesion_2026-10-12/brecha_ampliada/`.

## La familia posterior: una ventaja que se aproxima al 11.11%

$n=4$, $B=G=2$, $q=(s,1/2,1/2,s)$, $u=(1,3,3,2)$, con $19/20\le s<1$. Aquí también cambian las utilidades respecto del ejemplo de tres personas.

$$V_{\rm bin}=\frac52s(1+s),\qquad V_{\rm lam}=3+\frac32s^2.$$

La política binaria abre BD: si sale negativo prueba AC; si sale positivo prueba AD. La laminar abre BC: si el conteo es 1, prueba B; si es 0 o 2, prueba AD. La primera aprovecha la posibilidad de reutilizar D en otro grupo; la segunda combina la resolución del par BC con la oportunidad de acreditar AD.

En $s=0.99$, binario obtiene 4.92525 y laminar 4.47015. Al acercar s a 1, el cociente binario/laminar tiende a $10/9$. Es una separación alcanzable como límite, no una prueba de que ningún caso la supere. No se sustituye $s=1$: entonces A y D ya son sanos conocidos y el valor laminar cambia.

La comprobación simbólica compara todas las primeras pruebas y continuaciones mediante 454 desigualdades en el intervalo. Se reproduce con `python -m pytest augmented/tests_busqueda_brecha_ampliada.py -q`; la nota de resultados explica las ramas y enlaza el certificado. La búsqueda numérica sugirió la familia; el certificado comprueba sus fórmulas.
'''

study=nb.v4.new_notebook(cells=[cell(study1),cell(study2),cell(study3,'regimen_local'),cell(study4),cell(study5)])
study.metadata.update(language_info={'name':'python'},generator='build_29_sesion.py')
from alinear_plan_29 import align
presentation,study=align(presentation,study,cell,images,ROOT)
from enfocar_log_g_29 import focus
presentation,study=focus(presentation,study,ROOT)
from base_29 import write_unless_base
write_unless_base(presentation,OUT/'29_tres_clases_y_contraejemplo.ipynb')
nb.write(study,OUT/'29_notas_de_estudio.ipynb')
from simplificar_29 import build as build_presentation
build_presentation()
for path in [OUT/'29_tres_clases_y_contraejemplo.ipynb',OUT/'29_notas_de_estudio.ipynb']:
    nb.validate(nb.read(path,as_version=4))
    print(path)
