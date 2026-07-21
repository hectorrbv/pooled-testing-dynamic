# Plan semanal: del notebook 22 al atlas de razones (21–27 de julio de 2026)

Plan para dos personas durante una semana. Sustituye al sprint de dos días del
mismo día. No hay especialistas: ambas personas alternan teoría, código y
redacción a lo largo de la semana (A: teoría–código–teoría–código–redacción;
B: código–teoría–código–teoría–código), y la propiedad es por entregable, no
por habilidad. La sincronía diaria de media hora al cierre revisa cruzado:
quien demostró revisa los tests del otro y quien programó revisa los
enunciados. Los bloques están pensados para que ninguna persona espere a la
otra.

## El marco que fijó Francisco

Al modular las características de la población (número de personas, tasas de
activos, utilidades), medir el desempeño de cuatro cantidades y las razones
worst-case/best-case entre ellas en función de esas características:

1. óptimo dinámico aumentado ($V^*$);
2. óptimo laminar aumentado ($V^{\mathcal L}$);
3. greedy laminar aumentado;
4. óptimo estático binario.

Este marco no sustituye la dirección previa: la robustece con dos giros. El
primero es que la razón $V^{\mathcal L}/V^*$ deja de buscarse como una
constante universal y pasa a estudiarse como función de la población: el
resultado deseado es un diagrama de fases con cotas por régimen, no un solo
número de peor caso. Los datos del 22 ya lo justifican — el peor caso laminar
(0.943) aparece con utilidades casi planas y tasas medias, no en el generador
de estrés — así que la búsqueda adversaria se convierte en la herramienta que
encuentra el fondo de cada región del mapa. El segundo giro es el ancla
estática binaria: las comparaciones dejan de ser internas al mundo dinámico
aumentado y se conectan con el linaje del paper. La razón compuesta greedy
laminar$/$estático-binario habilita el enunciado central del proyecto: una
política adaptativa barata y estructurada domina al mejor diseño estático
binario en casi todo el espacio de poblaciones.

La cadena teórica no cambia de contenido, cambia de papel: el Lema A
(inferencia laminar exacta) y la Proposición B (rollout domina al greedy con
esperanzas exactas) pasan de ser el resultado a ser la teoría que explica las
regiones del atlas, y la Conjetura C se reformula por régimen. La escalera de
treewidth y la afirmación de complejidad "el óptimo laminar es más fácil de
encontrar" quedan en trabajo futuro; lo segundo lo cubre operativamente que el
greedy laminar figure como cantidad propia.

## Entregables al cierre de la semana

1. Lema A (inferencia laminar) enunciado y demostrado en LaTeX, con el
   corolario de que la distribución de $R_t$ es exacta para pools compatibles
   con la familia.
2. Proposición B demostrada, con el experimento del 22 §7 como ilustración y
   la hipótesis de exactitud ligada explícitamente al Lema A.
3. `laminar_inference.py` y `scenario_milp.py` extraídos del cuaderno con
   tests contra enumeración; el óptimo estático binario cableado como cuarta
   función con la misma firma `(p, u, B, G) -> valor`.
4. Atlas v1: las cuatro cantidades exactas sobre una malla de características
   de población, mapas de calor de las cuatro razones y peor/mejor caso por
   celda.
5. Búsqueda adversaria sembrada en las peores celdas del atlas, con la peor
   instancia por región documentada.
6. Al menos una cota por régimen cerrada (candidatos: $B\le 2$, priors
   homogéneos con utilidades planas) o una familia adversaria que la
   descarte.
7. Diagnóstico del gap de independencia por prueba: distribución de $R_t$
   bajo el posterior real contra el producto de marginales, en historiales
   laminares y no laminares.
8. Pipeline partículas → MILP → conteo → átomos → rollout corriendo de punta a
   punta en $n=40$, contra greedy plano y MILP miope como controles.
9. Paquete para Francisco y decisión sobre el eje del notebook 23.

## Lunes — cimientos

Persona A: enunciar y demostrar el Lema A. Las tres partes están en el 22 §3:
(i) los átomos residuales parten la población cubierta, (ii) los conteos de
nodos y átomos se determinan mutuamente por la resta
$c(D_A)=c(A)-\sum_C c(C)$, (iii) el prior producto condicionado a sumas sobre
bloques disjuntos factoriza. Incluir el corolario de complejidad
$O(\sum_A |D_A|\,c_A)$ dado el bosque padre-hijos y el corolario sobre la
distribución exacta de $R_t$. Los `ValueError` del código enumeran los casos
degenerados que el enunciado debe excluir.

Persona B: extraer `augmented/laminar_inference.py` (recibiendo la jerarquía,
sin el parser cúbico) y `augmented/scenario_milp.py`. Tests: identidad contra
`bayesian_update_by_counting` en familias ramificadas aleatorias ($n\le 12$),
identidad del MILP contra fuerza bruta en priors exactos ($n\le 10$), y las
validaciones de historial no laminar y conteos incompatibles.

## Martes — la Proposición B y las cuatro cantidades

Persona B: enunciar y demostrar la Proposición B por policy improvement: el
rollout evalúa cada acción con la continuación del greedy, la acción del
greedy está entre las candidatas, y la desigualdad se propaga por inducción
hacia atrás en el presupuesto. La parte con contenido propio es la hipótesis:
la exactitud de las esperanzas es la que el Lema A garantiza dentro de una
biblioteca laminar, y fuera de ella el 21 §7 documenta el fallo empírico.

Persona A: dejar las cuatro cantidades del marco como funciones con la misma
firma. Tres existen en el 22; el óptimo estático binario se cablea
reutilizando el solver estático de `classical/` o, si el acople tarda, por
enumeración directa de diseños no adaptativos para el $n$ pequeño del atlas.
Verificar en instancias chicas las desigualdades esperadas entre las cuatro.

Sincronía: A revisa que el enunciado de B coincida con lo que exige el código
del lunes; B revisa que las cuatro funciones de A expresen las cantidades del
marco.

## Miércoles — el atlas

Persona B (bloque principal del día): correr el atlas v1. Malla sobre
características de población — tasa homogénea $p\in\{0.05,\dots,0.9\}$,
dispersión de tasas (homogéneo contra Beta bimodal), dispersión de utilidades
(planas contra log-uniformes) — con $n\in\{4,5,6\}$, $B\in\{2,3\}$,
$G\in\{2,3\}$ y réplicas por celda. En cada celda, las cuatro cantidades
exactas y el peor/mejor caso de las cuatro razones. Salidas: mapas de calor y
un CSV por instancia para reanálisis.

Persona A: disecar a mano la peor instancia conocida (razón 0.9433, $n=5$,
utilidades casi planas): qué pools usa el óptimo libre que ninguna familia
laminar puede imitar. Con los primeros mapas de B, delimitar la región mala y
formular la conjetura por régimen que el jueves intentará cerrarse.

## Jueves — adversaria y cotas por régimen

Persona A: búsqueda adversaria sembrada en las peores celdas del atlas:
descenso por perturbación local en $(p,u)$ (coordenada a coordenada o
Nelder-Mead) minimizando $V^{\mathcal L}/V^*$; reportar trayectorias y las
instancias límite de cada región. En paralelo o al cierre, el diagnóstico del
gap de independencia: variación total entre la distribución exacta de $R_t$ y
la Poisson-binomial de marginales, en historiales laminares y no laminares; la
hipótesis es que el gap es cero para pools compatibles con la familia y crece
con el traslape.

Persona B: atacar la cota en el caso especial más plausible según el atlas:
$B=1$ es trivial (razón 1), $B=2$ acota el árbol de historias, y el caso
homogéneo $p_i=p$, $u_i=1$ reduce el óptimo a tamaños de pool por simetría. El
objetivo realista es un enunciado del tipo "para $B\le 2$,
$V^{\mathcal L}\ge\alpha(p) V^*$ con $\alpha$ explícita" o una familia donde
la razón baje de 0.9, lo que decidiría el rumbo. Los mínimos locales de A son
los candidatos a contraejemplo.

## Viernes — pipeline y paquete

Persona B: pipeline completo en `augmented/`: partículas del posterior → MILP
por escenarios elige el pool raíz → conteo simulado → inferencia laminar
exacta → rollout dentro de la jerarquía. Correrlo en $n=40$, $S=100$ contra
greedy plano con independencia y MILP miope sin rollout. Cerrar la validación
pendiente del 22 §2: pool del MILP contra el óptimo exacto para $n\le 12$
barriendo el número de partículas $S$.

Persona A: ensamblar el paquete para Francisco — lema y proposición en LaTeX,
el atlas comentado con sus razones peor/mejor caso por región, la tabla
adversaria, la figura del gap de independencia y la corrida del pipeline.

Sincronía de cierre (ambas): decidir el eje del notebook 23 — el atlas, la
cota por régimen o la familia adversaria — y qué se le presenta a Francisco
como resultado y qué como pregunta.

## Metas de estiramiento

Solo si el viernes deja margen: prototipo de la escalera de treewidth
(permitir un único pool cruzado sobre la biblioteca fija y medir cuánta razón
recupera en la región mala del atlas), y una figura del enunciado compuesto
greedy laminar contra estático binario sobre toda la malla.

## Riesgos

El costo del atlas crece rápido: cada celda exige el DP exacto por réplica;
con $n\le 6$ y la malla propuesta cabe en horas de cómputo, y si no, se
recorta la malla antes que las réplicas, porque el peor/mejor caso por celda
necesita muestras. El acople del solver estático puede tardar más que
escribir la enumeración directa; la enumeración es el plan B declarado. La
búsqueda adversaria puede estancarse en ~0.94: eso también es resultado
(la clase parece robusta) y desplaza el esfuerzo hacia la cota por régimen.
El MILP puede volverse lento al crecer $S$; el barrido del viernes lo detecta
y el `time_limit` acota el daño.
