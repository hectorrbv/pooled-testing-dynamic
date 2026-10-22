"""One presentation question: the price of requiring laminar policies."""


def add_presentation(add):
    add(r'''
## 1. La pregunta: ¿cuánto cuesta exigir laminaridad?

Comparamos el mejor resultado posible en dos clases de políticas:

- General: permite cualquier grupo de hasta $G$ personas, sin exigir laminaridad.
- Laminar: los grupos probados deben ser disjuntos o estar contenidos uno en otro.

Por ejemplo, AB seguido de AC no es laminar. AB seguido de A, o de CD, sí respeta la laminaridad.

Ambas clases pueden agrupar y adaptar la siguiente prueba a los resultados. Usan las mismas probabilidades, utilidades y presupuesto. Lo único que cambiamos es la restricción sobre los grupos.

$n=6$, $B=1,\ldots,6$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$, $p=1-q$, $u_i=1$.
$q$ es la probabilidad de estar sano; los estados son independientes. Observamos el conteo de infectados $R$ y usamos soft clearing: acreditamos a quien sabemos sano con certeza.
''', 65, cell_id='sesion28-pregunta', notes='''
Frase oral: queremos aislar cuánto cuesta una sola restricción: que los grupos formen una familia laminar.
Un óptimo es el mejor valor esperado entre todas las políticas admisibles de su clase; no es el nombre de una heurística. La clase general contiene a la laminar, por eso su valor no puede ser menor. Ambas son dinámicas, observan conteos y usan soft clearing. También comparten el presupuesto por rama y el tope G. No comparar aquí contra individuales ni introducir el mecanismo del par: esa es otra pregunta, guardada en el material de consulta. Empezamos sin pruebas previas y todas las personas tienen el mismo q. Los ejemplos AB/AC y AB/A ilustran únicamente la compatibilidad de los grupos, no su optimalidad. El destino de nuestra investigación puede seguir siendo dinámico; no estamos fijando una lista de pruebas ni estudiando la brecha dinámica-estática del paper en este pase.
Transición: con la comparación fijada, ahora mostramos dónde cambia el valor.
''')

    add(r'''
## 2. El resultado: dónde aparece la separación

228 casos: 19 probabilidades × 6 presupuestos × 2 tamaños máximos de grupo.

El color indica qué porcentaje del valor general se pierde al exigir laminaridad. Blanco significa empate.

![Pérdida del óptimo laminar frente al general en la misma malla de 228 casos con n=6.](attachment:general_vs_laminar.png)

En esta malla, con $B=1,2,6$ ambos óptimos coinciden. Con $B=3,4,5$ aparecen separaciones; las mayores pérdidas se encuentran con $G=3$ y probabilidades de salud intermedias.

Hay 118 empates y 110 separaciones. La mayor pérdida observada es 16.67%.
''', 110, 'general_vs_laminar', cell_id='sesion28-mapa', notes='''
Frase oral: la restricción no siempre cuesta; en esta malla el costo aparece con presupuestos intermedios.
Recorrer primero las filas B1 y B2, después B3 a B5, y terminar en B6. El porcentaje es 100(1−V_laminar*/V_general*). Se comparan exactamente las mismas personas, probabilidades, utilidades, presupuesto y G. B3 separa 34 de38casos; B4 yB5 separan38de38 cada uno. B1, B2 yB6 empatan38de38 cada uno. B6=n permite resolver a todos, lo que explica ese control; los empates B2 son observados en esta malla y no una afirmación universal. Las mayores pérdidas por cada combinación B,G aparecen en q=.45 o .5. Las políticas laminares se evaluaron sobre los64perfiles; paraB<6 los190valores coinciden entre dos solvers, y paraB6 hay38certificados por cota alcanzada. Se reutilizan los generales ya verificados en esos mismos casos. No es un mapa de π_L o π_ratio, ni una garantía universal de retener5/6. La malla n4 permanece como consulta; su máximo12.5% no describe esta figura.
Transición: fijemos una sola celda para expresar la pérdida en unidades de utilidad.
''')

    add(r'''
## 3. Un caso del mapa: 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=0.5$, $p_i=0.5$, $u_i=1$.

- Óptimo general: 3 de utilidad esperada.
- Óptimo laminar: 2.5 de utilidad esperada.

La restricción cuesta 0.5 unidades:

$$\frac{3-2.5}{3}\times100=16.67\%.$$

Los dos valores incluyen la mejor decisión posible después de cada resultado. Por tanto, esta diferencia persiste incluso eligiendo la mejor política laminar.

Esta es la separación que queremos entender: qué información aprovecha una política no laminar que no puede aprovechar igual una política laminar.
''', 90, cell_id='sesion28-caso', notes='''
Frase oral: incluso haciendo todo lo mejor posible dentro de la clase laminar, aquí dejamos media unidad respecto al óptimo general.
Es la celda B4,G3,q.5 del mismo mapa n6. No cambiamos de población ni de modelo. General3 coincide con toda la utilidad esperada disponible, nq=3; laminar2.5 es óptimo por ambos solvers. El denominador del16.67% es3. Estos números no describen una sola rama ni una política heurística. Evitar introducir aquí individuales2, la conversión1.5 del caso n4 o el par. Si preguntan cómo produce la separación la información de pruebas no laminares, la realización ABC=1,ABD=1,ACD=2 de las notas es una ilustración en una instancia distinta n4; no atribuirle el valor de esta celda n6. La identificación de un mecanismo simple en el caso n6 es un siguiente paso explicativo, no algo que la figura ya haya demostrado.
Transición: conocer los dos óptimos mide la pérdida; el siguiente objetivo es construir una transformación.
''')

    add(r'''
## 4. La dirección: convertir una política general en laminar

La malla mide el costo de la restricción. Queremos ahora un procedimiento que produzca políticas laminares y conserve una parte garantizada del valor original.

El próximo experimento seguirá una política general hasta que proponga una prueba incompatible con la laminaridad. Antes de ejecutarla, cambiará a la mejor continuación laminar compatible con lo ya probado.

Compararemos esa política con el óptimo laminar desde el inicio. La diferencia mostrará cuánto cuesta conservar las decisiones anteriores de la política general.

Estado: la comparación de óptimos está calculada; este experimento de conversión sigue pendiente.

Pregunta para discutir: ¿conviene conservar las decisiones iniciales y cambiar justo antes de perder laminaridad, o hay que modificar decisiones anteriores?

[Guion para presentarlo](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Consultas separadas: individuales, reglas y paper](28_notas_de_estudio.ipynb).
''', 95, cell_id='sesion28-direccion-prueba', notes='''
Frase oral: ya medimos el costo de la restricción; ahora queremos saber cómo construir una política que se acerque al mejor laminar.
Mantener esta cuarta pantalla como próximo paso, no como otro resultado de la malla. Seguir la general hasta antes de la primera prueba no compatible conserva un prefijo laminar; continuarlo con acciones compatibles da una política laminar. Su valor no puede superar al óptimo laminar desde la raíz. La diferencia mide el compromiso con ese prefijo, además del costo de clase que ya se conoce. Si la política general ya es laminar, ambas ejecuciones coinciden. Fijar antes la definición tras reducir ancestros y el desempate favorable a acciones laminares entre óptimas. Empezar por los ocho casos acordados. La comparación adicional con el cociente será una evaluación de heurística aparte; no introducirla hasta haber definido la transformación. Tampoco introducir muestreo de caminos ni el recorte de Nick en este pase. Si se pregunta por el paper, ir a la sección separada de estudio: su garantía se refiere a estáticas sin restricción laminar, en binario y hard clearing, y no establece nuestra desigualdad de laminaridad.
''')
