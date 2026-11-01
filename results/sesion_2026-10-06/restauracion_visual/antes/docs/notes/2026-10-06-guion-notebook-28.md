# Notebook 28 · guion oral después de la revisión con Opus

[Presentación](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) · [Consultas](../../augmented/notebooks/28_notas_de_estudio.ipynb) · [Revisión de Opus](../../results/sesion_2026-10-06/revision_opus_pedagogia/revision.md).

Seis minutos más discusión. Una comparación durante todo el pase: óptimo general frente a óptimo laminar. El orden es definición mínima → caso concreto → mapa → qué podemos garantizar.

## Apertura · unos veinte segundos

“Francisco, hoy traigo una sola comparación. Si exigimos que la política sea laminar, ¿cuánto valor perdemos frente a la mejor política sin esa restricción? Mantuvimos los mismos datos. Primero te muestro un caso y luego el mapa.”

## 1. Definir solo lo necesario · un minuto

“Una política laminar usa grupos disjuntos o contenidos unos en otros. AB y después A respeta esa condición; AB y después AC no. La clase general permite cualquier grupo de hasta G personas, incluidas las políticas laminares. Las dos clases pueden adaptarse a los resultados. Solo cambia la restricción.”

El setup está escrito en pantalla. Explicar B como presupuesto por rama y G como tamaño máximo de grupo si hace falta. q es probabilidad de salud, p=1−q; los estados son independientes, observamos conteos y cada sano identificado con certeza aporta 1. ‘Óptimo’ es el mejor valor esperado dentro de cada clase.

Transición: “Antes del mapa, un caso que da significado a la pérdida.”

## 2. Dar significado al resultado · un minuto

“Seis personas, cuatro pruebas, grupos de hasta tres y q=.5. El óptimo general acredita tres sanos en promedio. El mejor laminar acredita dos y medio. Mismas personas, mismos recursos: exigir laminaridad cuesta una sexta parte del valor.”

Hacer una pausa después de los dos valores. La diferencia persiste incluso escogiendo la mejor política laminar. No hace falta leer varias versiones del mismo cálculo: 16.7% ya está en pantalla. Los números son promedios, no el resultado de una sola rama. El mecanismo detallado de esta política n=6 queda por explicar; el mapa cuantifica su diferencia.

Transición: “¿Es un caso aislado, o aparece un patrón al cambiar el presupuesto y las probabilidades?”

## 3. Leer el mapa, no sus números · dos minutos

“Esta celda señalada es el caso anterior. Ahora ampliamos a 228 casos. Con una, dos o seis pruebas, los dos óptimos coinciden en esta malla. En los presupuestos intermedios, exigir laminaridad casi siempre cuesta. Las mayores pérdidas aparecen con grupos de hasta tres y probabilidades de salud intermedias.”

Señalar en este orden: la etiqueta 16.7%, las filas B=1 y B=2, la fila B=6 y la región delimitada entre B=3 y B=5. Blanco significa empate. El contorno incluye todas las pérdidas estrictamente positivas, incluso las que tienen un color casi blanco. No interpretar sus bordes como umbrales continuos: cada celda es un caso calculado.

Si preguntan: 19 probabilidades × 6 presupuestos × 2 cotas = 228; 118 empates y 110 separaciones. De los 114 casos con B=3,4,5, hay separación en 110. El porcentaje siempre divide por el óptimo general. B=6=n permite resolver a todos; el empate con B=2 se presenta como observación de esta malla.

Transición: “Esta pérdida también limita lo que una conversión podría prometer en todos los casos.”

## 4. Cerrar con una pregunta de investigación · minuto y medio

“En el caso que vimos, ninguna política laminar puede superar dos y medio, mientras que la general llega a tres. Por eso una conversión no puede prometer conservar más de cinco sextos en todos los casos. Lo que todavía falta es demostrar cuánto sí podemos garantizar.”

“La propuesta es conservar las primeras decisiones de la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar. La compararemos con el óptimo laminar desde el inicio para medir cuánto cuesta conservar esas primeras decisiones.”

Estado: ese experimento todavía está pendiente. Antes de ejecutarlo hay que fijar la condición de laminaridad tras reducir ancestros y el desempate entre acciones óptimas; se empieza con los ocho casos acordados.

Pregunta para Francisco: “Para intentar una garantía, ¿conservarías el prefijo de la política general o buscarías transformar sus decisiones desde el inicio?”

## Cierre · unos veinte segundos

“En esta malla, el costo aparece con presupuestos intermedios. En el caso destacado, exigir laminaridad pierde una sexta parte incluso haciendo lo mejor posible. Ahora queremos estudiar una conversión concreta y medir qué parte del valor logra conservar.”

## Para contestar una pregunta sobre cinco sextos

Es una cota superior sobre una posible garantía universal; no una garantía conseguida. Si una transformación admisible prometiera conservar una fracción c del valor de cualquier política general, al aplicarla al óptimo general de este caso tendría que obtener al menos 3c. Pero cualquier política laminar de esta instancia vale como máximo 2.5. Así, 3c ≤ 2.5 y c ≤ 5/6. Aleatorizar entre políticas laminares tampoco permite superar su óptimo. Puede haber casos peores fuera de la malla; esto no demuestra que 5/6 sea alcanzable para todos.

Esta observación no necesita otro experimento: es una consecuencia de los dos óptimos exactos ya calculados.

## Qué queda en las consultas

Agrupar frente a individuales, el mecanismo del par, las reglas heurísticas y la sección 5 de Nick responden preguntas diferentes. Se abren aparte si surgen. La comparación del paper es dinámica frente a estática, con observación binaria y hard clearing; no establece nuestra garantía de laminaridad.

## Evidencia y reproducción

[Datos exactos](../../results/sesion_2026-10-06/tres_referencias_28.json). Los 190 óptimos laminares con B<6 coinciden entre dos solvers; las 228 políticas se evaluaron sobre 64 estados y se verificó su laminaridad. Para B=6, las individuales alcanzan la cota de utilidad total. Los generales proceden de la misma malla exacta ya verificada.

Reconstruir desde los datos guardados: `python augmented/notebooks/build_28_sesion.py`. La revisión pedagógica no relanzó experimentos. El contenido principal está en `presentacion_28.py`; el resto, en `contexto_28.py` y el generador.
