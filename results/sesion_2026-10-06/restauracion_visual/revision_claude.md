## A) Qué recuperar y qué dejar aparte

Recomiendo recuperar el árbol del par y el mapa de laminar frente a individuales en la presentación principal. Responden a la duda que Francisco planteó el 29-sep («no me queda claro que siempre sea mejor agrupar»). Además, el árbol es el intercambio de la proposición en una instancia concreta: (q²+p²)·q = 0.58·0.3 = 0.174. Si los quitamos, la presentación pierde su hallazgo más claro.

Coincido en usar tres figuras. Dejaría aparte, listos por si pregunta:
- Los conteos ABC/ABD/ACD. Son el único mecanismo concreto de separación general/laminar, pero con n=4 confunden el tamaño de población. Si pregunta qué hace el general, se responde de palabra aclarando «en n=4».
- Una prueba con n=2: sobra, porque la fila B=1 ya la muestra.
- El mapa con n=4, el piloto de conversión, la regla del cociente y Nick: se quedan en las notas.

Un cambio respecto a tu propuesta: usar la celda B4 G3 q=0.5 como puente entre las dos preguntas. Con números absolutos (individuales 2, laminar 2.5, general 3) se ven ambas comparaciones sin mezclar porcentajes. En el primer mapa esa celda marca 25 %; en el segundo, 16.7 %.

## B) Secuencia

1. Convenciones y dos preguntas
> n=6 personas, cada una con utilidad 1, estados independientes. q es la probabilidad de estar sano y p=1−q. Cada prueba de hasta G personas devuelve el conteo de infectados R. Acreditamos a quien sabemos sano con certeza (soft clearing). B es el número de pruebas. Todas las políticas adaptan.
> 1. ¿Puede ser óptimo probar solo individualmente?
> 2. ¿Cuánto cuesta exigir laminaridad: grupos disjuntos o uno dentro del otro en cada rama?

Oral: «Son las dos dudas que quedaron abiertas el 29.»

2. Hallazgo y mecanismo (árbol)
> Con B≥2 y n>B, usar solo individuales nunca es óptimo. Caso B=2 demostrado; caso general derivado y verificado de forma exacta, con la prueba escrita pendiente.
> n=6, B=2, G=2, q=0.3. Individuales A y B: 0.6. Probar AB y luego A solo si R=1 da la misma información; si R=0 o R=2, sobra una prueba para C.
> 0.6 + 0.58×0.3 = 0.774.

Oral: «Agrupar no gana por acreditar más en AB, sino porque a veces ahorra una prueba.»

3. Regímenes (mapa 1)
> 228 casos: B=1…6, G∈{2,3}, q=0.05…0.95. Mejora del óptimo laminar sobre individuales, (laminar − individuales)/individuales.
> B=1 es el modelo binario: mejora solo si q>0.5. B=2…5: mejora en los 152 casos. B=6: empate.

Oral: «La primera fila es tu teorema. Desde la segunda, el conteo cambia el régimen, también con pocos sanos. Y basta con políticas laminares.»

4. Transición
> Mismo caso, tres valores: individuales 2, óptimo laminar 2.5, óptimo general 3 (n=6, B=4, G=3, q=0.5).

Oral: «Ya vimos el primer salto. Ahora cambia la referencia: medimos contra el general.»

5. Precio de laminaridad (mapa 2)
> Pérdida del óptimo laminar, (general − laminar)/general. Empate con B=1, 2 y 6; separación en 110 de 114 casos con B=3, 4 y 5. Máximo: 16.7 %.

Oral: «Así que ninguna garantía uniforme puede pasar de 5/6. Cuánto se puede garantizar sigue abierto.» Esto basta; no le daría un bloque propio.

6. Siguiente pregunta
> Seguir la política general y cambiar a la mejor continuación laminar antes de perder laminaridad. ¿Cuánto conserva frente al óptimo laminar?

## C) Cambios mínimos a las figuras

Árbol:
- Título con el setup: «n=6, B=2, G=2, q=0.3».
- Hoja izquierda: «R=0 (9 %) o R=2 (49 %)» y «+0.3 esperado con C».
- Hoja derecha: «1 sano acreditado».
- Pie: «0.6 con individuales · 0.774 con el par».

Mapa 1:
- Marcar con un recuadro solo la celda G=2, B=2, q=0.3. El 29 % aparece también con G=3, y dos marcas confunden.
- Añadir un contorno de mejora > 0 con la misma convención que el mapa 2. Las filas B=5 (15–20 %) casi no se distinguen del blanco, y sin contorno parecen empate.
- Escribir el denominador en la barra de color.

Mapa 2:
- Usar otra familia de color (cálida). Con la misma paleta y escalas de 0–170 y 0–16, el oscuro de un mapa parece equivaler al del otro.
- Escribir «empate» en las filas B=1, 2 y 6, que hoy parecen datos faltantes.
- Etiqueta: «16.7 % · 3 frente a 2.5».
- Denominador en la barra de color.

## D) Riesgos

- El árbol es una construcción y el mapa muestra óptimos. En esta celda coinciden; en otras el óptimo supera al intercambio. Hay que decirlo para que el 29 % no parezca la ganancia del par en todas partes.
- Tamaño de población: la misma cifra, 0.774, aparece en el material con n=4. Comprobar que el pie del árbol dice n=6 y no reutilizar `adaptacion_y_muestreo.png`.
- «Empate con B=2» vale desde la raíz en esta malla homogénea. Tu propia tabla tiene un estado condicionado con B=2 que sí separa (1.30 frente a 1.45). No decir «con dos pruebas nunca se separan».
- El 5/6 es un techo para cualquier garantía, no una garantía. Y la cota 1/G vale desde el historial vacío, no para cualquier historial.
- Sin mecanismo para la brecha con n=6: decir «aún no extraemos el árbol del general en ese caso». Puede presentarse como tarea, no como explicación.
- No decir «agrupar siempre conviene»: lo que no es óptimo es la política de solo individuales. La primera acción puede empatar.
- Etiquetas: no usar «Q1/Q2», que en el proyecto significan otra cosa.

## E) Preguntas útiles para Francisco

1. ¿Este enunciado es el que te parecía sorprendente? ¿Lo pondrías en la sección de separación o como resultado aparte?
2. Para la garantía laminar frente a general, ¿apuntarías a una constante independiente de G, o es aceptable una que dependa de G? Sabemos 1/G desde el inicio y un techo de 5/6.
3. En tu molde de §5, ¿la distribución debe construirse desde la política completa, o es razonable conservar el prefijo real hasta el primer paso no laminar? Así nuestro siguiente experimento probaría algo que a él le sirva para la demostración.