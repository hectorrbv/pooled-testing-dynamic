# Revisión para una charla de 6 minutos: general frente a laminar

## A) Diagnóstico

1. **El oyente no sabe por qué escucha hasta la pantalla 4.** La pantalla 1 da la definición, el ejemplo AB/AC, ocho parámetros, R y soft clearing antes de que haya una pregunta en juego. La razón para que importe aparece al final: queremos una conversión a laminar con garantía. Pero hay un argumento fuerte que hoy no se dice. El caso 3 frente a 2.5 pone un techo de 5/6 a cualquier garantía de conversión. Ese techo une la medición con el siguiente paso, y es lo que vuelve memorable el resultado.

2. **Hay demasiados números y cada idea tiene varios nombres.** En la pantalla 2 se oyen 228, 118, 110 y 16.67 % mientras la figura muestra unos 40 porcentajes. La misma pérdida aparece como 0.5, 16.67 % y 3−2.5, y en las notas también como 5/6. Al oírlo una sola vez, el oyente se queda con «muchos números». Basta con una sola expresión: «medio sano de tres, una sexta parte». En la figura, además, un blanco de empate y un 0.1 % se ven casi iguales.

3. **El orden va del mapa al caso, y debería ir al revés.** El color solo significa algo cuando ya se ha visto una pérdida concreta. Además, la pantalla 3 termina invitando a preguntar por el mecanismo («qué información aprovecha…»), justo lo que el mapa no establece. Las notas cargan con muchos «no decir X». Conviene dejar una sola línea: «si preguntan por el mecanismo, está pendiente».

## B) Cuatro pantallas

El orden que propongo es: definición mínima, caso, mapa y techo con siguiente paso. No se puede abrir con 3 frente a 2.5 porque sin «laminar» no se entiende. Tampoco hace falta defender ante Francisco la importancia de una línea que propuso él. El caso va antes del mapa para que el color ya tenga significado. El techo cierra y motiva la conversión.

**Pantalla 1. ¿Cuánto cuesta exigir laminaridad?**

> Seis personas; cada una está sana con probabilidad q (p = 1 − q), de forma independiente. Probamos grupos de hasta G personas; cada prueba devuelve R, el número de infectados del grupo. Hay como máximo B pruebas en cada rama. Ganamos 1 por cada persona que sabemos sana con certeza (soft clearing).
>
> Laminar: dentro de una rama, dos grupos probados son disjuntos o uno contiene al otro. AB y después A, o CD: sí. AB y después AC: no.
>
> General: cualquier grupo, incluidas las políticas laminares.
>
> Comparamos el mejor valor esperado de cada clase con los mismos datos.

- Oral: «La pregunta es una sola: si obligamos a que los grupos sean laminares, ¿cuánto perdemos frente a no restringirlos?»
- Transición: «Antes del mapa, un caso.»

**Pantalla 2. Un caso: 3 frente a 2.5**

> B = 4, G = 3, q = 0.5.
> Mejor política general: 3 sanos acreditados en promedio, todos los esperados.
> Mejor política laminar: 2.5.
> Pierde medio sano de tres: 1/6 ≈ 16.7 %.

- Oral: «No es una heurística mala: son los dos óptimos. Aun eligiendo lo mejor dentro de la clase laminar, queda medio sano por acreditar.»
- Transición: «¿Es un caso aislado?»

**Pantalla 3. La pérdida aparece con presupuestos intermedios**

> 228 casos: 19 valores de q (0.05 a 0.95), B de 1 a 6, G = 2 y 3.
> Color: porcentaje del valor general que se pierde. Blanco: empate.
> B = 1, 2 y 6: empate en todos.
> B = 3, 4 y 5: pérdida en 110 de 114.
> Para cada B y G, la mayor pérdida está en q = 0.45 o 0.5.

- Oral: «Con muy pocas pruebas, o con seis, la restricción no cuesta nada en esta malla. En medio, casi siempre cuesta, y las mayores pérdidas están con grupos de tres.»
- Transición: «Esto limita lo que puede prometer una conversión.»

**Pantalla 4. Siguiente paso: convertir una política general en laminar**

> Ninguna regla de conversión puede garantizar en todos los casos más de 5/6 del valor general: con B = 4, ni la mejor laminar lo logra.
> Propuesta: seguir la política general y, antes de su primera prueba incompatible con laminaridad, pasar a la mejor continuación laminar.
> Pendiente: compararla con el óptimo laminar.

- Oral: «Ya sabemos cuánto cuesta la clase. Falta medir cuánto cuesta, además, conservar las primeras decisiones de la general.»

El texto de pantalla suma unas 240 palabras.

## C) Figura

El heatmap basta. Le haría dos cambios con los mismos datos:

1. **Quitar todas las etiquetas numéricas menos una.** Enmarcar la celda B4, G3, q.5 y rotularla «16.7 %», para que sea visiblemente el caso de la pantalla 2. Así la figura deja de ser una tabla y pasa a ser un dibujo con un punto de referencia.
2. **Distinguir el empate de la pérdida pequeña.** Trazar un contorno fino alrededor de las celdas con separación, o dar a los empates una trama suave. Hoy 0.1 % y 0 % se ven iguales, y la leyenda «blanco = empate» engaña en las filas B4 y B5 cerca de q = 0.9.

## D) Apertura, cierre y pregunta

**Apertura:** «Francisco, hoy traigo una sola comparación. Si obligamos a que los grupos sean laminares, ¿cuánto valor perdemos frente a la mejor política sin esa restricción? Lo calculé con seis personas y los mismos datos para ambas. Primero te muestro un caso y luego el mapa.»

**Cierre:** «En esta malla, la laminaridad no cuesta nada con una, dos o seis pruebas. Con tres a cinco casi siempre cuesta, hasta una sexta parte. Eso pone un techo de 5/6 a cualquier garantía de conversión. Lo siguiente es medir cuánto se pierde al seguir la general y cambiar antes de su primera prueba incompatible.»

**Pregunta:** «Si 5/6 es el techo y la pérdida depende tanto de B, ¿buscarías una garantía de fracción fija o una que dependa del presupuesto?»

## E) Afirmaciones revisadas y corregidas

- **El techo de 5/6** es una cota superior para cualquier garantía uniforme dentro de este modelo (conteos, soft clearing, G ≤ 3). No es una garantía de retener 5/6; la frase de la pantalla 4 lo dice así.
- **«Todos los esperados»** se refiere al promedio: 3 = 6 × 0.5. De ahí se deduce que la general acredita a todos los sanos en cada uno de los 64 estados. Si quieres decirlo así, compruébalo antes en las evaluaciones guardadas; no lo afirmes sin esa comprobación.
- **G = 3:** solo dije «las mayores pérdidas están con G = 3». No afirmé que G = 3 pierda más que G = 2 en cada celda, porque eso no lo verifiqué.
- **Empates en B = 1 y 2:** se presentan como observados en la malla, no como un teorema.
- **La política general óptima de este caso no es laminar**, necesariamente, porque supera al óptimo laminar. Pero no expliqué por qué, porque el mecanismo no está establecido.
- **Laminaridad por rama:** la condición de la conversión se refiere a lo ya probado en esa rama. Antes de ejecutarla hay que fijar la definición tras reducir ancestros, como dicen las notas.
