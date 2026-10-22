# Notebook 28 · guion con las gráficas restauradas

[Presentación](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) · [Notas de estudio](../../augmented/notebooks/28_notas_de_estudio.ipynb) · [Revisión de Claude](../../results/sesion_2026-10-06/restauracion_visual/revision_claude.md).

Unos nueve minutos y medio más discusión. Siete bloques y cinco figuras. Resultado principal: en la malla, usar solo individuales deja de ser óptimo cuando 2≤B<n; el par explica el mecanismo. Después, como segunda cuestión, general frente a laminar. Las mismas seis personas en todos los ejemplos principales.

## 1. La pregunta de Francisco · cuarenta segundos

“Francisco, la sesión pasada te sorprendía que con dos pruebas y una población mayor las individuales pudieran dejar de ser óptimas para cualquier probabilidad. Traemos el mecanismo del par y un mapa donde la mejora aparece en todos los casos con 2≤B<n. Después te muestro cuánto cuesta exigir laminaridad.”

La pregunta está registrada en el [acta del 29-sep, D4](2026-09-29-sesion-francisco.md), 19:18–20:20. Distinguir el caso homogéneo con dos pruebas, demostrado, de la extensión general por intercambio, derivada y verificada pero pendiente de cierre escrito y revisión humana.

Una política laminar usa, dentro de cada rama, grupos disjuntos o contenidos unos en otros. AB → A es laminar; AB → AC no. La clase general permite ambas posibilidades. Las dos clases pueden adaptarse. Comparamos los mejores valores de cada clase, no heurísticas.

Las convenciones están una sola vez en pantalla: n=6, utilidad 1 por sano identificado con certeza, estados independientes, q sano y p=1−q. B limita el número de pruebas por rama; G, el tamaño de cada grupo. R cuenta infectados. Todas las instancias principales empiezan sin pruebas previas.

Transición: “Empecemos con dos pruebas y un par.”

## 2. El mecanismo del par · minuto y medio

“Con B=2, G=2 y q=.3, dos individuales obtienen .6. Si probamos AB, siempre podemos recuperar toda la información de ambos. Cuando el resultado es cero o dos, ya están resueltos: usamos la segunda prueba en C. Cuando es uno, probamos A e identificamos al sano.”

Señalar primero las dos decisiones del árbol. Después explicar el cálculo: .6 por los sanos de AB, más .58 de probabilidad de llegar a C multiplicado por .3 de utilidad esperada al probarlo. Total .774, un 29% más.

No decir que ambas ramas del dibujo cobran lo mismo: R=0 y R=2 comparten la siguiente acción, pero el primero acredita a dos personas de AB y el segundo a ninguna. Cada rama de la política completa usa dos pruebas. D, E y F quedan sin probar.

Si pide el cálculo completo: .09×2.3 + .42×1 + .49×.3 = .774. Las utilidades de cada rama ya incluyen su continuación. Esta política alcanza ambos óptimos en esta instancia; no es el árbol usado en cada celda del mapa.

La ganancia del mismo plan frente a dos individuales es q[q²+(1−q)²]>0 para todo 0<q<1, con n≥3, G≥2 y utilidades1. Ese es el resultado demostrado que el árbol permite explicar. Para otros q, el plan puede ser mejorable; no hace falta que sea óptimo para demostrar la ventaja sobre individuales.

Transición: “Este mecanismo responde a tu pregunta. El mapa muestra los regímenes y los extremos del presupuesto.”

## 3. Resultado principal: individuales no óptimas · dos minutos

“Esta es la respuesta a tu pregunta: en los 152 casos donde hay al menos dos pruebas pero no alcanzan para todos, usar solo individuales pierde para todas las probabilidades probadas. El par muestra por qué. Con una prueba hay mejora solo para q mayor que .5; con seis podemos probar a todos y empatamos.”

El azul mide mejora respecto a individuales. Se calcula dividiendo la ganancia por el valor de individuales, Bq. Dentro del contorno hay mejora, incluso si el tono es casi blanco; fuera hay empate. El contorno permite distinguir mejoras pequeñas de empates.

Son 19 probabilidades × 6 presupuestos × 2 tamaños máximos = 228 casos, con 170 mejoras y 58 empates. B=2…5 aporta 152 mejoras. Los contornos siguen celdas calculadas, no umbrales interpolados.

La observación no dice que cualquier grupo mejore ni que toda primera acción óptima agrupe. El argumento de intercambio de dos pruebas está demostrado; el caso general heterogéneo sigue pendiente de cerrar por escrito y de revisión humana, según el plan.

Si pide la extensión: partir de los B mejores individuos según q_i u_i. Sustituir dos pruebas individuales por el par y su continuación; mantener las otras B−2. En R=1 conservamos toda la utilidad de esas dos personas. En R=0/2 probamos a alguien c fuera de la selección inicial. Ganancia: (q_a q_b+p_a p_b)q_c u_c>0. Requiere estados independientes, 0<q_i<1, u_i>0, 2≤B<n, G≥2, soft clearing y comenzar sin historial. La construcción es laminar. Este argumento ya está derivado; su revisión formal aún no está cerrada.

Pregunta: “¿Este enunciado recoge el fenómeno que te sorprendía y conviene usarlo como resultado de separación entre modelos?”

Transición: “Esa es nuestra respuesta sobre individuales. Como segunda cuestión, medimos cuánto se pierde al exigir laminaridad.”

## 4. Los árboles de 3 frente a 2.5 · minuto y medio

“Con n=6, B=4, G=3 y q=.5, individuales obtiene 2 y el óptimo laminar 2.5. Ahora lo comparamos con el general, que obtiene 3. Los dos árboles empiezan con AB. Cuando AB es mixto, el general prueba ACD y el laminar prueba A: aquí aparece la mayor parte de la diferencia.”

Señalar los dos nodos resaltados. Cada caja inferior resume una continuación óptima; su valor es la utilidad final esperada de toda esa rama, incluido lo que ya sabemos sano. Las probabilidades de los tres resultados de AB son 1/4, 1/2 y 1/4.

General: .25×4 + .5×3 + .25×2 = 3. Laminar: .25×3.75 + .5×2.25 + .25×1.75 = 2.5. No leer toda la cuenta salvo que la pida: basta mostrar que las tres ramas están incluidas. La rama mixta aporta .5×(3−2.25)=.375 de la diferencia total .5; las otras dos aportan .0625 cada una.

La primera mejora relativa era .5/2=25%; ahora la pérdida es .5/3≈16.7%. No comparar directamente esos porcentajes.

Mostramos una política óptima de cada clase. Hay desempates en algunas acciones, por lo que no se debe atribuir toda la pérdida a probar C frente a CD en las ramas extremas. Los árboles completos están en las consultas y sus valores se verificaron sobre los 64 perfiles.

Transición: “Sigamos esas dos políticas sobre una misma población para ver qué información consiguen.”

## 5. Una realización que muestra las decisiones · minuto y medio

“Supongamos que solo B y D están infectados. Las políticas no conocen ese perfil: lo vamos descubriendo al seguir sus pruebas. Ambas empiezan con AB y reciben uno. La laminar prueba A para resolver AB. La general prueba ACD: conserva A e incorpora dos personas nuevas.”

Señalar que AB y ACD comparten A y ninguno contiene al otro: ACD no es admisible bajo la restricción laminar después de AB.

“La laminar sigue A, CD y C; identifica a A y C y se queda sin pruebas para E y F. La general combina los conteos de AB, ACD y BCE para deducir que C es sano sin probarlo individualmente. Su última prueba, AEF, identifica a los otros tres sanos.” Señalar el resultado de BCE y explicar la inferencia escrita en la celda. La figura conserva solo pruebas, resultados y hojas; el texto explicativo queda fuera.

Si pregunta cómo se deduce C: los tres primeros resultados del general son AB=1, ACD=1, BCE=1. Si C estuviera infectado, ACD=1 obligaría a A y D sanos, y BCE=1 a B y E sanos; eso contradice AB=1. Por tanto C es sano. La última prueba AEF=0 acredita a A, E y F.

En la laminar: A=0 acredita a A e identifica B infectado; CD=1 y C=0 acreditan a C e identifican D infectado. E y F siguen sin probarse.

Esta realización tiene probabilidad 1/64 y da 4 frente a 2. No es el promedio 3 frente a 2.5. Tampoco afirma que cualquier óptimo laminar cobraría 2 en este perfil; describe los dos árboles elegidos. El promedio completo y sus ramas están en la pantalla anterior.

Pregunta de investigación: “¿Cómo conservar parte de esta información compartida al transformar la política en laminar?”

Transición: “El mapa muestra en qué otros casos aparece la separación.”

## 6. El mapa del precio de laminaridad · un minuto y veinte segundos

“Esta celda es el caso de los árboles: 3 frente a 2.5, pérdida del 16.7%. Con una, dos o seis pruebas el óptimo laminar coincide con el general en esta malla. Con presupuestos intermedios aparecen separaciones en casi todos los casos.”

Señalar B=4 y después B=2. “El blanco de B=2 no contradice el mapa azul: agrupar ya mejora a individuales, pero permitir la clase general no añade valor en esos casos.”

El tono cálido mide la pérdida dividida por el general, con una escala distinta de la azul. Fuera del contorno hay empate; dentro hay pérdida, incluso si el color es tenue. B=3 separa 34 de 38 casos; B=4 y B=5, todos. Total: 110 separaciones en los 114 casos intermedios. El empate B=2 es una observación desde el inicio de la malla homogénea, no un teorema para cualquier historial condicionado.

Transición: “Los árboles muestran qué información aprovechamos; falta diseñar la transformación.”

## 7. Una pregunta para la siguiente etapa · un minuto

“Proponemos seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar compatible con lo ya probado. La compararemos con el óptimo laminar desde el inicio: eso mide cuánto cuesta conservar esas primeras decisiones.”

Ese experimento sigue pendiente. Empezará en los ocho casos acordados después de fijar la condición de laminaridad y el desempate entre acciones óptimas. Extraer los árboles de este caso no sustituye ese experimento.

Pregunta para Francisco: “¿Conviene conservar ese comienzo, o para intentar una garantía habría que transformar las decisiones desde el inicio?”

Cierre: “El par explica una ventaja frente a individuales. Los árboles localizan qué decisiones aportan al general, y el mapa sitúa la separación. El siguiente paso es ver cuánto conserva una transformación concreta.”

## Para la discusión, fuera del pase principal

- Sobre 5/6: es un techo para una posible garantía universal, no una garantía conseguida. Toda política laminar en el caso destacado vale a lo sumo 2.5, también si se aleatoriza sobre políticas laminares. Conservar una fracción c del general de valor 3 requiere 3c≤2.5. Puede haber casos peores fuera de esta malla.
- Sobre el paper: la construcción de §5 produce planes estáticos con pruebas binarias y hard clearing. Nuestro destino puede seguir siendo dinámico, con conteos y soft clearing. La identidad de dos perfiles independientes sirve como herramienta; falta la desigualdad de utilidad para la transformación laminar. Pregunta: “¿Qué propiedad de la transformación necesitaríamos para adaptar tu argumento?”
- El árbol con n=4, el piloto de conversión, las heurísticas y los dibujos del paper siguen en las notas de estudio. Anunciar el cambio de pregunta y de setup antes de abrirlos.
- La lectura del paper y los comentarios siguen siendo el encargo principal de Francisco. Estas figuras aportan evidencia para discutir; no sustituyen el memo de §5, el experimento pendiente ni la revisión matemática con Vladimir.

## Evidencia y reproducción

[Datos exactos](../../results/sesion_2026-10-06/tres_referencias_28.json) · [Verificación del artefacto](../../results/sesion_2026-10-06/verificacion_presentacion.json).

Los 190 óptimos laminares con B<6 coinciden entre dos solvers; las 228 políticas se evaluaron sobre 64 estados y se verificó laminaridad. Con B=6, las individuales alcanzan la cota total. Los generales proceden de la misma malla exacta. La malla conserva todos sus datos. Para explicar la separación se resolvió nuevamente solo el caso n=6, B=4, G=3, q=.5 y se auditaron ambos árboles nodo por nodo con filtrado independiente de perfiles.

Reconstrucción: `python augmented/notebooks/build_28_sesion.py`. Validación: `python results/sesion_2026-10-06/verifica_presentacion.py`. La extracción del caso es `python -m augmented.extrae_separacion_28`. El contenido principal está en `presentacion_28.py`; las consultas, en `contexto_28.py` y el generador. Los nuevos árboles se dibujan con `figuras_separacion_28.py`.
