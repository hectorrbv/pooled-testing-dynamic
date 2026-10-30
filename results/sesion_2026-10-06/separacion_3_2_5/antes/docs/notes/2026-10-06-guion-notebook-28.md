# Notebook 28 · guion con las gráficas restauradas

[Presentación](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) · [Notas de estudio](../../augmented/notebooks/28_notas_de_estudio.ipynb) · [Revisión de Claude](../../results/sesion_2026-10-06/restauracion_visual/revision_claude.md).

Unos ocho minutos y medio más discusión. Seis bloques y tres figuras. Primero agrupación frente a individuales; después general frente a laminar. Las mismas seis personas en todos los ejemplos principales.

## 1. Dos preguntas · un minuto

“Francisco, retomamos tu pregunta sobre las individuales. Primero muestro por qué un par puede mejorarlas; luego medimos cuánto cuesta exigir laminaridad.”

Una política laminar usa, dentro de cada rama, grupos disjuntos o contenidos unos en otros. AB → A es laminar; AB → AC no. La clase general permite ambas posibilidades. Las dos clases pueden adaptarse. Comparamos los mejores valores de cada clase, no heurísticas.

Las convenciones están una sola vez en pantalla: n=6, utilidad 1 por sano identificado con certeza, estados independientes, q sano y p=1−q. B limita el número de pruebas por rama; G, el tamaño de cada grupo. R cuenta infectados. Todas las instancias principales empiezan sin pruebas previas.

Transición: “Empecemos con dos pruebas y un par.”

## 2. El mecanismo del par · dos minutos

“Con B=2, G=2 y q=.3, dos individuales obtienen .6. Si probamos AB, siempre podemos recuperar toda la información de ambos. Cuando el resultado es cero o dos, ya están resueltos: usamos la segunda prueba en C. Cuando es uno, probamos A e identificamos al sano.”

Señalar primero las dos decisiones del árbol. Después explicar el cálculo: .6 por los sanos de AB, más .58 de probabilidad de llegar a C multiplicado por .3 de utilidad esperada al probarlo. Total .774, un 29% más.

No decir que ambas ramas del dibujo cobran lo mismo: R=0 y R=2 comparten la siguiente acción, pero el primero acredita a dos personas de AB y el segundo a ninguna. Cada rama de la política completa usa dos pruebas. D, E y F quedan sin probar.

Si pide el cálculo completo: .09×2.3 + .42×1 + .49×.3 = .774. Las utilidades de cada rama ya incluyen su continuación. Esta política alcanza ambos óptimos en esta instancia; no es el árbol usado en cada celda del mapa.

Pregunta posible: “¿Este intercambio es el fenómeno que te parecía sorprendente con dos pruebas y una persona adicional?”

Transición: “Veamos ahora los regímenes donde el mejor laminar supera a individuales.”

## 3. El mapa de agrupación · minuto y medio

“Este 29% es el ejemplo del árbol. Con una prueba hay mejora para q mayor que .5. Con dos, tres, cuatro o cinco pruebas, el mejor laminar supera a individuales en todos los casos de esta malla. Con seis podemos probar a todos, y hay empate.”

El azul mide mejora respecto a individuales. Se calcula dividiendo la ganancia por el valor de individuales, Bq. Dentro del contorno hay mejora, incluso si el tono es casi blanco; fuera hay empate. El contorno permite distinguir mejoras pequeñas de empates.

Son 19 probabilidades × 6 presupuestos × 2 tamaños máximos = 228 casos, con 170 mejoras y 58 empates. B=2…5 aporta 152 mejoras. Los contornos siguen celdas calculadas, no umbrales interpolados.

La observación no dice que cualquier grupo mejore ni que toda primera acción óptima agrupe. El argumento de intercambio de dos pruebas está demostrado; el caso general heterogéneo sigue pendiente de cerrar por escrito y de revisión humana, según el plan.

Transición: “Agrupar ya ayuda dentro de la clase laminar. Ahora cambiamos la referencia para medir cuánto cuesta quedarnos en ella.”

## 4. Un caso que conecta las comparaciones · tres cuartos de minuto

“Con n=6, B=4, G=3 y q=.5, individuales obtiene 2, el óptimo laminar 2.5 y el general 3. La primera comparación gana media unidad; la segunda pierde media unidad. Vamos a medir esta segunda diferencia respecto al general.”

La primera mejora relativa es .5/2=25%; la segunda pérdida es .5/3≈16.7%. Por eso no se comparan directamente los porcentajes de los dos mapas. Mantener las mismas personas y recursos hace visible el cambio de referencia.

La cifra general 3 alcanza toda la utilidad esperada disponible, nq. Los resultados son exactos; aún no hemos extraído un árbol breve del óptimo n=6 que explique la separación. Si se abre el ejemplo ABC/ABD/ACD de las notas, anunciar que corresponde a n=4 y es otra instancia.

Transición: “El siguiente mapa mantiene los 228 casos; ahora compara general con laminar.”

## 5. El mapa del precio de laminaridad · casi dos minutos

“El caso anterior está marcado: 3 frente a 2.5, una pérdida del 16.7%. Con una, dos o seis pruebas ambos óptimos empatan en esta malla. Con presupuestos intermedios aparecen separaciones en casi todos los casos.”

Señalar el caso B=4; después B=2. “Esta fila blanca no contradice el mapa azul: con dos pruebas agrupar ya mejora a individuales, pero permitir la clase general no añade valor en esos casos.”

El tono cálido mide la pérdida dividida por el óptimo general. Blanco significa igualdad y todas las celdas tienen datos. La escala es distinta de la azul. B=3 separa 34 de 38 casos; B=4 y B=5, todos. Total: 110 separaciones en los 114 casos intermedios. El empate B=2 es una observación desde el inicio de la malla homogénea, no un teorema para cualquier historial condicionado.

Transición: “Estos resultados miden la separación; ahora queremos estudiar una transformación concreta.”

## 6. Una pregunta para la siguiente etapa · un minuto

“Proponemos seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar compatible con lo ya probado. La compararemos con el óptimo laminar desde el inicio: eso mide cuánto cuesta conservar esas primeras decisiones.”

Ese experimento sigue pendiente. Empezará en los ocho casos acordados después de fijar la condición de laminaridad y el desempate entre acciones óptimas. El mapa presentado no es una evaluación de esa transformación.

Pregunta para Francisco: “¿Conviene conservar ese comienzo, o para intentar una garantía habría que transformar las decisiones desde el inicio?”

Cierre: “El par explica una ventaja frente a individuales. Los mapas distinguen esa ganancia del costo de exigir laminaridad. El siguiente paso es ver cuánto conserva una transformación concreta.”

## Para la discusión, fuera del pase principal

- Sobre 5/6: es un techo para una posible garantía universal, no una garantía conseguida. Toda política laminar en el caso destacado vale a lo sumo 2.5, también si se aleatoriza sobre políticas laminares. Conservar una fracción c del general de valor 3 requiere 3c≤2.5. Puede haber casos peores fuera de esta malla.
- Sobre el paper: la construcción de §5 produce planes estáticos con pruebas binarias y hard clearing. Nuestro destino puede seguir siendo dinámico, con conteos y soft clearing. La identidad de dos perfiles independientes sirve como herramienta; falta la desigualdad de utilidad para la transformación laminar. Pregunta: “¿Qué propiedad de la transformación necesitaríamos para adaptar tu argumento?”
- El árbol con n=4, el piloto de conversión, las heurísticas y los dibujos del paper siguen en las notas de estudio. Anunciar el cambio de pregunta y de setup antes de abrirlos.
- La lectura del paper y los comentarios siguen siendo el encargo principal de Francisco. Estas figuras aportan evidencia para discutir; no sustituyen el memo de §5, el experimento pendiente ni la revisión matemática con Vladimir.

## Evidencia y reproducción

[Datos exactos](../../results/sesion_2026-10-06/tres_referencias_28.json) · [Verificación del artefacto](../../results/sesion_2026-10-06/verificacion_presentacion.json).

Los 190 óptimos laminares con B<6 coinciden entre dos solvers; las 228 políticas se evaluaron sobre 64 estados y se verificó laminaridad. Con B=6, las individuales alcanzan la cota total. Los generales proceden de la misma malla exacta. La restauración conserva todos los datos; no relanza experimentos.

Reconstrucción: `python augmented/notebooks/build_28_sesion.py`. Validación: `python results/sesion_2026-10-06/verifica_presentacion.py`. El contenido principal está en `presentacion_28.py`; las consultas, en `contexto_28.py` y el generador.
