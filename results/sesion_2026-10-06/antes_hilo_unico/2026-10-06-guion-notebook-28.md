# Notebook 28 · guion para Francisco

[Presentación](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) · [Material de estudio](../../augmented/notebooks/28_notas_de_estudio.ipynb).

Cuatro bloques, unos ocho minutos. El hilo es: cuánto ganamos agrupando sin cruces, por qué ganamos, cuánto valor adicional aportan los cruces y qué debería hacer una buena conversión. Los dos mapas usan exactamente las mismas 228 instancias con n=6. La simulación del paper se discute por separado.

## 1. «Con dos pruebas aparece otra forma de aprovechar un grupo»

“Calculamos 228 casos exactos con seis personas. Incluso exigiendo laminaridad, agrupar mejora con dos a cinco pruebas en toda la malla. Con una sola prueba, mejora si la probabilidad de salud supera un medio. Con seis, las individuales ya resuelven a todos.”

Señalar tres filas del mapa: B=1, B=2 y B=6. Un 50% significa una utilidad laminar 1.5 veces la de individuales. Son 19 probabilidades × 6 presupuestos × 2 tamaños máximos. En total: 170 mejoras y 58 empates. El mapa compara óptimos, no evalúa cuál heurística funciona mejor.

Pregunta posible: “¿Cómo cambia la magnitud de la mejora cuando las probabilidades son distintas entre personas?” Eso queda para otro experimento; esta malla es homogénea.

## 2. «Agrupar permite resolver a los mismos dos y a veces probar a un tercero»

“En q=.3, probar individualmente A y B vale .6. Si probamos AB primero, también podemos resolver a ambos usando como máximo dos pruebas. Cuando AB tiene cero o dos infectados, ya está resuelto y la segunda prueba se dedica a C. Esa oportunidad aparece el 58% de las veces y aporta .58×.3=.174. Por eso llegamos a .774.”

Pausa para señalar las dos ramas. No mezclar aquí simulación, planes estáticos ni λ. El .6 es una contribución esperada sobre todas las ramas: no cobramos .6 cada vez. Si AB tiene dos infectados, no cobramos nada por ellos, pero podemos cobrar por C.

Este ejemplo explica el 29% del mapa para n=6, B=2, G=2, q=.3. Solo usa A, B y C; las otras tres personas no se necesitan. No afirma que cualquier grupo ni cualquier apertura agrupada sea óptima.

## 3. «El empate entre general y laminar no es un empate con individuales»

“La segunda pregunta es cuánto perdemos al prohibir cruces. Ambas clases pueden agrupar y adaptar. En los mismos 228 casos, el general supera al laminar en 110 y empata en 118. Las separaciones aparecen con tres, cuatro o cinco pruebas. La mayor pérdida observada es 16.67%.”

Cruce en una frase: AB seguido de AC comparte A, pero ninguno de esos grupos contiene al otro. AB seguido de A, o de CD, sí es laminar.

Si surge la confusión: en n=6, B=2, G=2, q=.3 las individuales valen .6 y ambos óptimos .774. El cero del gráfico solo compara estos dos últimos. Aquí el porcentaje divide la pérdida por el óptimo general, mientras que el primer mapa divide la ganancia por el valor individual.

Ejemplo que une ambos mapas: n=6, B=4, G=3, q=.5. Individuales=2, laminar=2.5, general=3. El primer mapa dice +25% respecto a 2; el segundo dice pérdida de 16.67% respecto a 3. No sumar esos porcentajes: sus denominadores son distintos. Para explicarlo sin porcentajes: agrupar sin cruces añade media unidad y permitir cruces añade otra media.

Pregunta posible: “¿Qué información aportan los cruces en esta ventana de presupuestos?” El empate con B=2 es observado en esta malla; el de B=n se explica por resolver a todos. B=3 separa 34/38 casos; B=4 y B=5 separan 38/38 cada uno. El mapa anterior n=4 queda en las notas: su máximo 12.5% ya no describe la malla principal.

## 4. «Una conversión puede perder más que la propia restricción»

“Para n=4, B=3, G=3 y q=.5, los tres conteos de ABC, ABD y ACD identifican a todos: valor 2. El mejor laminar vale 1.75. Una conversión simple —ABC, después D y después A si falta resolver ABC— se queda en 1.50. Hay .25 de pérdida inevitable por la clase y .25 adicional por la regla. Conservar ABC y continuar con el cociente recupera la segunda parte en este piloto.”

Los tres valores son utilidades esperadas de políticas completas. La regla de cociente compara utilidad esperada por prueba esperada y reconsidera después de cada resultado. Este caso no demuestra que siempre alcance el óptimo. Si Francisco pide el mecanismo, abrir el árbol de continuación en las notas: después de ABC mixto, empezar por A permite decidir si la última prueba conviene en D o dentro de ABC.

Propuesta para discutir: “Queremos seguir las decisiones de la política general y cambiar antes del primer cruce. Compararemos la mejor continuación laminar con el cociente, y también con sus versiones desde el inicio.” Esto separa el costo de laminaridad, el de conservar decisiones previas y el de la heurística. El experimento de primer cruce sigue pendiente; los valores del piloto no lo sustituyen.

## Si hablamos del paper

“Entendimos la construcción de §5: una población ficticia selecciona un recorrido, se sortean las muestras que permanecen en cada grupo y el plan fijo se evalúa sobre otra población independiente. La garantía es en promedio; implica que existe una buena lista fija.”

El destino del paper es estático y permite cruces, con observación binaria y hard clearing. Nuestro destino sería dinámico y laminar, con conteos y soft clearing. La simetría entre dos perfiles independientes puede reutilizarse; falta un argumento que controle la utilidad al imponer laminaridad. No presentar el factor dos como una garantía nuestra. El memo de transferencia paso a paso aún está pendiente.

## Verificación y archivos

La malla unificada contiene 228 instancias con los tres valores. Para B<6, los 190 óptimos laminares coinciden exactamente entre Bellman por perfiles y solver de átomos; las 228 políticas laminares se evaluaron sobre los 64 estados y se comprobó que no cruzaran. Para B=6, las individuales alcanzan la cota de utilidad total. Se reutilizan los óptimos generales ya calculados y verificados de la misma malla; no se mezclan poblaciones ni presupuestos.

La malla anterior n=4, conservada como consulta, tiene 114 casos: políticas evaluadas sobre los 16 estados, valores laminares cotejados con el solver de átomos y generales con la auditoría independiente. La evaluación independiente certifica el valor de las políticas; la optimalidad procede de los enumeradores exactos y de las cotas alcanzadas.

[Tres referencias en los mismos casos](../../results/sesion_2026-10-06/tres_referencias_28.json) · [Malla anterior n=4](../../results/sesion_2026-10-05/mapa_laminar_28.json) · [Pilotos](../../results/sesion_2026-10-04/evidencia_28.json).

Para añadir los valores laminares a la malla general: `python -m augmented.experimento_tres_referencias_28`. Reconstruir ambos notebooks sin relanzar experimentos: `python augmented/notebooks/build_28_sesion.py`. El contenido principal está en `augmented/notebooks/presentacion_28.py`; las derivaciones y tareas se conservan en el generador.
