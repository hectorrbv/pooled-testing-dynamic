# Lectura de los loops y siguiente experimento — 20 de octubre

La dirección de diseños generales sí merece continuar, cambiando la familia de pruebas que explora. La regla laminar nueva merece conservarse como candidata, pero no invertiría otra noche en ajustarla sobre la misma batería. El resultado relevante para la sesión es separar estas dos decisiones.

## Diseños: valores confirmados, familia todavía pequeña

Se reevaluó `design_best.py` (`694fc32231`) en los seis objetivos. El nuevo evaluador recorre todos los perfiles y acredita los sanos al final de cada trayectoria, sin llamar al juez original ni usar el valor interno del optimizador para calcular utilidad. Reutiliza el generador de acciones: es una verificación independiente del valor del diseño fijo, no una optimización independiente de toda la clase general.

Las seis diferencias son cero. Para n=16 y B=4, los valores son 11/4, 107/32, 3459/1024 y 3459/1024 en G=2,4,8,16. Para B=8, G=8 y G=16 dan 6769/1024. Los árboles completos y el SHA están en [la auditoría](../../results/sesion_2026-10-20/loops_revision/loop1_audit.json). Hay controles independientes con individuales y con el detector de cuatro personas.

El diseño mejora la línea base, pero todos sus bloques tienen a lo sumo seis personas y todos sus pools, cinco. Pasar de G=8 a G=16 deja idéntica la familia que optimiza. La curva plana no informa contra la conjetura de log G. Con estos pools, la propia cota de información limita V/B por log₂6, independientemente del G nominal. Además, n fijo impone el techo V≤n/2; en n=16, B=8, ya tenemos V/B≤1.

La ganancia encontrada es una construcción alcanzable, no el óptimo general ni una separación general/laminar. Para esta última sigue faltando una cota que alcance a todas las políticas laminares.

## Heurísticas: utilidad parecida, costo muy distinto

Se recalcularon los porcentajes y errores desde los CSV pareados de n=50, B=5. Rollout mejora a greedy 2.01% en G=3 y 5.20% en G=5. La regla añade sobre rollout 0.16% y 0.34%; sus intervalos aproximados del 95% son [-0.11%, 0.43%] y [-1.01%, 1.69%]. Ninguno separa la diferencia de cero. Esto no demuestra equivalencia.

La regla cuesta 3.82 y 9.66 veces el tiempo de rollout. Son cocientes de tiempos medios por instancia, acumulados sobre todos los sorteos y con cachés: 24 sorteos en G=3 y 12 en G=5. No son tiempos por decisión ni una medida de complejidad asintótica. Mantendría rollout como referencia práctica con estos datos.

Los errores se calculan sobre diferencias pareadas por instancia. El porcentaje divide la diferencia media entre la utilidad media de la referencia. Los intervalos t de la figura se construyen para la diferencia y se escalan por esa media observada; son aproximados, no intervalos exactos de un cociente. Datos y fórmulas quedan en [interpretacion.json](../../results/sesion_2026-10-20/loops_revision/interpretacion.json).

La comparación es dentro de una partición fija: personas ordenadas por qᵢuᵢ en bloques de tamaño G. Se consideran los bloques tocados más los tres intactos con mayor ganancia inmediata. Los tres métodos comparten esta poda. Los qᵢ y uᵢ son heterogéneos; este experimento no corresponde al régimen q=1/2, u=1 de la conjetura. No se conoce el óptimo laminar de estas instancias.

La comprobación del harness a 10⁻⁶ en casos de un bloque es útil, pero su alcance es ese control de implementación. No demuestra que se evalúen todas las acciones laminares ni certifica la regla para todas las instancias.

## Corrección de la comparación «0.3% frente a 5%»

El scorer conservado (`cc6dae29d4`) alcanza 124 de las 125 celdas de selección; su peor pérdida es 0.2808%. La búsqueda reutilizó esa batería para escoger la regla. La cifra anterior de rollout, 5.26%, procede de la malla de 32 casos y una jerarquía fija con menos acciones permitidas. No es un contraste controlado entre ambas reglas. El 0.9495 que aparece en otro ensayo del loop 2 corresponde a una regla de profundidad tres, tampoco a ese rollout.

El cara a cara disponible es el de n=50. Para afirmar una ventaja en casos pequeños, falta evaluar rollout y la regla nueva en los mismos casos, con las mismas acciones. Para afirmar que la ventaja se traslada, hacen falta casos reservados antes de ajustar la regla. No debe confundirse una pérdida máxima en la batería de selección con una mejora media fuera de ella.

## Propuesta concreta antes de otra noche

1. Usar como referencia los seis diseños ya auditados. El criterio inicial es mejorar su utilidad con el mismo n, B y G, no ganar un score por cambiar la población.
2. Comparar por separado dos ampliaciones: grupos que combinen bloques, y bloques mayores. Mantener el diseño previo disponible, el mismo evaluador, semillas de búsqueda y límites de tiempo y memoria fijados de antemano. Una búsqueda truncada puede rendir peor aunque su familia sea mayor: reportarlo como límite del algoritmo, no del modelo.
3. Para cada mejora, guardar el árbol, la primera decisión distinta, el tamaño máximo de pool y el tamaño de los componentes conectados por las pruebas. Reevaluar el árbol completo independientemente. El mecanismo importa más que una mejora agregada sin explicación.
4. Si el piloto aporta un mecanismo útil, definir una familia creciente de n, B, G compatible con las matrices y con la cota laminar buscada. Repetir bloques de tamaño fijo no basta. Tampoco basta comparar contra individuales para afirmar una separación de toda la clase laminar.
5. Para heurísticas, congelar una batería nueva y comparar a tiempo limitado antes de volver a optimizar. Medir por separado el efecto de la profundidad, la diversidad de tamaños y la partición. Mantener este frente como secundario respecto al argumento log G.

El piloto y la nueva noche no se han ejecutado. No se ha contactado a A. La decisión que conviene llevar es si esta ampliación de familia merece el presupuesto de búsqueda, con un piloto acotado como primer paso.

## Notebook y reproducción

El notebook principal conserva sus figuras anteriores y añade los bloques 5 y 6: diseños y costo de planificación. El bloque 7 reúne las propuestas. Las definiciones, limitaciones y respuestas para estudiar están en las dos celdas nuevas al principio de las notas de estudio.

Desde la raíz del repositorio, con el entorno del proyecto:

```sh
python augmented/auditar_loop_logg.py
python augmented/notebooks/actualizar_loops_29.py
```

El segundo comando solo lee resultados guardados. Los generadores generales del notebook también incluyen esta actualización. No hace falta repetir las simulaciones de n=50 para regenerar las gráficas.
