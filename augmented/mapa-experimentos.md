# Mapa de experimentos: qué se corrió, sobre qué instancias y para qué

## Glosario de las cantidades comparadas

Casi toda la confusión de la última sesión viene de que hay cinco cantidades
distintas y dos de ellas se nombran "estático". Todas están definidas en
`laminar_benchmarks.py`.

| Símbolo en los CSV | Función | Qué es |
|---|---|---|
| `V_star` | `dynamic_augmented_value` | Óptimo dinámico irrestricto. Programación dinámica exacta sobre todos los pools de tamaño ≤ G. Es el techo. |
| `V_laminar` | `laminar_augmented_value` | Óptimo dinámico restringido a la mejor biblioteca laminar. Mide cuánto cuesta exigir jerarquía. |
| `V_opt_en_arbol_practico` | `optimal_value(balanced_laminar_library)` | Óptimo dinámico dentro del árbol balanceado que se construye sin ver el óptimo. Es el árbol que un algoritmo real usaría. |
| `V_greedy_laminar` | `greedy_laminar_value` | Greedy miope dentro del árbol práctico. La política implementable. |
| `V_rollout_laminar` | `rollout_laminar_value` | Un paso de anticipación sobre el greedy anterior, misma biblioteca. |
| `V_static_binary` | `static_binary_value` | Diseño estático con modelo binario: se fijan los B pools de antemano y solo se distingue negativo de no negativo. |
| `V_static_greedy` | `static_greedy_value` | Diseño estático elegido de forma golosa, modelo de conteo. |

Las dos últimas coinciden solo en el 69% de las instancias del barrido. Cuando en
la sesión se dijo que greedy va mejor en tasas altas, la comparación implícita era
contra `V_static_binary` en el atlas; la objeción que se recibió era sobre
`V_static_greedy`. Ambas afirmaciones son ciertas por separado y describen cosas
distintas. Conviene no volver a decir "el estático" sin decir cuál.

## Los artefactos, uno por uno

| CSV | Generador | Instancias | Pregunta | Qué salió |
|---|---|---|---|---|
| `subset_tables.csv` | `experiments_laminar_week.py` (etapa tables) | G ∈ {4,6,8,10,12}, 12 réplicas | ¿La caché Φ de convoluciones acelera construir el tensor de subpools, y da lo mismo que recalcular? | Error máximo 0.0 exacto. Aceleración 1.3× a 1.76×, creciente en G. Es el respaldo de que el tensor es correcto y reusable. |
| `independence_gap.csv` | `independence_gap.py` | 80 réplicas × 4 categorías de pool | ¿Cuánto se equivoca la heurística de independencia (producto de marginales) frente al posterior exacto? | Sobre pools disjuntos el error es cero. Sobre un nodo ya observado la distancia en variación total llega a 0.60, media 0.275; anidado compatible y cruzado no laminar quedan en 0.14 de media. El átomo condicionado da error numérico cero en todas. Es la justificación cuantitativa de usar el tensor. |
| `atlas_instances.csv` | `experiments_laminar_week.py` (etapa atlas) | n ∈ {4,5,6}, B ∈ {2,3}, G ∈ {2,3}, 18 valores de base_p, 2 modos de tasa, 2 de utilidad, 3 réplicas. 2592 instancias con semilla | Barrido maestro: las cuatro cantidades y sus cocientes en cada celda del espacio | Es la fuente de la que salen los demás resúmenes. Guarda p, u, la mejor biblioteca y los tiempos por instancia. |
| `atlas_cells.csv` | mismo | 864 celdas agregadas del anterior | Resumen por celda: mínimo, máximo, media y mediana de cada cociente | Permite localizar la celda peor y la mejor sin releer 2592 filas. |
| `showcase_regions.csv` | mismo | 7 recortes del atlas | ¿En qué régimen gana la política dinámica al diseño estático binario? | Prevalencia alta es el régimen dominante: greedy supera al estático en 96.0% de las instancias y rollout en 98.2%, con ganancia máxima 1.2433 en p=0.9, n=4, B=3, G=2. Prevalencia baja es el peor: 40.5% y 50.9%. |
| `homogeneous_b2.csv` | mismo | perfiles homogéneos, 35 valores de p, n ∈ {4,5,6}, B ∈ {1,2}, G ∈ {2,3} | Con presupuesto pequeño y perfiles homogéneos, ¿la restricción laminar cuesta algo? | `ratio_laminar_opt` vale 1.0 en las 420 filas. Con B ≤ 2 y perfiles homogéneos la mejor biblioteca laminar alcanza el óptimo irrestricto. Es una afirmación sobre la clase de diseños, no sobre greedy. |
| `adversarial_trajectories.csv` y `adversarial_minima.csv` | mismo (etapa adversarial) | n ∈ {4,6}, B=3, G ∈ {2,3}, tres regiones de prevalencia | Búsqueda por coordenadas del peor caso: ¿qué instancia hace más chica la razón laminar sobre óptimo? | 75 pasos de trayectoria, 3 mínimos. Da los perfiles adversarios concretos, entre ellos uno de prevalencia alta muy dispersa. |
| `milp_particle_sweep.csv` | `experiments_laminar_week.py` (etapa milp) | n ∈ {6,8,10,12}, S ∈ {25,50,100,250,500} partículas, 4 réplicas | ¿Cuántas partículas hacen falta para que el MILP escenario elija el pool que elegiría el cálculo exacto? | Registra coincidencia de pool, arrepentimiento y brecha MIP. Es el puente entre lo exacto y lo escalable. |
| `pipeline_n40_*.csv` | `laminar_pipeline.py` | n=40, B=3, G=3, S=100 partículas, 250 perfiles | Tubería completa a escala grande: partículas, MILP para elegir raíz, conteo, átomo condicionado, rollout | Cuatro métodos. `flat_independence` 11.267, `myopic_milp` 11.460, `laminar_greedy` y `laminar_rollout` 10.069 ambos. Ver la advertencia de abajo. |
| `nesting.csv` | `experiments_nesting.py` | n ∈ {5,6}, B=3, G=3, 9 valores de base_p, 2 modos, 4 réplicas | ¿El greedy exacto, con el conjunto de acciones completo, elige alguna vez volver a probar dentro de un pool ya observado? | Con perfiles homogéneos y p ≥ 0.4 nunca anida: territorio virgen 1.000 exacto. Por debajo de 0.4 anida en un tercio de las decisiones. En modo heterogéneo anida entre 4% y 20%. |
| `arbol_vs_miopia.csv` | sin generador versionado | misma rejilla que el atlas, 2160 filas | Descomponer la pérdida de greedy en dos factores: elegir el árbol práctico en vez del mejor, y ser miope dentro de él | `costo_arbol` = V_práctico / V_laminar, `costo_miopia` = V_greedy / V_práctico. El reparto es 39% árbol y 61% miopía en homogéneo, y se invierte a 69% y 31% en heterogéneo. El rollout cierra el 86% de la brecha de miopía. |
| `greedy_vs_static_greedy.csv` | sin generador versionado | misma rejilla, 2160 filas | ¿Greedy laminar le gana al greedy estático, que es la comparación justa? | Con perfiles homogéneos y p ≥ 0.5 el cociente es exactamente 1.0000 en las 432 filas. La ventaja real está en p entre 0.25 y 0.45, con media 1.018 a 1.044. En heterogéneo con prevalencia baja greedy laminar pierde en 54% a 64% de los casos. |

## Tres cosas que el mapa deja a la vista

La rejilla es siempre la misma y es chica: n entre 4 y 6, B entre 2 y 3, G entre 2
y 3. Todo lo que se ha concluido sobre regímenes vive en poblaciones de seis
personas o menos. La única corrida grande es la de n=40, y ahí la política laminar
queda 11% por debajo del control más simple.

En esa corrida de n=40, `laminar_greedy` y `laminar_rollout` producen media, error
estándar, mediana y tasa de cero idénticos hasta el último dígito. El rollout no
está haciendo nada a esa escala. Cualquier número de n=40 debería esperar a que se
entienda por qué.

Los dos CSV que sostuvieron la discusión de la última sesión, `arbol_vs_miopia` y
`greedy_vs_static_greedy`, no tienen script generador en el repositorio. Son los
únicos del conjunto que no se pueden reproducir.
