# Encargos del 6-oct: binario-gana-a-laminar y el precio de la laminaridad en q = ½ (diagnóstico IA, 2026-10-12)

**Estatuto §25: diagnóstico en casos exactos chicos.** Solver nuevo `augmented/dp_perfiles_binario_conteo.py` (DP exacta sobre conjuntos de perfiles, canal binario o de conteo, pools libres ≤ G, posterior-zero, Fractions), escrito por el soporte IA y **no validado por B**; reproduce cinco valores conocidos: 387/500, 1137/1000 y 2 (aumentado irrestricto, `results/precio_laminar_2026-09-21/`) y 3/5, 9/10 (binario dinámico, Thm 3.3(i) en q = 0.3). El óptimo laminar aumentado sale de `augmented.bellman_perfiles.BellmanPerfiles(laminar=True)` (validado en septiembre). Scripts: `augmented/busqueda_binario_vs_laminar.py`, `augmented/malla_precio_laminar_q.py`, `augmented/arbol_testigo_binario_vs_laminar.py`. Datos: `results/encargos_2026-10-06/`.

## 1. Encargo: un ejemplo donde el binario dinámico óptimo gana a laminar [D 35:58, 57:22]

Por V4 del acta del 6-oct el ejemplo solo puede tener Λ > 1 (con Λ ≤ 1, Thm 3.3(i) da Binary = NonOverlap ≤ Laminar). Se muestrearon instancias con Λ > 1: n ∈ {3, 4, 5}, B ∈ {2, 3}, G ∈ {2, 3}, q_i y u_i heterogéneos (tres semillas, 240 s cada una).

| Instancias con Λ > 1 | Binario > laminar | Con q homogénea | Razón mínima laminar/binario |
|---|---|---|---|
| 6,800 | 63 (0.9%) | 0 | **0.978** (n = 4, B = 2, G = 2) |

Testigo más limpio: n = 3, B = 2, G = 3, u ≡ 1, q = (0.9, 0.7, 0.7), Λ = 1.75.

| Clase | Valor | Árbol óptimo |
|---|---|---|
| Binario dinámico | 2079/1000 = 2.079 | {a,b}; negativo (0.63): {c}; positivo (0.37): **cruce {a,c}**, paga 2 con prob. 0.511 |
| Laminar aumentado | 1031/500 = 2.062 | {a,b}; conteo 0: {c}; conteo 1 (0.34): refina {a}, paga exactamente 1 por deducción; conteo 2: {c} |
| Aumentado irrestricto | 2121/1000 = 2.121 | {a,b}; conteo 1: cruce {a,c} (vale 1.1735 con conteos); conteo 0 o 2: {c} |

Mecanismo: tras el positivo, a es el sano probable (0.9 contra 0.7) y c es fresco y probablemente sano; el cruce {a,c} apuesta por dos personas (2 × 0.511 = 1.02) y gana a la deducción segura de una (1.00). Con todo q_i ≤ ½ la deducción segura siempre gana: ese es el régimen de la Proposición. Son las "dos fuerzas" de Francisco [D 31:41–32:31], medidas. Lo que **no** se dice: nada sobre el peor caso ni sobre la constante; el ¼ no se acerca en n ≤ 5.

## 2. Encargo: el factor laminar vs aumentado irrestricto en q = ½ [D 42:21–45:13, 57:50]

Malla exacta, u ≡ 1, q ∈ {0.3, 0.5, 0.7}, n ∈ {4, 5, 6}, G ∈ {2, 3, 4}, B ∈ {2, 3, 4} (81 celdas; `precio_laminar_malla_n4_5.csv`, `precio_laminar_malla_n6.csv`). Peor razón laminar/irrestricto por (n, q):

| n | q = 0.3 | q = 0.5 | q = 0.7 |
|---|---|---|---|
| 4 | 0.895 (G3,B3) | **0.875** (G3,B3) | 0.937 (G3,B3) |
| 5 | 0.874 (G3,B3) | **0.857** (G3,B3) | 0.895 (G4,B3) |
| 6 | 0.860 (G3,B4) | **0.833** (G3,B4) | 0.879 (G4,B3) |

q = ½ es el peor de los tres en todo n, y la pérdida crece con n y con B (0.875 → 0.857 → 0.833). Con B = 2 la razón es 1 en toda celda (con dos pruebas no hay cruce rentable, cf. caso par_B2 del 21-sep). Consistente con la conjetura de Francisco; no dice nada sobre el límite ni sobre G grande (la cota de información V5 sugiere O(log G); sin evidencia aún).

## 3. Lo que sigue

- B valida el solver (dos vías: contra `BellmanPerfiles` sin laminar en la malla; contra `static_binary_value` en instancias con B pequeño) antes de que cualquier número de aquí entre al guion con etiqueta distinta de "diagnóstico".
- Extender la malla en n (7–8) y G = 5 para ver si la razón en q = ½ sigue bajando; buscar el peor testigo binario-vs-laminar con n ≤ 6 y u libre.
- Preguntas para el 13-oct: (31) el ejemplo debe tener Λ > 1 — confirmado en la búsqueda; (32) ¿O(log G) en q = ½?; (33) la referencia de las matrices decodificadoras.
