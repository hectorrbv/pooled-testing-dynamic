# Contraejemplo universal de las heuristicas greedy — keep de la mision (2026-09-01)

**Procedencia:** mision contraejemplo de dapts-autoresearch (encargo de la sesion 1-sep, Phase 4 del companion §11); juez y politicas de este repo (bm17_toy_solver, densidad_companion, C3). Scoreboard: results/contraejemplo_scoreboard.tsv. Verificado por re-corrida independiente. **Estatuto: hallazgo diagnostico §25.**


**Fecha:** 2026-09-01. **Estado: KEEP.** Score 0.6576 < 0.75, 0 violaciones,
verificado por `run_contraejemplo.py` con el juez intacto (gate de tests en
verde; fila `keep` en `results_contraejemplo.tsv`). Etiqueta: **hallazgo
diagnóstico §25** — guía para la teoría, no adopción.

## La instancia (la que queda en `instancia_candidata.py`)

```python
p = {0: 0.9, 1: 0.825, 2: 0.875, 3: 0.8, 4: 0.95, 5: 0.85}   # prob de INFECCIÓN
u = {0: 2,   1: 1,     2: 1,     3: 1,   4: 4,    5: 2}
B, G = 3, 4        # n = 6
```

OPT = 1.0645. Desglose (ratio vs OPT): **inmediato 0.6576, committed 0.6576,
receding 0.6576, C3 0.6567**. Score = max = **0.6576**: las cuatro pierden
más del 34% a la vez. Encontrada por microbúsqueda aleatoria (versión 0.7434)
y pulida por descenso local; el pulido fino (paso 1/80, 40 vecinos sin
mejora) la deja como óptimo local.

## El mecanismo: utilidad atrapada tras infección casi segura

Toda la población está casi seguramente infectada (p en 0.8–0.95) y la
utilidad se concentra en los peores (u4=4 con p4=0.95). El único modo de
cosechar es el crédito por deducción de posterior-zero, encadenado y
condicionado al conteo:

- **OPT abre el trío {0,4,5}** (z0 ≈ 0.001: "sin esperanza" para todo score
  de cobro inmediato) y juega el conteo observado:
  - R=1 (pr 0.025): refina {4} — el refine de doble filo: si 4 está infectado
    (lo probable), 0 y 5 quedan deducidos limpios y cobra u0+u5 = 4; si 4
    resulta limpio, cobra su u4 = 4. Paga por ambos lados: V = 4.98.
  - R=2 (pr 0.247): refina {5}; misma lógica en cascada, V = 2.41.
  - R=3 (pr 0.727): pivota al par {1,3}, otro mini doble filo
    (R=1 → refinar {1} vale 1.0).
- Las cuatro golosas rascan lo poco "seguro": E[W] = 0.70 en 3 pruebas.

## Por qué engaña a cada una

- **pi_M (inmediato):** su score es z0·u(S); el trío óptimo tiene z0≈0.001,
  invisible. Junta cobros marginales q_i·u_i por prueba.
- **pi_C (committed) y pi_R (receding):** la densidad ρ = max_c H_c/c ve el
  valor local del trío, pero dividir entre c y comparar componentes aislados
  hace ganar a los cobros sueltos; además el valor del trío exige reasignar
  el presupuesto restante según R (interleaving entre el átomo y el par
  {1,3}), cosa que ningún índice local captura. Las tres coinciden con pi_M
  en la trayectoria completa (0.6576 idéntico).
- **C3:** su término inmediato también es p_limpio·u_S ≈ 0; la "promesa"
  (v_mágico − inmediato) no modela el crédito por deducción (v_mágico suma
  q_i·u_i del pool, y aquí el pago viene de deducir limpios a los NO probados
  cuando el conteo se localiza en el más infectado); y su reserva de vírgenes
  usa el sano medio del propio pool como proxy del resto, inflando aperturas
  "seguras". Abre {0,3,4,5} y termina en 0.6567.

Es el obstáculo 3 de §10.3 (interleaving) más raíces endógenas (obstáculo 4):
el pool óptimo mezcla escalones de p (0.85/0.9/0.95) y ningún score lo arma;
la ganancia viene de decidir DESPUÉS de ver el conteo a qué pedazo va el
presupuesto.

## Mapa de la búsqueda (dónde bajó y dónde no)

Corridas oficiales en `results_contraejemplo.tsv`; exploración local
(importando el juez en solo-lectura) en el scratchpad de la sesión.

| Región | Mejor score | Nota |
|---|---|---|
| Homogéneas (baselines) | ≈ 1.00 | C3 óptima; receding 0.998 |
| Dos bloques homogéneos, u=1 (interleaving puro) | 0.988 | no muerde sin heterogeneidad en u |
| Premio u alta + relleno (p bajas-medias) | 0.977 | C3 la resuelve |
| Escalera p con u pro-correlacionada, B=4–5, G=4 | 0.9598 | primer síntoma: OPT fabrica átomos |
| Gadget premio+acompañante (p medias), B=2 | 0.877–0.897 | unánime, pero un solo error no baja de ~0.88 |
| Par de doble filo (p altas asimétricas), B=2 | 0.80 | aparece el crédito por deducción |
| **Todo-alta-p, deducción en cadena, B=3, G=4** | **0.6576 (keep)** | el error se compone: apertura + refinamiento + pivote |

Progresión: barridos gruesos (bloques, escaleras, premio+relleno) → búsqueda
aleatoria n≤8 → microbúsqueda masiva n≤6, B≤3 (miles de evaluaciones exactas)
→ descenso local en dos rondas. B=2 tiene piso empírico ≈0.88 (una sola
decisión errada y todas son óptimas en b=1); B=3 permite componer el error.
Ningún candidato disparó `ratio_gt_1` ni `opt_trivial`.

## Conclusión honesta

El contraejemplo EXISTE en n ≤ 8: en la instancia keep las cuatro heurísticas
de la batería pierden ≥ 34.2% simultáneamente (score 0.6576). El ingrediente
que a todas les falta es el que anticipaba el companion (§10.3): valor de
información de aperturas sin esperanza de cobro + reasignación del
presupuesto condicionada al conteo, aquí en su forma posterior-zero: cadenas
de deducción de doble filo. Consecuencia para la Conjetura 10.7: cualquier
garantía uniforme ≥ 3/4 para esta batería queda refutada en el rango n≤8,
G≤4, B≤5; un score V-hat con garantía necesitará un término explícito de
crédito por deducción condicionada al conteo, no solo p_limpio·u y promesas
de bisección.
