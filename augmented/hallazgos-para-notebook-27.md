# Hallazgos nuevos para el notebook 27 (2026-08-31)

Material acumulado desde el cierre del notebook 26. El notebook 27 debe contar
la brecha de convención con el solver como testigo; cada número de abajo ya
tiene artefacto que lo regenera (`augmented/bm17_toy_solver.py`, corrida
2026-08-31) o cuenta a mano verificada
(`docs/notes/2026-08-31-ratificacion-G0-cuentas-B.md`).

Convención del solver y del companion: R = conteo de infectados, q =
probabilidad de estar sano. Los notebooks 24–26 cuentan sanos; el 27 debe
declarar cuál habla y traducir una sola vez.

## 1. La brecha de convención cambia la política, no solo el valor

Instancia de la spec B-M16 (n = 4, q = 0.3, u ≡ 1, B = 2, G = 2), solver
exacto en fracciones:

| Convención | Óptimo laminar | Primera acción óptima |
|---|---|---|
| estricta (hard clearing) | 3/5 = 0.60 | singleton — nunca agrupa |
| posterior-zero (G0) | 387/500 = 0.774 | **abrir el par {a,b}** |

El "par primero con continuación perfecta" pasa de 0.564 (estricta, spec de A)
a 0.774 (posterior-zero). Es la evidencia computacional de la Proposición de
brecha de convención de A: la deducción que acredita vuelve rentable el
agrupamiento donde antes no lo era.

## 2. Los tres números de G0, ahora por máquina

- Reentrada del contraejemplo: 0.5 (estricta) → 1.0 (posterior-zero),
  verificada por el solver forzando la acción.
- Ancla del acid test (q = 0.05, G = 16, B = 7): k pasa de 2 a 3 y el
  bienestar de 1 − 0.95³² = 0.806289 a 1 − 0.95⁴⁸ = 0.914742; baseline 0.35u
  sin cambio. (Para reproducirla con el solver hace falta la compresión por
  tipos, Prop 6.2 — etapa 2 de B-M17.)
- El colapso S₁ = S₀ queda adscrito a la variante estricta (S₁ᵖᶻ = 1.0 ≠ 0.5).

## 3. Consistencia cruzada con el notebook 24

Con B = 3 (estricta) el solver da 1011/1000 = 1.011 = q(3q² − 3q + 4): el
óptimo del caso de sesión, reproducido por un artefacto independiente. Bajo
posterior-zero sube a 537/500 = 1.074, y ahí el par y el singleton **empatan
exactos** como primera acción (verificado forzando ambas): la ventaja
estricta de agrupar que existe en B = 2 (+0.174) se disuelve cuando el
presupuesto deja de ser escaso. La ventaja no es monótona en B; eso merece su
propia celda.

## 4. Hallazgo de diseño: la forma atom-normal depende de la convención

El Remark 4.2 del companion usa posterior-zero para probar la forma
atom-normal; bajo estricta el menú necesita además las pruebas de átomos
deducidos de conteo 0 (sin información, pero acreditan). El solver ya lo
implementa; el notebook debe mostrarlo con un ejemplo de dos ramas.

## 5. Conexiones pendientes de validación (estatuto §25: [SIN VALIDAR] hasta A-M23)

- La martingala del presupuesto mágico (Prop 8.6 del companion) es nuestro
  A-M11b: el score del contraejemplo de no-reentrada no contiene VOI.
- ρ_b = max_c H_c/c (ec. 8.12) es la candidata F con α = 1 y filtro
  incorporado — el barrido α del notebook 26 ganaría una columna exacta local.
- Salvaguarda singleton (Cor 8.3): toda candidata en versión portafolio con
  top-B singletons conserva 1/G gratis — columna futura del harness.
- El mecanismo de densidad requiere G ≥ 4 (Prop 9.1): la malla del barrido
  (G ∈ {2,3}) no puede exhibirlo; anotar, no corregir.

## 6. Qué le toca al notebook 27 (según el plan de semana)

1. Validación dos vías del solver contra el enumerador pathwise (n ≤ 5,
   B ≤ 3, ambos flags) — miércoles.
2. La tabla dual completa de B-M16 (estricta vs posterior-zero, todas las
   acciones del menú) — B-M18 mínimo, jueves; alimenta la Proposición de A.
3. J^π_b sobre el solver: evaluar al menos π_M (greedy inmediato) y comparar
   contra el óptimo laminar por convención — viernes.
4. Si la validación cierra: primeros mapas homogéneos chicos (q, B, G) —
   extra, Phase 3 del companion.
