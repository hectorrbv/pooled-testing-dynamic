# Revisión C-M1 — qué cambió en el v4 de arXiv:2206.10660 y qué perfiles reales trabajó (2026-09-01)

Complementa la extracción del 2026-08-20 (que cubrió Prop 1, Thm 4 y el
despacho a C1/A-M22). Aquí: el diff v3→v4 completo, las ideas del update que
tocan a este proyecto (con pregunta para Francisco cada una), y los perfiles
de infección reales del piloto contra nuestra cobertura experimental.
Fuentes: v3 local (`papers:study/Welfare Maximizing Group Counting.pdf`,
2023-09-20) y v4 HTML (arXiv, 2026-08-15), leídos hoy. **A valida antes de
citar cualquier enunciado en el paper propio.**

## 1. El diff v3 → v4 (lo relevante)

| Pieza | v3 (EC'23) | v4 (2026) |
|---|---|---|
| Costo de prohibir solapes | factor ≤ 4 (peor ejemplo 7/6) | **factor ≤ 2 universal** (Thm 1; ejemplo 19/16; gap abierto), más fino con tope de pool en q alto (Thm 2), y **cero si todo q ≤ 1/2 (Prop 1)** |
| Complejidad | NP-dureza citada de literatura previa | suite propia: **#P-duro evaluar solapes** con G ≥ 3 (Prop 2), NP-duro + **sin FPTAS** con o sin solape (Thm 4, Cor 1), **W[1]-duro el test único en G** (Thm 3); poli para G ≤ 2 vía matching |
| Garantía del greedy | 1/5 del óptimo sin solape | **1/(e+1) contra el óptimo CON solape** (Prop 4); familia que tiende a 1/e (gap abierto); Thm 7 más fuerte vs sin-solape; OrderedGreedy 1/e con utilidades homogéneas (E.1) |
| Oráculo de test único | heurística | **FPTAS + programa cónico entero mixto (MICP, ε = 1e-7)**; post-proceso por subconjuntos ("subset-domination", Thm 6) que restaura la constante con oráculo aproximado |
| Poblaciones con tipos | — | **E.2: clusters de (u, q) idénticos** — greedy casi-óptimo repitiendo el test casi-óptimo; MILP con clusters |
| Empírico | descripción del piloto | MILP validado contra fuerza bruta; greedy ≥ **99.37%** en datos del piloto (B hasta 34, G ∈ {5,10}); sintético n = 200 con q ~ U[0.5,1] y u normal; RCT con análisis formal (ANCOVA + **equivalencia TOST** en d = 0.5 para satisfacción/desempeño/productividad/aprendizaje; estrés no establecido, p = 0.079) |

## 2. Ideas del v4 que sirven aquí — tres preguntas para Francisco

(Versión simplificada tras revisión cruzada 2026-09-01; las cinco técnicas
originales quedan absorbidas o descartadas abajo. Cada pregunta: qué se
pregunta, por qué importa, y qué desbloquea la respuesta.)

**P1 — La de dirección.** *"Con lo que ya tenemos — el solver exacto chico,
el contraejemplo y las heurísticas — ¿cuál es la siguiente evidencia que de
verdad cambiaría lo que creemos sobre una garantía constante: encontrar una
familia de instancias mala, demostrar una desigualdad nueva, o medir un
benchmark grande por tipos?"*
Por qué importa: evita implementar mejoras sin saber qué resultado mueve el
proyecto. Qué desbloquea: la meta de las próximas 4–6 semanas y el criterio
para priorizar teoría vs contraejemplos vs cómputo.

**P2 — La de la barrera.** *"Tu greedy estático nuevo garantiza una constante
(1/(e+1)) sin importar el tamaño de pool; nuestro greedy dinámico solo
garantiza 1/G. ¿En qué paso concreto se rompe tu prueba cuando la prueba
revela un conteo y deja decidir después? ¿Ese es EL obstáculo para pasar de
1/G a constante?"*
(La jerga en cinco palabras: "valor de opción" = beneficio de decidir
después.) Por qué importa: localiza el hueco matemático real en vez de
perseguir analogías. Qué desbloquea: B puede instrumentar el solver para
buscar las instancias que exhiben exactamente ese fallo.

**P3 — La del oráculo y el benchmark realista.** *"Para atacar esa barrera
con cómputo: ¿el score local debe valorar también las decisiones futuras —
no solo el mejor test estático — y lo validamos primero sobre la población
del piloto (los 6 grupos demográficos con sus utilidades encuestadas) como
benchmark canónico del solver por tipos?"*
Por qué importa: conecta la Conjetura 10.7 con una implementación evaluable
y con datos reales. Qué desbloquea: la especificación del oráculo (el MICP
estático queda como baseline de comparación, no como respuesta) y el
benchmark que define el trabajo de B con tipos.

**Descartadas, con razón:** la analogía factor-2-solapes ↔ factor-G-laminar
(bonita, pero son restricciones distintas y no produce decisión ni
experimento); y el MICP como pregunta independiente (optimiza un test sin
valorar la información futura — vive dentro de P3 como baseline). El truco de
subset-domination queda dentro de P3 (tolerancia a oráculos aproximados).
La metodología TOST se anota para cuando haya datos propios; sin acción hoy.

## 3. Perfiles de infección reales del piloto, y nuestra cobertura

**Lo que trabajaron en la vida real (IPICYT, San Luis Potosí, sept 2022):**
n = 130 participantes (RCT de dos brazos), presupuesto semanal B = 30,
G = 5 en despliegue (benchmarks hasta B = 34, G ∈ {5,10}). Probabilidades de
infección estimadas por **6 grupos demográficos** ({hombre, mujer} × {15–29,
30–59, ≥60}) con actualización bayesiana de datos IHME de SLP + agregados
nacionales — **incidencia baja** (q_sano alto, régimen ~0.9+). Utilidades
**heterogéneas por encuesta** en 4 dimensiones (desventaja socioeconómica,
dependencia de recursos institucionales, necesidad psicosocial, desempeño
reciente), escaladas ×50 a enteros. Sintético: n = 200, q ~ U[0.5, 1], u
normal.

**¿Ya lo cubrimos? En su mayor parte, no.** Nuestra malla vive en el otro
extremo (prevalencia alta, q_sano ≤ 0.7, homogéneo, u = 1, G ≤ 3; el ancla
G = 16 es analítica). Del régimen del piloto solo tocamos q = 0.7 y el
sintético q ~ U[0.5,1] coincide en espíritu con nuestro barrido 23.1 superior.
Faltan: q_sano ∈ [0.9, 1), utilidades heterogéneas, G ∈ {5,10}, n ~ 130 con
tipos.

**Primer parche, corrido hoy (rebanada homogénea, n ≤ 5, G ≤ 3, u = 1,
posterior-zero, juez exacto):** en q_sano ∈ {0.90, 0.95} todas las
heurísticas quedan cerca del óptimo (S0 0.9906 de media, mágico puro 0.9935,
C3 0.9908 con el mejor peor-caso 0.9637) — las patologías que estudiamos
(no-reentrada, degeneración del costo) casi no muerden en incidencia baja — y
agrupar paga fuerte: óptimo laminar 4.29 contra 2.70 de singletons en n = 5,
B = 3, G = 3, q = 0.9 (+59%). Consistente con que su piloto agrupara con
G = 5 y su Prop 1 no aplique ahí (q > 1/2).

**Qué falta para cubrir el perfil completo (ruta):** etapa de tipos del
solver (Prop 6.2, ya en la nota de diseño B-M17) para llegar a n ~ 130 con
los 6 grupos; malla con G = 5 (además separa densidad de bisección, Prop
9.1); utilidades heterogéneas tras G4b (cap declarado de la matriz 23.1). El
formato de población del piloto es el candidato natural a benchmark realista
del solver — pregunta 5 de arriba.

## Acciones

- Preguntas 1–5 al buzón de §34 para la próxima sesión (A las modera).
- La rebanada q ∈ {0.9, 0.95} se incorpora a la malla del barrido y de la
  misión V̂ como región "perfil piloto" (etiquetada; homogénea por ahora).
- A valida el diff v3→v4 contra los PDFs antes de citar en el paper propio.
