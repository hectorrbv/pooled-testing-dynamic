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

## 2. Ideas del v4 que sirven aquí — preguntas para Francisco

1. **Subset-domination (Thm 6).** Su greedy con oráculo aproximado pierde la
   garantía exponencialmente en B; la restauran evaluando los 2^G
   subconjuntos del test devuelto. *Pregunta: ¿ese post-proceso se transfiere
   al greedy dinámico del companion (§8) para tolerar oráculos aproximados de
   pool virgen — el obstáculo de raíces endógenas de 10.3?*
2. **MICP/FPTAS como oráculo de raíz.** La Conj. 10.7 permite explícitamente
   un oráculo del objetivo local. *Pregunta: ¿el MICP del test único estático
   sirve como ese oráculo para elegir pools raíz en poblaciones grandes y
   heterogéneas, casando las dos cajas de herramientas?*
3. **1/(e+1) estático vs 1/G dinámico.** Su greedy estático tiene constante
   independiente del tamaño de pool; nuestro inmediato dinámico es 1/G
   ajustado. *Pregunta: ¿qué rompe el análisis telescópico de su Prop 4 en el
   canal de conteo — es el mismo mecanismo (option value sin cobro inmediato)
   de la Cor 8.4?* Señalaría exactamente dónde debe vivir el teorema nuestro.
4. **Factor 2 de solapes ↔ factor G laminar.** Analogía de programa: su
   pregunta "¿cuánto cuesta la restricción operativa?" con respuesta
   constante; la nuestra (10.6) sigue abierta con piso 1/G. *Pregunta: ¿el
   argumento del factor 2 (duplicar al sano en dos tests) tiene contraparte
   laminar-aumentada, o el conteo rompe la simetría?*
5. **Clusters (E.2) ↔ tipos (Prop 6.2 del companion).** Mismo movimiento en
   estático y en dinámico. *Pregunta: ¿heredamos el formato exacto de
   clusters del piloto (los 6 grupos demográficos × utilidades encuestadas)
   como benchmark canónico del solver por tipos?*
6. **Metodología TOST** para claims empíricos de equivalencia — anotada para
   cuando el proyecto llegue a datos; sin acción hoy.

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
