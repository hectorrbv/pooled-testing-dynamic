# Inventario del companion — checklist viva de A-M23 (2026-08-30)

**Fuente:** `dynamic_augmented_laminar_companion.pdf` ("Dynamic Augmented Welfare-Maximizing Pooled Testing: A Laminar Theory Companion", working theory notes, agosto 2026; 24 pp.).
**Estatuto global: [SIN VALIDAR — §25]** hasta validación de A, resultado por resultado.
**Prioridad de verificación (sesión 2026-08-25):** §8–§10 primero (lo "mucho más nuevo"; Francisco toma la lectura de §8+); ≤§7 declarado por él "prácticamente seguro" — se valida igual, después.
Leyenda: ✓ = verificado por el equipo (se indica qué y cuándo); ∅ = pendiente.

| § | Resultado | Enunciado (una línea) | Confianza de Francisco (2026-08-25) | Verificado por el equipo | A-M23 |
|---|---|---|---|---|---|
| 2 | Modelo; Def 2.1–2.3 | conteo exacto $R(T)$, pools $\le G$; posterior-zero clearing; clase **pathwise** laminar | alta (≤§7) | coherencia con §5 del plan tras G0 ✓ (2026-08-30); ojo: clase ex post → pregunta (22) | ∅ revisar definiciones |
| 3 | Prop 3.1 | átomos residuales particionan lo testeado; conteos de pools ⟺ conteos de átomos | alta | = Lema A Partes I–II (confluencia; nuestra prueba propia sigue en A-M5–A-M9) | ∅ leer prueba |
| 3.1 | Ec. 3.3–3.5; Prop 3.2; Cor 3.3 | $Z(A,r)$ = coef. de $\prod(q_i+p_iz)$; posterior factoriza sobre átomos; $O(\|A\|^2)$ por convolución; átomo conteo-0 acredita | alta | = A2.3/A2.8; implementado y testeado en `laminar_tables.py` (dos vías, G1) ✓ | ∅ cotejo 1:1 de enunciados |
| 4 | Thm 4.1; Rem 4.2 | forma atom-normal WLOG: solo pool virgen o subconjunto propio de UN átomo; exige clearing epistémico | alta | responde (17); A-M21 = caso $D=\varnothing$ | **∅ validar (cierra A-M21)** |
| 5 | Thm 5.1; Rem 5.2 | estado suficiente $(U,\mathcal A,b)$; $Q^{open}/Q^{ref}$; recursión exacta 5.5; lo difícil = presupuesto compartido | "prácticamente seguro… hasta Bellman" | Rem 5.2 = advertencia de separabilidad ya escrita en §14.5 ✓ | ∅ leer prueba (B-M17 da validación computacional indirecta) |
| 6 | Prop 6.1 | algoritmo finito exacto; cota $O([(G{+}1)(n^G{+}B2^G)]^B\,\mathrm{poly})$; W[1]-dureza de un test estático (cita [1]) | alta | ∅ | ∅ |
| 6.1 | Prop 6.2; (6.2)/(6.3) | compresión por tipos: acciones por composición | alta | aritmética ✓ (2026-08-30): $297{,}968{,}931=\sum_g\binom{130}{g}$; 55/251/3,002 para M=3/5/10; átomo m=16: 15 vs 65,534 | ∅ prueba de simetría |
| 7.1 | Thm 7.1; Rem 7.2 | $q_i\le1/2$ ⇒ binario adaptativo = top-$B$ singletons por $q_iu_i$, no-adaptativo | "casi seguro" (forma parecida a teoremas del otro trabajo) | ∅ (Harris 7.1 + potencial normalizado 7.3 por leer) | **∅ validar (cierra media §18; toca cuarentena C1)** |
| 7.2 | Lem 7.3; Thm 7.4; Ej 7.5 | bisección guiada por conteo ($\lceil\log_2\|A\|\rceil$); bienestar $\ge 1-(1-q)^{kG}$, $k=B-\lceil\log_2G\rceil$; ratio→$kG/B$ | alta | Ej 7.5 aritmética ✓ ($1-0.96^{32}=0.72918$; razón 3.0383); cotejo con §16 ✓: mismo objeto, off-by-one disuelto bajo posterior-zero ($k{=}3$, $1-0.95^{48}\approx0.9147u$) | ∅ prueba |
| 8.1 | (8.1)–(8.3); Lem 8.1; Thm 8.2; Cor 8.3/8.4 | $M_h$ (= $S_0$ en virgen + complemento libre en átomo); cota de exposición $\mathbb E[W]\le A_{BG}$; greedy inmediato = $G$-aprox **tight** incluso vs irrestricto; **salvaguarda singleton**; $\mathrm{OPT}^D_{lam}\ge\frac1G\mathrm{OPT}^D_{aug}$ | por checar (§8+) | ∅ | **∅ PRIORIDAD** |
| 8.2 | Prop 8.5; Prop 8.6 | testigo directo anti-AS ($n{=}2$); masa posterior sana = martingala (el presupuesto mágico no contiene VOI) | por checar | testigo: aritmética ✓ bajo ambas convenciones para $q<1/2$ (guion 2026-08-25 §B.1); Prop 8.6 = nuestro A-M11b ✓ | **∅ prueba de A → C5 sube a [DEMOSTRADO]** |
| 8.3 | (8.10)–(8.13) | $H_c^\circ/H_c$ = menú local exacto por presupuesto; $\rho_b=\max_c H_c/c$; políticas committed/receding; índice Lagrangiano $I_\lambda$ | por checar | ∅ (= candidata F exacta local, §14.8) | **∅ PRIORIDAD** |
| 9 | Prop 9.1; Lem 9.2; Thm 9.3 | densidad de bisección > singleton **sii** $G>1+\lceil\log_2G\rceil$ (⇒ $G\ge4$); cota de árbol de decisión; **separación de tres vías** $1/G$ · $1/\log G$ · $\to1$ (rare-health) | por checar | condición $G\ge4$ verificada aritméticamente ✓ (G=2,3 fallan; G=4 pasa) | **∅ PRIORIDAD** |
| 10.1 | Thm 10.1 | proyectos disjuntos de costo determinista: prefijo por densidad ∨ mejor proyecto ≥ ½ del óptimo adaptativo | por checar | ∅ | **∅ PRIORIDAD** |
| 10.2 | Thm 10.2 | certificado Lagrangiano de tiempo de paro: $\mathrm{OPT}_{hard}\le\lambda B+\sum I_\lambda$; $\inf_\lambda$ = relajación de presupuesto esperado (dualidad LP) | por checar | ∅ | **∅ PRIORIDAD** |
| 10.2b | (10.4)–(10.9); Thm 10.3 | evaluador forward $J_b^\pi$; portafolio best-of-three ($1/G$ siempre; $\to1$ rare-health; ½ restringido); certificado por instancia $W_{best}/U$ | por checar | $J_b^\pi$ ≈ nuestro `ExactPolicyEvaluator` (reutilizable en B-M17) | **∅ PRIORIDAD** |
| 10.3 | Abiertas 10.4–10.6; Conj 10.7 | 4 obstáculos (raíces endógenas, costos aleatorios, interleaving, complementariedad); conjetura: constante $\alpha$ vs óptimo laminar con $G$ fijo | abiertas (digestión, no validación) | mapeo obstáculos→flags del falsificador ✓ (abajo) | ∅ alinear con A-M22 |
| 11 | Phases 1–6 | programa de prueba y cómputo propio del companion | — | mapeo a milestones ✓ (abajo) | — |
| 12 | Balance | qué afirma y qué NO (sin constante $G$-independiente vs laminar general ni laminar-vs-irrestricto) | — | leído ✓; marco compatible con §25 | — |

## Mapeos útiles

**Phases §11 → plan:** Phase 1 ≈ A-M23 + tests estilo G1 (unit-test del update de conteos vs enumeración) · Phase 2 = B-M17 (solver 5.5, árboles de decisión completos, $J_b^\pi$ para $\pi_M/\pi_C/\pi_R$) · Phase 3 = mapas homogéneos $(q,B,G)$ — diagnóstico exacto post-B-M17; **no** es el atlas de §23.4 · Phase 4 = B-M11, familia "menor-instancia-separadora-por-par" · Phase 5 ≈ A-M12 (teoría de raíz fija, $W_c(\Gamma)$: monotonicidad, split balanceado) · Phase 6 ≈ §20/§35 (extender 10.1 a costos estocásticos vía 10.2).

**Obstáculos §10.3 → falsificador (§17):** raíces endógenas → selección de pool (contiene el problema original de un solo test); costos efectivos aleatorios → colas de $C$ (pregunta 11 de §34); interleaving → flags multiátomo / átomo–virgen; complementariedad → Prop 8.5 / clase "valor perdido por separabilidad".

**Hallazgos de lectura (IA, 2026-08-30 — A valida; no son directrices de sesión):**
1. Phases 1–6 mapean 1:1 a milestones (arriba).
2. El companion trabaja con la clase **pathwise (ex post)**; la normativa del plan era ex ante → pregunta (22), comparador like-with-like en B-M17.
3. **Salvaguarda singleton** (Cor 8.3): toda candidata en versión portafolio con top-$B$ singletons conserva $1/G$ gratis → columna del harness (B-M9/B-M10).
4. $G\ge4$ para el mecanismo de densidad (Prop 9.1): la malla 23.1 y el barrido de §17 ($G\in\{2,3\}$) **no pueden exhibirlo**; las familias densidad-vs-inmediato requieren $G\ge4$ (23.2 las tiene).
5. $I_\lambda$ (8.13) como candidata G declarada (estatuto tipo D); $U_{exp}$ (10.8) y $U_{Lag}$ conectan con la vara de certificados de §30.1.

**Choque resuelto:** convención → posterior-zero vía G0 (sesión [43:44–43:53]; fila §32 2026-08-25; ratificación de B pendiente). Off-by-one del acid test disuelto.
**Choque pendiente:** ninguno conocido; el desfase paráfrasis↔enunciado de las "tres garantías ante lo óptimo en laminar" (sesión) vs comparadores formales del companion (Thm 8.2/9.3) se resuelve en A-M23.

**Referencias del companion:** [1] Finster et al., arXiv:2206.10660v4 · [2] nuestro working draft (enero 2026) · [3] Harris 1960.
