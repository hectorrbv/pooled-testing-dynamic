# Guion de sesión 2026-08-25 — mapeo del *companion* de Francisco contra el plan maestro

**Documento de entrada:** `dynamic_augmented_laminar_companion.pdf` ("Dynamic Augmented Welfare-Maximizing Pooled Testing: A Laminar Theory Companion", *Working theory notes*, agosto 2026), enviado por Francisco. 24 pp., seis direcciones, referencias [1] Finster et al. arXiv:2206.10660v4, [2] nuestro working draft de enero 2026.

**Estatuto de todo lo que sigue: [SIN VALIDAR — §25].** Se verificó aritmética y consistencia interna, no las demostraciones. Nada entra a §9, §5 ni al paper antes de la validación de A.

**Procedencia de nuestro lado (git, todo anterior al PDF):** plan maestro `3750ffc` 2026-08-02 · ciclo §34-bis `387d551` 2026-08-02 · despacho 11-ago `97ee156` 2026-08-18 · despacho 18-ago + A-M21/A-M22/B-M16 `ed323af` 2026-08-20 · mantra de sobre-entrega `7438ac0` 2026-08-20.

---

## Frase de apertura

> "Leímos el companion completo y lo mapeamos contra el plan. Cinco de las seis direcciones caen sobre milestones que ya estaban escritos y fechados en el plan; dos de ellas resuelven preguntas que teníamos explícitamente marcadas como abiertas. Lo que no teníamos son cuatro objetos de la §10 y la compresión por simetría de la §6. Y hay un choque de convención que cambia los números en una prueba y tenemos que decidir hoy."

---

## Tabla maestra — las seis direcciones

| # | Dirección del PDF | ¿Lo teníamos? | Slot en el plan | Cómo lo íbamos a hacer |
|---|---|---|---|---|
| **1** | Forma normal por átomos (Thm 4.1) | **Sí, más débil** | A-M21, A-M22, §5.5, §14.10, pregunta (17) §34 | Demostrar solo el caso de repetición idéntica; podar por regla declarada mientras tanto |
| **2** | Caracterización Bellman (Thm 5.1) | **Sí, por otra ruta** | §14.5 $\Phi_b$, §15 rollout, §5 | Lema A (A-M5–A-M9) para estado exacto + rollout exacto como oráculo, no teorema escrito |
| **3** | Cómputo exacto en instancias chicas (Prop 6.1/6.2) | **Parcial** | B-M2, B-M13, B-M14, G9, §23 | Caché + consultas por demanda + perfilado; el PDF ataca otro eje |
| **4** | Greedy laminar (§8) | **Sí, cuatro veces** | A-M11, C5/A-M17, §14.6–14.8 cand. F, §34 q(10) | Ver desglose abajo |
| **5** | Densidad de extracción y separación de tres vías (Thm 9.3) | **Sí, como ancla finita** | §16 acid test, §2.4, G4a/G4b | Ancla ejecutable + nueve checks; el PDF da el teorema asintótico |
| **6** | Garantías de aproximación (§10) | **Solo el marco** | §20, §35 Sobresaliente B/C, §4 q(23) | Reducción submodular o, si falla, policy improvement / regret / régimen |
| **(§7)** | Separación reforzada (Thm 7.1) | **Sí, como celda abierta** | §18, C4, A-M15 | DP exacto + reemplazo pool→singleton con continuación |

---

## A. Lo que ya teníamos y coincide — decir esto primero

**A.1 · La martingala (Prop 8.6) = nuestro A-M11b, ya cumplido.**
El PDF prueba que la masa posterior sana es una martingala y que por lo tanto un score que la promedia no contiene valor de información. Eso es literalmente §14.4: $\Phi_2$ global muere por *tower property* [A-M11a] y $\Phi_2^{cov}$ es martingala bajo subdivisión [A-M11b]. **A-M11 está cumplido y validado en sesión** (§1, estado heredado). Coincidencia independiente, no aprendizaje.
→ *Y la moraleja que sacamos de ahí es la misma que él escribe en §8.2:* el potencial debe ser **realizable bajo presupuesto**. Eso es lo que produce $S_3$ (§14.5).

**A.2 · El estado suficiente (Thm 5.1) = nuestro $\Phi_b(H)$.**
Su $\mathcal S=(U,\mathcal A,b)$ — virgen, átomos no resueltos con conteo interior, presupuesto — es exactamente la firma de §14.5. Más importante: su **Remark 5.2** ("un valor escalar por átomo no descompone, por sí solo, el Bellman global; lo difícil es el presupuesto compartido") es palabra por palabra la advertencia que ya está escrita en §14.5 bajo el título *"Separabilidad como aproximación falsificable"*, con el falsificador (§17) midiendo específicamente el regret de acciones multiátomo y átomo–virgen.
→ Punto a marcar: **no dimos la separabilidad por buena; la declaramos aproximación y le pusimos instrumento de medición.**

**A.3 · La forma normal (Thm 4.1) ⊃ nuestro A-M21.**
A-M21 nació de su propia directriz del 18-ago ("tras una prueba grupal, la interacción es por subpruebas laminares; la repetición idéntica queda excluida"). Lo que el plan pedía era la prueba que §5.5 exige: recompensa y transición idénticas ⟹ podable. Su Thm 4.1 es el teorema general del que eso es el caso $D=\varnothing$.
→ **Y de paso responde nuestra pregunta (17) de §34**, que estaba abierta: los ancestros (testear un superconjunto) no quedan *excluidos*, quedan **reducibles sin pérdida** a su parte residual, porque $R(D)=R(T)-R(K)$. Es una respuesta mejor que la que íbamos a pedirle.

**A.4 · $H_c$ y $\rho_b$ (§8.3) = nuestra candidata F, con el menú que él mismo nombró.**
$H_c(C)$ = rollout local exacto dentro de una raíz con presupuesto $c$. Eso **es** el "menú valor-por-presupuesto" que él llamó el objeto ideal en la sesión del 18-ago (§32) y que el plan ya había escrito como $\varphi(D,c,b)$ indexada por presupuesto (§14.5–14.6, desde el 2-ago). Y $\rho_b(C)=\max_{c\le b}H_c(C)/c$ es nuestra candidata F con $\alpha=1$ y el filtro $C\le b$ incorporado.
→ *Cómo lo íbamos a hacer:* barrido **diagnóstico** de $\alpha\in\{1/2,1,3/2\}$ con filtro de factibilidad sobre la matriz §23.1, $\alpha$ congelado antes del atlas (§31), adopción solo vía B-M9 **después** de G4a. El estatuto diagnóstico está fechado en §32 (18-ago) precisamente para no elegir candidata por barrido.

**A.5 · El acid test (§16) es el punto finito de su Thm 9.3.**
Nuestra ancla: $(q,G,k,B)=(0.05,16,2,7)$, cota $u[1-(1-q)^{kG}] = 0.806u$ contra $0.35u$ del baseline singleton. Su Ejemplo 7.5: $(0.04,16,2,6)$ → $0.72918$ contra $0.24$. **Misma familia, misma fórmula, off-by-one por convención** (ver §E).

---

## B. Lo que teníamos como pregunta abierta y él responde — los dos golpes fuertes

**B.1 · Prop 8.5 nos da el testigo directo que A-M17 estaba cazando.**
Estado en el plan: C5 es **[DERIVACIÓN CONDICIONAL]**, por modus tollens indirecto ($V^{S_0}/V^*\le 0.434 < 1-1/e$), con A-M17 pendiente para subirlo a [DEMOSTRADO], y §20 diciendo que el falsificador busca *además* el testigo directo $(\psi,\psi',t)$.
Él lo da con **dos individuos**: $\Delta(\{1\}\mid\varnothing)=q_1$, pero tras observar $R(\{1,2\})=1$, $\Delta(\{1\}\mid\psi')=1$.
→ **Verificado por nosotros (aritmética, no prueba):** su testigo usa *posterior-zero clearing*; comprobamos que **sobrevive bajo nuestra convención estricta de hard clearing siempre que $q<1/2$** — es decir, exactamente en el régimen del acid test:

```
q=0.05:  Δ(ψ=∅)=0.0500   Δ(ψ′)=0.5000   → violación de AS
q=0.49:  Δ(ψ=∅)=0.4900   Δ(ψ′)=0.5000   → violación de AS
q=0.50:  Δ(ψ=∅)=0.5000   Δ(ψ′)=0.5000   → frontera
q=0.60:  Δ(ψ=∅)=0.6000   Δ(ψ′)=0.5000   → no hay violación
```

→ Es enumerable exacto con $n=2$, $B=2$. **Es el artefacto más barato del tablero y cierra la primera mitad de A-M17** (interés prioritario declarado de A, §0-bis b).

**B.2 · Thm 7.1 cierra la mitad de la celda dinámico-binaria (§18).**
§18 está escrita explícitamente **"sin resultado predicho"**, y C4 dice que la separación es *conjunta* porque la celda dinámico-binaria está pendiente. Su Thm 7.1: con $q_i\le 1/2$, utilidades arbitrarias y priors heterogéneos, **el binario adaptativo no vale nada** — el óptimo son los $\min\{B,n\}$ singletons de mayor $q_iu_i$, alcanzado no-adaptativamente. Es la extensión dinámica de la Prop 1 del arXiv que C-M1 acaba de extraer (nota `2026-08-20-extraccion-arxiv-2206.10660.md`).
→ **Caveat que hay que decir en voz alta:** no dice nada sobre $q>1/2$, y ese es justo el lado donde nuestro C1 muestra al greedy ganando solo **40.4%** en prevalencia baja. Cierra la mitad que ya sospechábamos y deja abierta la mitad disputada.

---

## C. Lo que NO teníamos — decirlo sin adornos

| Objeto | Qué es | Dónde habría entrado |
|---|---|---|
| **Prop 6.2** compresión por simetría | Acciones por vector de conteo de tipos: átomo homogéneo de tamaño 16 → **15** refinamientos en vez de 65,534; virgen $\binom{G+M}{M}-1$ en vez de $\binom{n}{G}$ | B-M14 / G9. **Ataca el número de acciones**, eje distinto de nuestra caché (C3: mediana 97,274 convoluciones evitadas en $G{=}10$, pero pared solo 1.3–1.8×). Complementario, no redundante |
| **Thm 10.1** garantía ½ | Proyectos disjuntos con costo determinista | §35 Sobresaliente B, en modelo restringido |
| **Thm 10.2** certificado Lagrangiano de tiempo de paro | **Cota superior computable** para componentes independientes fijos | §4 pregunta (23) "¿certificado por instancia?" — no teníamos ningún objeto así |
| **Thm 10.3** portafolio best-of-three + $W_{best}/U$ | Certificado por instancia, mejor que $1/G$ en poblaciones concretas | Línea de certificados, que §1 mantiene congelada hasta cerrar G4b |
| **Cor 8.3** $\mathrm{OPT}^D_{lam}\ge \tfrac1G\mathrm{OPT}^D_{aug}$ | Primera **cota** real sobre $\rho_{lam}$ | §9-C8 hoy dice *"evidencia finita, jamás cota"* (0.928 malla / 0.9069 adversaria). Débil, pero es cota |
| **Eje commit-vs-recompute** (Thm 9.3) | Mismo índice, dos implementaciones: reserva comprometida → orden $1/\log G$; horizonte recedente → $\to 1$ | Nuestro barrido de sábado sweepea $\alpha$ y **no tiene ese flag**. Es un hueco de diseño que él expone gratis |

---

## D. Lo que tenemos y el PDF no toca — para que la conversación sea de ida y vuelta

1. **Lema A y el tensor** (A-M5–A-M9): el estado exacto por partes, implementado y validado. Él asume la inferencia; nosotros la construimos y la testeamos.
2. **El falsificador como instrumento** (§17): clases de acción (repetida / descendiente / ancestro / unión de átomos / mixta / virgen / cruzada / dominada) y **flags de separabilidad** (intraátomo, multiátomo, átomo–virgen, valor perdido por separabilidad). Él razona en papel; nosotros medimos la frecuencia con la que cada patología ocurre.
3. **La descomposición de pérdidas** $\rho_{plan}\cdot\rho_{tree}\cdot\rho_{lam}$ (§8) con números: 0.928 / 0.9069 / greedy balanceado 0.747.
4. **La frontera $B{=}2$/$B{=}3$** (§19): su eje es $G$, el nuestro es el presupuesto. No se pisan.
5. **Las relajaciones convexas** (Lovász, multilineal — §20 preguntas 2–3, §21): Prop 8.5 mata la vía AS **directa**, que es la pregunta 1. Las preguntas 2 y 3 siguen abiertas, y son la línea que A declaró prioritaria (§0-bis b).
6. **La celda estático-conteos** (notebook 25 §2: 0.800 por pesado de monedas) que la Prop 1 del arXiv **no cubre** — su Thm 7.1 tampoco, porque es canal binario.
7. **§9, §22, §25**: registro de claims corregidos, reproducibilidad y disciplina de etiquetas. Es la razón por la que podemos decir hoy qué está demostrado y qué no.
8. **B-M16**: el artefacto del contraejemplo de no-reentrada (0.5 / 0.6 / retest) con semilla y CSV.

---

## E. El choque de convención — hay que decidirlo hoy (pasa por G0)

- **Él:** *posterior-zero clearing* (Def 2.1) — epistémico: acredita a quien la historia **prueba** sano, aunque no haya estado en un tubo negativo.
- **Nosotros:** *strict hard clearing* (§5.6–5.7) — solo acredita pertenencia física a un pool con conteo cero.

**Su Remark 4.2 dice que la forma normal (Thm 4.1) *requiere* la versión epistémica.** Bajo nuestra convención, reemplazar $T$ por $D$ puede cambiar quién cobra utilidad ⟹ **Thm 4.1 no transfiere literal.**

El desfase es exactamente **una prueba** (el test acreditador), y se ve en las fórmulas:

| | Fórmula de $k$ | Ancla | Cota |
|---|---|---|---|
| PDF | $k=B-\lceil\log_2 G\rceil$ | $B{=}6,G{=}16,q{=}0.04$ | $1-0.96^{32}=0.72918$ (razón 3.0383) |
| §16 | $k=\max\{0,B-\lceil\log_2 G\rceil-1\}$ | $B{=}7,G{=}16,q{=}0.05$ | $1-0.95^{32}=0.80629$ |

Misma familia, off-by-one. §14.9 ya tenía identificado el fenómeno: *"algunas ramas terminan con un singleton cuyo cero fue observado; otras identifican al sano por descarte y requieren una prueba adicional para acreditarlo."*

**Decisión pedida:** ¿adoptamos posterior-zero como convención normativa (§5.7 → fila §32 → G0), o mantenemos hard clearing estricto y aceptamos que Thm 4.1 se debilite a una versión con test acreditador?

---

## F. Preguntas para él

1. **Convención** (§E): ¿posterior-zero o hard clearing estricto? Toca §5.7, el acid test y la validez de Thm 4.1.
2. **Estatuto del companion:** ¿es insumo para nosotros, material del paper, o un documento paralelo? Concretamente: ¿cómo entra en A-M19 (outline) y A-M22 (documento formal laminar), que son los dos entregables que quedaron fechados el 18-ago?
3. **Eje del índice:** ¿lo decisivo es $\alpha$ o commit-vs-recompute? Si es lo segundo, el barrido de §23.1 necesita un flag más antes de congelar (§31).
4. **Prioridad en §20:** con Prop 8.5 cerrando la pregunta 1, ¿vamos por las relajaciones convexas (preguntas 2–3) o por policy improvement / regret acotado?
5. **Certificados:** §1 los tiene congelados hasta cerrar G4b. ¿Thm 10.2 y el portafolio adelantan esa línea ahora, o respetamos el orden de la cadena?
6. **Reemplaza a la pregunta (17)**, que su Thm 4.1 ya respondió: los ancestros son reducibles, no excluidos. ¿Lo confirma?

---

## G. Extras no pedidos (mantra §32 2026-08-20)

- **Verificación del testigo de Prop 8.5 bajo nuestra convención** (§B.1): sobrevive con $q<1/2$; tabla arriba. Enumerable exacto en $n{=}2$. Etiqueta: aritmética verificada, prueba pendiente de A.
- **Cotejo numérico de las dos anclas** (§E): confirmado que es la misma fórmula con desfase de un test, no dos resultados distintos.
- **Cotejo del PDF contra la extracción C-M1 del arXiv v4** (`2026-08-20-extraccion-arxiv-2206.10660.md`): su Thm 7.1 es la extensión dinámica de la Prop 1 estática; el punto 3 de esa nota (piso de dureza NP para $G\ge3$ en la rebanada estática del espacio laminar) sigue siendo insumo nuestro para la dicotomía algoritmo-vs-barrera de A-M22, y el PDF **no** lo cubre.
