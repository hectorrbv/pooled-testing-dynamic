# Extracción C-M1 — arXiv:2206.10660v4 "Welfare-Maximizing Pooled Testing" (2026-08-20)

**Identificación:** Finster, González Amador, Lock, **Marmolejo-Cossío**, Micha, Procaccia. arXiv:2206.10660**v4** [cs.GT], subido el **15 de agosto de 2026** — es "la versión más nueva" que Francisco anunció en la sesión del 18-ago. PDF descargado y leído (pp. 1–12); **A valida contra el PDF antes de que nada de esto toque C1 o el paper propio.**

**El modelo (Sección 2) es el ancestro estático exacto del nuestro:** q_i = P(sano) (misma convención), utilidad u_i por ser despejado, presupuesto B, tope de pool G, pruebas perfectas, prior producto; bienestar u(T) = Σ u_i P_i^T con despeje por pool negativo (= hard clearing); **sin seguimiento adaptativo** (nota al pie 2: no hay retesteo tras positivo). Nuestro proyecto es literalmente ese modelo + dinámico + canal de conteo (la espina, §0-bis a).

## Pregunta (15) de §34 — el teorema estático

> **Proposición 1** [prueba en Apéndice B.2]: *If every individual in the population satisfies q_i ≤ 1/2, then G(B,J) = 1.*

donde G(B,J) = OPT*(B,J)/OPT(B,J) es la razón entre el óptimo **con solapes** y el óptimo **sin solapes** (Definición 1, p. 7). Y dentro de la prueba, el enunciado que citábamos de oídas:

> "The strict case q_i < 1/2 is direct: we first show that **every optimal test must be a singleton**. All optimal allocations are therefore non-overlapping." El caso frontera q_i ≤ 1/2 sale por perturbación y continuidad de OPT*/OPT.

**Hipótesis exactas:** estático una etapa, canal binario (negativo despeja el pool), pruebas exactas, infecciones independientes, bienestar Σ u_i P_i^T. **Alcance:** con q_i < 1/2 estricto, el óptimo estático binario son singletons — el pooling binario no paga.

**Caveat de canal (importante para nosotros):** la Proposición 1 es del canal **binario**. Nuestra celda estático-conteos con acreditación por inferencia (notebook 25 §2: 0.800 vía pesado de monedas) **no está cubierta** por este teorema; bajo hard clearing estricto sí coincide (vuelve a 0.600 = singletons).

## Consecuencias para el plan

1. **Auditoría C1 (cuarentena, fila §32 2026-08-02):** el lado-teorema queda confirmado y citable: en prevalencia alta (q<1/2) el baseline estático binario correcto ES la prueba individual. Falta el lado-trazas (B: primera decisión en p>0.5, contabilidad de empates en `showcase_regions.csv`) para cerrar.
2. **Ancla del acid test (§16):** "baseline singleton S₀ coincide con el óptimo estático en esta ancla" (q=0.05) deja de ser observación propia y gana cita formal (Prop 1, caso estricto).
3. **Pregunta 6 de §20 / documento A-M22 — piso de dureza estático:** *Theorem 4*: computar la asignación óptima **sin solapes** es NP-completo para todo G ≥ 3 fijo (poli para G ≤ 2 vía matching); *Corollary 1*: sin FPTAS salvo P=NP. Las asignaciones disjuntas son el caso de un nivel de una biblioteca laminar ⟹ **la rebanada estática del espacio laminar ya es NP-dura con G ≥ 3**: piso heredado para la dicotomía algoritmo-vs-barrera del documento formal. [Etiqueta: implicación nuestra sobre teorema de ellos; formalizar el embedding antes de afirmarlo en el PDF.]
4. **Eco del cruce:** *Proposition 2*: evaluar el bienestar de asignaciones **con solape** es #P-duro para G ≥ 3 (poli para G ≤ 2). Analogía con nuestra motivación laminar (los cruces rompen la factorización); es canal binario estático — se cita como eco, no como equivalencia.
5. **Umbral G=3 en todo el paisaje** (evaluación, optimización, FPTAS): rima con nuestra frontera B=2/B=3 — ejes distintos (tamaño de pool vs. presupuesto); anotado como pregunta, no como conexión.
6. **Otros datos:** Theorem 1: no-solape cuesta ≤ factor 2 universal (ejemplo 19/16; gap abierto). Theorem 2: cotas afinadas en q alto. Greedy con garantía 1/(e+1); FPTAS para prueba única; MILP como benchmark; 99.37% del óptimo no-solapado en datos del piloto IPICYT.

## Acciones

- A valida esta extracción → se marca la pregunta (15) de §34 como respondida y se habilita la cita en la auditoría C1.
- B: lado-trazas de la auditoría C1 (pendiente, cadena B).
- El embedding "disjunto ⊂ laminar" del punto 3 se formaliza en A-M22 antes de usarse en el PDF de problema concreto.
