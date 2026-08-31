# Plan operativo — lunes 2026-08-31 (día 1 de la semana §1)

**Contexto en una línea:** el despacho de la sesión del 25-ago quedó aplicado el 30-ago (commit en esta rama); hoy se cierra G0 (ratificación de Héctor), A arranca la migración de convención (A-M24) y B arranca el diseño del solver de Bellman (B-M17). Autocontenido: Héctor puede leer esto sin el chat.

**Lectura previa mínima (15–20 min cada quien):**
- Acta: `docs/notes/2026-08-25-sesion-francisco.md` (sobre todo D2, D3).
- Plan maestro: §5 nuevo (posterior-zero), §16 re-derivado, filas §32 del 2026-08-25.
- Checklist del companion: `docs/notes/2026-08-30-inventario-companion.md`.

---

## Bloque conjunto A+B (primera hora, ~30–45 min): ratificación de G0

Agenda:

1. **La directriz** (5 min, A presenta): Francisco, ante la pregunta explícita del equipo sobre hard clearing — "¿tenemos que probarlo?" — respondió "**No, no, digamos soft clearing es mejor**" [43:44–43:53]. Es la decisión que el guion §E pedía. El §5 del plan ya está reescrito: posterior-zero normativo, strict como variante nombrada de comparación.
2. **Los tres números que cambian** (verificados, 10 min):
   - Acid test: el test acreditador desaparece → $k = \max\{0, B-\lceil\log_2 G\rceil\}$; ancla $(0.05,16,B{=}7)$: $k$ pasa de 2 a 3, cota de $0.806u$ a $1-0.95^{48}\approx 0.9147u$.
   - Contraejemplo B-M16 ($q_{\text{sano}}{=}0.3$, $u{\equiv}1$, par AB con conteo 1): la reentrada (testear A) pasa de **0.5 a 1.0** — ambas ramas acreditan (cero observado o deducción del complemento).
   - §14.3: el colapso $S_1^{hard}=S_0$ es de la variante estricta; bajo posterior-zero el greedy inmediato es $M_h$ del companion (ecs. 8.1–8.3, con complemento libre).
3. **Decisión de B**: ratifica u objeta.
   - Ratifica → actualizar la fila §32 del 2026-08-25 (quitar "ratificación de B pendiente").
   - Objeta → la objeción se registra en §32 y se procesa como R10 (la directriz no se descarta sin decisión registrada). **El trabajo del día no se bloquea**: B-M17 implementa la ec. 5.5 del companion, que está definida en posterior-zero de todos modos.
4. **Estado heredado de B** (10 min): ¿qué quedó de la semana 20–24 — B-M6 ext. (re-medición $C(T)$ local), barrido $\alpha$, B-M16? Lo que falte reordena los días 4–5 de la tabla §1.
5. **Logística**: si llega el mensaje de Francisco con la nueva hora de sesión (da clases mar/jue), responder considerando eso.

---

## Cadena A (Vladimir) — A-M24: migración de convención

**Entregable del día:** `docs/notes/2026-08-31-AM24-migracion-posterior-zero.md` con dos partes.

**Parte 1 — Validar el §5 nuevo como propio** (modo propedéutico: re-derivar, no solo leer):
- Re-derivar a mano la recompensa incremental en el ejemplo ABCD del acta [42:04–42:50]: test $\{A,B\}$ con conteo 1, luego test $\{A\}$ → verificar que ambas ramas pagan exactamente 1 (cero observado en una, deducción del complemento en la otra) y que nadie paga dos veces.
- Verificar el ancla nueva: $k=3$, $1-0.95^{48}$, y que el baseline singleton $0.35u$ no cambia.

**Parte 2 — Inventario cerrado de impacto** (lista exhaustiva; cada ítem con *qué cambia / quién re-deriva / etiqueta*). Mínimo a cubrir:

| # | Ítem | Qué cambia bajo posterior-zero | Re-deriva |
|---|---|---|---|
| 1 | §14.2 $S_0$ | pasa a $M_h$: término de complemento libre en átomos (ec. 8.3) | A (mano) |
| 2 | §14.3 | colapso $S_1=S_0$ → solo variante estricta | A (nota) |
| 3 | §14.6 caso $c=0$ | trivial (átomo conteo-0 ya acreditó); κ queda solo en estricta | A (nota) |
| 4 | §14.9 caso $c=\|D\|-1$ | el test acreditador extra desaparece; costo esperado/peor caso se re-derivan | A (mano) |
| 5 | §16 ancla y checks | hecho el 30-ago — verificar; "baseline = óptimo estático" pendiente (cadenas estáticas anidadas deducen) | A verifica; B-M18 computa |
| 6 | Spec B-M16 (`2026-08-20-spec-BM16-contraejemplo.md`) | números objetivo en ambas convenciones: reentrada 1.0; totales de política V̂ a re-derivar por enumeración | B-M18 (enumeración), A coteja |
| 7 | C5 / ancla $0.35u$ vs $0.806u$ | el modus tollens usa números estrictos → re-derivar bajo posterior-zero o etiquetar como estricta | A |
| 8 | Assertions/notebooks | listar (p. ej. `build_resultados_y_peticiones_notebook.py:608`) — **solo inventariar**, arreglar es B-M18 | B-M18 |

Criterio de cierre: la lista es **cerrada** (sin "etc."), y el soporte IA la arbitra (C-M2) cuando esté.

**Si queda tiempo (prep del martes):** decidir las secciones del esqueleto A-M22 — modelo posterior-zero · espacio laminar pathwise · contraejemplo · pregunta algoritmo-vs-barrera alineada con las abiertas 10.4–10.6 y la Conjetura 10.7 del companion.

---

## Cadena B (Héctor) — B-M17: diseño del solver de Bellman (diseño, no código de producción)

**Entregable del día:** nota de diseño (media página + firmas de funciones). El esqueleto de implementación es el del companion §6 (Steps 1–6): codificar canónico → caso base → enumerar acciones → tablas de coeficientes → recursión memoizada → guardar acción maximizante para reconstruir la política.

A fijar en la nota:

1. **Estado canónico** $(U, \mathcal A, b)$: encoding de $U$ y del multiconjunto de pares (átomo, conteo interior) — canonicalización (orden por tamaño/contenido) para la memoización; conteos extremos se normalizan con ν (se cosechan/descartan antes de guardar el estado).
2. **Etapas**: primero labeled con $n\le 6$ (valida contra enumerador); compresión por tipos (Prop 6.2) como segunda etapa — no antes.
3. **Comparador de validación** (para el miércoles): enumerador pathwise (ex post) de fuerza bruta, $n\le5$, $B\le3$ — recursión sobre historias laminares con acciones en forma normal (pool virgen $\le G$ o subconjunto propio de un átomo), bienestar posterior-zero. **Ojo:** no comparar contra el $V^{*,\mathcal L}$ ex ante del atlas sin etiquetar la clase (§6.6, pregunta 22).
4. **Reuso**: `laminar_tables.py` — $f_A$ por convolución **es** $Z(A,r)$ del companion — para las transiciones; `ExactPolicyEvaluator` como base del evaluador forward $J^\pi_b$ (ec. 10.4), que después evalúa $\pi_M/\pi_C/\pi_R$.
5. **Casos ancla del cotejo**: (i) instancia B-M16 ($n{=}4$, $G{=}2$, $B{=}2$, $q_{\text{sano}}{=}0.3$): bajo estricta el óptimo registrado es 0.6 (singletons, acta 18-ago); bajo posterior-zero el valor se fija **por enumeración**, no a mano; (ii) $B{=}1$ → debe coincidir con el mejor test único; (iii) el ejemplo motivador chico.
6. **Ambas convenciones como flag** del solver (`clearing="posterior_zero" | "strict"`): barato ahora, responde la pregunta (21) después y da la columna de comparación.

---

## Cierre del día — checklist

- [ ] G0: ratificado por B (o objeción registrada en §32 → R10).
- [ ] Fila §32 del 2026-08-25 actualizada si hubo ratificación.
- [ ] A-M24: nota con §5 validado + inventario cerrado.
- [ ] B-M17: nota de diseño (estado canónico, etapas, comparador, casos ancla, flag de convención).
- [ ] Estado heredado de B confirmado (B-M6 ext. / barrido α / B-M16) y días 4–5 reordenados si hace falta.
- [ ] Si escribió Francisco: hora de sesión acordada (evitar mar/jue en horario de su clase).

**Qué NO hacer mañana:** tocar scorers/candidatas (B-M9 espera G4a) · citar resultados del companion como validados (todo sigue [SIN VALIDAR → A-M23]) · re-derivar todos los números del inventario (mañana se inventaría y reparte; la re-derivación es el resto de la semana) · mezclar números de convenciones sin etiqueta.

**Soporte IA disponible hoy mismo a pedido:** arbitraje del inventario A-M24 (C-M2) · nota Harris 1960 para §21 · enumerador pathwise chico de referencia para el cotejo de B-M17 (C-M3).
