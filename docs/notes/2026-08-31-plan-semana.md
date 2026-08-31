# Plan de la semana 31-ago → 4-sep — objetivo: super-paper (SODA)

> **Actualización 2026-08-31:** la sesión con Francisco quedó para el **jueves 3-sep, 11:00 CDMX** (no el martes). Corrimiento: el guion se congela el **miércoles 2-sep** (borrador ya renombrado a `2026-09-03-guion-sesion-BORRADOR.md`); mar–mié absorben la nota A-M24 y el arranque del solver; los rótulos de día de la tabla se leen con ese desfase y el despacho post-sesión (§34-bis) reescribe el resto.

**Regla de la semana:** seguimos la dirección de Francisco (checar el companion, implementar Bellman, escribir) **empujando a nivel de primer congreso**. Presupuesto realista: **3–4 h/día por persona**. Sesión con Francisco: **martes 1-sep**. Autocontenido: Héctor puede leer esto sin el chat.

**Lectura previa (15–20 min):** acta `docs/notes/2026-08-25-sesion-francisco.md` (D2, D3) · §5 y §16 nuevos del plan maestro · checklist `docs/notes/2026-08-30-inventario-companion.md`.

---

## Qué hace al paper de primer nivel (y qué de eso se juega esta semana)

1. **Cero hoyos en las pruebas** — SODA se cae por un gap, no por falta de ambición. → A-M23 adversarial (§8–§10 del companion), con lupa IA generando lista de gaps.
2. **El esqueleto que Francisco ya enunció** [49:26–50:32]: modelo práctico + algoritmo exacto + separación + tres garantías. → B-M17 (el algoritmo exacto corriendo) y A-M22/A-M19 (la escritura) son ese esqueleto.
3. **Al menos una pieza que suba el nivel**: cerrar o avanzar una abierta (10.6 razón exacta en $G,B$; 10.4; Conj. 10.7) o aportar el instrumento único del equipo (instancias separadoras mínimas, hallazgo $G\ge4$). → escalera de piezas de A (abajo).
4. **Restricción vigente:** el reparto con el paper de Nick está pendiente (fila §32) — las piezas nombrables de esta semana viven en *nuestro* modelo (brecha de convención, clase ex-ante/pathwise), inmunes a ese reparto.

**Escalera de piezas nombrables de A** (una por semana, se declara a Francisco): (1) esta semana: **Proposición de brecha de convención** (estricta vs posterior-zero) → (2) próxima: gap-y-reparación en Thm 9.3 (la prueba menos amarrada) → (3) después de A-M23: intento serio en 10.4 / Conj. 10.7 (la estrella). Advertencia §31: la dureza "fácil" vía $B{=}1$ es corolario de [1] — no se llama teorema; la versión valiosa es la no trivial (homogéneo / $G$ fijo, $B$ creciente).

---

## LUNES 31 — preparar la sesión y arrancar las dos cadenas

**Bloque conjunto A+B (0.5 h): ratificación de G0.**
1. A presenta la directriz [43:44–43:53] ("soft clearing es mejor") y el §5 nuevo (5 min).
2. Los tres números que cambian (verificados): ancla acid test $0.806u \to 1-0.95^{48}\approx 0.9147u$ ($k$: 2→3); reentrada B-M16 **0.5 → 1.0**; $S_1^{hard}{=}S_0$ queda solo en la variante estricta.
3. B ratifica u objeta (objeción → §32, R10; el día no se bloquea: B-M17 implementa la ec. 5.5 del companion, definida en posterior-zero de todos modos).
4. Confirmar estado heredado de B (B-M6 ext., barrido α, B-M16) — 10 min.

**A (~3 h más):**
- **A-M20 — guion del martes, congelado hoy (1 h; IA entrega borrador para editar):** paquete = despacho aplicado (acta + G0 ejecutado) + inventario/checklist del companion + números de brecha + diseño B-M17 (+ toy si corre). **Preguntas congeladas (§34):** (18) estatuto/fusión companion↔outline y calendario arXiv/SODA; (19) reparto con el paper de Nick; (20) ¿el eje del índice es $\alpha$ o committed-vs-receding?; (21) ¿el harness conserva la variante estricta como columna?; (22) ¿clase normativa ex ante o pathwise? **Y la declaración "yo tomo X":** A anuncia la Proposición de brecha de convención (en curso) y su interés de mediano plazo en 10.4/Conj. 10.7.
- **A-M24 núcleo (1.5 h):** validar §5 como propio — re-derivar a mano el ejemplo ABCD (test {A,B} conteo 1, luego {A}: ambas ramas pagan 1.0, nadie paga dos veces) + verificar ancla ($k{=}3$, $1-0.95^{48}$, baseline $0.35u$ sin cambio). Estos cálculos son la **semilla de la Proposición del jueves**. (El inventario cerrado completo: miércoles/jueves, tabla de 8 ítems en el plan maestro.)

**B (~3 h más):**
- **B-M17 nota de diseño (1.5 h):** esqueleto = Steps 1–6 de Prop 6.1 (codificar canónico → caso base → enumerar acciones en forma normal → tablas de coeficientes → recursión memoizada → guardar argmax para reconstruir política). Fijar: encoding canónico de $(U,\mathcal A,b)$; etapas (labeled $n\le6$ primero, tipos Prop 6.2 después); reuso de `laminar_tables.py` ($f_A$ = $Z(A,r)$) y `ExactPolicyEvaluator` (base de $J^\pi_b$); **flag de convención** `posterior_zero | strict` (barato hoy, responde (21) gratis).
- **Prototipo toy (1.5 h):** $n\le4$, $B\le2$, labeled, contra enumeración a mano de la instancia B-M16. Si corre → **demo en la sesión del martes** (extra del mantra); si no, la nota de diseño es la base y no pasa nada.

**IA (hoy mismo, a pedido):** borrador del guion A-M20 · enumerador pathwise de referencia (C-M3) para el cotejo del miércoles · arbitraje de lo que produzcan.

---

## MARTES 1-sep — SESIÓN (≈1 h) + media jornada de trabajo

- **Pre (0.5 h, juntos si se puede):** repasar guion; congelado desde el lunes — en sesión no se improvisan claims (§25).
- **Sesión:** presentar paquete; hacer las 5 preguntas; A declara "yo tomo X". Grabar/transcribir como siempre.
- **Post (0.5 h):** A me pasa el transcript → yo preparo acta + tabla de despacho para aprobación (protocolo §34-bis).
- **Trabajo restante:** A (1–1.5 h) arranca Thm 8.2 (la prueba del coupling, ≈2 pp. — el teorema de la salvaguarda). B (1.5 h) completa la recursión memoizada del solver.

---

## MIÉRCOLES–VIERNES (provisional: el despacho del martes la reescribe si Francisco redirige — §34-bis)

| Día | A (3–4 h) | B (3–4 h) | Soporte IA |
|---|---|---|---|
| Mié 2 | Aprobar acta+despacho (0.5) · **A-M22 bloque 1**: modelo posterior-zero + espacio laminar pathwise (3) | **B-M17 validación dos vías**: solver vs enumerador pathwise, $n\le5$, $B\le3$ — ojo: no comparar contra $V^{*,\mathcal L}$ ex ante sin etiquetar clase (§6.6/pregunta 22) | acta + despacho de la sesión · lupa adversarial §8: lista de gaps de Thm 8.2/9.3 para A |
| Jue 3 | **A-M22 bloque 2**: contraejemplo + algoritmo-vs-barrera (2) · **Pieza nombrable**: enunciar y probar la Proposición de brecha de convención, etiqueta §25 (1.5) | Ancla §16 reproducida por el solver + **flag de convención** → números B-M16 duales (B-M18 mínimo): la tabla estricta-vs-posterior-zero que alimenta la Proposición de A | C-M1: cotejo del teorema $p>1/2$ (levantar cuarentena C1 si da) |
| Vie 4 | **A-M23**: cerrar Thm 8.2 + Cor 8.3/8.4 con la lista de gaps (2) · **A-M19**: outline SODA v0 — con miras a draft arXiv-able temprano (1.5) | $J^\pi_b$ sobre el solver (evaluar $\pi_M$ al menos) · si la validación cerró: primeros mapas homogéneos chicos (extra, Phase 3) | arbitraje de la Proposición de A · paquete de cierre de semana |

**Sinergia clave jueves:** los números duales de B (solver con ambos flags) son la evidencia computacional de la Proposición de A — la pieza queda enunciada (A, a mano) **y** verificada (B, por enumeración) el mismo día. Así se etiqueta [DEMOSTRADO + VERIFICADO n≤5], no aspiracional.

---

## Cierre de semana — checklist

- [ ] G0 ratificado (o objeción en §32).
- [ ] Guion congelado lunes; sesión con 5 preguntas hechas y "yo tomo X" declarado.
- [ ] Acta + despacho de la sesión del 1-sep aplicados (miércoles).
- [ ] B-M17 validado contra enumerador pathwise ($n\le5$, $B\le3$) con flag de convención.
- [ ] A-M22 bloques 1–2 escritos bajo posterior-zero.
- [ ] Proposición de brecha de convención: enunciada, probada y verificada por enumeración.
- [ ] A-M23: Thm 8.2 + Cor 8.3/8.4 leídos con gaps anotados (o "sin gaps").
- [ ] A-M19 outline SODA v0.

**Qué NO hacer:** tocar scorers/candidatas (B-M9 espera G4a — **G4a pasa a la semana entrante por asignación de horas, ningún gate se reordena**) · citar el companion como validado · llamar teorema al corolario de dureza $B{=}1$ (§31) · mezclar números de convenciones sin etiqueta · comprometer resultados del companion como nuestros antes del reparto con Nick (pregunta 19).
