# Guion de sesión para A — 2026-09-29, 19:00 Praga (escrito 18:58, estatus honesto)

**Tesis (dilo primero):** "La separación ya es un teorema, no un ejemplo. El lado clásico es tu Teorema 3.3(i) en arXiv:2601.22419v2: con q ≤ ½ el óptimo dinámico son B pruebas individuales. Verificamos que el ancla cae dentro de su régimen (Λ = 1 en q = 0.05, así que el óptimo clásico es exactamente 0.35). El lado aumentado es nuestra construcción y el solver exacto de Héctor. La razón crece sin cota a lo largo de una familia."

**Números (verificados):**
| Celda | clásico exacto | laminar | factor |
| ancla (q=0.05, G=16, B=7), n=48 | 0.35 | ≥0.9147 CBS · 1.1233 exacto G≤8 | 2.61 · 3.21 |
| frontera (0.05, 8, 6), n=24 | 0.30 | 0.8654 exacto | 2.88 |
| familia j=8: G=256, B=9, q=1/256 | 0.035 | ≥0.633 | ≈18, crece como G/log G |

**Sobre su paper (v2, 25-sep):** el Lema 3.1 es el paso de Harris que íbamos a escribir; Λ es el parámetro que hoy reconstruí a mano; el Thm 3.3(i) es nuestro lado clásico, se cita; el Thm 5.4 (factor 2) sustituye nuestro remark condicional. Validamos §3 contra el PDF antes de citar.

**Mi estatus, honesto:** hoy trabajé a mano la tabla de rendimientos y Λ; la prueba escrita del caso homogéneo (paso de Harris) está en curso, como dominio propio y como remark; ya no bloquea el paper. El esqueleto main.tex existe con cada enunciado y su estatus.

**Preguntas:**
1. (28) Tu Prop 3.6 + Thm 5.4: en el modelo binario todo el valor dinámico viene de re-agrupar tras un positivo y vale a lo más 2. Con conteos el mismo movimiento, refinar un átomo, es no acotado. ¿Es esa la frase correcta para nuestra introducción? [preguntar, no afirmar]
2. (29) Reparto: resultados binarios = tu paper; laminar aumentado = nuestro; Thm 7.1 del companion = tu Thm 3.3(i). ¿De acuerdo?
3. (18) ¿Construimos el paper SODA sobre nuestro esqueleto de seis secciones, citando tu paper para el lado binario?
4. (23) ¿En qué paso se rompe el coupling del Thm 5.4 cuando la prueba devuelve un conteo?
5. (24) Para π_L, ¿qué λ?
6. ¿Recibiste el PDF de la brecha de convención? (enviar después si no)

**Respuestas listas:** "¿Esto es todo dentro de laminar?" → la separación no depende de laminar, es un fenómeno de conteos; laminar la hace computable. Umbral: en binario el par le gana al singleton sii 2q > 1; con conteos y soft clearing el par le gana a dos singletons para todo q (Cor. 1 del PDF de la brecha).

**No decir:** "probamos la cota clásica" (la citamos; la prueba propia está en curso). Ningún número fuera de la tabla. La frase "solo-cubrir = estático no traslapado" es una derivación por verificar.

**Mañana:** Partes 1–4 en el cuaderno, leer §3 del PDF, seccion_separacion.tex.
