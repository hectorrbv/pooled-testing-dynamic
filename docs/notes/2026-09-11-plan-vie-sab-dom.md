# Plan vie 11 → dom 13-sep — A (Vladimir) y B (Héctor), manos cruzadas

**Presupuesto:** vie 4 h · sáb 3 h · dom 3 h por persona (10 h cada uno). **Regla de manos cruzadas:** toda entrega tiene **autor** y **verificador del otro lado**; el verificador la reproduce (corre el código o re-deriva la cuenta) y firma con cinco líneas en la nota del día (`docs/notes/2026-09-1X-sync.md`). **Sync diario, 20 min (chat o llamada):** cada uno explica de vuelta la entrega del otro ("teach-back"); si no la puede explicar, no está cerrada. Lunes 14 = congelar guion (ya en `docs/notes/2026-09-10-plan-semana.md`).

**Ajuste del 11-sep respecto al plan de semana:** la pieza de A es la **Proposición de brecha de convención en tres partes** (dominación · simulación con sobrecosto · agudeza), tal como quedó diseñada en la nota de A del 31-ago ("The convention gap"). El "dividendo de deducción" con forma cerrada en B=2 es solo el testigo de juego completo de la parte 3. Y el Mapa 3 de B cambia de pregunta: no "¿alcanza el solver el ancla?" sino **"¿el plan cover-then-bisect es el óptimo posterior-zero en el ancla?"**.

**Qué sale el domingo en la noche:**
- A: Proposición de brecha de convención (3 partes) en `.tex`, parte 2 probada, partes 1 y 3 probadas y verificadas · sección de modelo v0 (3 pp.) · acta del 1-sep aprobada · forma cerrada B=2 reproducida por su mano contra `bellman_tipos` · reconstrucción del argumento de Francisco (extra).
- B: higiene de claims (6 ítems, incluye la k del ancla) · `tests_bm17.py` en verde · Mapas 1–3 + nota (Mapa 3 = ¿es óptimo el plan del ancla?) · re-derivación a mano del testigo B=2 (como verificador) · lectura del Thm 8.2 explicada en sync · regla de λ (extra).
- IA (a pedido, mismo día): `augmented/pathwise_enum.py` para B · borrador de acta para A · lupa sobre la parte 2 (sábado) · lista de gaps del Thm 8.2 · esqueleto del guion.

---

## La Proposición de brecha de convención (lo que A escribe; etiquetas §25)

Convenciones: **estricta** = acreditado solo quien estuvo físicamente en un pool probado con conteo 0; **posterior-zero** = acreditado en cuanto la historia prueba que está sano (P(infectado | historia) = 0), incluyendo deducciones por conteo. Presupuesto duro en toda rama.

1. **Dominación.** Para toda historia h, C_strict(h) ⊆ C_pz(h); luego OPT_strict(B) ≤ OPT_pz(B) para todo B. *(casi inmediata; se prueba el viernes)*
2. **Simulación con sobrecosto.** Toda política posterior-zero se replica bajo estricta añadiendo, por cada conjunto D acreditado por deducción, ⌈|D|/G⌉ pruebas de cero garantizado; el sobrecosto es el **máximo sobre trayectorias** (el presupuesto es duro en cada rama, no en promedio). Luego OPT_pz(B) ≤ OPT_strict(B + sobrecosto_max). *(la prueba real; sábado; lupa de IA)*
3. **Agudeza.** El sobrecosto no es un artefacto. Tres testigos: (a) **estado**: par con conteo 1 y una prueba — valor 1 vs ½ (Ej. 1 de la nota de A); (b) **juego completo mínimo**: n≥3, G=2, B=2, u≡1: V_pz(par primero) − V_strict(par primero) = p·q·u, y V_pz(par) − 2q·u = q·(q²+p²)·u > 0 para todo q ∈ (0,1) [verificar: q=0.3 → 0.774 / 0.564 / 0.6]; (c) **escala**: en el ancla (q=0.05, G=16, B=7) la reserva estricta cuesta un pool entero: 1−0.95⁴⁸ = 0.9147 vs 1−0.95³² = 0.8063 como valores del plan cover-then-bisect. *(a y b: [DEMOSTRADO] + [VERIFICADO por enumeración]; c: cota realizable — sube a exacta si el Mapa 3 de B muestra que el plan es el óptimo pz)*

Corolario para el martes: la región q_sano ≤ ½ donde [Nick v4, Prop 1] hace óptimos a los singletons **no existe** bajo conteos aumentados con posterior-zero (testigo b, todo q).

---

## VIERNES 11 (4 h c/u) — cerrar la deuda y arrancar la pieza

| Bloque | A (Vladimir) | B (Héctor) |
|---|---|---|
| 1 (1.5 h) | **Partes 1 y 3 a mano.** Dominación en tres líneas. Agudeza (b): V(par primero) y V(dos singletons) **como función de q**, posterior-zero y estricta; la diferencia entre convenciones debe salir de una sola rama (R=1). Control (verificar, no copiar): q=0.3 → 0.774 / 0.564 / 0.6. | **Higiene 1–3.** `.gitignore` (`TRANSCRIPCION_*`, `pasted-text*`) y propuesta sobre `*.tex` (raíz :74). `augmented/tests_bm17.py` con el enumerador que IA entrega (`pathwise_enum.py`): n≤5, B≤3, ambas convenciones + 48 celdas contra `experiments_separacion_n10.laminar_value`. |
| 2 (1 h) | **Acta del 1-sep**: aprobar el borrador de IA; añadir lo que la IA no puede saber. Filas §32: encargo cumplido, reparto propuesto, resultado soñado como norte, W[1] citable, Francisco manda su prueba. | **Higiene 4 + 6.** `run_showcase` con tres cuotas (gana / empata / pierde, 1e-9) en `experiments_laminar_week.py:603-608`; regenerar `showcase_regions.csv`; reescribir C1 en el plan maestro §9 (global 10.9/56.1/33.0; alta 1.4/97.5/1.1). **Y la k del ancla:** `acid_test.py:69-71` sigue con el `−1` del test acreditador (k estricta); bajo G0 `k_from_budget(B, G) = B − ⌈log₂G⌉` con la variante estricta como parámetro. `anchor_instance()` pasa de k=2 (0.8063) a k=3 (0.9147). Revisar qué tests de `tests_acid_falsificador.py` asumen k=2. |
| 3 (1 h) | **Cruce → código.** Leer `docs/notes/2026-09-01-contraejemplo-universal-greedy.md` y `2026-09-02-constante-empirica-portafolio.md`. Escribir **5 preguntas** para B. Correr `python -m augmented.bellman_tipos` y anotar qué imprime y por qué n=120 == n=500. | **Cruce → teoría.** Leer transcript [41:07–47:15] y la nota de A "The convention gap" (Ej. 1 y 2). Re-derivar por tu cuenta V(par) en q=0.3 bajo posterior-zero (debe dar 387/500) **sin correr el solver**. Escribir **5 preguntas** para A sobre la parte 2 (¿qué pasa con el sobrecosto cuando D se acredita en varias ramas distintas?). |
| 4 (0.5 h) | Abrir `augmented/paper/proposicion_brecha_convencion.tex`: enunciado de las tres partes + testigos (sin prueba de la parte 2 aún). | **Higiene 5.** Adversaria con presupuesto real (5k evals/celda, reinicios, semilla en CSV) **o** borrar 0.9069 y dejar 0.928 + testigo 0.8906. Commit + push de todo lo del día. |
| **Sync (20 min)** | A explica a B el mecanismo del contraejemplo (por qué el trío {0,4,5} con z0≈0.001 es óptimo). | B explica a A la rama R=1 y por qué vale u con certeza bajo posterior-zero y u/2 bajo estricta. |

**Criterio de hecho (vie):** `pytest augmented/tests_bm17.py` verde y pusheado · `showcase_regions.csv` con tres columnas · `anchor_instance()` en k=3 con tests ajustados · acta en `docs/notes/2026-09-01-sesion-francisco.md` · cuaderno de A con partes 1 y 3(b) y las dos listas de 5 preguntas cruzadas en `docs/notes/2026-09-11-sync.md`.

---

## SÁBADO 12 (3 h c/u) — la parte difícil, probada y verificada el mismo día

| Bloque | A | B |
|---|---|---|
| 1 (1.5 h) | **Parte 2 (simulación con sobrecosto).** Definir D(h) = acreditados por deducción en la historia h; construir la política estricta que replica la pz y, al final de cada trayectoria, añade ⌈|D|/G⌉ pruebas de cero garantizado (todos sanos con certeza, conteo 0 seguro). Argumentar que el presupuesto duro exige el **máximo** de ⌈|D|/G⌉ sobre trayectorias, no la esperanza. Escribirlo en el `.tex`. Lupa de IA por la tarde (ataca: ¿pueden las pruebas de acreditación mezclar D de ramas distintas? ¿qué pasa si G < |D| y las pruebas extra cambian la información?). | **Mapa 1** con `bellman_tipos.py` (guardar argmax de la raíz: hoy solo devuelve valor): primera acción óptima y cociente óptimo/(B·q·u) sobre q ∈ {0.05…0.95}, G ∈ {2,3,4,5,8}, B ∈ {2…8}. `results/mapa_homogeneo_1.csv` + `.meta.json` limpio (commit antes de generar). |
| 2 (1 h) | **Cruce → código.** Con un script de 15 líneas (IA o B lo pasa): correr `bellman_tipos` en G=2, B=2, n=4 para q ∈ {0.1,…,0.9} y **comprobar la forma cerrada de 3(b) contra el solver** en 9 puntos. La tabla va en la nota del día: es tu verificación del Mapa 1 en la fila que te toca. | **Cruce → teoría (verificador de 3(b)).** Re-derivar la fórmula general V(par)(q) − 2q por tu cuenta y compararla con la de A; si coincide, firmar "verificado a mano por B" en el `.tex`; si no, la discrepancia es el tema del sync. |
| 3 (0.5 h) | Esqueleto de `augmented/paper/seccion_modelo.tex`: títulos de subsecciones y las definiciones que van en cada una (sin prosa aún). | **Mapa 3 — ¿es óptimo el plan del ancla?** `bellman_tipos` en q_sano=0.05, G=16, B=7 (posterior-zero): comparar V* con 0.9147 (valor del plan cover-then-bisect, k=3). Si coinciden, la parte 3(c) de A pasa de "cota realizable" a "exacta" y el ancla deja de ser analítica. Si el muro en B/G muerde, reportar la frontera (G=8, B=6: comparar con 1−0.95²⁴ = 0.7080). **Mapa 2** (piloto: q ∈ {0.90, 0.95}, G=5, B≤10, n=130) si queda tiempo. Convención `posterior_zero` en el `.meta.json`. |
| **Sync (20 min)** | A presenta la parte 2 en 5 min; B objeta como referee con sus 5 preguntas del viernes. | B presenta Mapa 3 en 5 min: ¿el plan es óptimo o el solver encontró algo mejor, y qué? A dice en una frase qué cambia en la parte 3(c). |

**Criterio de hecho (sáb):** `proposicion_brecha_convencion.tex` con las tres partes y firma de B en 3(b) · `mapa_homogeneo_1.csv` con verificación de A en 9 puntos anotada · Mapa 3 con veredicto (óptimo sí/no o frontera declarada).

---

## DOMINGO 13 (3 h c/u) — texto y lectura cruzada

| Bloque | A | B |
|---|---|---|
| 1 (2 h) | **Sección de modelo v0** (3 pp.): modelo (Z_i, R(T), historia, laminar pathwise, posterior-zero como regla de acreditación) · Lema 1 = Lema A(i)–(iii) (traer de `lemma_A_laminar_inference.tex`) · Teorema 1 = recursión 5.5 del companion (algoritmo exacto; citar companion hasta el reparto) · Prop 6.2 como escalabilidad (citar `bellman_tipos.py`: n=500 en segundos) · **Proposición de brecha de convención** como primer resultado propio · Figura 1 = Mapa 1. `git add -f` si `.tex` sigue ignorado. Al escribir Teorema 1 estás validando §3–§5 del PDF de Francisco: anotar en A-M23 las filas Thm 4.1 y Thm 5.1 como "leídas por A, implementadas por B, dos vías". | **Cruce → teoría.** Leer Thm 8.2 + Cor 8.3 del companion (la garantía 1/G del greedy inmediato) con la lista de gaps de IA. Escribir media página en `docs/notes/2026-09-13-thm82-lectura-B.md`: qué dice, qué supone (regla de paro), qué columna del harness lo mide. Es lo que explicas en el sync. |
| 2 (0.5 h) | **Cruce → código.** Leer `docs/notes/2026-09-11-mapa-homogeneo.md` de B y reescribir la frase de cada mapa con tus palabras debajo de la de B. Si no puedes, pregunta. | **Cruce → texto.** Compilar `seccion_modelo.tex` y leerla como lector externo: lista de **5 puntos que no se entienden** (contenido, no estilo). Va al sync. |
| 3 (0.5 h) | **Extra:** reconstrucción del argumento perfil/antiperfil de Francisco [05:05–39:13], una página: tres pasos + dónde crees que se rompe en augmented (pregunta P2). | **Extra:** regla de λ sin rejilla (λ = q̄·ū; λ = OPT_tipos/B; dual Thm 10.2) sobre las 10 instancias de `results/constante_scoreboard.tsv`. Una tabla. Luego regenerar CSVs con `.meta.json` limpio y push. |
| **Sync (30 min)** | A explica a B el Thm 8.2 de vuelta (lo que B leyó). | B explica a A la sección de modelo de vuelta (lo que A escribió). Ambos firman la lista del paquete del lunes. |

**Criterio de hecho (dom):** `seccion_modelo.tex` compila y B firmó sus 5 puntos · nota de mapas con las dos versiones de cada frase · `docs/notes/2026-09-13-thm82-lectura-B.md` · paquete del lunes listado y firmado por los dos.

---

## Reparto en horas (10 h c/u)

| | Teoría / texto | Código / datos | Cruce (leer, reproducir, verificar) | Acta / sync |
|---|---|---|---|---|
| A | 5.5 | 1.5 | 2 | 1 |
| B | 2 | 5 | 2 | 1 |

Cada uno pasa 2 h en el terreno del otro y 1 h explicando de vuelta. Al domingo los dos pueden explicar: el contraejemplo, la proposición (tres partes), los mapas, el Thm 8.2 y la sección de modelo.

## Qué NO se hace este fin de semana

- No diseñar el score nuevo (S3/G4a): el contraejemplo ya dice qué término le falta (crédito por deducción con abandono); se diseña **después** de la proposición, no antes.
- No abrir misiones nuevas de contraejemplos: la evidencia diagnóstica existe; el Mapa 3 es la única corrida nueva y responde una pregunta concreta.
- No citar el companion como nuestro ni como validado; no mezclar convenciones sin etiqueta.

## Válvula

Si A cae a 2 h/día: vie = partes 1 y 3 a mano + acta; sáb = parte 2 en `.tex`; dom = esqueleto de modelo (IA rellena prosa, A edita el lunes). Si B cae a 2 h/día: vie = higiene 1–3 + k del ancla; sáb = Mapa 3 + verificación de 3(b); dom = lectura Thm 8.2 + push. Los cruces no se cortan: son lo que hace que el martes los dos puedan responder cualquier pregunta de Francisco.
