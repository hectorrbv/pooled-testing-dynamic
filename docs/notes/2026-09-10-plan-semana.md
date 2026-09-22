# Plan jue 10-sep → lun 14-sep — objetivo: el salto al paper (sesión con Francisco: martes 15-sep)

> **Ajuste 2026-09-11:** el detalle operativo de vie–dom con manos cruzadas está en `docs/notes/2026-09-11-plan-vie-sab-dom.md`. Cambian dos cosas respecto a este plan: la pieza de A es la **Proposición de brecha de convención en tres partes** (dominación · simulación con sobrecosto · agudeza; el "dividendo de deducción" B=2 es solo el testigo de la parte 3), y el Mapa 3 de B pregunta si el plan cover-then-bisect **es el óptimo posterior-zero en el ancla**. Se añade una corrección de código: `acid_test.py:69-71` sigue con la k estricta (`−1`); bajo G0 el ancla es k=3 (0.9147), no k=2 (0.8063).

**Autocontenido:** Héctor puede leerlo sin el chat. **Presupuesto:** 3–4 h/día por persona (válvula al final si A está en tránsito a Praga). **Fuentes de este plan:** revisión externa del 5-sep (`REVIEW-2026-09-05.md`, local de A; se sube a pedido); transcript de la sesión del 1-sep (`TRANSCRIPCION_FRANCISCO_1_SEPTEMBER.md`, 50:43, no versionado); los 15 commits de B del 1 al 9-sep (`69ad0a3..14770c3`, ya en local); `docs/notes/2026-08-31-plan-semana.md`.

---

## 0. Dónde estamos hoy (hechos, no opiniones)

**Lo que Francisco pidió el 1-sep** [48:10–49:53]: *más análisis empírico de varios greedy, en el régimen donde ya hay separación; buscar una población donde a todos les vaya mal; si no aparece, "vamos bien"; y "corre tres greedy y toma el mejor" como candidato a algoritmo eficiente con factor.* Y el resultado soñado [46:24–47:15]: *laminar dentro de un factor constante del óptimo dinámico, y dentro de laminar un algoritmo eficiente aproximadamente óptimo.*

**Lo que B entregó (1–9 sep, ya en GitHub):**

| Commit | Qué | Estado |
|---|---|---|
| `50eb431` | π_M, π_C, π_R exactas sobre el solver (`augmented/densidad_companion.py`) | instrumento |
| `3bb9415` | **Contraejemplo universal**: n=6, B=3, G=4 heterogénea, las cuatro (π_M, π_C, π_R, C3) caen a **0.6576** del óptimo | hallazgo diagnóstico §25 |
| `c3070ee` + `5889672` | **π_L**: índice Lagrangiano I_λ (ec. 8.13) con abandono gratis y no-parálisis (`augmented/indice_lagrangiano.py`); rescata el contraejemplo a **0.9641** | instrumento + hallazgo |
| `db9712c` | **Constante empírica del portafolio best-of-5: α ≈ 0.9307**, ~7,000 instancias, n≤8, G≤4, B≤5, sin contraejemplo; la política ganadora cambia por instancia; **λ es parte de la especificación** (rejilla gruesa → 0.794) | hallazgo diagnóstico §25 |
| `20f498f` | **Bellman por tipos (Prop 6.2), homogéneo**: óptimo exacto a n=120 y n=500 en segundos; el muro es exponencial en B (B=10 ≈ 33 s; B=12 o G=10 > 120 s) (`augmented/bellman_tipos.py`) | instrumento |
| `d9c3c87`/`b0db25c` | C-M1: diff v3→v4 de arXiv:2206.10660 + tres preguntas para Francisco (P1 dirección, P2 barrera, P3 oráculo/benchmark) + perfiles reales del piloto (n=130, 6 grupos, G=5, q_sano ≈ 0.9+) | nota |
| `e9f1769`/`f0f9e93`/`dca44e6` | notebook 24 (existe ya), 25-II, 26_esencial; revisión QGT versionada; material de estudio | docs |

Traducción: **el encargo empírico de Francisco está cumplido y con sobre-entrega** (contraejemplo → ingrediente faltante → política que lo repara → constante empírica con su cautela metodológica). La cadena B va por delante de la cadena A.

**Lo que A entregó (1–9 sep):** nada en el repo. El transcript del 1-sep existe (3-sep) pero no hay acta ni despacho (§34-bis). Lema A: el `.tex` de julio ya contiene la factorización (línea 120) — falta cerrarlo, no empezarlo.

**Lo que la revisión externa exige antes de citar nada** (REVIEW-2026-09-05): C1 cuenta empates como victorias (96.0% en prevalencia alta → 1.4% de victorias estrictas, 97.5% empates; global 67.0% → 10.9% / 56.1% / 33.0% pierde); `seccion_anatomia_del_gap.tex:170` declara normativa la convención estricta (G0 dice posterior-zero); 0.9069 no es mínimo (0.8906 en 32 s con 400 propuestas más); C3 "sin sobreajuste" con 4 held-out dentro del régimen de entrenamiento (en G=4: media 0.872, peor 0.633); `bm17_toy_solver` sin pytest (un enumerador pathwise independiente ya corrió y **coincide en todo**: 3/10, 3/5, 1011/1000 estricta; 3/10, 387/500, 537/500 posterior-zero; 240 estados heterogéneos, 0 discrepancias); 23 transcripciones privadas sin `.gitignore`.

**Lo que cambia con el paper de Nick v4 + lo que Francisco dijo el 1-sep:** el no-augmented dinámico es territorio de Francisco/Edwin (factor 2 de adaptabilidad, greedy 1/(e+1), Prop 1: singletons óptimos si q_sano ≤ 1/2, W[1] en G — Thm 3 de v4, ahora citable). Francisco lo dijo literalmente: ese argumento **"no va a funcionar en augmented"** [37:04–38:15]. **Nuestro territorio es augmented + laminar + posterior-zero**, y ahí tenemos ya el hecho que a Francisco le pareció "qué interesante" [41:07–45:19]: bajo conteos aumentados y posterior-zero, **agrupar gana a singletons incluso donde su Prop 1 dice que no** (n=4, q_sano=0.3: 0.774 vs 0.6, verificado por solver y por enumerador independiente).

---

## 1. Regla de la semana

**Convertir evidencia en enunciados.** Ningún instrumento nuevo salvo el que un enunciado necesite para etiquetarse [DEMOSTRADO] o [VERIFICADO n≤…]. La semana termina con **texto de paper bajo el modelo vigente** — lo único que hoy no existe.

**Escalera de piezas nombrables de A (esta semana):**
1. **Base (jue–vie): Proposición del dividendo de deducción** (= la "brecha de convención", ya declarada a Francisco). Enunciado y prueba a mano; verificación por enumeración (IA ya la tiene); corolario explícito: *la Prop 1 de [Nick v4] / Thm 7.1 del companion no sobrevive a conteos aumentados con posterior-zero.*
2. **Base (vie–sáb): A-M22 bloque 1** — la sección de modelo en `.tex` (posterior-zero, espacio laminar pathwise, átomos, Lema A como Lema 1, recursión 5.5 como algoritmo exacto, Prop 6.2 como escalabilidad). Tres páginas arXiv-ables para que Francisco reaccione el martes.
3. **Extra (dom–lun): reparación del Thm 9.3** del companion (la expansión (9.6)/(9.9) omite el presupuesto sobrante B − m(ℓ+1); en G=4, B=4 el cociente es 5/8, no 1/2; el límite 1/(1+log₂G) sobrevive). Pieza chica, atribuible, y es la (2) de la escalera que ya anunciamos.
4. **Extra de sobre-entrega (1 h, cualquier día): reconstrucción escrita del argumento perfil/antiperfil de Francisco** [05:05–39:13], una página: "esto entendí". Él pidió explícitamente que revisemos su prueba cuando la mande [49:58]; llegar con la reconstrucción hecha es la manera de que la revisión sea nuestra. Y es la pregunta P2 (la barrera): por qué el coupling no pasa a augmented.

---

## 2. Calendario

### JUEVES 10 — cerrar lo abierto y arrancar la pieza

**Bloque conjunto A+B (0.5 h, en chat si no coinciden):**
1. B presenta en cinco líneas lo del 1–9 sep (tabla §0). A confirma que lo leyó (los tres `docs/notes/2026-09-0*.md`).
2. Acordar las cinco correcciones de la revisión que B ejecuta hoy (abajo) — ninguna cambia un número, todas cambian una frase.
3. Fijar la sesión del martes: hora (¿11:00 CDMX sigue?; Francisco está entregando el paper de Nick el 14), quién presenta qué (A: pieza 1 + tex; B: contraejemplo → π_L → constante).

**A (3 h):**
- **Acta + despacho del 1-sep** (§34-bis; IA entrega borrador; A aprueba en 30 min). Filas §32 que salen del transcript: (i) encargo empírico de greedies — cumplido por B; (ii) reparto con el paper de Nick — *propuesta*: no-augmented dinámico = ellos; augmented laminar = nosotros; se pregunta el martes (pregunta 19); (iii) el resultado soñado como norte de A-M22/A-M23; (iv) el W[1] del companion ahora es Thm 3 de v4 — citable; (v) Francisco compartirá su prueba escrita para revisión.
- **Pieza 1, ejercicio numérico (1.5 h, a mano, sin mirar al solver):** instancia B-M16 (n=4, G=2, B=2, q_sano=q, u=1) bajo posterior-zero. Calcular V(par primero) y V(dos singletons) como función de q — no solo en q=0.3. Después, bajo estricta. La diferencia entre convenciones tiene que salir como una sola rama (R=1). **Valores de control** (IA; verificar, no copiar): en q=0.3: pz par = 387/500 = 0.774; estricta par = 0.564; singletons = 0.6. Forma cerrada esperada para B=2, G=2, n≥3: pz par − singletons = q·(q² + p²)·u > 0 para todo q ∈ (0,1); pz − estricta = p·q·u. En n=2 hay empate exacto (no queda virgen para el segundo test): el enunciado necesita n ≥ 3.
- **Pieza 4 si sobra energía (0.5 h):** releer [05:05–39:13] y anotar los tres pasos del argumento de Francisco (muestra → rama del árbol = asignación estática; thinning 50%; perfil/antiperfil condicionado en desacuerdos ⇒ uniforme).

**B (3 h) — correcciones de la revisión, todas de horas:**
- `.gitignore`: `TRANSCRIPCION_*`, `pasted-text*`. **Primero, 5 minutos.** Y decidir con A si `augmented/paper/*.tex` deja de ignorarse (hoy `*.tex` está ignorado globalmente en la raíz, línea 74; el paper no puede vivir fuera del repo).
- `augmented/tests_bm17.py`: (i) enumerador pathwise independiente — sin importar el solver, perfiles explícitos 2^n con `Fraction`, historia = tupla de (pool, conteo), legalidad sobre la historia, crédito terminal bajo ambas convenciones — contra `SolverLaminar.V` en n≤5, B≤3, ambas convenciones (IA lo re-entrega en el repo como `augmented/pathwise_enum.py`; la versión de la revisión corrió en 3.3 s y coincidió en todo); (ii) estados heterogéneos aleatorios (p∈{.1..0.9}, u∈{1..5}, raíz y átomo); (iii) `SolverLaminar(G=n, posterior_zero)` contra `experiments_separacion_n10.laminar_value` en n∈{3..6}, B∈{1,2,3}, q∈{.2,.3,.5,.7} (48 celdas; la revisión midió diff máx 4.4e-16). Con esto el "dos vías" de la adenda del 1-sep queda **verdadero y en CI**.
- `augmented/experiments_laminar_week.py:603-608`: `run_showcase` emite tres cuotas (gana / empata / pierde, tolerancia 1e-9) en vez de `>=`. Regenerar `showcase_regions.csv`. C1 se reescribe en el plan maestro §9: *global 10.9 / 56.1 / 33.0; prevalencia alta 1.4 / 97.5 / 1.1* — el resultado real es "greedy empata al estático en prevalencia alta", que es otra frase y sigue siendo interesante.
- Columna `convencion` (`strict|posterior_zero`) y `clase` (`ex_ante|pathwise`) en cada CSV citable (atlas, separación n=10, scoreboards). Un banner de una línea en `augmented/paper/seccion_anatomia_del_gap.tex:170` ("esta sección está en la convención estricta; la normativa del paper es posterior-zero desde G0").
- Búsqueda adversaria con presupuesto real (`experiments_laminar_week.py:301-318` hoy son 24 propuestas por semilla; subir a 5k evaluaciones por celda, reinicios, semilla y presupuesto en el CSV) o borrar el 0.9069 y dejar solo el mínimo de malla 0.928 con el testigo 0.8906 (p=[0.7467, 0.01, 0.8156, 0.7849, 0.7805, 0.6722], u=[1.0944, 0.7855, 0.9246, 1.4188, 1.2068, 0.57]).

**IA:** borrador de acta + despacho del 1-sep · `pathwise_enum.py` + cotejo heterogéneo para `tests_bm17.py` (commit aparte) · parche de tres cuotas para `showcase` · esqueleto del guion del martes.

---

### VIERNES 11 — la pieza queda enunciada y verificada el mismo día

**A (3–4 h):**
- **Pieza 1, borrador propio (2 h):** enunciado + prueba para B=2, G=2, n≥3 (dos ramas: R∈{0,2} coinciden en ambas convenciones; R=1 es el dividendo: bajo posterior-zero el refinamiento paga u con certeza, bajo estricta con probabilidad 1/2). Corolario: *"Bajo conteos aumentados con posterior-zero, para todo q ∈ (0,1) el par domina estrictamente a los singletons; en particular la región q_sano ≤ 1/2 donde [Nick v4, Prop 1] hace óptimos los singletons no existe en el modelo aumentado."* Etiqueta: **[DEMOSTRADO B=2] + [VERIFICADO n≤6, B≤3 por enumeración]**.
- **Lupa de revisor (0.5 h):** IA ataca el enunciado (¿u heterogénea? ¿G>2? ¿qué pasa en B=3 donde par y singleton empatan exacto — 537/500 — y la ventaja no es monótona en B?). A decide qué entra al enunciado y qué queda como observación.
- **A-M22 bloque 1, arranque (1 h):** esqueleto de la sección de modelo: definiciones (Z_i, R(T), historia, clase pathwise laminar, posterior-zero como regla de acreditación), Lema 1 (= Lema A(i)–(iii), ya en `augmented/paper/lemma_A_laminar_inference.tex`), y el enunciado de la recursión 5.5 como Teorema 1 (algoritmo exacto) citando el companion como fuente hasta el reparto.

**B (3–4 h) — el mapa homogéneo que hace visible el mecanismo (todo con `augmented/bellman_tipos.py`, segundos por celda):**
- **Mapa 1 — "primera acción óptima":** para q_sano ∈ {0.05, 0.1, …, 0.95}, G ∈ {2, 3, 4, 5, 8}, B ∈ {2, …, 8}: ¿el óptimo abre pool o singleton? y el cociente óptimo-laminar / (B·q·u). Es la **evidencia computacional de la Pieza 1 a escala** (y de que la región de la Prop 1 de Nick no existe aquí). CSV con `.meta.json`, commit limpio antes de generar. (Requiere guardar el argmax de la raíz en `bellman_tipos.py`: hoy solo devuelve el valor.)
- **Mapa 2 — régimen del piloto:** q_sano ∈ {0.90, 0.95}, G = 5, B ≤ 10, n = 130 (gratis por tipos). Es la respuesta a P3 sin esperar la etapa de M tipos: "así se comporta el óptimo exacto en la población de su piloto, con una sola q".
- **Mapa 3 — el ancla del acid test:** familia §16 (q_sano = 0.05, G = 16, B = 7, k = 2). Probar si el solver por tipos la alcanza; si el muro en B/G muerde, reportar la frontera (G = 8, B = 6 …). Es la primera vez que el ancla dejaría de ser analítica.
- Guardar los tres como `results/mapa_homogeneo_{1,2,3}.csv` + nota corta `docs/notes/2026-09-11-mapa-homogeneo.md` con **una** frase por mapa y sin adjetivos.

**IA:** lupa sobre la Pieza 1 · lista de gaps del Thm 8.2 (el 1/G) para A-M23 · verificación cruzada de los mapas contra el solver etiquetado en n ≤ 8 (medido: n=8, B=4, G=4 tarda 28 s; n=9 ya no entra en 100 s).

---

### SÁBADO 12 (medio día) — texto

**A (2–3 h):** A-M22 bloque 1 completo: la sección de modelo en `.tex` (3 pp.). Incluir la Proposición 1 como primer resultado propio y el mapa 1 de B como Figura 1 ("la primera acción óptima es un pool en toda la región homogénea explorada"). Es el primer texto de paper bajo el modelo vigente. `git add -f` si el `.gitignore` sigue ignorando `.tex`.

**B (2 h, opcional):** **regla de λ** para π_L (la grieta que la constante deja abierta, pregunta 10.5): probar tres reglas sin rejilla — λ = q̄·ū (baseline singleton por prueba), λ = OPT_tipos/B, λ = dual del LP de presupuesto esperado (Thm 10.2) — sobre las 10 instancias registradas en `results/constante_scoreboard.tsv`. Reportar si alguna mantiene el portafolio ≥ 0.93 sin rejilla. Un día, un número, una tabla.

---

### DOMINGO 13 (opcional) — extras de la escalera

**A (2 h):** **Pieza 3 — reparación del Thm 9.3**: re-derivar (9.6)/(9.9) con el término sobrante B − m(ℓ+1) ∈ {0,…,ℓ}; comprobar el caso G=4, B=4 (π_C = 5q + O(q²) ⇒ cociente 5/8); enunciado corregido con "(ℓ+1) | B" como hipótesis de la versión exacta. Media página; se la mandamos a Francisco el martes como hallazgo, no como crítica.

**B:** descanso o CI mínima (GitHub Actions: suite base ~80 s, 219 passed / 16 skipped hoy, + regeneración del atlas).

---

### LUNES 14 — congelar el guion (Francisco entrega su paper ese día; el martes llega con la prueba escrita)

**A + IA (2 h):** guion congelado (§25: en sesión no se improvisan claims). **Paquete:**
1. Acta + despacho del 1-sep aplicados (filas §32).
2. **Proposición 1** enunciada, probada y verificada (A presenta, 5 min, con la fórmula).
3. Sección de modelo v0 en `.tex` (3 pp.) — se comparte antes de la sesión.
4. B: contraejemplo → π_L → constante 0.93 → mapas homogéneos (10 min, tres figuras).
5. Higiene de claims hecha: C1 reescrita como gana/empata/pierde; convención etiquetada en todo.

**Extras declarados** (mantra): Thm 9.3 reparado · regla de λ · `tests_bm17` en verde con dos vías · reconstrucción del argumento perfil/antiperfil.

**Preguntas congeladas (5):**
- (19) **Reparto**: proponemos no-augmented dinámico = paper de Nick; augmented laminar (modelo, algoritmo exacto, dividendo de deducción, separación, portafolio) = nuestro. ¿De acuerdo?
- (18) **Estatuto del companion** tras v4: ¿se funde en nuestro paper con Francisco como coautor, o queda como nota técnica citada?
- **P2 (barrera)**: ¿en qué paso concreto se rompe el coupling perfil/antiperfil cuando la prueba devuelve un conteo y el crédito es por deducción? (A llega con su reconstrucción.)
- **10.5 / λ**: ¿λ = OPT/B estimado, o el dual del Thm 10.2? ¿Vale como especificación de política para un enunciado de factor?
- **Thm 9.3**: el término sobrante — ¿lo corrige él o lo escribimos nosotros?

**B (2 h):** regenerar todo lo citado con `.meta.json` limpio (commit antes de generar); `git push`.

---

## 3. Qué NO hacer esta semana

- Ninguna misión nueva de búsqueda de scores ni de contraejemplos: la evidencia diagnóstica ya existe; falta el enunciado.
- No citar el companion como validado ni como nuestro antes de la respuesta a (18)/(19).
- No citar Nick v4 sin que A coteje el PDF (la nota C-M1 lo exige).
- No mezclar convenciones sin etiqueta; no reutilizar "0.928" sin decir cuál de los dos objetos es (mínimo de malla estricto ex-ante 0.928022 en n=6,B=3,G=2 vs valor laminar posterior-zero pathwise 0.9282432 en n=10,q=0.2,B=3).
- No llamar teorema al corolario de dureza B=1 (§31).
- No dejar el `.tex` fuera del repo.

## 4. Válvula

Si A tiene < 2 h/día por el viaje: la base se reduce a **acta + Proposición 1** (jueves–viernes); la sección de modelo pasa a la semana del 15 con el esqueleto hecho por IA para que A lo edite. Los mapas de B son independientes y se presentan igual. La sesión no se mueve.

## 5. Checklist de cierre (lunes 14, noche)

- [ ] `.gitignore` cubre transcripciones; `.tex` del paper versionado.
- [ ] Acta + despacho del 1-sep en `docs/notes/2026-09-01-sesion-francisco.md` + filas §32.
- [ ] `augmented/tests_bm17.py` en verde (dos vías reales).
- [ ] C1 reescrita (gana/empata/pierde) y `showcase_regions.csv` regenerada.
- [ ] Columna `convencion`/`clase` en CSVs citables; banner en `anatomia_del_gap.tex`.
- [ ] **Proposición 1**: `.tex` con enunciado, prueba y etiqueta §25.
- [ ] **Sección de modelo v0** (3 pp.) en `.tex`.
- [ ] `results/mapa_homogeneo_{1,2,3}.csv` + nota.
- [ ] Guion del martes congelado con 5 preguntas y extras declarados.
- [ ] (extra) Thm 9.3 corregido · regla de λ · reconstrucción perfil/antiperfil.

## 6. Nota honesta sobre el ritmo

B cerró en nueve días lo que Francisco pidió, con sobre-entrega. A lleva 40 días sin artefacto en el repo mientras la escalera de piezas se anuncia y se reprograma. Esta semana la asimetría se corrige con una sola cosa: **la Proposición 1 escrita el viernes**. Es corta, es nuestra, tiene la evidencia lista y Francisco ya dijo "qué interesante". Si el viernes existe, el paper existe.
