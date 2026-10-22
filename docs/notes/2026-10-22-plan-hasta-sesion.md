# Plan hasta la sesión — jue 22-oct → mar 27-oct, 19:00 Praga (por confirmar con Francisco)

**Autocontenido para Héctor.** Supersede a `2026-10-14-plan-semana.md`. Fuentes: acta del 20-oct (`2026-10-20-sesion-francisco.md`), plan maestro §1 (semana del 22-oct) y §34-bis. **Presupuesto:** A ≈ 2 h/día; Héctor ≈ 2–3 h/día. **Regla de la semana:** la base de cada quien sale hoy por WhatsApp; nada se abre antes. Ningún número se dice el martes si no está en el repo con script, CSV o PDF y etiqueta §25. **Reparto (pedido por A el 22-oct):** Héctor lleva todo el programa y, como pesa poco con Claude Code, también dos piezas teóricas; cada día el programa va primero y la teoría ocupa el resto del bloque.

## 0. Dirección (lo que Francisco dijo el 20-oct y qué nos toca)

Francisco afirma, de palabra y "casi seguro", que en $u\equiv1$, $q\equiv\tfrac12$ el aumentado irrestricto gana a laminar por un factor $\log G$:

- **Lado laminar, suyo:** función potencial sobre átomos, $\phi(m,r)=(m-2r)^2/m$ ($m$ personas, $r$ infectadas), sumada sobre átomos; es $\ge$ la utilidad ya obtenida; una prueba virgen la sube 1 en promedio; partir un átomo la sube una constante; telescopio ⟹ laminar $=O(B)$. Él lo formaliza y lo sube al Overleaf. Nuestra verificación (acta, V1–V3): el split sube exactamente $4r(m-r)/(m(m-1))\le2$ ⟹ $V^{*}_{lam}\le2B$ [DERIVACIÓN IA, por validar].
- **Lado general, suyo:** matrices detectoras (Lindström) identifican a todos con $B$ pruebas ⟹ $\Omega(B\log G)$. Los números están "en un papel perdido"; los enviará.
- **Nuestra mitad:** la cota superior $B\log_2(G+1)$ para **toda** política (lema de información): hechos 1–3 ya en palabras de A (versión "siempre"); hecho 4 (promedio) es la pieza de A esta semana. Más los ejemplos y mediciones que acompañan: contraejemplos escritos, tabla $q=\tfrac12$, familia de matrices detectoras.
- **Lo que el equipo debe poder leer y cotejar:** las dos mitades de Francisco cuando lleguen escritas. Por eso Héctor reconstruye el argumento laminar y fija el enunciado de Lindström con constantes; ninguna de las dos se presenta como nuestra.
- **Conjetura de Francisco:** $\log G$ es el factor óptimo de pérdida laminar. No se afirma; es la pregunta (35).
- **Directriz:** "las diferencias grandes se verán con poblaciones grandes": las familias crecen $n$ y $B$ con $G$ siguiendo diseños detectores; la malla $n\le7$ ya no se extiende por sí misma.

## 1. Piezas de la semana

| # | Pieza | Dueño | Estado hoy | Etiqueta el martes |
|---|---|---|---|---|
| 1 | Párrafo de respuesta al mensaje del 13-oct (hechos 1–3, "siempre"; promedio pendiente), **WhatsApp hoy antes de 20:00** | A | trabajado en chat, no escrito por A | en palabras de A |
| 2 | Los dos contraejemplos escritos (árboles, valores exactos, mecanismo) + tabla $q=\tfrac12$, **WhatsApp hoy antes de 20:00** | Héctor | números en el repo con segunda vía; falta el escrito | VERIFICADO dos vías (solver IA + `BellmanPerfiles`/enumerador) |
| 3 | Firma de la nota "individuales nunca óptimas" (lectura de la prueba de A, tests y barrido) | Héctor | pendiente desde el 12-oct | firmada u objetada |
| 4 | Verificación exacta de V2 (identidad del split del potencial) | Héctor | derivación IA | VERIFICADO exacto $m\le12$ o refutado |
| 5 | **Reconstrucción escrita del argumento laminar de Francisco** (potencial domina utilidad; virgen sube 1; partir sube $\le2$; telescopio; $V^{*}_{lam}\le2B$), una página | Héctor | solo de palabra en la sesión | "reconstrucción, por cotejar con el escrito de Francisco" |
| 6 | Lema de información en palabras de A: $\mathbb E[\text{despejados}]\le B\log_2(G+1)+1$; corolario precio $\le2\log_2(G+1)+2/B$; **WhatsApp lun 26 antes de 20:00** | A (prueba), IA (typeset), Héctor (árbitro) | hechos 1–3 cerrados; 4 en curso | DEMOSTRADO si cierra; si no, "en escritura" |
| 7 | Familia de matrices detectoras: laminar exacto frente a $n/2$ por diseño $(n,B,G)$, $n\le16$; curva en $G$ | Héctor | $n=15$, $B=9$, $G=7$: 0.828 | diagnóstico, exacto por diseño |
| 8 | **Enunciado de Lindström con constantes:** para qué $(n,B)$ existe matriz detectora y qué peso de fila $G$ exige; el lado $\Omega(B\log G)$ explícito | Héctor | Francisco perdió sus números; pregunta (33) abierta | "borrador de literatura, por cotejar con Francisco" |
| 9 | Overleaf: enlace en el repo, aviso a Francisco, estructura de secciones | Héctor | abierto el 20-oct | — |

**Extras, solo con la base cerrada:** el potencial de Francisco como ejercicio de A (lun 26); piloto de diseños con cruces entre bloques (nota de Héctor del 20-oct). **Al pozo:** $T_{BG}$ en palabras de A; nota "individuales" pasos 3–4; comentarios a la §3 del paper (compromiso: fin de octubre); PDF de la brecha.

## 2. Hoy (jue 22)

**A (2 h).** Escribe las seis líneas en sus palabras (hechos 1–3: historiales $\le(G+1)^B$; revelar $m$ personas siempre exige $2^m$ historiales; luego $m\le B\log_2(G+1)$; "el promedio es el paso que sigue"). Las envía por WhatsApp antes de las 20:00 Praga. Avisa a Héctor que el párrafo lo manda A, no Héctor.

**Héctor (2–3 h).**
1. `git pull` en `augmented/correctness-and-paper` (acta, plan maestro y este plan ya están en el remoto).
2. *Programa.* Redacta los dos contraejemplos para Francisco: (a) $n=3$, $B=2$, $G=3$, $u\equiv1$, $q=(0.9,0.7,0.7)$: binario 2.079, laminar 2.062, irrestricto 2.121, los tres árboles; (b) familia $n=4$, $B=G=2$, $q=(s,\tfrac12,\tfrac12,s)$, $u=(1,3,3,2)$: $V_{bin}=\tfrac52 s(1+s)$, $V_{lam}=3+\tfrac32 s^2$, cociente $\to10/9$, con la observación de Francisco (la primera prueba del binario es un diseño detector). Adjunta la tabla $q=\tfrac12$ (`results/precio_q05_G/`, peor 41/52 en $n=7$, $B=4$, $G=3$) con su etiqueta. Envía antes de las 20:00. Manda a Francisco el enlace del Overleaf; déjalo en `docs/notes/2026-10-22-overleaf.md` (una línea).
3. *Teoría (30 min).* Firma de la nota "individuales": `pytest augmented/tests_brecha_convencion.py`, `PYTHONPATH=. python augmented/barrido_proposicion_individuales.py <semilla> 120 6 4 4` (espera `fails 0`), lectura de `augmented/paper/proposicion_individuales_construccion.md`; objeta o firma como fila en `2026-10-12-encargos-6-oct.md`.

## 3. Día por día

| Día | A (≈ 2 h) | Héctor, programa (primero) | Héctor, teoría (resto del bloque) |
|---|---|---|---|
| **jue 22** | §2 | §2 | §2 |
| **vie 23** | Hecho 4a: una historia que revela $m$ personas solo puede venir de mundos donde esas $m$ personas tienen exactamente esos estados: a lo sumo $1/2^m$ de los mundos. En sus palabras, con una línea de las tablas A y B. | **Pieza 4 (1 h).** Enumeración exacta (`fractions.Fraction`): $m=2..12$, $r=0..m$, $k=1..m-1$, $X$ hipergeométrica$(m,r,k)$; $\Delta=\phi(k,X)+\phi(m-k,r-X)-\phi(m,r)$; comprobar $\mathbb E[\Delta]=4r(m-r)/(m(m-1))$ y $\le2$. Fila en una nota con el script en `augmented/`. | **Pieza 5, parte 1.** Por qué el potencial domina la utilidad bajo soft clearing: los despejados son exactamente los átomos con $r=0$ (forma normal por átomos, companion Thm 4.1; `docs/notes/2026-08-25-mapeo-companion-francisco.md`), y un átomo con $r=0$ aporta $m$. Después el paso virgen: $R\sim\mathrm{Bin}(M,\tfrac12)$, $\mathbb E[(M-2R)^2]=M$. En sus palabras. |
| **sáb 24** | Hecho 4b: de la rareza al promedio. Con $K$ historiales y $P(m\ge j)\le\min(1,K/2^j)$: $\mathbb E[m]=\sum_{j\ge1}P(m\ge j)\le\log_2K+1$. En sus palabras, con $K=9$ como ejemplo. | **Pieza 7 (1–2 h).** Diseños detectores con $n\le16$ (Söderberg–Shapiro, Lindström, el detector ABC/ABD/ACD de $n=4$): $B$ = filas, $G$ = peso máximo de fila; general $=n/2$ si identifica todo perfil; laminar exacto con `bellman_tipos`; tabla $(n,B,G,\text{lam},n/2,\text{razón})$ y figura razón vs $G$. | **Pieza 5, parte 2.** El paso de partir: media y varianza hipergeométricas ⟹ $\mathbb E[\Delta]=4r(m-r)/(m(m-1))$ (su enumeración como control); telescopio sobre las $\le B$ pruebas; $V^{*}_{lam}\le2B$ y $\ge B/2$ por singletons. Una página en `augmented/paper/`, etiqueta "reconstrucción del argumento de Francisco, por cotejar con su escrito". |
| **dom 25** | **Pieza 6.** Enunciado + prueba en una página (molde: la nota "individuales"): hechos 1–4 y el corolario. Entrega el texto a la IA antes de las 16:00 para typeset. | **Segunda vía del lema (1 h).** Para las políticas óptimas de la malla $q=\tfrac12$, $\mathbb E[\text{revelados}]$ contra $\log_2$ del número de historiales alcanzados; ninguna celda puede violar el lema. | **Árbitro del lema de A** (objeta o firma). Después **pieza 8:** Lindström 1964/65, Cantor–Mills 1966, Söderberg–Shapiro 1963 — enunciado exacto: para qué $(n,B)$ existe matriz detectora, qué peso de fila exige, qué da con tope $G$; el lado $\Omega(B\log G)$ con constantes, como borrador para cotejar con los números de Francisco (responde la (33)). |
| **lun 26** | Extra: el potencial como ejercicio (virgen de tamaño 2 sube 1; partir un par con conteo 1 sube 2; identidad V2 en $m=3$). Dos preguntas para el martes. **WhatsApp antes de las 20:00:** lema (PDF) + pregunta (35). | Piloto de cruces entre bloques **solo si** 4, 5, 7 y 8 cerraron. Push. | Línea de la tabla de matrices para el WhatsApp; estructura del Overleaf: modelo · lema de información (equipo) · potencial [Francisco] · construcción [Francisco] · ejemplos y mediciones; dos preguntas para el martes. |
| **mar 27** | 16:00 guion congelado (`docs/notes/2026-10-27-guion-sesion.md`); ensayo 5+5; commit y push 17:30; **19:00 sesión** | Ensayo; commit y push 17:30 | |

## 4. WhatsApps a Francisco

- **jue 22, antes de 20:00:** A: párrafo de respuesta (hechos 1–3). Héctor: contraejemplos escritos + tabla $q=\tfrac12$ + enlace del Overleaf.
- **lun 26, antes de 20:00 (víspera, §34-bis):** A: el lema (PDF) con la pregunta (35). Héctor: la tabla de matrices detectoras con su etiqueta, en una línea.

## 5. Válvulas

- La base del jueves no sale → nada más se abre hasta que salga (regla del 2026-08-20, cuarta sesión seguida).
- El hecho 4 no cierra el domingo a las 16:00 → el lunes sale solo la tabla de matrices y "el lema está en escritura"; A presenta hechos 1–3 y dice que el promedio está en curso.
- V2 falla en la enumeración → es el resultado de la semana: se escribe la discrepancia y se le manda a Francisco antes de que formalice.
- La reconstrucción (pieza 5) no cierra la parte 2 → se entrega la parte 1 sola, etiquetada; no se extrapola.
- Los diseños con $n\le16$ no dan curva (pocos puntos) → se reporta la tabla tal cual; no se extrapola.

## 6. Soporte IA (solo con aprobación de A)

Typeset del lema desde el texto de A (dom); guion del 27-oct (lun); script de enumeración de V2 si Héctor lo pide (lleva "sin validar por B" hasta que lo corra); figura de la familia de matrices. Nada va al repo ni a Francisco sin que A lo haya leído.
