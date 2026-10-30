# Plan hasta la sesión — vie 30-oct → mar 3-nov, 19:00 Praga (por confirmar con Francisco) — semana de recuperación

**Autocontenido para Héctor.** Supersede a `2026-10-22-plan-hasta-sesion.md`, cuyas piezas siguen siendo las mismas: la semana pasada no salió ninguna pieza de A ni los dos WhatsApps; las tuyas de programa están en el notebook 30 (`results/sesion_2026-10-26/notebook_30/`). Fuentes: acta del 27-oct (`2026-10-27-sesion-francisco.md`), plan maestro §1 (semana del 30-oct) y §34-bis. **Horas:** A: vie 1 h, sáb 3–4 h, dom 3–4 h, lun 6 h, mar 6 h; Héctor ≈ 2–3 h/día. **Regla:** la base de cada quien sale el viernes por WhatsApp; nada se abre antes; nada nuevo esta semana.

## 0. Qué dijo Francisco el 27-oct y qué cambia

- **No cambia la dirección** (20-oct): en $u\equiv1$, $q\equiv\tfrac12$ el gap aumentado/laminar es $\log G$; su lado laminar es el potencial, su lado general las matrices detectoras; la mitad del equipo es el techo $B\log_2(G+1)$ para toda política (lema de información, A).
- **Confirmó tu paso de partir:** "para un átomo abierto es contar todas las distribuciones, pero es un factor constante también" [26:19]. Tu enumeración exacta (638 tripletas) es la segunda vía de la identidad $4r(m-r)/(m(m-1))\le2$.
- **El paper en cuatro secciones** [22:17–23:48]: formulación · separación factor $G$ (no aumentado vs aumentado) · separación $\log G$ (laminar vs óptimo en $q=\tfrac12$) · algoritmos laminares de aproximación. "Super paper, primer capítulo de esta dirección." Esa es la estructura del Overleaf.
- **Cree que $\log G$ es el peor caso** y sugiere "ver si salen ideas para atacar eso". No es encargo fechado; es la pregunta (35).
- **Compromisos de él:** la estrategia de los códigos óptimos (el detalle que faltaba el 20-oct) y un documento con el argumento. Si llegan esta semana, se leen antes que cualquier otra cosa teórica.
- **Destino:** SODA es el mejor fit pero lejos; mandó una lista de congresos por mensaje (A la pega en `docs/notes/2026-10-30-congresos.md`); hay que investigar fechas.
- **Política de IA acordada:** herramientas como extensión; "el bar es poder reescribirlo todo"; nada se entrega que no se pueda explicar. Coincide con §25: tus números llevan segunda vía; los textos van en palabras de quien firma.

## 1. Piezas de la semana

| # | Pieza | Dueño | Estado hoy | Etiqueta el martes |
|---|---|---|---|---|
| 1 | Párrafo de respuesta al mensaje del 13-oct (hechos 1–3), **WhatsApp vie 30 antes de 20:00** | A | trabajado en chat, no escrito | en palabras de A |
| 2 | WhatsApp con los dos contraejemplos escritos, la tabla $q=\tfrac12$, la tabla de diseños detectores (notebook 30 §2) y la línea de V2, **vie 30 antes de 20:00** | Héctor | todo está en el repo; falta el escrito | VERIFICADO dos vías |
| 3 | Firma de la nota "individuales nunca óptimas": lectura de la prueba de A (`augmented/paper/proposicion_individuales_construccion.md`); el barrido ya corrió (1,766 casos, 0 fallas) | Héctor | lectura pendiente | firmada u objetada |
| 4 | Tabla de congresos: de la lista de Francisco, fecha límite, notificación, fit (el soporte IA borra; tú cotejas en los sitios) | Héctor | lista por pegar | borrador para la pregunta (37) |
| 5 | Reconstrucción escrita del argumento laminar de Francisco (potencial domina utilidad; virgen sube 1; partir sube $\le2$; telescopio; $V^{*}_{lam}\le2B$), una página | Héctor | el paso de partir ya está exacto en el notebook 30 | "reconstrucción, por cotejar con el escrito de Francisco" |
| 6 | Lema de información en palabras de A: $\mathbb E[\text{despejados}]\le B\log_2(G+1)+1$; corolario precio $\le2\log_2(G+1)+2/B$; **WhatsApp lun 2 antes de 20:00** | A (prueba), IA (typeset), Héctor (árbitro) | hechos 1–3 cerrados; 4 en curso | DEMOSTRADO si cierra; si no, "en escritura" |
| 7 | Enunciado de Lindström con constantes (para qué $(n,B)$ existe matriz detectora, qué peso de fila $G$ exige; el lado $\Omega(B\log G)$) — **solo si no llega el documento de Francisco** | Héctor | pregunta (33) abierta | "borrador de literatura, por cotejar" |
| 8 | Overleaf: enlace en el repo, aviso a Francisco, estructura con las cuatro secciones | Héctor | abierto el 20-oct | — |

**Extras, solo con la base cerrada:** el potencial como ejercicio de A (lun); $T_{BG}$ en palabras de A (lun, si el lema cerró antes de las 16:00); fronteras de la nota "individuales" (mar, si sobra). **No esta semana:** comentarios al paper de Francisco (la promesa de fin de octubre se re-fecha en sesión); $q\ne\tfrac12$; loops; extensiones de §30.7–30.8.

## 2. Hoy (vie 30)

**A (1 h).** Las seis líneas en sus palabras (hechos 1–3: historiales $\le(G+1)^B$; revelar $m$ personas siempre exige $2^m$ historiales; luego $m\le B\log_2(G+1)$; "el promedio es el paso que sigue"). WhatsApp antes de las 20:00 Praga. Pega la lista de congresos de Francisco en `docs/notes/2026-10-30-congresos.md`.

**Héctor (2–3 h).**
1. `git pull` en `augmented/correctness-and-paper` (acta del 27-oct, plan maestro y este plan ya están en el remoto).
2. *Base (programa, 1.5 h).* Redacta para Francisco: (a) testigo $n=3$, $B=2$, $G=3$, $u\equiv1$, $q=(0.9,0.7,0.7)$: binario 2.079, laminar 2.062, irrestricto 2.121, los tres árboles; (b) familia $n=4$, $B=G=2$, $q=(s,\tfrac12,\tfrac12,s)$, $u=(1,3,3,2)$: $V_{bin}=\tfrac52 s(1+s)$, $V_{lam}=3+\tfrac32 s^2$, cociente $\to10/9$, con su observación de que la primera prueba del binario es un diseño detector; (c) la tabla $q=\tfrac12$ (`results/precio_q05_G/`, peor 41/52 en $n=7$, $B=4$, $G=3$); (d) la tabla de diseños detectores del notebook 30 (0.875–0.958 en $n\le8$; 0.828 en $n=15$; 0.858 en $n=16$); (e) una línea: "el paso de partir sube exactamente $4r(m-r)/(m(m-1))\le2$, comprobado en 638 casos". Cada número con su etiqueta. Envía antes de las 20:00. Manda el enlace del Overleaf; déjalo en `docs/notes/2026-10-30-overleaf.md`.
3. *Teoría (45 min).* Firma de la nota "individuales": lee la prueba de A y escribe tu veredicto como fila en `2026-10-12-encargos-6-oct.md` (objeción concreta o firma).

## 3. Día por día

| Día | A | Héctor, programa (primero) | Héctor, teoría (resto) |
|---|---|---|---|
| **vie 30** | §2 (1 h) | §2 | §2 |
| **sáb 31** | Hecho 4a (rareza) y 4b (suma de colas ⟹ $\log_2K+1$), en sus palabras, con las tablas A y B del chat (3–4 h) | **Pieza 4 (1 h).** Tabla de congresos: nombre, fecha límite, notificación, fit (teoría vs aplicado), con enlace; el soporte IA entrega el borrador, tú cotejas cada fecha en el sitio del congreso. | **Pieza 5, parte 1.** Por qué el potencial domina la utilidad bajo soft clearing (despejados = átomos con $r=0$; forma normal por átomos, companion Thm 4.1, `2026-08-25-mapeo-companion-francisco.md`) y el paso virgen $\mathbb E[(M-2R)^2]=M$. En tus palabras. |
| **dom 1** | **Lema, borrador 1** (3–4 h): enunciado + prueba, una página; lupa de revisor en el chat | — | **Pieza 5, parte 2.** Partir desde media y varianza hipergeométricas ⟹ $4r(m-r)/(m(m-1))$ (tu enumeración como control); telescopio sobre las $\le B$ pruebas; $V^{*}_{lam}\le2B$ y $\ge B/2$. Una página en `augmented/paper/`, etiqueta "reconstrucción del argumento de Francisco, por cotejar con su escrito". |
| **lun 2** | Versión final del lema (texto a la IA antes de las 14:00 para typeset); el potencial como ejercicio (1 h); $T_{BG}$ si el lema cerró antes de las 16:00. **WhatsApp antes de las 20:00:** lema (PDF) + preguntas (35)(38) (6 h) | **Segunda vía del lema (30 min):** tu CSV `lema_informacion_malla.csv` contra el enunciado final de A (ninguna celda puede violarlo). Push. | **Árbitro del lema** (objeta o firma). Si llegó el documento de Francisco: leerlo y lista de dudas. Si no: **pieza 7**, Lindström 1964/65, Cantor–Mills 1966, Söderberg–Shapiro 1963. **Pieza 8:** estructura del Overleaf con las cuatro secciones. |
| **mar 3** | Ensayo del lema en voz (10 min); fronteras de la nota "individuales" solo si sobra; **16:00 guion congelado (`docs/notes/2026-11-03-guion-sesion.md`); ensayo 5+5; commit y push 17:30; 19:00 sesión** (6 h) | Ensayo; commit y push 17:30 | |

## 4. WhatsApps a Francisco

- **vie 30, antes de 20:00:** A: párrafo (hechos 1–3). Héctor: contraejemplos + tabla $q=\tfrac12$ + diseños detectores + línea de V2 + enlace del Overleaf.
- **lun 2, antes de 20:00 (víspera, §34-bis):** A: el lema (PDF) con (35)(38). Héctor: la tabla de congresos, en una línea con el enlace al repo.

## 5. Válvulas

- Las seis líneas no salen el viernes → nada más se abre hasta que salgan.
- El lema no está final el lunes a las 16:00 → el WhatsApp dice "lema en escritura"; A presenta hechos 1–3 y la rareza.
- Llega el documento de Francisco (códigos óptimos o argumento) → Héctor lo lee en lugar de Lindström; la reconstrucción se coteja contra él.
- La reconstrucción no cierra la parte 2 → se guarda la parte 1 sola, etiquetada.

## 6. Soporte IA (solo con aprobación de A)

Borrador de la tabla de congresos (sáb); typeset del lema desde el texto de A (lun); guion del 3-nov (lun); lupa del lema y de la reconstrucción. Nada va al repo ni a Francisco sin que A lo haya leído.
