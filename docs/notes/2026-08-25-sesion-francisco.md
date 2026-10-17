# Sesión con Francisco — Tiendas 3B La Esperanza (2026-08-25)

**Fuente:** `TRANSCRIPCION_TIENDAS_3B_LA_ESPERANZA.md` (53:16; habla recuperada 00:00–53:13). Francisco confirmado por nombre dentro del audio; Participante 1 y Participante 2 son etiquetas sin identidad confirmada en la diarización. P1 es internamente consistente con A — «justo con Héctor, en la semana, le propuse» [03:02] y el relato del mapeo cinco-de-siete [03:29–04:21] que coincide con la tabla del guion — **confirmado por testimonio directo de A (2026-08-30)**. Disciplina: a Francisco solo lo inequívoco; lo demás, "el equipo reporta".
**Fecha:** 2026-08-25, confirmada por A (guion fechado ese día; §1 del plan: "sesión del 25-ago"; cadencia de martes).
**Documento de la sesión:** `dynamic_augmented_laminar_companion.pdf` (24 pp., seis direcciones), enviado por Francisco antes de la sesión. Guion previo congelado del equipo: `docs/notes/2026-08-25-mapeo-companion-francisco.md` (A-M20 de esta sesión). **Estatuto de todo claim del companion: [SIN VALIDAR — §25] hasta A-M23** (checklist: `docs/notes/2026-08-30-inventario-companion.md`). Despacho aplicado al plan maestro el 2026-08-30.

## Resumen ejecutivo

La sesión es la primera sobre el companion. Francisco lo presenta como dirección del tipo de algoritmo a implementar, declara su estado de verificación (seguro hasta Bellman/§7; §8 en adelante "mucho más nuevo", a checar a fondo) y reparte el trabajo: él se dedica a leer §8+; el equipo checa, digiere, escribe y experimenta. Deja dos encargos concretos (implementar el Bellman exacto y cotejarlo en el ejemplo chico; seguir leyendo el companion), un destino de publicación (congreso de algoritmos; SODA nombrado; arXiv antes como opción; Edwin de King's College en la conversación) y una decisión de modelo pedida por el guion y resuelta en nueve segundos: **"soft clearing es mejor"** — la convención pasa a posterior-zero vía G0. El reparto de resultados con el paper de Nick queda pendiente de un update que Francisco compartirá. El mapeo del equipo (5 de 7 puntos ya cubiertos por el plan) se reporta en sesión y Francisco confirma alineación ("muy on the same page"; la gran pregunta del equipo es "bastante alineada").

## D1 — Estatuto del companion y encargo de verificación [00:58–01:32, 50:49–51:47]

Francisco: ha leído "prácticamente como bien, bien la mitad de esto, porque lo he estado como que generando en conjunto, como unos resultados más recientes que tengo que checar bien. Y que valdría la pena que, de hecho, **todos los checáramos bien** para ver si está bien, porque nos da como una muy buena dirección del tipo de algoritmo que podríamos implementar y experimentar" [00:58–01:32]. Contexto de generación asistida: "Mi ChatGPT también está hablando de los residual atoms… estamos en la misma página" [02:37].

Gradiente de confianza, al cierre: "la parte hasta Bellman, definitivamente yo estoy prácticamente seguro que todo eso está bien. Y también estoy prácticamente seguro que lo que habíamos hasta la sección 8… incluso es el mismo ejemplo que habíamos visto… el nuevo teorema es como bastante parecido a otros teoremas que tenemos en el otro trabajo… **lo que sí es como que mucho más nuevo es todo lo que está en la sección 8 en adelante**, así que entre más nos empleemos a fondo a checar que eso está bien y a digerirlo y a escribirlo…" [50:49–51:47]. (La frontera exacta "hasta la sección 8" queda ambigua entre inclusivo/exclusivo; ambas lecturas coinciden en que §8+ es lo nuevo — A-M23 prioriza §8–§10.)

Despacho: A-M23 nuevo (validación dirigida del companion); estatuto [SIN VALIDAR] por resultado hasta la validación de A.

## D2 — La convención: «soft clearing es mejor» [43:44–43:53]

En la explicación de la utilidad del complemento libre (ec. 5.1: "tenemos de hecho una otra prueba gratis por el conteo perfecto… la utilidad que sacamos de la prueba gratis del complemento" [43:19–43:44]), el equipo pregunta: "¿aquí no estamos imponiendo esto porque lo estamos manejando así con el hard clearing, tenemos que probarlo?" [43:44–43:53]. Francisco: "**No, no, digamos soft clearing es mejor.**"

Es la decisión que el guion §E pedía ("Decisión pedida: ¿posterior-zero o hard clearing estricto?"): la convención normativa pasa a **posterior-zero clearing** (Def. 2.1 del companion; su Remark 4.2 hace de esta convención requisito para que Thm 4.1 transfiera literal). Procesada por G0 con fila en §32 (§5 no se toca sin eso). Cascada identificada: §16 pierde el test acreditador ($k = \max\{0, B-\lceil\log_2 G\rceil\}$; ancla re-derivada $1-0.95^{48}\approx 0.9147u$, aritmética verificada); en B-M16 la reentrada vale 1.0 en vez de 0.5 (ambas ramas acreditan: cero observado o deducción del complemento); el colapso $S_1^{hard}=S_0$ (§14.3) y la maquinaria κ (§5.8/§14.6) quedan adscritos a la variante estricta. Inventario completo: A-M24/B-M18.

## D3 — Encargo: el Bellman exacto, implementado y cotejado [07:49–09:19, 46:24–47:22]

"En el caso laminar en este documento, la ecuación de Bellman es como bastante sencilla… nos da un algoritmo para el cómputo exacto de una estrategia óptima laminar, que es una recursión… es complejo el algoritmo, pero es mejor que la enumeración… podríamos correr el algoritmo de Bellman en casos pequeños, más grandes de los que hicimos con brute force, definitivamente" [08:03–08:48]. Estado y transición: "tu estado son las personas que no tienen prueba y los átomos residuales… tus acciones es hacer una prueba, ya sea de la parte sin pruebas o de los átomos residuales. Ahora tenemos también la ecuación convolucional para el cómputo posterior… hay como una manera eficiente de describir esa transición… Si no teníamos antes" [09:19–10:11]. Recorrido de §5 del companion: estado $(U,\mathcal A, B)$ [37:28–38:24], normalizador de extremos ν [38:24–38:53], función g y las dos vías de utilidad [39:22–40:19, 42:50–43:44], recursión 5.5 y reconstrucción del árbol de abajo hacia arriba [40:19–41:35].

El encargo, literal: "la complejidad computacional no es eficiente pero es más eficiente que brute force… esto ya lo podríamos implementar y **de hecho un ejercicio interesante sería implementar con la ecuación de Bellman el cómputo óptimo y checarlo en el ejemplo que dimos para un caso pequeño**" [46:24–47:22]; "cualquier experimento que podamos hacer, por ejemplo la implementación de Bellman… también nos ayuda" [51:47–51:54].

Ganancia por tipos (Prop 6.2): "si nada más tenemos como que tres utilidades y tres probabilidades… nueve o diez tipos, podemos correr el algoritmo de Bellman en casos más grandes todavía" [08:48–09:04]; "imagínate que hay como que 10 tipos… población de 100 personas pero lo único que tienes que tomar en cuenta es un vector de tamaño 10… tu espacio de representación disminuye bastante" [47:52–48:30]. (Aritmética de Prop 6.2 verificada por el equipo 2026-08-30: 297,968,931 acciones etiquetadas vs 55/251/3,002 composiciones para M=3/5/10 con n=130, G=5; átomo homogéneo m=16: 15 vs 65,534.)

Despacho: B-M17 nuevo; no reordena gates (G5/rollout intactos; el solver es la segunda vía exacta de $V^{*,\mathcal L}$, clase pathwise etiquetada).

## D4 — Convolución/polinomios = nuestro tensor, por ruta independiente [00:00–00:24, 16:05–24:45, 27:15–35:34]

"el factor normalizador, la partition function… tenemos un algoritmo dinámico que nos da eso… la ecuación 3.4, que todo esto es un polinomio. Entonces al ser un polinomio hay una convolución… se puede hacer todo esto de una manera muy eficiente" [00:00–00:24]. Alcance: "¿esto depende del régimen de… probabilidades homogéneas? no, no, no, esto es completamente arbitrario" [18:44–19:12]. Las dos estructuras explotadas: "la probabilidad prior es independiente y tiene como que esta estructura binomial que se presta al coeficiente de un polinomio y que las pruebas son aumentadas… para que tengas 10 personas infectadas en el conjunto principal y 7 en el chico tienes que tener 7 y 3, no hay ninguna otra opción… ganancias enormes en términos del cómputo posterior" [19:12–20:29]. Y el cierre: "¿se acuerdan cuando estamos haciendo lo de Gibbs y que las configuraciones complejas…? no tenemos que hacer el conteo, nada más tenemos que hacer el cómputo del polinomio una vez… y sale el resultado" [35:05–35:34].

Confluencia exacta con lo nuestro: su $Z(A,r)$ es nuestra $f_A(r)$ (Poisson-binomial); su ec. 3.5 es la forma cerrada A2.3 ($Q = f_S(r)f_{T\setminus S}(R-r)/f_T(R)$); su caché por convolución es A2.8. El equipo aclara en sesión que su nota previa de Obsidian usaba "convolución" en otro sentido (suma de contribuciones independientes) [20:29–21:56]. Confirma la formalización (§0-bis c) sin cambio de contenido.

## D5 — Separación reforzada: la distinción aumentado / no-aumentado [06:16–07:24, 48:30–48:58]

"hay un resultado que demuestra que… si las probabilidades de todo el mundo son suficientemente altas de infección, en el algoritmo dinámico no usa pruebas grupales… **eso a nosotros nos da una distinción muy concreta entre aumentado y no aumentado. Porque eso demuestra que incluso en lo dinámico, sin aumentado, todo es individual.** Y tenemos este ejemplo particular que como que separa las dos cosas" [06:16–06:51]. Y sobre §7: "7.1 es lo que había dicho que es nuevo… el 7.2 es nuestro ejemplo pero formalizado… la diferencia… no tiene límite, es **potentially unbounded**" [48:30–48:58].

Lectura contra el plan: Thm 7.1 (companion) afirma exactamente la mitad $q\le 1/2$ de la celda dinámico-binaria (§18, escrita "sin resultado predicho"); el lado $q>1/2$ — el disputado en C1 — sigue abierto. C4/§18 no cambian de lenguaje hasta validar la prueba (Harris + clearing normalizado) en A-M23. Dónde vive el teorema (companion vs paper de Nick) queda pendiente del reparto (D10).

## D6 — Forma normal: «nunca cosas por encima» — responde la pregunta (17) [36:02–36:59]

"técnicamente algo laminar podría ser… hacer una prueba de dos personas, dos personas y luego aplicar una tercera prueba que es las cuatro… pero es muy obvio que eso no es correcto… **para qué aplicaría eso si ya [sabes] el resultado de eso**… lo de cuatro simplemente es como de manera rigurosa de escribir eso de que siempre una prueba laminar nada más va como a hacer subdivisiones, **nunca va a hacer como que cosas por encima**… una vez que tienes un átomo solamente lo vas haciendo más fino" [36:02–36:59].

Respuesta a (17): los ancestros no se *prohíben* — son **reducibles sin pérdida** (Thm 4.1: $T = K \mathbin{\dot\cup} D$ con $R(D)=R(T)-R(K)$; la parte informativa es $D$). Coincide con la lectura A.3 del guion. A-M21 (dominación de la repetición) queda contenido como el caso $D=\varnothing$; su cierre pasa a "validar Thm 4.1" dentro de A-M23.

## D7 — Híbrido: endgame exacto + heurísticas; el menú local sí es computable [14:15–15:49]

Sobre la gran pregunta del equipo: "bastante alineado porque de hecho hasta me pone a pensar en… algo híbrido donde tenemos algo que como que hacia el final del algoritmo tienen como una solución explícita y utilizar heurísticas como que anteriores a la solución explícita… Cuando tú tienes cinco personas… corres el algoritmo óptimo de como con uno a cuatro para una población de cinco y con eso, por ejemplo, **ya sabes por grupo exactamente lo óptimo que podrías hacer después**… eso para como tamaños de grupos pequeños, creo que incluso hasta podríamos enumerar la estrategia laminar óptima" [14:15–15:49].

Lectura contra el plan: el menú global sigue incomputable (fila 2026-08-18), pero el **menú local por componente es computable exacto en grupos chicos** — es $W_c/H_c^\circ$ del companion (§8.3), la instancia exacta local de $\varphi(D,c,b)$ (§14.5–14.6), y $\rho_b = \max_c H_c/c$ es la candidata F con $\alpha{=}1$ y filtro incorporado. Refuerza A-M12/B-M9 sin milestone nuevo; el barrido α conserva su estatuto diagnóstico.

## D8 — Tres greedies con garantías distintas; Francisco se dedica a §8+ [11:25–12:14, 48:30–50:32]

Sobre el presupuesto mágico, reportado por el equipo (asume presupuesto infinito, siempre agrupa): "Exacto, que es incorrecto en algunos casos… Eso conecta con el último punto de este documento también. **Pero sí que voy a tener un ejemplo de eso también**" [12:03].

"la dirección 8 es lo que yo sí quiero como que empezar a ver más a fondo… el artículo supuestamente describe tres [estrategias greedy]: una es como que el greedy inmediato, otra es como que una medida de greedy que incorpora un poquito más el presupuesto a largo plazo y hay otra que habla como que de factores de probabilidad de infección… **supuestamente cada uno tiene una diferente garantía de aproximación ante lo óptimo en laminar**… esto yo sí me voy a dedicar… tratar de leer esto lo más posible" [48:30–49:55]. (Paráfrasis de sesión; en el companion las tres políticas son inmediato/committed-density/receding-density con fracciones límite $1/G$, $1/\log G$, $\to 1$ frente a $\mathrm{OPT}^D_{\mathrm{aug}}$ en la familia rare-health — Thm 9.3 — y factor $G$ general para el inmediato — Thm 8.2. El desfase paráfrasis↔enunciado formal queda anotado para A-M23.)

El esqueleto de publicación, condicionado a la verificación: "si todo esto resulta ser cierto ya tenemos como que de hecho todos los resultados que necesitaríamos para publicar… un modelo que está inspirado por la práctica… un algoritmo exacto… tres diferentes algoritmos de optimización… un caso que demuestra esta separación… y tres distintos algoritmos que tienen diferentes garantías de aproximación… **pero sí tenemos que checar todo esto**" [49:26–50:32].

## D9 — Destino de publicación: SODA; arXiv antes; Edwin en la conversación [04:21–05:59]

"yo he estado hablando con esto con mi colega Edwin… acaba de tener un hijo… profesor en King's College en Londres" [04:21–04:35] (verosímilmente E. Lock, coautor del arXiv 2206.10660 — inferencia del equipo, no dicho en sesión). "el sabor que yo he estado como que metiendo y organizando ha sido como que más de la perspectiva de algoritmos y la aproximación… hay unos congresos como que muy prestigiosos de algoritmos, en particular hay uno que se llama **SODA**… probablemente como que el mejor de como algoritmos discretos… la fecha límite es como que hasta julio… **bien podríamos mandar algo para como que julio del año que entra. Nos da bastante tiempo**… podemos poner algo públicamente en arXiv desde antes" [04:35–05:59].

## D10 — El paper de Nick y el reparto de resultados [05:59–07:24]

"en paralelo estoy trabajando el paper de Nick para como mejorarlo para publicación. Lo que le faltaba al paper de Nick también eran los resultados teóricos… ese ahorita es un poquito mi prioridad… y como que hacer un update a lo que está en arXiv. Y cuando tenga eso se los comparto" [05:59–06:16]. Sobre dónde vive cada resultado (p. ej. el teorema de D5): "tengo que ver como que dónde pongo cosas… en el argumento de Nick también hay como un factor interesante que entra en la aproximación… cuando lo tenga se los comparto para como que veamos bien **cómo repartir cosas**" [06:51–07:24]. (Identidad de "Nick" no establecida en sesión; no coincide con autores de [1].) Consecuencia operativa: ningún resultado del companion se compromete como nuestro en A-M19/A-M22 hasta ese reparto.

## D11 — Lo que el equipo reporta (contexto, no directriz)

Semana del equipo: recopilación + escritura en el documento nuevo (enfoque laminar, Overleaf compartido) [01:41–02:31]; master plan refinado con Héctor y mapeo contra el companion: "de los siete puntos… hay cinco… que tenemos prácticamente similar… un punto que nosotros estamos abarcando de forma parcial y uno en el que no… lo de las garantías de aproximación" [03:02–04:21] — coincide con la tabla del guion (1, 2, 4, 5, §7 ≈ / 3 parcial / 6 fuera de radar). Francisco: "creo que estamos como que muy on the same page para estas dos cosas" [02:57].

La gran pregunta, formulada por el equipo: "¿**qué sustituto de esa utilidad que sí sea computable puede capturar suficiente información futura y puede capturar suficiente información de los test que ya hemos realizado para guiar la política?**" [13:40–14:01]; Francisco: "bastante alineado" [14:15]. El equipo también reporta: extremos greedy/presupuesto-mágico como dos caras de la misma moneda [12:14–12:35], costo por greedy sobre átomos y territorio virgen [12:50–13:40], y la lectura fina de átomos ("cada subtest… solamente va a ser más fino al átomo" [10:42–10:56]) que Francisco confirma vía §4 (D6).

## D12 — Logística [52:40–53:05]

Francisco da clases desde la semana entrante ("los martes y jueves"; "una de Economía y Computación"); enviará mensaje para fijar nueva hora de sesión durante el semestre.

## Preguntas de §34

- **(17)** ancestros bajo la regla de subpruebas → **RESPONDIDA** (D6): reducibles sin pérdida, no prohibidos.
- **(13)** enunciado objetivo del PDF → **RESPONDIDA** (companion §10–§12 + [07:24–07:49]): el programa completo — Bellman exacto + garantías frente al óptimo laminar; la dureza como telón fácil.
- **(1)** ¿$V^{*,\mathcal L}$ o $V^*$? → **PARCIAL**: comparador escalonado (factor $G$ frente al irrestricto ya enunciado; la meta viva es constante frente al óptimo laminar, Conj. 10.7).
- **(4)** garantía valorada → **PARCIAL**: aproximación frente al óptimo laminar como pieza que completa el paper (D8); la celda: teorema (Thm 7.1), condicionado a validación.
- **(12)** contraejemplo de referencia → **PARCIAL**: §7.2 formaliza el de separación; el rol del de no-reentrada espera el ejemplo que Francisco anunció [12:03].
- **(9), (11), (14), (15), (16)**: no tocadas. Del guion: F.1 respondida (D2); F.2 parcial (D9/D10); F.3 no alcanzó a hacerse → pregunta nueva (20).
- **Nuevas (18)–(22)** añadidas a §34 (estatuto de escritura; reparto con Nick; eje α vs committed/receding; variante estricta en el harness; clase ex ante vs pathwise — esta última por hallazgo de lectura, no de sesión).

## Compromisos y pendientes

- **De Francisco (2026-08-25):** compartir el update del paper de Nick y el reparto [06:51–07:24] · un ejemplo propio de la falla del presupuesto-infinito [12:03] · lectura a fondo de §8+ [48:30–49:55, 51:17] · mensaje con nueva hora de sesión [52:40–52:52].
- **Del equipo (2026-08-25):** checar el companion completo (A-M23; "sigamos leyendo este" [52:00]) · implementar Bellman y cotejar en el ejemplo chico (B-M17) · migración de convención tras G0 (A-M24/B-M18).
- **Siguen vigentes (2026-08-18):** A-M22 (documento formal laminar) · A-M19 (outline, ahora formato SODA) · B-M6 ($C(T)$ local) · barrido diagnóstico $V/C^\alpha$ · cotejo del teorema estático $p>1/2$ (§21) · reformulación de la Conjetura C (P21-A8).
- Despacho §34-bis aplicado el 2026-08-30: 8 filas en §32; A-M23/A-M24/B-M17/B-M18 nuevos; G0 reabierto y aprobado por A (ratificación de B pendiente); §5 y §16 re-derivados; §34 actualizado con (18)–(22); §0-bis (a) ampliado; §1 reescrita (31-ago→4-sep); checklist del companion en `docs/notes/2026-08-30-inventario-companion.md`. Los hallazgos de lectura del 2026-08-30 (Phases 1–6, clase pathwise, salvaguarda, $G\ge4$, $I_\lambda$) quedan etiquetados como hallazgos IA — A valida — y no como directrices de sesión.
