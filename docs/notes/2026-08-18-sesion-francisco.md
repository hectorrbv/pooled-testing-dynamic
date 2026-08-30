# Sesión con Francisco — Avenida de La Pradera 2 (2026-08-18)

**Fuente:** `TRANSCRIPCION_AVENIDA_DE_LA_PRADERA_2_ACTIVE.md` (56 segmentos; diagnóstico de diarización en `TRANSCRIPCION_AVENIDA_DE_LA_PRADERA_2.md.diagnostics.json`). `speaker 2` corresponde a Francisco con confianza 0.80–0.92 en todas las citas que aquí se le atribuyen (verificado segmento a segmento); Hablantes B y C son agrupaciones de voz del equipo, no identidades confirmadas. Disciplina de atribución: lo no inequívoco se cita como "la conversación plantea…". **Corrección por testimonio directo:** el bloque [10:25–10:50] quedó etiquetado como equipo pero contiene un cambio de turno que la diarización no separó; A, presente en la sesión, confirma que la frase sobre subpruebas laminares es de Francisco (ver D2). Despacho aplicado al plan maestro el 2026-08-20.

## Resumen ejecutivo

El equipo reporta el avance de redacción de los dos Hechos del colapso del score y presenta un **contraejemplo nuevo**: con un par AB ya testeado y conteo 1 — utilidad viva de 1, un sano seguro pero no acreditado — el score de presupuesto mágico nunca reentra a la prueba individual que cobraría esa utilidad; prefiere un par virgen, y en tercera instancia se atasca retesteando el par idéntico. Francisco lo califica de "definitivamente subóptimo", excluye la repetición idéntica del menú de candidatas y fija la disciplina laminar sobre los pools testeados. Hallazgo estructural del equipo, validado en la mesa: **el score sin planificación y el de presupuesto mágico son los dos extremos**, y el score correcto los interpola con el presupuesto restante.

Francisco aporta cuatro cosas nuevas: (1) el objeto ideal es un **menú valor-por-presupuesto** — tenerlo equivale a resolver el problema; (2) el **arreglo a la medición del costo**: simular greedy *dentro del conjunto y después de la prueba*, no globalmente desde cero; (3) el colapso valor/costo a una dimensión es un problema tipo **knapsack sin respuesta canónica** — familia de heurísticas $V/C^\alpha$ a barrer empíricamente; (4) dos encargos de escritura: un documento formal nuevo sobre lo laminar con el contraejemplo, y el outline del paper en documento aparte. Además: el paper hermano ya está en arXiv.

## D1 — Reconfirmación de la dimensión faltante [02:49–02:55, 05:08–05:26]

Francisco: "había como una dimensión que faltaba que eran de **cuántas pruebas se necesitan para realizar ese valor**" — reconfirma la directriz del 2026-08-11 sin modificarla. Complemento [05:08–05:26]: "esto es como que la diferencia entre el puntaje de un pool versus el valor… el caso lineal versus [el subexponencial]".

El equipo reporta el caso de enmedio [03:45–04:31]: con conteo 1 en un par "sabes que hay un sano, pero no sabes quién va a ser… aunque en teoría lo puedes inferir, eso no te genera cobrar utilidad" — lectura correcta y espontánea del strict hard clearing (§5.7–5.8): la deducción informa, no acredita.

## D2 — El contraejemplo de no-reentrada y la disciplina sobre pools testeados [07:04–11:04]

Hallazgo del equipo (utilidad uniforme 1, probabilidad de actividad 0.3, par AB con conteo 1, utilidad viva 1). Tres opciones y el orden que produce el score de presupuesto mágico:

1. **reentrar** y testear A individual → score 0.5 — cobra la utilidad con certeza ("al hacer la prueba individual, yo con certeza voy a saber quién es el sano" [08:45–09:39]);
2. **abrir un par virgen** CD → score dictado 0.6; "retestear ahora un nuevo par… siempre daba un score más alto" [07:04–08:00];
3. **retestear el par AB idéntico** → score aún mayor, "porque **puntúa la utilidad viva completa**" [08:23–08:33].

Resultado: "nunca terminaba de ni siquiera cobrar la utilidad que ya sabía que ya existía… se quedaba retesteando el mismo par" [07:04–08:00]. Francisco: "**Es algo bueno saberlo porque eso es definitivamente subóptimo**" [10:20–10:25].

**La directriz, con atribución corregida por testimonio directo.** La frase de [10:25–10:50] — "en el momento que se hace una prueba grupal, no vuelves a hacer nada más, puedes hacer subpruebas de forma laminar" — **es de Francisco** (testimonio directo de A; la diarización etiquetaba el bloque como equipo sin separar el cambio de turno). Y su frase siguiente [10:50–11:04]: "**dentro de las pruebas que no son la prueba total, cuál es la óptima, porque en ningún momento haríamos eso**."

Alcance despachado: (i) la **repetición idéntica queda excluida** del menú; (ii) la interacción con un pool testeado es por **subpruebas laminares** — la historia permanece laminar (§6.6). Abrir territorio virgen sigue permitido: los pools disjuntos son laminar-compatibles, la propia frase siguiente de Francisco mantiene el par virgen entre las candidatas ("las pruebas que no son la prueba total" eran la individual y el par virgen), y el ejemplo motivador que él mismo fijó como criterio de validación (k pools raíz + búsqueda binaria) testea un segundo pool virgen tras el primero. La lectura global fuerte ("nunca nada más tras una grupal") contradiría todo eso; de sostenerse, iría por R10. Queda la pregunta de precisión (17) en §34: si "subpruebas" excluye también a los ancestros.

**Cifras:** el 0.5 es reconstruible analíticamente (con conteo 1 en un par homogéneo, P(A sano | R=1) = 1/2, independiente de p). El 0.6 del par virgen y el score del retest **no tienen artefacto en el repo**: quedan como dictados en sesión hasta B-M16. **Nota post-acta (2026-08-20, verificación a mano de A, dos vías — estructura y enumeración de los 16 mundos):** los números dictados reproducen exactamente con la convención **q = P(sano) = 0.3**, u≡1, G=2, B=2 — el 0.5 (reentrada), el 0.6 (par virgen, = q+q), el retest derivado en 1.0 ("más alto" en sesión), y los pesos del árbol 0.09/0.42/0.49 (dictados como "0.9, 0.42"). La frase del transcript "probabilidad de actividad del punto 3" es habla suelta; la convención confirmada es q sano = 0.3. Totales de política derivados: V̂ con retest 0.2844; V̂ con poda 0.36; singletons (= óptimo en esta instancia) 0.6; par-primero jugado perfecto 0.564. B-M16 fija esta instancia por enumeración.

## D3 — El menú valor-por-presupuesto es el objeto ideal [12:55–14:33]

Francisco: "en el mundo perfecto… nos diría el valor que le sacamos a ese grupo, como óptimo… **tendríamos un menú, por presupuesto, cuánto le puede sacar**… digamos que sale uno, nos diría que con presupuesto uno sacas tanto, dos, tanto, tres, tanto. **Al tener eso, sabes exactamente la solución del problema.** Es un poquito como en reinforcement learning, cuando tienes la Q-table."

Valida de forma independiente la forma $\varphi(D,c,b)$ indexada por presupuesto (§14.5–14.6) y fija por qué no se calcula exacta: tenerla es resolver el problema. Es la misma moraleja que el equipo alcanzó por otra ruta [05:26–07:04]: el score sin planificación y el de presupuesto mágico son **los dos extremos**, y el score correcto los interpola con el presupuesto restante — la interpolación $S_0 \to V$ de A-M12.

Estatuto de lo que hoy se calcula, dejado explícito [14:33–15:44]: el valor actual "es como la utilidad total que hay extraíble, si tú no tuvieras que preocuparte por el presupuesto"; greedy "usa una diferente cantidad de pruebas"; "son medidas diferentes" — dos medidas imperfectas, y la invocación de greedy es "una manera súper imprecisa". El framing honesto queda registrado.

## D4 — Arreglo a la medición del costo: greedy local y posterior a la prueba [15:44–16:34]

La directriz operativa de mayor valor inmediato:

> "una manera de hacerlo es **no necesariamente correr greedy desde el principio**, sino como que primero tú tomas un conjunto, luego haces esa prueba y tiene cierto valor. Y luego tú como **simulas a greedy dentro de este conjunto, no global**… si quieres simular a greedy, tienes que primero simular el resultado y luego hacer greedy… aplicar greedy después de que aplicaste una prueba global. Que es básicamente simulando a futuro lo que haría si fuera aplicar greedy después de aplicar esta prueba aquí."

Procedimiento resultante para $C(T)$: fijar $T$ → aplicar la prueba → **simular el conteo $R$** → correr greedy **restringido a $T$** condicionado a $R$ → contar pruebas → promediar sobre $R$.

Responde la pregunta abierta del notebook 25 §8 ("¿el costo debería medirse con la continuación golosa, o con el plan de cobertura y búsqueda binaria?") con una tercera opción, y ataca la causa verificada de la degeneración: la medición actual (`costo_greedy_simulado`) corre greedy global "sobre m personas frescas" con presupuesto ilimitado, por lo que con $q<0.5$ el argmax de $q^k k$ es $k=1$ y el costo colapsa al número de personas sin resolver (assertion en `build_resultados_y_peticiones_notebook.py:608`).

## D5 — La tijera, confirmada [17:15–18:14]

Propuesta del equipo: "que el costo funcionara solamente contando el valor que todavía podemos alcanzar a cobrar de utilidad con las pruebas que te quedan. Si por ejemplo, si algo te cuesta más de lo que tienes, disponible de presupuesto, entonces en realidad ese valor no existe, porque nunca lo vas a poder alcanzar a cobrar". Francisco: "Sí, por eso **necesitamos la medida de cuántas pruebas cuestan**" [18:10–18:14]. Es el filtro de factibilidad de la fila del 2026-08-11 y la acreditación parcial de §14.6.

## D6 — Knapsack: el colapso 2D → 1D no tiene respuesta canónica [18:14–20:20]

La pregunta del divisor la abre el equipo ("¿qué pasa si el costo lo hacemos como un tipo divisor?" [18:14–18:31]). Francisco la ancla en literatura: "el ejemplo más cercano que yo conozco es en **subastas**… tienes el valor de conjuntos en subastas, pero tienes el tamaño… **todo esto es como que 'knapsack optimization'**". Las heurísticas: "tomas un conjunto y lo divides por el tamaño… es como que el '**bang per buck**'… hay otras donde, por ejemplo, si lo divides por la **raíz cuadrada** del tamaño… hay ejemplos en donde, si lo divides por el tamaño **al exponente 1.5**, es mejor". El diagnóstico: "tenemos dos objetivos… no son como que necesariamente, inmediatamente comparables. **Lo que tenemos que es como que colapsar esa optimización en dos dimensiones, a una dimensión.** Y **no hay una respuesta canónica**… es un espacio interesante. Y justo, ese es el problema."

Consecuencia: la regla de decisión no es un objeto único sino una **familia declarada** $V/C^\alpha$ ($\alpha \in \{1/2, 1, 3/2\}$ mencionados), más el filtro de factibilidad. Responde la pregunta (10) de §34. El equipo dejó formulados los dos paradigmas [20:20–21:12]: factibilidad ("si te está costando más de lo que tienes… eso no existe") y devaluación ("si esa utilidad viene a un costo muy alto de sacar… vale poco cada prueba que puedas realmente invertir").

## D7 — Correr las heurísticas ya, en paralelo a la escritura [23:21–24:02]

"lo bueno de todas estas heurísticas es de que debería de ser ojalá como que implementables de manera como medio sencilla… **aunque no sepamos cuál sea la respuesta correcta en teoría, podemos como que correr experimentos** con dividir por el tamaño o dividir por el presupuesto que toma o el presupuesto como que cuadrado… lo que podemos hacer es como que jugar con ellas por ahora."

Despacho: el barrido corre **ya, con estatuto de diagnóstico** (como la candidata E de §14.8); la adopción de un $\alpha$ como candidata $S_3$ sigue pasando por G4a/G4b. Sin reordenamiento de gates (fila en §32).

## D8 — Encargo: documento formal nuevo sobre lo laminar [21:22–23:21]

"siento que valdría la pena ahora como que escribir algo más formal sobre lo laminar… ahorita lo que estás haciendo son muchos apuntes, diferentes cosas. Yo creo que podemos como que volver a reenfocar" [21:22–22:03]. Forma concreta: "le podemos como que preparar un PDF, decir como que este es el modelo, esto es lo que quisiéramos hacer… **este es el espacio laminar**. Y la pregunta podría ser como que **existe un algoritmo de optimización dinámica que hace algo aquí, si no… hay un reto computacional fundamental**" [22:37–23:21]. Contenido pedido: "**reescribir el documento con el enfoque laminar en particular y con el contraejemplo que tenemos también** y simplemente como que irlo preparando".

**Sugerencia explícitamente opcional** [22:03–22:33]: usar demostración asistida por IA como punto de partida ("no necesariamente necesitamos hacerlo con Lean" [22:37–23:21]); "**lo pongo ante ustedes** por si quieren hacer algo o si quieren por ejemplo seguirle pensando sin eso, adelante… no lo digo por como rendirme del reto". Se registra como opcional, no vinculante; lo que produzca entra solo con validación de A y etiqueta de §25.

## D9 — Outline del paper, en documento nuevo [24:50–25:29]

"definitivamente reescribamos cosas… **destilemos** lo que hay en algo más. Ya tenemos mucho de que destilar ahí. **Incluso podemos escribir como que el outline de un paper ya.** Ya tenemos para la introducción, ya tenemos trabajo relevante. Tenemos el modelo… todo ya prácticamente está, lo que falta es irlo llenando con resultados." ¿Mismo documento u otro? — "**En el otro, yo diría, dejemos ese tal cual y hagamos otro**" [25:26–25:29].

## D10 — El paper hermano ya está en arXiv [24:10–24:43]

"les acabo de mandar la versión más nueva del paper la acabamos de poner en arXiv… ahora sí le pueden echar un ojo a eso… si tienen alguna pregunta de como los resultados más recientes, encantado de platicar de eso también."

Consecuencia: el cotejo del teorema estático $p>1/2$ deja de depender de Francisco — la fuente primaria es accesible al equipo. De ese cotejo depende levantar la cuarentena de C1 en prevalencia alta (fila §32 del 2026-08-02).

## Preguntas de §34

- **(10)** ¿filtro de factibilidad, cociente V/C, o knapsack? → **Respondida:** es knapsack; el filtro se confirma y el cociente no tiene forma canónica — familia $V/C^\alpha$ a barrer ($\alpha = 1, 1/2, 3/2$ mencionados).
- **(2)** reconfirmada [02:49–02:55], sin cambio.
- **(9)** ¿forma cerrada de $C(T)$ con $u \equiv 1$? → **no respondida**: Francisco reformuló la medición (D4), no la forma cerrada. Sigue abierta.
- **(11)** esperanza vs. cuantiles → no tocada.
- **Nuevas (12)–(17)** añadidas a §34: contraejemplo de referencia del documento (12), enunciado objetivo del PDF (13), congelación de $\alpha$ (14), teorema $p>1/2$ del arXiv (15), alcance del costo local (16), ancestros bajo la regla de subpruebas (17).

## Compromisos y pendientes

- **Nuevos del equipo, fechados 2026-08-18:** documento formal laminar (A-M22) · outline del paper en documento aparte (A-M19) · re-medición de $C(T)$ con greedy local post-prueba (B-M6) · barrido diagnóstico $V/C^\alpha$ (estatuto diagnóstico; adopción vía B-M9 tras G4a).
- **Modificado:** el cotejo del teorema estático $p>1/2$ pasa de Francisco al equipo (paper en arXiv, D10).
- **Sigue vigente:** reformulación de la Conjetura C (P21-A8).
- Despacho §34-bis aplicado el 2026-08-20: 7 filas en §32; A-M21/A-M22/B-M16 nuevos; candidata F en §14.8; pregunta 6 en §20; matriz 23.6; §1 reescrita para el 20–24 de agosto.
- **Adenda post-sesión (2026-08-20, mensaje al grupo):** Francisco envía la referencia **"Knapsack auctions"** e insiste en revisarla. Identificación C-M1 (por búsqueda; fuente primaria pendiente de lectura): Aggarwal–Hartline, SODA 2006 — subasta tipo knapsack con valuaciones privadas y tamaños públicos, benchmark de pricing óptimo omnisciente. Despachada a §21 (prioridad temprana C-M1) con nota de revisión en `docs/notes/2026-08-20-revision-knapsack-auctions.md`; A valida antes de incorporar cualquier claim.
