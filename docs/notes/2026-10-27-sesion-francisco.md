# Sesión con Francisco (2026-10-27, 19:00 Praga / 11:00 CDMX)

**Fuentes.** Fathom de la llamada (44 min; "Francisco Marmolejo Cossio" = Francisco; "Héctor BV" = el micrófono del equipo, A y Héctor juntos; se atribuye por contenido: A en [4:48–5:42, 23:48–24:44, 28:34, 29:19, 42:48], Héctor presenta el notebook 30 en [25:00–28:34]). Se cita [m:ss]. Sin guion congelado del equipo (el de Héctor, `2026-10-27-guion-hector.md`, sirvió de base); sin WhatsApp la víspera.

**Entregado:** notebook 30 de Héctor (paso de partir el potencial, exacto; diseños detectores $n\le16$; techo de información en la malla; barrido de la Proposición con semilla propia). **No entregado:** la respuesta escrita de A al mensaje del 13-oct, el lema de información, los contraejemplos escritos, la tabla $q=\tfrac12$, el enlace del Overleaf. Quinta sesión seguida sin pieza escrita de A. A lo dijo de frente [23:48–24:44, 42:48]; Francisco tampoco escribió su argumento [5:48, 24:32].

## Resumen ejecutivo

Sesión corta y honesta: ninguna de las dos partes avanzó (A: viaje a París, 18 h de camión por trayecto; Francisco: tres papers entregados el lunes). Francisco abre la conversación sobre el uso de IA en el trabajo: la usa como conversación, nunca entregaría algo que no entendiera y pudiera reescribir, y pide el nivel de confort del equipo; el equipo: extensión, no reemplazo, con el mismo límite. Reafirma el paper: formulación, separación factor $G$ (no aumentado vs aumentado), separación $\log G$ (laminar vs óptimo en $q=\tfrac12$) y algoritmos laminares de aproximación — "ya es un super paper… el primer capítulo de esta dirección"; cree que el ejemplo de $\log G$ es el peor caso y sugiere ver si salen ideas para atacarlo. Héctor muestra el paso de partir del potencial: Francisco confirma que para un átomo abierto es "contar distribuciones" y también factor constante. Compromisos de Francisco: compartir la estrategia de los códigos óptimos (el detalle que faltaba) y redactar un documento con el argumento. Abre el tema del destino (SODA es el mejor fit pero "de aquí a junio cambia mucho el mundo"; mandó una lista de congresos por mensaje) y tres extensiones futuras de pool testing (pruebas imperfectas; dictamen con cuatro utilidades; infecciones dependientes), más la intención de pedir un grant sobre la agenda. Próxima sesión: la semana entrante.

## D1. Semana sin avance [0:20–6:31, 23:48–24:44, 42:48]

A: "esta semanita sí fallamos… no tenemos mucha sustancia" [5:42]; "accountability mío por irme a la fuga del juego" [28:34]; "la siguiente es un bounce back… reponer lo que no hice… te mando el whats esta semanita" [42:48]. Francisco: "también de parte mía… ni siquiera tuve el tiempo de estarme escribiendo el argumento" [5:48]; "no se preocupen… no hay prisa" [43:50].

## D2. Uso de IA: la postura de Francisco y la del equipo [8:54–22:14]

Contexto: resultado anunciado por OpenAI sobre la conjetura del tiempo exponencial (SAT en tiempo subexponencial), "de hace 10 años hubiera sacado el Turing" [9:31–12:09]. Francisco: "quería checar qué tanto quisieran que usemos o no" [15:00]; su filosofía: conversaciones, no delegar el problema entero; "yo jamás entregaría algo para publicación que no entendiera exactamente cómo funciona, que pudiese exponer, explicar de dónde viene" [15:00–17:18]; "el bar: poder reescribirlo todo… haber internalizado las ideas a tal punto de que uno lo pudiese reescribir" [20:38–22:14]; analogía con equipos grandes (CERN); "en cuanto a este proyecto, vamos a empezar a ver qué se nos genera también" [22:17]. El equipo: "una extensión y no un reemplazo… poner un límite de hasta aquí la estoy usando" [17:18–19:47]. **Lectura:** coincide con §25 (dos vías; código IA sin segunda vía = diagnóstico) y con la regla de A de reescribir todo en sus palabras; se registra como política acordada, no como directriz nueva.

## D3. El paper y el peor caso [22:17–23:48, 28:57]

"La pregunta de que hay un factor logarítmico entre laminar y óptimo: al menos tenemos un ejemplo de esa separación; sí creo que es el peor de los casos; sería interesante ver si salen ideas para atacar eso" [22:17]. Contenido del paper: "la formulación del problema, la separación logarítmica, la separación con factor de $G$ entre no aumentado y aumentado, y algunos algoritmos de aproximación laminares; eso ya es para mí un super paper… el primer capítulo de esta dirección de investigación; lo importante es poner algo para que la gente vea que se está haciendo" [22:17–23:48]. "¿A dónde mandamos un trabajo así? Estamos acercando a tener algo" [28:57].

## D4. El notebook 30 de Héctor y la respuesta de Francisco [25:00–28:34]

Héctor: el potencial implementado y graficado en $q=\tfrac12$, $u=1$; "el valor esperado cada que se abre un [grupo virgen] es 1" [26:02]; experimentos con varios $n$, $G$, $B$; $n=6$ y $n=15$ frente al óptimo laminar; "sin estrategia" [27:07]. Francisco: "es de los grupos vírgenes; para un átomo abierto con resultado $R$ es un poco más complicado, es contar todas las distribuciones, pero es un factor constante también" [26:13–26:19]; "la comprobación nos va a decir que en probabilidades intermedias ninguna prueba laminar va a tener mejor que un factor constante del presupuesto" [26:19–27:07]; "qué padre ver empíricamente el promedio del cambio en el potencial" [27:58]. **Ítem de acción:** "el otro detallito que faltaba la vez pasada: la estrategia que utilizan los códigos óptimos; lo lleno y se los comparto" [27:58].

Evidencia en el repo (notebook 30, `results/sesion_2026-10-26/notebook_30/`): V2 confirmada en 638 tripletas $m\le12$; nueve diseños detectores con razón laminar/$(n/2)$ 0.875–0.958 en $n\le8$, 0.828 en $n=15$, 0.858 en $n=16$; techo de información en 24 celdas, ninguna por encima de $\log_2K$; barrido de la Proposición, semilla 20261026, 1,766 casos, 0 fallas (lectura y firma pendientes).

## D5. Destino y tiempos [38:08–42:48]

"El congreso más apto es SODA, el simposio de algoritmos discretos, pero es hasta muchísimo después; de aquí a junio cambia mucho el mundo; no sé cuál va a ser el barómetro en seis meses" [38:08]. Mandó por mensaje una lista de otros congresos (no está en el repo; A la pega) [41:07–42:35]. **Ítem de acción (equipo):** investigar congresos y fechas límite.

## D6. Líneas futuras y doctorado [29:19–37:50]

El equipo: interés en blockchain/subastas "cuando acabemos esta línea" [29:19]. Francisco: tres extensiones abiertas de pool testing, "preguntas muy profundas, posiblemente un paper cada una": (1) pruebas aumentadas imperfectas ("cambia completamente todo; no podemos hacerlo laminar porque ya no se reparten los átomos así"); (2) dictamen final con cuatro utilidades (atinar sano/infectado y los dos errores; "nadie lo ha hecho"); (3) infecciones dependientes [30:49–34:51]. Varias publicaciones en una agenda demuestran expertise para doctorados; pedirá un grant sobre pool testing ("no prometo ser exitoso"), que daría funding para doctorados [35:13–37:04]. Nada de esto es para este año [34:51].

## D7. Cierre [43:09–43:47]

Francisco: "yo genero algún documento para que lo empecemos a ver con este argumento… quizá eso nos da algo que digerir durante varias semanas" [43:09]. El equipo: manda el Overleaf [43:34]. Próxima reunión la semana entrante.

## Verificaciones del soporte IA (2026-10-29)

- **V1.** Las cifras del notebook 30 coinciden con sus CSV; única discrepancia: el guion de Héctor dice 1,774 casos del barrido y el archivo registra 1,766.
- **V2.** En las 24 celdas, $\mathbb E[\text{despejados}]\le\log_2K$ sin el $+1$: coherente con la versión por entropía del lema ($\mathbb E[m]\le H(\text{historial})\le\log_2K$). La versión elemental de A (suma de colas) da $+1$; la pregunta (38) para Francisco es si su argumento quita el $+1$.
- **V3.** "Para un átomo abierto es factor constante" [26:19] es compatible con V2 del acta del 20-oct ($\le m/(m-1)\le2$); él no dio número.

## Preguntas de §34

(35): **REFORZADA** — "sí creo que es el peor de los casos" [22:17], sin prueba. (36): sin número; sigue. (33): sigue; él compartirá "la estrategia de los códigos óptimos" [27:58]. (1), (4): **REFORZADAS** — contenido del paper [22:17–23:48]. **Nuevas:** (37) destino y fecha: de tu lista, ¿cuál congreso y qué deadline fijamos? (38) ¿tu argumento del potencial quita el $+1$ del techo promedio?

## Compromisos y pendientes

- **De Francisco (27-oct):** compartir la estrategia de los códigos óptimos [27:58] · documento con el argumento y direcciones [43:09] · lista de congresos (enviada por mensaje). Siguen: escrito del potencial (20-oct); números de Lindström; (33).
- **Del equipo (27-oct):** WhatsApp con contraejemplos y párrafo (A dijo "esta semanita") · Overleaf · investigar congresos y fechas · **todo lo del 22-oct sigue vigente**: párrafo de A, lema de A, reconstrucción del potencial (Héctor), enunciado de Lindström (Héctor), firma de la nota "individuales" (Héctor: barrido hecho, lectura pendiente).
- **Válvula.** Quinta sesión seguida sin pieza escrita de A. Semana del 30-oct: *bounce back*: la base sale el viernes; A no promete nada fuera del párrafo y del lema.
- Despacho §34-bis **aplicado el 2026-10-30** con aprobación de A: tres filas en §32, §1 (semana del 30-oct), §9 (C12), §26 (A-M19, B-M11, B-M17), §30 (30.7, 30.8), §34 (35) y nuevas (37)(38), compromisos de §34-bis; plan `docs/notes/2026-10-30-plan-hasta-sesion.md`.
