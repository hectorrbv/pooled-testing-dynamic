# Sesión con Francisco (2026-10-20, 19:00 Praga / 11:00 CDMX)

**Fuentes.** Fathom de la llamada (49 min; Speaker 1 = Francisco; el micrófono del equipo aparece como "Héctor BV": la sesión se tomó desde la computadora de Héctor con A y Héctor juntos, así que lo atribuido al equipo es de ambos; en voz, Héctor). No hay transcripción del celular. Se cita [m:ss]. Lo que Francisco escribió en la tablet no está en la transcripción: las fórmulas de abajo son reconstrucción a partir de sus palabras, verificadas por el soporte IA el 2026-10-21 (ver "Verificaciones").

**Guion previo:** `docs/notes/2026-10-13-guion-sesion.md`, no re-fechado ni congelado; no hubo WhatsApp la víspera. **Entregado por el equipo [38:06–46:47]:** dos ejemplos donde el binario óptimo gana a laminar (testigo $n=3$ y familia $n=4$ con límite $10/9$); la cadena del techo $B\log_2(G+1)$ dicha de palabra; la medición $n=15$, $B=9$, $G=7$ con matriz detectora frente al óptimo laminar; la regla laminar del loop Q1 frente a rollout. **No entregado:** la tabla $q=\tfrac12$ (está en el repo), la respuesta escrita de A al mensaje del 13-oct, el argumento $T_{BG}$ en palabras de A, la nota "individuales nunca óptimas". Cuarta sesión seguida sin pieza escrita de A enviada.

## Resumen ejecutivo

Francisco llega con la prueba, de palabra y "casi seguro", de que en $u\equiv1$, $q\equiv\tfrac12$ el aumentado irrestricto gana a laminar por un factor $\log G$: (i) toda política laminar vale $O(B)$, por una **función potencial** sobre los átomos cuyo cambio esperado por prueba es una constante; (ii) con matrices detectoras (Lindström, generalización de la búsqueda binaria) una política general identifica a todos y obtiene $\Omega(B\log G)$. Conjetura que ese $\log G$ es el factor óptimo de pérdida laminar frente al óptimo. Su ítem de acción: formalizar la prueba del potencial y subirla al Overleaf. El equipo presenta los dos ejemplos binario > laminar (Francisco: la primera prueba del binario "es una matriz decodificadora"), la cadena del techo $B\log_2(G+1)$ y la medición $n=15$; Francisco: las diferencias grandes se verán con poblaciones grandes, "lo óptimo crece como $\log G$ por presupuesto y lo otro como el presupuesto". Compromiso del equipo (ítem de Fathom): WhatsApp con los contraejemplos escritos y el párrafo del techo promedio. Próxima sesión la semana entrante.

## D1. Logística [0:16–2:42, 37:34, 47:14–49:19]

Semana pesada de Francisco; entrega otros papers el sábado/domingo [0:31, 47:18]. Sale temprano: "tengo unos cinco minutitos" [37:34]. El equipo: auditoría administrativa la semana pasada; con menos clases hay más espacio para esto [0:46–1:48]. Overleaf: "ya empezamos el overleaf a meter todo lo nuevo" (Héctor) [46:49]; Francisco: "avísenme y yo también le voy a ir agregando, ya cuando formalice bien este argumento para el logaritmo" [46:58]. "Quedamos para vernos la semana que entra" [47:18].

## D2. La tesis: $q=\tfrac12$ es el mal ejemplo para laminar; el gap es $\log G$ [1:48–5:05]

"Estoy casi seguro de que tengo una comprobación de que el ejemplo que estamos contemplando, de la mitad de la probabilidad de infección, es precisamente un mal ejemplo [para laminar]" [1:48]; "podemos comprobar que el dinámico tiene un factor de logaritmo de [G] mejor… [que] laminar" [2:42–2:58]. Setting: población de $n$ personas, todas con probabilidad $\tfrac12$, utilidad 1 [3:49]. Dos mitades: "acotar el desempeño de las estrategias laminares, lo más difícil de los dos lados", y "demostrar que una estrategia adaptativa tiene mejor desempeño", la fácil: "haces esas pruebas óptimas de lo de compressing… el paper de peso de monedas: con $\log G$ pruebas puedes saber exactamente quién… sabes con certeza quién está infectado y sacas toda la utilidad de la población… es una generalización de lo que hicimos con binary search" [3:49–5:05].

## D3. La función potencial [5:05–17:13]

Método de "Potential Function… que se usa mucho en diseño de algoritmos" [5:42]. Estado de una política laminar: átomos (pedazos disjuntos con conteo conocido) más la población no tocada; "el algoritmo Bellman que tenemos aprovecha esta descomposición" [6:56]. Potencial por átomo: premiar "casi saber que personas están sanas": átomo de 10 con 9 sanos "está medio caliente" [7:20–8:21]; la medida es la discrepancia sanos menos infectados, al cuadrado "para no lidiar con valor absoluto" y normalizada por el tamaño [8:32–9:51]. Con $m$ personas y $r$ infectados, discrepancia $m-2r$ [12:22–13:49] y potencial del átomo

$$\phi(m,r)=\frac{(m-2r)^2}{m},$$

[20:14]; potencial del historial = suma sobre átomos [15:18–16:11]. Primera propiedad: el potencial es $\ge$ la utilidad ya obtenida, porque un átomo totalmente negativo ($r=0$) aporta exactamente $m$, su tamaño [13:52–14:28, 16:15–17:11]; equivalentemente, "utilidad obtenida con certeza más la suma sobre átomos ambiguos" ($0<r<m$) [16:15–17:13].

## D4. Prueba virgen: el potencial sube 1 en promedio [17:13–25:09]

Con el prior balanceado, "en promedio tu potencial sube cero" [17:13–18:50]; corrección inmediata: "no es cero, por el cuadrado" [18:52]. Prueba virgen $T$ de tamaño $M$: el conteo $R$ es binomial$(M,\tfrac12)$ [21:42–21:56]; cambio del potencial $(M-2R)^2/M$ [22:15]; "usas los momentos de la binomial… esto es igual a $M$… la expectativa es igual a 1" [22:54–23:32]. Por qué solo 1: "es un proceso súper balanceado… casi todo está concentrado en la media… una prueba virgen no te va a dar más que uno en promedio" [23:58–25:09].

## D5. Partir un átomo: también constante; telescopio [25:10–29:12]

"El paso más sutil… partes un átomo que ya tiene cierta información" (siete sanos de diez: discrepancia cuatro); "nada más aplicación de identidades de distribuciones hipergeométricas… no desvía mucho de la media por ser completamente unbiased… tu discrepancia no va a crecer mucho en los subátomos" [25:10–28:00]. Cierre: el cambio total es la suma de los cambios por prueba, cada uno $\le$ constante en promedio, luego el potencial crece lineal en $B$; como el potencial acota la utilidad por arriba, "las pruebas laminares no pueden obtener más que un factor lineal en el presupuesto" [11:02–11:39, 28:01–29:12]. Héctor, paráfrasis aceptada: "ni el territorio virgen ni refinar átomos aportan información sustancial" [29:12–29:46].

## D6. Por qué el solapamiento escapa; Lindström [29:46–38:06]

"Cuando sí permites que solapen las cosas… pocas pruebas te pueden dar mucha información… por estas combinaciones lineales complejas… por eso este es el ejemplo problemático para lo laminar" [29:46–31:03]; Héctor: "el potencial no puede crecer tanto está relacionado con la restricción de no permitir overlaps" — "Exacto" [31:03–31:56]. "Los números de la construcción esta de Lindström que les compartí… es una generalización de lo que hicimos en pruebas binarias… las pruebas binarias son dentro de un factor constante del presupuesto en este caso" [31:58–33:07]; con presupuesto y población en cierta desigualdad "puedes saber exactamente quién está infectado y quién no… con las matrices detectoras" [33:30–36:12; detalles inaudibles; "no me acuerdo bien los detalles"]. "Hice los números en algún papel que tengo perdido… voy a llenar los detalles después… estoy casi, casi seguro que da un factor de logaritmo en G de diferencia concreta entre laminar y no laminar" [37:34–38:06]. **Conjetura:** "eso es de hecho el factor óptimo… nuestro factor de pérdida para laminar versus óptimo, pero habría que comprobar eso y es más complejo todavía" [38:11]. Observación de régimen: el caso donde laminar "es muy bueno" (tasa baja; frase con sonido ambiguo, "tasa de protección/infección muy baja") "no es para nada este caso" [33:07–33:30].

## D7. El equipo: dos ejemplos binario > laminar [38:06–41:25]

Testigo $n=3$, $B=2$, $G=3$, $u\equiv1$, $q=(0.9,0.7,0.7)$: binario 2.079 contra laminar 2.062 (irrestricto 2.121) [38:06–40:18; cifras del repo, `results/encargos_2026-10-06/`, segunda vía de Héctor 14-oct]. Familia $n=4$, $B=G=2$, $q=(s,\tfrac12,\tfrac12,s)$, $u=(1,3,3,2)$, $19/20\le s<1$: $V_{bin}=\tfrac52 s(1+s)$, $V_{lam}=3+\tfrac32 s^2$, cociente $\to10/9$; en $s=0.99$, 4.925 contra 4.470 (laminar/binario 0.9076) [39:02–40:18; certificado simbólico, `results/sesion_2026-10-12/brecha_ampliada/`]. Francisco: "en la primera prueba de la izquierda estoy casi seguro que con eso sabes exactamente quién está sano, no importa el resultado; es una de estas matrices codificadoras" [40:18]; "justo este es el ejemplo de lo que estaba mencionando" [41:15]. Héctor: "lo vamos a escribir bien para mandártelo" [40:56].

## D8. El equipo: el techo $B\log_2(G+1)$ y la medición $n=15$ [41:25–46:47]

Héctor dice la cadena: $m$ personas ⟹ $2^m$ estados; una política tiene a lo sumo $(G+1)^B$ historiales; luego el techo promedio $B\log_2(G+1)$; "ahí nos quedamos en cómo empezar la escritura"; "el parrafito… te lo mando hoy o mañana" [41:25–43:56, 44:44–46:47]. Francisco: "Ok, perfecto" [46:47]. Medición [43:56]: $n=15$, $B=9$, $G=7$, $q=\tfrac12$, $u\equiv1$, matriz detectora de Söderberg–Shapiro (1963): general $15/2$ (identifica a todos; $n/2$), óptimo laminar $203479/32768\approx6.210$ por Bellman pathwise con replay sobre los $2^{15}$ perfiles, individuales $9/2$; laminar/general $\approx0.828$ (`results/sesion_2026-10-20/matriz15_laminar/resultados.json`). Francisco: "las diferencias más grandes las vamos a ver conforme hay una población más grande… lo óptimo va a crecer como el logaritmo de $G$ multiplicado por el presupuesto y el otro como el presupuesto" [43:56]. Heurística laminar nueva (regla del loop Q1: mirada hacia adelante, diversidad por tamaño de pool) comparada con rollout y Bellman: "todavía no entendemos bien por qué" [44:44]; en $n=50$ no se distingue de rollout (`2026-10-20-lectura-loops-y-siguiente-experimento.md`).

## D9. Cierre [46:47–48:11]

"El plan inicial: si logramos demostrar entre laminar y dinámico ciertos bounds, ese sería el primer paso de la agenda de investigación; y adentro de laminar cuáles son los buenos algoritmos" [47:44–48:00]. Ítems de acción (Fathom): Francisco formaliza la prueba del potencial y la sube al Overleaf [46:47]; siguiente reunión por agendar [47:14]; el equipo manda por WhatsApp los contraejemplos y el párrafo del techo promedio [47:44–48:11].

## Verificaciones del soporte IA (2026-10-21; nada de esto se cita como de Francisco)

- **V1. Prueba virgen.** $R\sim\mathrm{Bin}(M,\tfrac12)$: $\mathbb E[(M-2R)^2]=4\,\mathrm{Var}(R)=M$, luego $\mathbb E[\Delta\phi]=1$ para todo $M$. Coincide con [22:54–23:32]. [VERIFICADO, aritmética]
- **V2. Partir un átomo.** Átomo $(m,r)$, subprueba de tamaño $k$, $X\sim\mathrm{Hipergeom}(m,r,k)$; sub-átomos $(k,X)$ y $(m-k,r-X)$. Con $\mathrm{Var}(X)=k\frac{r}{m}\bigl(1-\frac rm\bigr)\frac{m-k}{m-1}$:

$$\mathbb E\bigl[\phi(k,X)+\phi(m-k,r-X)-\phi(m,r)\bigr]=\frac{4\,r\,(m-r)}{m\,(m-1)}\le\frac{m}{m-1}\le2,$$

  independiente de $k$, igualdad a 2 solo en $m=2$, $r=1$. [DERIVACIÓN IA, por validar: Héctor con enumeración exacta en racionales, $m\le12$]
- **V3. Consecuencia.** Potencial inicial 0; cada prueba lo sube a lo sumo 2 en promedio; la utilidad bajo soft clearing es la suma de tamaños de átomos con $r=0$, que el potencial domina. Luego, en $q\equiv\tfrac12$, $u\equiv1$: $V^{*}_{lam}\le2B$, y con los singletons $V^{*}_{lam}\ge B/2$. [DERIVACIÓN IA sobre el argumento de Francisco; A lo reescribe; nada se cita hasta su prueba escrita]
- **V4. Lado general.** Nuestro techo $B\log_2(G+1)$ para cualquier política (hechos 1–3 en palabras de A, versión "siempre"; hecho 4, promedio, pendiente) y la construcción de Lindström son compatibles: el gap es $\Theta(\log G)$ si (ii) alcanza $cB\log G$. La medición $n=15$ no dice nada del orden: ambos valores están lejos de sus techos ($2B=18$; $B\log_2 8=27$).
- **V5. Lo que no está escrito.** Nada de Francisco está escrito; su ítem es formalizar. Lindström 1964/1965 y Cantor–Mills 1966 (coin weighing) son las fuentes probables; Söderberg–Shapiro 1963 es la matriz usada por Héctor. Pregunta (33) sigue abierta en la referencia exacta.

## Preguntas de §34

- (1), (4): **REFORZADAS** — la agenda en dos pasos [47:44–48:00]. (23): **PARCIAL → REFORZADA** — el punto de ruptura con conteos queda localizado en el potencial: sin cruces, cada prueba añade a lo sumo una constante [29:46–31:56]. (32): **RESPONDIDA de facto** — Francisco afirma $\log G$ por ambos lados y lo conjetura óptimo [1:48, 37:34–38:11]. (33): **PARCIAL** — "Lindström" nombrado [31:58]; referencia exacta y desigualdad presupuesto/población pendientes ("papel perdido"). (30), (31), (34): no se hicieron; siguen. (18), (24), (28), (29): siguen.
- **Nuevas:** (35) ¿$\log G$ es el peor caso sobre todo $(q,u)$ (tu conjetura del 20-oct)? ¿qué haría falta para probarlo? (36) ¿la constante del lado laminar es 2 (V2–V3) o mejor? ¿el potencial sirve con $q\ne\tfrac12$?

## Compromisos y pendientes

- **De Francisco (20-oct):** formalizar la prueba del potencial y subirla al Overleaf; los números de Lindström; agendar la siguiente sesión (la semana entrante). Siguen: referencia del paper de matrices (33); explicar la §5 si hay dudas; contacto con Alexei.
- **Del equipo (20-oct):** WhatsApp con los dos contraejemplos escritos y el párrafo del techo promedio ("hoy o mañana", Héctor); enlace del Overleaf en el repo. **Siguen:** respuesta escrita de A al mensaje del 13-oct (ahora: hechos 1–3 en sus palabras, promedio pendiente) · lema del techo en palabras de A (hecho 4) · $T_{BG}$ en palabras de A · nota "individuales nunca óptimas" (pasos 3–4, firma de B) · comentarios al paper antes de fin de octubre · PDF de la brecha · cotejo de A de la §3 · esqueleto `main.tex` frente al Overleaf.
- **Válvula.** Cuarta sesión seguida sin pieza escrita de A. Se mantiene la base mínima de una pieza por persona; la de A esta semana es el párrafo de respuesta ya trabajado en chat, no el lema completo.
- Despacho §34-bis **aplicado el 2026-10-22** con aprobación de A (2026-10-22): dos filas en §32, §0-bis (a), §7 (cadena en $q=\tfrac12$), §9 (C12), §20 (6), §26 (A-M17, A-M19, A-M22, B-M11, B-M17), §34 (23)(32)(33) y nuevas (34)(35)(36), compromisos de §34-bis, §1 (semana del 22-oct) y `docs/notes/2026-10-22-plan-hasta-sesion.md`.
