# Acta — Sesión con Francisco (Lowell House), 2026-08-02

**Fuente:** `TRANSCRIPCION_LOWELL_HOUSE_ACTIVE.md` (Whisper large-v3-turbo; rótulos de
hablante inferidos — audio mono). **Disciplina de atribución:** se atribuye a Francisco
solo lo inequívocamente directivo; los pasajes donde la transcripción fusiona voces
(reportes de resultados nuestros dentro de turnos rotulados "Francisco") se citan como
"la conversación registra". El guion pre-sesión (A-M20) no se ejecutó para esta sesión:
las preguntas de §34 no se plantearon formalmente; toda respuesta es incidental.

**Caveat de transcripción:** [43:07] dice "utilidad alta y cierta probabilidad chica de
actividad", pero el mecanismo discutido ([36:38] "R = T−1 … tienes una persona sana";
[40:51–42:07] "sale 3, todos activos, el valor posterior … es 0") es el del régimen de
actividad alta del ancla (§16). Se toma como lapsus/ruido de Whisper: el ejemplo es el
motivador de siempre (q = P(sano) pequeño, u alta). Confirmado por A (2026-08-02): sí.

## Resumen

Sesión de trabajo sobre el estado del tensor y el greedy laminar. Francisco valida la
línea (tensor como objeto suficiente, greedy solo necesita la columna R=0), cuestiona
el resultado reportado de "mejor desempeño en tasas altas" por chocar con el teorema
estático de p>1/2, desarrolla en vivo una familia de heurísticas de planificación que
converge al diseño S3 del plan (valor por componente, selección de rama, descuento por
sanos detectados-no-localizados, costo log G de extracción), identifica el bootstrap
como el crux ("¿cómo tomas el primer paso inteligente si eres miope?") y deja UNA tarea
explícita: evaluar el objetivo V(T)=E_R[B(R)] en el ejemplo motivador, con cómputo
exacto, para la próxima sesión.

## Directrices de Francisco (citas fechadas)

**D1 — Tarea explícita para la próxima sesión.** Definición dictada [39:14–43:07]:
para una prueba T con resultado R, B(R) = Σ_{i∈T} u_i·P(sano_i | R) ("cuánta utilidad
puedes extraer de este grupo total en promedio", bajo presupuesto mágico); el objetivo
es V(T) = E_R[B(R)] — "esto no es utilidad inmediata, es un valor de planificación"
[42:07–43:07]. Encargo: "Una primera tarea, si quieren, para la próxima vez es esto…
¿qué tal el desempeño en nuestro ejemplo?" [38:56–39:14]. Hipótesis de Francisco: "es
posible que sea mejor tomar una prueba grande… cuando R es igual a 1 este valor explota…
quizá esto pueda regenerar el comportamiento del ejemplo que tenemos" [37:39–38:38,
42:07–43:07]. Reconoce el cambio de objeto: "antes estábamos maximizando la utilidad en
expectation… ahora estamos maximizando un objeto donde como que cambia" [38:40–38:54].

**[DERIVACIÓN — a validar por A]** Con utilidad homogénea, B(R) = u·(G−R) y por tower
property V(T) = u·|T|·q: el score es lineal en el tamaño y ciego al historial.
Consecuencia doble para presentar: (i) la intuición de Francisco acierta el bootstrap —
el score elige el pool máximo, el primer paso grande que S₀ nunca da; (ii) pierde la
extracción — en el ancla, abrir virgen puntúa 0.8u contra 0.5u de subdividir un pool
con sano detectado: nunca subdivide, nunca cobra (la martingala de §14.4b). El mensaje:
"tu objetivo resuelve el bootstrap y pierde la extracción; S3 conserva lo primero
valorando el plan de continuación realizable (φ_virgin), no el conteo esperado de
sanos".

**D2 — Auditoría del resultado "mejor en tasas altas".** "En la tasa de actividad
alta… no sé si puedes explorar las pruebas que se hacen… estoy casi seguro que si la
tasa de actividad es mayor al .5, la primera prueba que tiene que [hacer] greedy
laminar tiene que ser una individual" [12:23–14:03]; base: "en las pruebas estáticas,
para cualquier presupuesto… si el valor de actividad es mayor al .5… pooling no te
sirve para nada" [14:03–15:05]; consecuencia: "me extraña que… le vaya mejor que los
otros… todos deberían de coincidir en ese régimen: Greedy Laminar, Greedy Estático,
Opt" [15:46–16:26]. Compromiso de Francisco: "Lo voy a checar bien… lo voy a entregar
bien el paper" [12:23–14:03].

**D3 — Familia de heurísticas de planificación (converge a S3).** Recompensa
descontada: "te voy a premiar por persona sana que me encuentres, pero también si me
encuentras un grupo de cinco dentro de los cuales hay dos sanos… descontado… me estás
ayudando a planificar" [17:35–18:42]; realizabilidad: "en el momento en que tú
identificas una persona sana en un grupo de G… con logaritmo de G más pruebas puedes
encontrar a esa persona sana, o sea que sí puedes obtener esa utilidad" [18:45–19:32];
valor por componente y selección de rama: "cada uno de estos componentes tiene su
valor… ¿cuál branch decido explorar?… la que tiene el valor máximo… ya tienen todo el
material para implementarla, porque tienen la tabla" [22:02–23:50]; y su propia
limitación: "esto no nos ayuda para el caso anterior… es un problema de bootstrapping…
¿cómo haces algo inteligente sin saber nada de antemano… con perfiles homogéneos?"
[23:50–24:36]; "¿cómo tomas el primer paso inteligente si eres medio miope? … es el
crux" [24:34–25:21].

**D4 — Rollout.** "Habíamos visto… un algoritmo… rollout… se valdría la pena verlo…
eso es lo que está faltando: inyectar algo de planificación" [25:03–25:27, 31:27–31:55];
preocupación de costo: "hay un paso computacional difícil… el rollout optimization va a
ser creo que difícil, pero posible… hay muchas states… ¿cómo optimizas sobre las
states?" [28:35–31:09].

**D5 — Las dos preguntas rectoras.** "¿Qué tanto la planificación es necesaria para
extraer este surplus?… ¿hay otros ejemplos [de separación]?… y ¿existen variantes de
greedy que planifican un poquito… sin tener algo que no pueda correr? Esas son como que
las dos preguntas que quedan pendientes" [35:03–35:39]; contexto: "el único ejemplo que
tenemos que separa las dos cosas es uno donde todo lo demás está acá [abajo]"
[32:15–32:52]; dirección confirmada: "seguimos en esta dirección, pero ahora el tema es
cómo atacamos la miopía… con esta nueva maquinaria" [32:54–34:03].

## Resultados reportados por nuestro lado (la conversación registra; EN CUARENTENA hasta verificación CSV — §25; nada de esto entra a §9 todavía)

- Tensor implementado y verificado contra fuerza bruta [02:32–03:19].
- Greedy exacto vs heurística de independencia: desempeño final parecido; la heurística
  sobreestima pools; gap CONSTANTE con probabilidades homogéneas, 2X–6X heterogéneas
  [05:33–06:15] (candidato a explicación analítica vía A2.4/A2.6: en el caso homogéneo
  p se cancela en la hipergeométrica).
- Con B=2, el greedy exacto fue el óptimo en todos los casos probados [06:15–07:39]
  (alimenta §19 y C10; pendiente enumeración trazable).
- Descomposición del gap opt–greedy: ~1/4 independencia, ~3/4 miopía; casos muy
  pequeños [07:41–08:43].
- Corre a N=40 en minutos; diferencias 2%–10% [07:41–08:43].
- "Mejor desempeño en tasas altas" — cuestionado por D2; en cuarentena reforzada.

**Ruta de salida de la cuarentena:** cada número se verifica contra CSV trazable
(cadena B); entra a §9 con etiqueta [VERIFICADO n≤X] o se retira. Destinos específicos:
la observación B=2-óptimo alimenta la evidencia de §19 (A-M16); el gap constante bajo
probabilidades homogéneas es candidato a mini-lema vía A2.4/A2.6 (en homogéneo la
hipergeométrica cancela p; el gap de independencia es el factor f_{T∖S}(R)/f_T(R)) —
se anota en la nota del tensor (A-M3).

## Preguntas de §34

- (2) "¿qué relajación expresa mejor 'valor futuro'?" — PARCIALMENTE RESPONDIDA:
  la preferencia revelada de Francisco es el valor posterior de sanos identificables
  (D1), pero su propio argumento log-G (D3) apunta a utilidad acreditable con
  presupuesto residual — la posición de S3. Se cierra al presentarle la obstrucción.
  Se añaden a §34 las preguntas (5)–(8) surgidas de esta sesión.
- (1), (3), (4): no tocadas (el paquete §34 no se presentó formalmente).

## Compromisos fechados

- Nuestro: derivación en papel del colapso de V(T)=E_R[B(R)] + confirmación
  computacional exacta en el ancla + curva V(G) ilustrativa + comparación contra
  S₀/planificador — próxima sesión (origen: sesión 2026-08-02).
- Francisco: cotejar en el paper el teorema estático p>1/2 ⟹ individuales óptimas.
- Vigente (sin cambio): reformulación de la Conjetura C (P21-A8).

## Despacho

Directrices D1–D5 despachadas el 2026-08-02 al plan maestro
(`docs/plans/2026-08-01-plan-maestro-politicas-laminares-planificacion.md`):
cuatro filas en §32 (formato completo), anotación de la pregunta (2) y preguntas
(5)–(8) en §34, compromisos fechados en §34-bis, familia
"separación-con-greedy-competitivo" en B-M11 (§26), línea de lectura en §14.8.
El buffer de §34-bis queda vacío (placeholder); la historia queda en §32, en esta
acta y en git.
