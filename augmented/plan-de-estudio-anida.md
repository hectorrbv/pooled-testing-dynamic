# Plan de estudio con Codex — el lema de anidamiento (misión ANIDA)

Siete lecciones para entender el resultado del 3 de agosto: el greedy
doble-homogéneo jamás anida por elección, por qué la prueba es más fina de lo
que parece, y qué compra eso para el proyecto. Mismo formato que
`plan-de-estudio-con-codex.md`: prompt que se pega tal cual, qué debe contener
una buena respuesta, y una pregunta de control con la respuesta ya escrita.
Material de referencia: la sección R3 del notebook 23 (todo se regenera ahí) y,
para las lecciones 6 y 7, `dapts-autoresearch/veredicto_anida.md`.

## Cómo arrancar cada sesión

> Estoy estudiando un lema sobre políticas golosas en pooled testing con
> conteos. Reglas: ejemplo numérico concreto antes de cualquier fórmula;
> números chicos; matemática en línea con ^ y *; si algo no se sigue de lo que
> te di, dilo. q_i es probabilidad de estar SANO (q < 0.5 es prevalencia
> alta). El score miope de un pool T es P(conteo de T = 0) por la utilidad no
> acreditada de T. No avances hasta que yo te lo pida.

## Lección 1. Las tres clases de acción

Prompt:

> Ya se probó el pool {a,b} y salió conteo 1. Quedan c, d, e sin tocar.
> Clasifícame las acciones posibles en tres clases: anidada (contenida en un
> pool ya probado), virgen (disjunta de todo lo probado) y mixta (lo demás), y
> calcula el score miope de {a}, de {c} y de {a,c} con todos a q = 0.3 y u = 1.

Buena respuesta: {a} es anidada y puntúa 1/2 (exactamente uno del par está
sano); {c} es virgen y puntúa q = 0.3; {a,c} es mixta y puntúa 2·(1/2)·q = 0.3.
El conteo 1 concentró la probabilidad: la anidada puntúa más que su prior.

Control: ¿por qué {a} puntúa 1/2 y no 0.3? Respuesta: porque el conteo 1 en el
par dice que exactamente uno de {a,b} está sano, y por simetría cada uno queda
con posterior 1/2.

## Lección 2. El lema en G=2, en dos líneas

Prompt:

> Con todos iguales (q homogénea, u = 1) y pools de a lo más 2: demuestra que
> el mejor score anidado posible es 1/2, que el mejor score virgen es
> max(q, 2q^2), y que max(q, 2q^2) >= 1/2 para todo q, con igualdad solo en
> q = 1/2. Concluye qué hace el greedy mientras quede territorio virgen.

Buena respuesta: la única anidada con score positivo es un miembro de un par
con conteo 1 (1/2); el virgen da q si q >= 1/2 y 2q^2... cuidado: max(q, 2q^2)
= q para q <= 1/2 y 2q^2 para q >= 1/2; su mínimo sobre q es 1/2, en q = 1/2.
El greedy nunca prefiere estrictamente anidar: con virgen disponible, anida
solo si hay empate exacto.

Control: el empate está en q = 1/2. ¿Ese estado lo alcanza el greedy?
Respuesta: no — en q = 1/2 el single y el par empatan a 1/2 en la raíz y el
desempate abre el single, así que el greedy nunca tiene un par observado ahí.
El punto tight de la desigualdad es frontera, no estado alcanzable.

## Lección 3. La reentrada en general: la hipergeométrica

Prompt:

> Pool de tamaño g ya probado con conteo R, todos iguales. Deriva que un
> subpool de tamaño k puntúa k * C(g-R, k) / C(g, k), y encuentra qué par
> (R, k) maximiza eso. Evalúalo en g = 4.

Buena respuesta: el subpool sale limpio si los R activos caen fuera de él,
que por conteo de combinaciones es C(g-R,k)/C(g,k). El máximo está en R = 1 y
k = piso(g/2), y vale piso(g/2)*techo(g/2)/g. En g = 4: k = 2 da
2*C(3,2)/C(4,2) = 1.

Control: en g = 4 con conteo 1, ¿cuánto puntúa reentrar con k = 3? Respuesta:
3*C(3,3)/C(4,3) = 3/4 — menos que con k = 2. Partir a la mitad es lo óptimo.

## Lección 4. Donde la desigualdad simple se rompe, y el rescate mixto

Prompt:

> n = 5, q = 4/5 homogénea, G = 4. El greedy abre {a,b,c,d} (4q^4 es el argmax)
> y sale conteo 1. Solo queda un virgen, {e}. Calcula el mejor score anidado,
> el virgen, y el de la acción mixta {a,e}. ¿Cuál es el argmax? ¿Qué le pasa a
> la "prueba de dos líneas" de la lección 2 aquí?

Buena respuesta: anidada {a,b} = 1.0; virgen {e} = 4/5; mixta {a,e} =
2*(3/4)*(4/5) = 6/5. La anidada SUPERA al virgen — la desigualdad simple se
rompe desde G = 4 — pero el argmax es la mixta, así que el greedy sigue sin
anidar. La razón general: score(anidada de tamaño k más un virgen reclutado) /
score(anidada) = q(k+1)/k, que es >= 1 exactamente cuando q >= k/(k+1).

Control: ¿la ruptura de la desigualdad simple refuta el lema? Respuesta: no —
refuta esa PRUEBA. El lema sobrevive porque donde la anidada gana al virgen,
la mixta le gana a la anidada.

## Lección 5. La pieza de alcanzabilidad, y el lema completo

Prompt:

> Demuestra que si el greedy homogéneo abre un pool de tamaño g (o sea, g*q^g
> fue el argmax sobre tamaños), entonces q >= (g-1)/g. Combínalo con la
> lección 4: si q >= (g-1)/g y k <= g-1, ¿qué pasa con q contra k/(k+1)?
> Ensambla el enunciado completo del lema.

Buena respuesta: g*q^g >= (g-1)*q^(g-1) da q >= (g-1)/g. Y k/(k+1) <= (g-1)/g
para todo k <= g-1, así que q >= k/(k+1) siempre: en todo pool que el greedy
mismo abrió, el reclutamiento mixto domina a cualquier reentrada. Lema: con p
y u homogéneas, mientras quede virgen, el argmax del greedy nunca es una
acción anidada — no porque anidar puntúe poco, sino porque el greedy solo
abre pools con q tan alta que reclutar domina.

Control: ¿el lema es sobre el score o sobre la trayectoria? Respuesta: sobre
la trayectoria — la restricción "el greedy mismo lo abrió" (q >= (g-1)/g) es
la que hace todo el trabajo. Esa distinción es la lección 6.

## Lección 6. H1': en estados ajenos, la cola greedy sí anida

Prompt:

> n = 3, q = 2/5, G = 2. Verifica que el greedy en la raíz abre un single. Ahora
> supón que OTRA política abrió el par {a,b} y salió conteo 1: calcula el score
> de la anidada {a}, del virgen {c} y de la mixta {a,c}, y di cuál es el argmax.
> Generaliza: ¿hasta qué q anida la cola greedy en un pool de tamaño g que le
> sembraron, y desde qué q lo abriría ella sola?

Buena respuesta: raíz: single 0.4 > par 0.32. En el estado sembrado: anidada
1/2 > virgen 2/5 = mixta 2/5 — el greedy anida por elección. En general anida
hasta q <= piso(g/2)/(piso(g/2)+1) y solo abre desde q >= (g-1)/g; las dos
regiones solo se tocan en g = 2, q = 1/2. Por eso el greedy nunca se
encuentra sus propios estados anidables.

Control: ¿qué tiene que ver esto con el ejemplo E4 del rollout? Respuesta:
todo — el rollout abre el par (un estado que el greedy solo jamás pisa) y su
cola miope reentra sola tras el conteo 1. La anticipación siembra; la cosecha
ya estaba en el greedy.

## Lección 7. Heterogeneidad: dónde y cuánto se rompe

Prompt:

> Muestra con números que basta heterogeneidad SOLO en u para que el greedy
> anide por elección: p_activo = 0.4 homogénea, u = (1, 1, 0.5, 0.5), n = 4,
> G = 2, B = 2. Después dime qué estructura mínima de heterogeneidad en q hace
> falta en general, y qué tan honda puede ser la cascada de reentradas.

Buena respuesta: el greedy abre el par de u altas (score 0.72), y tras conteo
1 la anidada puntúa 0.5 contra 0.36 del mejor virgen (el par de u bajas) —
anida. En general basta que dos individuos cabalguen el umbral 1/2 (uno
arriba para abrir, uno abajo para que el virgen no rescate); no hay
heterogeneidad mínima, el ínfimo es 0. La cascada llega a profundidad
piso(log2 g), con escalera exacta: profundidad D alcanzable cuando
q <= g/(g + 2^D).

Control: ¿por qué "no hay heterogeneidad mínima" no contradice el lema
homogéneo? Respuesta: porque el lema vive exactamente en el punto de
heterogeneidad cero, y ahí el empate cae en estados que el greedy no alcanza.
La frontera es discontinua en el límite, no el lema.

## Qué compra todo esto

Tres frases para tener a la mano. La miopía homogénea no es no-saber-volver:
es que volver nunca es el mejor pago inmediato mientras quede virgen (lema,
con prueba). La propiedad es de la trayectoria, no del score: sembrado el
estado por otra política, la cola miope cosecha sola — el mecanismo exacto por
el que un paso de anticipación recupera todo el excedente en E4. Y la
heterogeneidad la rompe por el umbral compartido 1/2, con cascada acotada por
piso(log2 g).
