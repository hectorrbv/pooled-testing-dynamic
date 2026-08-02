# Plan de estudio con Codex

Nueve lecciones, una por sesión, en orden. Cada una trae el prompt que se pega
tal cual, lo que la respuesta tiene que contener para ser buena, y una pregunta
de control cuya respuesta ya está escrita aquí. La pregunta de control es lo
importante: si Codex la falla, la lección no sirvió y hay que repetirla.

## Cómo arrancar cada sesión

Antes de la primera lección, pegar esto una vez:

> Estoy estudiando un proyecto de investigación sobre pooled testing con pruebas
> de conteo. Vas a explicarme conceptos de uno en uno. Reglas: empieza siempre
> con un ejemplo numérico concreto antes de cualquier fórmula; usa números
> chicos; no uses LaTeX, escribe la matemática en línea con ^ y *; si algo no se
> sigue de lo que te di, dilo en vez de inventarlo. Contexto del proyecto en
> `augmented/el-proyecto-desde-cero.md`. No avances al siguiente concepto hasta
> que yo te lo pida.

Para las lecciones 7 a 9 conviene además darle acceso al repositorio, porque hay
que mirar código y datos.

## Lección 1. El problema y la regla de cobro

Prompt:

> Explícame el setup: n personas, cada una con utilidad u_i y probabilidad q_i de
> estar sana, un budget de B pruebas, grupos de tamaño máximo G. Solo se cobra la
> utilidad de alguien si queda certificado sano con certeza. Dame un ejemplo con
> 4 personas y 2 pruebas, y calcula el welfare de dos estrategias distintas.

Buena respuesta: distingue certificar de adivinar, y deja claro que una persona
con posterior 0.99 de estar sana vale cero si no está certificada.

Control: si pruebo a Ana sola y sale sana, ¿cobro su utilidad? ¿Y si pruebo a Ana
y Beto juntos y sale que hay un infectado, cobro algo? Respuesta: sí en el
primero; cero en el segundo, porque no sé cuál de los dos es.

## Lección 2. Por qué agrupar no sirve con pruebas binarias

Prompt:

> Con pruebas binarias (solo dicen limpio o sucio) y todos con la misma
> probabilidad q de estar sanos, compara probar a una persona sola contra probar
> a un grupo de g. Deriva cuándo conviene agrupar y evalúalo en q = 0.3 y en
> q = 0.8.

Buena respuesta: individual da u*q, grupo da g*u*q^g, y agrupar conviene solo si
g*q^(g-1) > 1. Con g = 2 eso pide q > 0.5.

Control: con q = 0.3, ¿qué da más, una individual o un par? Respuesta: la
individual, 0.3u contra 0.18u.

Ojo con la trampa de notación: q es probabilidad de estar SANO. q < 0.5 significa
prevalencia alta. Si Codex lo invierte, corregirlo ahí mismo.

## Lección 3. Qué cambia con las pruebas de conteo

Prompt:

> Ahora la prueba devuelve R, el número exacto de infectados en el grupo, en vez
> de limpio o sucio. Con un grupo de 5 y R = 4, ¿qué sé que no sabría con una
> prueba binaria, y qué puedo hacer con eso?

Buena respuesta: sé que hay exactamente un sano ahí adentro, y con binary search
lo encuentro en unas log2(5) ~ 3 pruebas más, con certeza.

Control: con la prueba binaria, ¿puedo distinguir "todos infectados" de "uno
sano"? Respuesta: no, y por eso no puedo planear nada.

## Lección 4. El ejemplo canónico y la separación

Prompt:

> Población grande, todos idénticos, u enorme y q = 0.001. Budget de 11 pruebas,
> grupos de hasta 1024. Compara el welfare de la mejor estrategia estática
> binaria contra una estrategia dinámica con pruebas de conteo que use una prueba
> grupal y luego binary search.

Buena respuesta: la estática da 11*u*0.001 = 0.011u; la dinámica cubre 1024
personas con una prueba, encuentra un sano con probabilidad
1 - 0.999^1024 ~ 0.64, y gasta las otras 10 en sacarlo. Factor de casi 60.

Control: ¿de dónde sale la ganancia exactamente? Respuesta: de que se cubren G
personas pagando log(G) pruebas. La cobertura crece exponencialmente contra el
budget.

## Lección 5. Greedy y el bootstrapping

Prompt:

> Greedy escoge en cada paso la prueba que maximiza la utilidad esperada
> inmediata. Aplícalo al ejemplo anterior y dime qué prueba escoge en el primer
> paso y por qué eso es un desastre. Después explícame qué quiere decir que el
> problema es de bootstrapping.

Buena respuesta: la individual da u*0.001 y la grupal da 1024*u*0.001^1024, que
es cero, así que greedy siempre hace individuales. Bootstrapping quiere decir que
greedy sabe perfectamente qué hacer una vez que ya hay una prueba grande con
conteo bajo sobre la mesa, pero nunca la hace él, porque la primera no paga nada
de inmediato. Sabe continuar y no sabe arrancar.

Control: ¿greedy calcula mal? Respuesta: no, calcula bien la cantidad
equivocada. Por eso el arreglo va en el objetivo, no en el algoritmo.

## Lección 6. Qué es V(T) y por qué se necesita

Prompt:

> Para un grupo T que sale con resultado R, define V(R) como la suma sobre los
> miembros de u_i por la probabilidad posterior de que i esté sano dado ese R, y
> define V(T) como el promedio de V(R) sobre los R posibles. Interprétalo en
> palabras, calcúlalo en un grupo de 4 con R = 3, y explícame qué problema del
> objetivo miope pretende resolver.

Buena respuesta: V(T) es cuánta utilidad podría cobrar de ese grupo si después
pudiera probar a todos gratis. Con 4 personas y R = 3 hay exactamente un sano,
cada uno con posterior 1/4, así que B(3) = 4*u*(1/4) = u. Sirve porque premia
utilidad localizada y no solo utilidad cobrada, que es justo lo que le falta a
greedy.

Control: la trampa está en la siguiente lección, no se la adelantes.

## Lección 7. Por qué V(T) tal como está no puede funcionar

Esta es la lección importante, y conviene hacerla como pregunta abierta antes de
dar la respuesta.

Prompt:

> Toma la V(T) de la lección anterior. Por la ley de esperanzas totales, ¿cuánto
> vale el promedio sobre R de P(i sano | conteo de T = R)? Sustituye eso en la
> definición de V(T) y dime a qué se reduce V(T). ¿Qué implica para su utilidad
> como objetivo?

Buena respuesta: el promedio de la posterior es la prior q_i, así que
V(T) = suma de u_i * q_i sobre el grupo. O sea que V(T) no depende del resultado
de la prueba en absoluto: es la utilidad esperada del grupo y nada más. Como
objetivo siempre dice "agarra el grupo más grande con las mejores u_i * q_i", sin
distinguir una prueba informativa de una que no lo es.

Si Codex no llega solo, empujarlo con: V(R) es lineal en las posteriores, y el
valor de la información necesita convexidad.

Control: ¿por qué entonces V(T) sí da la respuesta correcta en el ejemplo
canónico? Respuesta: porque ahí la respuesta correcta es agarrar el grupo más
grande, así que acierta por accidente. En cuanto haya heterogeneidad va a
preferir siempre el grupo de q altas sobre el grupo informativo.

Segunda parte, en el mismo hilo:

> Proponme tres variantes de V(T) que sean convexas en las posteriores y explica
> para cada una por qué sí premiaría una prueba informativa.

Las tres que ya están sobre la mesa son: el promedio del máximo u_i*q_i(R) del
grupo; la utilidad localizada descontada por las log(g) pruebas que cuesta
extraerla; y el peor caso dentro del grupo.

## Lección 8. Laminar y el tensor de subpools

Prompt:

> Explícame qué quiere decir que una familia de pruebas sea laminar, y por qué
> con pruebas de conteo esa restricción sale gratis en información. Después lee
> `augmented/laminar_tables.py` y `augmented/demo_tensor.py` y dime qué calcula el
> tensor de subpools y qué guarda la caché.

Buena respuesta sobre lo conceptual: laminar es que dos pruebas cualesquiera o
son disjuntas o una contiene a la otra, nunca se cruzan a medias. Con conteo, si
sé que en T hay R infectados y pruebo un pedazo S con resultado r, sé gratis que
en el resto hay R - r. Esa contabilidad solo cierra si los grupos se anidan.

Control: ¿por qué no se puede hacer lo mismo con dos grupos que se cruzan?
Respuesta: porque la resta ya no identifica un conjunto, y las posteriores dejan
de factorizarse.

## Lección 9. Submodularidad y qué resultado se espera

Prompt:

> Explícame qué es una función submodular con un ejemplo de cobertura, qué
> garantiza el algoritmo goloso sobre funciones submodulares monótonas, y por qué
> en este proyecto interesaría demostrar que el objetivo por paso es submodular.

Buena respuesta: submodular es rendimientos decrecientes, meter un elemento a un
conjunto chico ayuda más que meterlo a uno grande. Sobre submodular monótona con
restricción de cardinalidad, el goloso garantiza 1 - 1/e del óptimo. Interesa
porque el paso goloso pasaría de heurística a algoritmo con garantía, y además
abre las relajaciones convexas para optimizarlo de verdad.

Control: ¿qué pregunta abierta contestaría eso? Respuesta: cómo se optimiza el
objetivo por paso, que es lo único que quedó sin respuesta al final de la sesión.

## Los resultados que el proyecto espera

Vale la pena tenerlos claros desde el principio, porque cada lección apunta a
uno. Son cuatro, en orden de ambición:

Un objetivo de un paso que recupere el ejemplo canónico. Es el mínimo: una regla
implementable que sí escoja la prueba grande cuando debe. Sin esto no hay nada.

Una demostración de que ese objetivo es submodular y monótono. Convierte la
heurística en algoritmo con garantía de 1 - 1/e y es lo que hace publicable el
método.

Más de un ejemplo de separación. Hoy toda la historia se apoya en uno solo. En
el barrido ya hay una segunda familia, la de prevalencia alta con perfiles
homogéneos, donde greedy no captura nada del excedente y un paso de anticipación
captura casi todo.

Evidencia empírica a escala. Que el método corra en poblaciones grandes y le gane
a los controles. Ahora mismo no le gana, y esa es la parte más lejana.
