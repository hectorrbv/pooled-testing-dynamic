# La dirección y qué priorizar

## Qué es el proyecto cuando se lo explica a alguien de fuera

La charla en Boston College es la versión limpia de la tesis del proyecto, porque
al explicarle a alguien que no conoce nada hay que decir en voz alta lo que
normalmente se da por supuesto. La estructura que se enunció ahí es una cadena de
inclusiones: lo no traslapado dentro de lo traslapado, lo estático dentro de lo
dinámico, y encima de todo eso el eje binario contra aumentado. El resultado
existente es una separación estricta: en el momento en que se tiene dinámico y
aumentado a la vez, se le gana al mejor estático traslapado binario.

El motor de esa separación es el régimen de infección alta. Con pruebas binarias y
probabilidad de sano por debajo de 0.5, agrupar no sirve de nada, y eso está
demostrado en el otro paper. Las pruebas de conteo rompen esa barrera: agrupar G
personas y observar $R = G-1$ dice que hay exactamente un sano, y a partir de ahí
una búsqueda binaria lo extrae con certeza. La separación viene de que agrupando se
cubre más gente, y cubriendo más gente sube la probabilidad de que haya alguien
sano que la búsqueda binaria pueda cobrar. Ese es todo el argumento, y explica por
qué el ejemplo canónico tiene utilidad idéntica y enorme con probabilidad de sano
diminuta.

Lo laminar no es una elección estética. Se adopta porque con pruebas de conteo,
probar un subconjunto entrega gratis el conteo del complemento, y esa propiedad
solo se sostiene si las intersecciones entre pruebas son subconjuntos. El tensor de
subpools es la implementación de esa propiedad. Es infraestructura terminada, no el
resultado.

## La dirección declarada

La pregunta abierta que se enunció en Boston College tiene una forma muy concreta,
y no es "hagamos mejores algoritmos". Es diseño de función objetivo. El espacio de
árboles es enorme y no se puede optimizar sobre él, así que la política tiene que
ser golosa de un paso; pero un paso goloso sobre utilidad inmediata no recupera el
ejemplo canónico, porque la prueba grande no paga nada de inmediato. La pregunta es
entonces cuál es el objetivo por iteración que incorpora planificación sin dejar de
ser de un paso.

El candidato que se nombró es el número esperado de personas sanas que se cobrarían
si después se pudiera probar a todos gratis. Ese objeto es exactamente el mismo que
se pidió calcular en la sesión de Lowell House: para una prueba $T$ con resultado
$R$, el valor $V(R) = \sum_{i \in T} u_i q_i(R)$ y su esperanza $V(T) =
\mathbb{E}_R[V(R)]$. Las dos sesiones convergen en el mismo objeto por caminos
distintos, y eso es la señal más fuerte de hacia dónde va todo.

Encima de eso apareció la contribución de la profesora de Boston College, y la
reacción fue inmediata: si ese objetivo resulta submodular, el paso goloso hereda
garantía y existen relajaciones convexas para optimizarlo. Eso responde la única
pregunta que quedó sin respuesta al final de Lowell House, que fue "how to
optimize". La cadena completa es: $V(T)$ recupera el ejemplo, $V(T)$ es submodular,
por lo tanto el paso goloso es tratable y tiene garantía.

## Prioridades

Primero, calcular $V(T)$ de forma exacta en el ejemplo canónico y verificar que
prefiere la prueba grande sobre la individual. Es barato, está pedido explícitamente
en una sesión y es el supuesto sobre el que descansa todo lo demás. Si $V$ no
recupera el ejemplo, las prioridades siguientes se caen y hay que rediseñar el
objetivo.

Segundo, probar submodularidad y monotonía de $T \mapsto V(T)$. Empezar por
verificación numérica exhaustiva en poblaciones chicas, que es donde la máquinaria
exacta ya funciona, y buscar contraejemplos antes de intentar demostrar nada. Es la
pieza que convierte un método en un resultado publicable, y es la que se identificó
como el paso difícil.

Tercero, consolidar la evidencia de separación. La historia entera se apoya hoy en
un solo ejemplo, y eso se dijo en las dos sesiones. El barrido ya contiene una
segunda familia: con perfiles homogéneos y prevalencia alta, greedy no captura nada
del excedente dinámico mientras que un paso de anticipación captura casi todo. Es
justo el régimen que la charla identifica como el corazón del proyecto, así que
convertirlo en un ejemplo presentable vale más que ampliar el barrido.

Cuarto, y solo como higiene, escribir el generador de los dos CSV que no lo tienen y
entender por qué el rollout está inerte en la corrida de n=40.

## Lo que conviene bajar de prioridad

La política de ordenar ramas por valor extraíble quedó descartada en la misma sesión
donde se propuso, por bootstrapping: presupone que ya se hizo una primera prueba
grande inteligente, que es precisamente el problema. La tubería de n=40 con
partículas y MILP no está lista para sostener conclusiones, porque la política
laminar pierde 11% contra el control plano y el rollout no está activo. Y ampliar el
atlas a más celdas no responde ninguna de las preguntas abiertas; el atlas ya
contiene más de lo que se ha leído.
