# Submodularidad, y si esto ya se hizo antes

## Qué es submodular, con antenas

Imagínense que tienen antenas y cada antena cubre un conjunto de casas. La
función f(S) es cuántas casas quedan cubiertas si prenden el conjunto S de
antenas.

Prendan la antena A sola: cubre 100 casas. Ahora prendan primero B y C, que
cubren un montón, y luego agreguen A: A ya solo aporta 20 casas nuevas, porque
las otras 80 ya estaban cubiertas. La ganancia de A se encogió al meterla en un
conjunto más grande.

Eso es todo. Submodular quiere decir rendimientos decrecientes:

  f(S + i) - f(S)  >=  f(T + i) - f(T)   siempre que S este contenido en T

La ganancia marginal de agregar i nunca crece cuando el conjunto ya es más
grande. Y monótona quiere decir que agregar nunca daña, f(S + i) >= f(S).

## Por qué le importa a alguien

Porque hay un teorema viejo y precioso, de Nemhauser, Wolsey y Fisher (1978): si
f es submodular, monótona y f(vacío) = 0, entonces el algoritmo goloso —agarrar
en cada paso el elemento de mayor ganancia marginal— consigue por lo menos
1 - 1/e, o sea 63%, del óptimo, con presupuesto de k elementos. Y Feige (1998)
mostró que ese 63% es lo mejor que puede hacer cualquier algoritmo eficiente,
salvo que P = NP.

O sea que la submodularidad convierte al greedy de heurística en algoritmo con
garantía. Ese es todo el atractivo.

## La versión adaptativa, que es la que nos toca

El teorema de arriba es para escoger un conjunto de golpe. Aquí no: aquí ustedes
prueban un pool, ven el conteo, y recién entonces deciden el siguiente. Eso es
adaptativo.

La generalización es de Golovin y Krause, "Adaptive Submodularity: Theory and
Applications in Active Learning and Stochastic Optimization", JAIR 2011. Las
condiciones se llaman adaptive monotone y adaptive submodular, y se enuncian
sobre la ganancia marginal ESPERADA condicionada al historial observado. Si se
cumplen, el greedy adaptativo consigue 1 - 1/e del óptimo adaptativo, o sea de la
mejor política del árbol completo, no del mejor conjunto fijo.

Eso es exactamente lo que este proyecto querría: greedy laminar con garantía
contra el mejor árbol.

## La tensión que hay que resolver antes de invertir

Vale la pena decirlo de frente, porque me parece el riesgo principal.

El ejemplo canónico funciona por complementariedad, que es lo contrario de los
rendimientos decrecientes. La prueba grande sola no paga nada: paga solo cuando
va acompañada del binary search que viene después. Dos acciones que juntas valen
mucho más que la suma de lo que valen por separado es justamente lo que rompe
submodularidad. El caso de libro es adivinar un número por bits: saber el bit
alto solo no sirve, saber el bit bajo solo tampoco, y juntos identifican.

Así que "meter planificación en el objetivo" y "que el objetivo sea submodular"
están en tensión, y no es una tensión menor. La literatura ya lo sabe: Krause y
Guestrin, "Near-Optimal Nonmyopic Value of Information in Graphical Models",
UAI 2005, muestran que el valor de información no es submodular en general, ni
siquiera en modelos Naive Bayes.

La salida conocida existe y es la receta que yo seguiría: Chen, Javdani, Karbasi,
Bagnell, Srinivasa y Krause, "Submodular Surrogates for Value of Information",
AAAI 2015. En vez de pelearse con el objetivo natural, construyen un sustituto
que sí es adaptive submodular y que domina al original. La pregunta de
investigación se vuelve entonces cuál es el sustituto submodular correcto para
el modelo de conteo con restricción laminar, y eso ya suena a un paper.

## Si esto ya se hizo antes

La respuesta corta es que las piezas existen todas por separado, la combinación
no existe, y el proyecto es literalmente la sección de trabajo futuro de su
propio paper anterior.

Lo que ya existe y nada más habría que aplicar. El teorema de Golovin y Krause
(JAIR 2011). El modelo estático de bienestar: Finster, González Amador, Lock,
Marmolejo-Cossío, Micha y Procaccia, "Welfare-Maximizing Pooled Testing", EC'23
(arXiv:2206.10660), que es el paper del que todo esto cuelga: población
heterogénea en utilidad y probabilidad, bienestar como suma de certificados
sanos, presupuesto fijo, tests binarios y asignación estática. Las pruebas de
conteo tampoco son nuevas como modelo: son el coin weighing problem con báscula,
planteado por Shapiro en el American Mathematical Monthly en 1960, y hoy se
llaman quantitative group testing.

Lo que es combinación nueva. Acoplar conteo, dinámico, bienestar y laminaridad no
lo ha hecho nadie. Y hay una continuación directa que conviene tener a la mano:
Lopez, Marmolejo-Cossío, Tello Ayala y Parkes, "Dynamic Welfare-Maximizing Pooled
Testing", arXiv:2601.22419, enviado en enero de 2026. Ese paper ya hace la parte
dinámica, pero con tests binarios y con un greedy puramente miope evaluado solo
empíricamente. En su trabajo futuro pide dos cosas explícitamente: extensiones a
resultados de prueba más ricos que el binario, y cotas teóricas sobre el valor
del testeo dinámico. Esos son exactamente los dos huecos que este proyecto
ataca.

Lo que sería genuinamente nuevo. Primero, la prueba de que el objetivo con valor
de planificación es adaptive submodular en el modelo de conteo con restricción
laminar. No existe en la literatura y, por lo de Krause y Guestrin, no es
corolario de nada: es riesgoso y puede salir falso. Segundo, una separación
formal, no nada más empírica, entre dinámico con conteo y el mejor estático
binario en prevalencia alta. Tampoco existe. Esas dos piezas son la contribución
real; lo demás es reensamblaje honesto.

Una nota sobre el régimen de prevalencia alta, corregida. El folclor de que
agrupar deja de servir cuando la prevalencia sube está por todos lados desde
Dorfman (1943) y no encontré un teorema externo citable que fije el umbral en 0.5
para este objetivo de bienestar. Pero en la charla de Boston College se dice
explícitamente que ese resultado ya está demostrado en el paper propio que se
estaba enviando. No es un hueco en la literatura esperando a que alguien lo
llene: es un resultado del grupo. Sirve como cimiento del argumento, no como
contribución nueva.
