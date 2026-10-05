# El proyecto desde cero

## 1. El problema

Imagínense que tienen 1000 personas en cuarentena. Cada una tiene una utilidad u
(lo que vale para el mundo dejarla salir) y una probabilidad q de estar sana.
Ustedes tienen un budget de B pruebas, y cada prueba se le puede aplicar a un
grupo de a lo más G personas.

La regla del juego es una sola: solo cobran la utilidad de alguien si lo
certifican sano con certeza. Si quedan dudas, se queda en cuarentena y no cobran
nada. Nadie se libera "probablemente sano".

Eso es todo el setup. La pregunta es a quién probar, y en qué grupos, para
cobrar la mayor utilidad posible.

## 2. Por qué agrupar casi nunca sirve con pruebas normales

Una prueba de las de toda la vida, binaria, le dice a un grupo: limpio o sucio.
Y como solo cobran cuando certifican, solo cobran cuando sale limpio, o sea
cuando todos los del grupo están sanos.

Hagan de cuenta que todos tienen la misma q. Sabemos lo siguiente:

- si prueban a una sola persona, cobran u con probabilidad q, o sea u*q;
- si prueban a un grupo de g, cobran g*u pero solo si todos salen sanos, o sea
  g*u*q^g.

Comparando las dos expresiones: agrupar conviene solo si g*q^(g-1) > 1. Con
g = 2 eso pide q > 0.5. Pónganle números: con q = 0.3, la individual da 0.3u y
el par da 2 * 0.09 = 0.18u. La individual gana.

De ahí sale el teorema del otro paper: si q < 0.5, agrupar no sirve nunca, y la
mejor estrategia estática es puras pruebas individuales. Y ojo con la
traducción, porque es donde uno se marea: q es probabilidad de estar sano, así
que q < 0.5 quiere decir que más de la mitad está infectada. El régimen de
infección alta es justo donde las pruebas binarias se rinden.

## 3. Qué cambia con las pruebas de conteo

Una prueba aumentada no dice limpio o sucio: dice R, exactamente cuántos
infectados hay en el grupo.

Imagínense un grupo de 5 y que sale R = 4. Con la prueba binaria eso hubiera
sido nada más "hay alguien infectado", inútil. Con conteo ustedes ya saben algo
enorme: hay exactamente una persona sana ahí adentro. Y con binary search, unas
log(5) ~ 3 pruebas más, la encuentran con certeza y cobran su u.

Esa es la grieta por donde entra todo el proyecto. Con conteo, un grupo sucio
sigue siendo informativo.

## 4. El ejemplo canónico, con números

Imagínense una población enorme donde todos son idénticos: u gigante y q
chiquita, digamos q = 0.001, o sea casi todo el mundo infectado. Budget de 11
pruebas, y grupos de hasta G = 1024.

Sabemos lo siguiente:

- la mejor estrategia estática son 11 individuales, y en expectativa cobran
  11 * u * 0.001 = 0.011u;
- la estrategia dinámica aumentada gasta 1 prueba en el grupo de 1024, y si hay
  aunque sea un sano ahí adentro, gasta las otras 10 en binary search y lo
  saca con certeza.

La probabilidad de que haya al menos un sano entre 1024 personas es
1 - 0.999^1024 ~ 0.64. Así que cobran 0.64u contra 0.011u. Es un factor de casi
60, y eso ya es una separación de verdad.

La idea de fondo, si se quedan con una sola frase, es esta: cubren G personas
pagando nada más log(G) pruebas. La cobertura crece exponencialmente contra el
budget, y mientras más gente cubren, más probable es que haya un sano; y cuando
lo hay, el binary search lo cobra seguro.

## 5. Por qué greedy no encuentra esa estrategia

Greedy es la regla obvia: en cada paso, hagan la prueba que maximiza la utilidad
esperada inmediata. Con los mismos números:

- la prueba individual da u * 0.001;
- la prueba de 1024 da 1024 * u * 0.001^1024, que es cero para efectos
  prácticos, porque necesitaría que las 1024 salieran sanas.

Entonces greedy siempre escoge la individual y nunca da el primer paso de la
estrategia buena. Ese es todo el problema, y por eso en las sesiones aparece una
y otra vez la palabra bootstrapping: en cuanto ya hicieron la prueba grande y ven
un conteo bajo, greedy sabe perfectamente qué hacer, y lo hace bien. Lo que no
sabe es arrancar.

También por eso la miopía es la palanca. No es que greedy calcule mal; es que
está midiendo la cosa equivocada.

## 6. La idea de arreglo: cambiar lo que se premia

Si el objetivo solo premia utilidad cobrada, la prueba grande vale cero. La
propuesta es premiar también la utilidad localizada.

Imagínense un grupo de 4 y que sale R = 3. No cobraron nada, pero ahora saben
que hay exactamente un sano entre esos cuatro, y cada uno tiene posterior 1/4 de
ser el sano. La utilidad que está ahí adentro es 4 * u * (1/4) = u, y con dos
pruebas más la cobran.

Eso se formaliza así: para una prueba T que sale con resultado R,

  V(R) = suma sobre i en T de u_i * q_i(R),   con q_i(R) = P(i sano | conteo = R)

y el valor de la prueba es V(T) = promedio de V(R) sobre los R posibles. En
palabras: si mágicamente pudiera probar gratis a todos los de este grupo después,
¿cuánta utilidad sacaría en promedio?

## 7. Un problema con esa idea, tal como está escrita

Aquí hay algo que conviene revisar antes de implementar nada.

Por la ley de esperanzas totales, el promedio sobre R de P(i sano | conteo = R)
es simplemente q_i, la prior. Y como V(R) es una suma lineal de esas
posteriores, al promediar queda

  V(T) = suma sobre i en T de u_i * q_i

y eso ya no depende del resultado de la prueba, ni de la estructura, ni de nada.
Es nada más la utilidad esperada del grupo. O sea que maximizar V(T) siempre dice
"agarra el grupo más grande con los mejores u_i * q_i", sin importar qué tan
informativa sea la prueba.

En el ejemplo canónico eso da la respuesta correcta, porque ahí la respuesta
correcta es justamente agarrar el grupo más grande. Pero la da por la razón
equivocada, y en cuanto las utilidades y las probabilidades sean heterogéneas va
a preferir siempre el grupo de q altas sobre el grupo informativo.

La causa es conocida: el valor de la información siempre necesita convexidad. Si
f es convexa, el promedio de f(posterior) es mayor que f(prior), y esa diferencia
es exactamente lo que vale enterarse. Si f es lineal, la diferencia es cero. V(R)
es lineal, así que V(T) no puede detectar información.

Lo que hay que hacer, entonces, es meterle convexidad. Hay varias formas
naturales y todas se mencionaron en las sesiones sin nombrarlas así:

- premiar el máximo en vez de la suma, o sea el promedio sobre R del mejor
  u_i * q_i(R) del grupo;
- descontar por lo que cuesta extraer, tipo la utilidad localizada dividida entre
  las log(g) pruebas que hacen falta para cobrarla;
- usar el peor caso dentro del grupo en vez del promedio.

Cualquiera de esas rompe la linealidad y sí premia la prueba grande por ser
informativa, no nada más por ser grande.

## 8. Dos piezas más que aparecen todo el tiempo

Lo laminar. La restricción es que si prueban un grupo, después solo prueban
subgrupos de él, nunca grupos que lo crucen a medias. Se ve arbitrario pero no lo
es: con conteo, si ya saben que en T hay R infectados y prueban un pedazo S y le
sale r, entonces saben gratis que en el resto hay R - r. Esa contabilidad solo
cierra si los grupos se anidan. Todo el trabajo del tensor de subpools es la
implementación de esa propiedad.

La submodularidad. Es la palabra que trajo la profesora de Boston College y es la
que amarra el proyecto. Una función es submodular si tiene rendimientos
decrecientes: meter una persona más a un grupo chico ayuda más que meterla a uno
grande. Si el objetivo por paso resulta ser submodular y monótono, entonces
maximizarlo golosamente ya viene con garantía de 1 - 1/e, y además hay
relajaciones convexas para optimizarlo de verdad. Eso contestaría la única
pregunta que quedó abierta al final de la otra sesión, que fue cómo diablos se
optimiza este objetivo.

## 9. El resumen en cuatro renglones

Hay un régimen, el de infección alta, donde las pruebas normales no pueden hacer
nada y las de conteo sí. Ahí se demostró una separación con un ejemplo. La
estrategia que la logra necesita planear, y greedy no planea, así que no la
encuentra. La apuesta del proyecto es diseñar un objetivo de un solo paso que sí
la encuentre, y ojalá demostrar que ese objetivo es submodular, porque entonces
se optimiza con garantía y eso ya es un buen paper.
