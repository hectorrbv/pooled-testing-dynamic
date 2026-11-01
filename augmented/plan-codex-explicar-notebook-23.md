# Plan con Codex: entender el notebook 23 para explicárselo a Francisco

Ocho lecciones, una por experimento, en el orden del notebook. El objetivo no es
entender el código sino poder defender el resultado en voz alta, así que cada
lección termina con la objeción que Francisco va a poner y con la respuesta.

La regla para saber si una lección funcionó: poder decir en una frase qué corre,
en una frase qué salió, y en una frase por qué importa, sin mirar el notebook.

## Cómo arrancar

Pegar esto una sola vez al abrir la sesión:

> Voy a estudiar un notebook de investigación para poder explicarle los
> resultados a un colaborador que conoce el problema pero no el código. El
> notebook es `augmented/notebooks/23_experimentos_pendientes.ipynb` y el script
> que lo genera es `augmented/notebooks/build_experimentos_pendientes_notebook.py`.
> Contexto del proyecto en `augmented/el-proyecto-desde-cero.md`. Vamos sección
> por sección; no avances hasta que yo lo pida. En cada sección: primero léela,
> después explícame qué computa el código en términos del modelo (no línea por
> línea), después qué significa el número que sale. Si algo del texto no se sigue
> del código, dilo.

## Lección 1 — E1, el colapso del valor de planificación

Qué corre: calcula V(T) con el tensor sobre 200 pools aleatorios y lo compara
contra la suma de u_i por (1 menos p_i).

Qué sale: error máximo 1.78e-15, o sea que son el mismo número.

Por qué importa: mata la propuesta de objetivo que salió de la sesión, antes de
que nadie invierta en implementarla.

Prompt:

> Lee la sección E1. Explícame qué es V(R), qué es V(T), y por qué el promedio
> sobre R de la posterior devuelve la prior. Después dime por qué eso vuelve
> inútil a V(T) como objetivo, y en qué sentido preciso "no depende de la
> prueba".

Objeción de Francisco: "pero el ejemplo canónico sí lo recupera". Respuesta: lo
recupera porque ahí la respuesta correcta es agarrar el pool más grande, así que
acierta por accidente; en cuanto haya heterogeneidad va a preferir siempre el
grupo de q altas sobre el grupo informativo.

## Lección 2 — E2, qué objetivo sí escoge el pool grande

Qué corre: cinco objetivos candidatos evaluados como función del tamaño del pool
en el ancla q = 0.001 con presupuesto 11 y pools de hasta 1024.

Qué sale: la utilidad inmediata y el promedio del máximo tienen su máximo en
g = 1; la V lineal, la descontada por log g y la utilidad extraíble lo tienen en
g = 1024.

Por qué importa: acota el espacio de diseño del objetivo a dos o tres candidatos
con sentido.

Prompt:

> Lee la sección E2. Para cada uno de los cinco objetivos dime qué mide, por qué
> crece o no con el tamaño del pool, y cuál de los tres que crecen lo hace por la
> razón correcta. Explícame por qué el promedio del máximo sale plano en el caso
> homogéneo.

Objeción: "la utilidad extraíble ya usa el presupuesto restante y el costo del
binary search, o sea que ya es medio valor de continuación". Respuesta: sí, y ésa
es justo la pregunta abierta de la sección; el límite entre un paso goloso
enriquecido y programación dinámica disfrazada no está trazado.

## Lección 3 — E3, la submodularidad adaptativa falla

Es la lección más importante y la que más cuesta. Vale la pena partirla en dos
sesiones si hace falta.

Qué corre: con cuatro personas y probabilidad de estar sano 0.05, compara la
ganancia marginal esperada de cada acción en la raíz contra la misma acción
después de observar un conteo, sobre 546 pares admisibles.

Qué sale: la monotonía adaptativa se cumple siempre; la submodularidad no, con
120 violaciones genuinas y 96 contables. La peor: observar conteo 1 en un pool de
3 sube la ganancia de probar un par de 0.0050 a 0.6667.

Por qué importa: cierra la ruta directa a la garantía de 1 menos 1/e para el
goloso adaptativo.

Prompt:

> Lee la sección E3. Primero explícame qué es adaptive submodularity en el
> sentido de Golovin y Krause, con un ejemplo de cobertura y sin fórmulas.
> Después dime qué compara exactamente el código, por qué excluye los pares donde
> la acción coincide con la observación, y verifica a mano la aritmética de la
> peor violación: con q = 0.05, ¿por qué la ganancia de un par en la raíz es
> 0.005, y por qué después de ver conteo 1 en un trío es 2/3?

Segunda parte, en el mismo hilo:

> ¿Por qué este contraejemplo es el mismo mecanismo que hace funcionar al ejemplo
> canónico? Relaciónalo con la idea de complementariedad y explícame por qué
> complementariedad y rendimientos decrecientes son opuestos.

Objeción: "las 96 contables son trampa". Respuesta: no son trampa pero tampoco
hacen falta; bajo la definición formal también son violaciones legítimas, y se
separan solamente para mostrar que el resultado no depende de la contabilidad de
acreditados. Con las 120 genuinas basta.

## Lección 4 — E4, la familia de separación con umbral exacto

Qué corre: con n = 5, presupuesto 3 y pools de hasta 2, perfiles homogéneos,
compara óptimo, greedy, rollout y greedy estático para p de 0.5 a 0.9.

Qué sale: el greedy iguala al greedy estático dígito por dígito; el óptimo está
entre 11% y 20% arriba; el rollout alcanza el óptimo exactamente desde
p = 2 menos raíz de 2, que es 0.5858. Las formas cerradas son 3q para el greedy y
q por (3q² menos 3q más 4) para el rollout.

Por qué importa: convierte un ejemplo suelto en un régimen con umbral analítico,
y es el segundo ejemplo de separación que hacía falta.

Prompt:

> Lee la sección E4. Explícame por qué el greedy vale 3q, por qué la política de
> probar primero un par y luego individuales vale q(3q² − 3q + 4), y deriva el
> umbral donde la segunda supera a la primera como óptima. Después dime por qué
> el greedy coincide exactamente con el greedy estático en todo el rango.

Objeción, y es la que Francisco ya puso en sesión: "en prevalencia alta todo
debería coincidir, greedy laminar, greedy estático y óptimo". Respuesta: los dos
primeros sí coinciden y tenía razón en eso; el óptimo no, y ahí está el resultado.
Su teorema es sobre diseños estáticos y el óptimo dinámico puede superarlos.

## Lección 5 — E5, el umbral de 0.5 con utilidades heterogéneas

Qué corre: 60 instancias heterogéneas donde todos tienen probabilidad de estar
sanos por debajo de 0.5, comparando el mejor diseño estático exacto contra tomar
las B mejores personas individualmente.

Qué sale: cero contraejemplos, más una demostración por cota de unión.

Por qué importa: es cimiento, no contribución. El enunciado ya está demostrado en
el paper que el grupo estaba enviando; conviene saberlo antes de presentarlo como
hallazgo.

Prompt:

> Lee la sección E5 y reconstruye la demostración: define a_i y x_i, aplica la
> cota de unión, explica por qué cada pool aporta a lo más |T| / 2^(|T|−1) cuando
> todos los q_j están por debajo de 1/2, y por qué eso acota el bienestar por la
> suma de los B mayores a_i. Después dime cuáles son los cuatro supuestos y da un
> contraejemplo mínimo al salirse de cada uno.

Objeción: "eso ya lo demostramos". Respuesta: exacto, y por eso la sección lo
presenta como verificación y no como resultado; lo único por confirmar es si la
versión enviada cubre utilidades heterogéneas y presupuesto.

## Lección 6 — B1, la línea base que faltaba

Qué corre: sobre la misma familia de E4, calcula los tres niveles: estático
binario, dinámico binario y dinámico aumentado.

Qué sale: el dinámico binario iguala exactamente al estático binario, o sea que
la adaptatividad sola aporta cero; el conteo aporta entre 9.3% y 24.3%.

Por qué importa: la auditoría de claims marcaba esta celda como pendiente y decía
que hasta tenerla la separación había que describirla como un cambio de dos
features. Ahora no solo se puede descomponer: una de las dos partes es cero.

Prompt:

> Lee la sección B1. Explícame por qué con prueba binaria y perfiles homogéneos
> la adaptatividad no compra nada, en términos del modelo y no del código. Después
> justifica las dos desigualdades de la cadena: por qué el estático es un caso
> particular del dinámico, y por qué la prueba binaria es un garbling de la de
> conteo.

Objeción: "eso es en n = 5, y la familia del paper es asintótica". Respuesta: es
cierto y está anotado como pregunta de la sección; hay que decidir si se reporta
el desglose en la instancia chica o se deriva el término dinámico binario en la
familia asintótica.

## Lección 7 — B2, la convención de acreditación

Qué corre: el ancla del paper, q = 0.1 con pools de hasta 16, bajo las dos
convenciones de presupuesto.

Qué sale: 0.9657u con crédito deductivo y 0.8147u con clearing estricto, contra
0.6u del estático. Las dos separan. Y la separación vive en una ventana de
presupuestos, de 6 a 9, porque la cota aumentada satura en u mientras el estático
crece lineal.

Por qué importa: desbloquea una decisión editorial y descubre un límite que no
estaba declarado.

Prompt:

> Lee la sección B2. Explícame de dónde sale la diferencia entre k + log2(G) y
> k + log2(G) + 1, por qué eso parte la cobertura a la mitad, y por qué la cota
> aumentada satura en u. Después dime en qué régimen de presupuesto deja de haber
> separación y si eso es un problema real del resultado o un artefacto de que la
> cota es floja.

Objeción: "la cota es floja, con presupuesto grande corres varias búsquedas
binarias". Respuesta: correcto, y por eso la saturación es del enunciado y no del
fenómeno; la pregunta abierta es si conviene refinar la cota o enunciar la
separación dentro de la ventana.

## Lección 8 — B3, lo que la suite verde no cubre

Qué corre: verifica que el archivo que valida el muestreador de Gibbs no tiene
funciones que pytest colecte, que dos CSV citados no tienen generador, y que en
n = 40 el greedy y el rollout devuelven números idénticos.

Qué sale: cero funciones test_ contra 136 en los otros trece archivos; los dos
CSV sin generador; greedy y rollout ambos en 10.069 contra 11.267 del control.

Por qué importa: es el único bloqueador que puede hundir un envío, porque las
afirmaciones a escala descansan en un muestreador cuya corrección no está
demostrada.

Prompt:

> Lee la sección B3. Explícame por qué que el rollout coincida con el greedy hasta
> el último dígito indica que no se está ejecutando, y no un empate. Después
> explícame el problema del muestreador: qué era el sesgo del swap-only, en qué
> consiste la corrección por caminos alternantes, y por qué que la irreducibilidad
> no esté demostrada es un riesgo distinto a que esté mal.

Objeción: "eso es higiene, no research". Respuesta: la parte del Gibbs no es
higiene; se pasó de un sesgo demostrado a una corrección no demostrada y no
vigilada por la suite, y toda afirmación a escala depende de ella.

## Cierre: el guion de diez minutos

Después de las ocho lecciones, pedirle a Codex esto:

> Con todo lo anterior, escríbeme un guion de diez minutos para presentarle estos
> resultados a un colaborador que conoce el problema. Orden por importancia, no
> por número de sección. Una frase por resultado. Que empiece por lo que cambia
> decisiones y termine por lo que hay que arreglar.

El orden que yo defendería: primero B1, porque cierra un pendiente de la
auditoría y la adaptatividad valiendo cero es más limpio de lo que nadie pedía.
Después E3, porque cierra una ruta y redirige el trabajo. Después E4, porque
convierte un ejemplo en un régimen con umbral. Después B2, que es una decisión que
alguien tiene que tomar. Y al final B3, que es lo que hay que arreglar antes de
enviar nada.
