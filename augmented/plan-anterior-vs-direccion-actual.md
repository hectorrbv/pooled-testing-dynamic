# El plan del 27 de julio contra la dirección de hoy

## Lo más importante: el plan predijo este pivote y lo disparó él mismo

El plan cerraba con un punto de decisión para el día 3, con dos ramas y una regla
para elegir entre ellas según lo que dijera el falsificador de anidamiento:

| Resultado del falsificador | Rumbo |
|---|---|
| Anida de forma no trivial en régimen relevante | Rama A: greedy laminar v1 completo |
| Anida solo en p muy alto o casi nunca | Rama B: re-centrar en alta prevalencia |

El falsificador corrió y dio lo segundo, y de forma más tajante de lo que el plan
contemplaba: con perfiles homogéneos y probabilidad de infección de 0.4 en
adelante, el greedy nunca anida, con fracción de territorio virgen igual a 1.000
exacta. En heterogéneo anida entre 4% y 20%.

Así que todo lo que se ha estado haciendo desde entonces es Rama B ejecutada al
pie de la letra, aunque nadie lo haya dicho con esas palabras. E4 es la familia
de alta prevalencia. B1 es la descomposición dentro de esa familia. La regla del
plan funcionó: el falsificador era un experimento diseñado para cambiar
decisiones, y las cambió.

El plan también anticipó el riesgo y lo desactivó por adelantado: "el
falsificador da nunca anida — no es fracaso, es EL resultado; la sesión lo
presenta como tal y el día 3 tiene rama B lista".

## La calibración de expectativas resultó correcta

El plan advertía, en las reglas de sesión, que había que decir en voz alta que el
greedy con tensor iba a dar solo uno o dos puntos, porque el peldaño de scoring
es alrededor de un cuarto del hueco, y que el premio real era habilitar rollout y
lookahead, que son los otros tres cuartos.

Los números de hoy confirman esa calibración con más nitidez de la que tenía
entonces. En la familia de alta prevalencia el greedy captura cero del excedente
dinámico y el rollout de un paso captura todo, exactamente, desde
p = 2 - raíz(2). El premio estaba donde el plan decía.

## Lo que se entregó, con una divergencia menor

El tensor con caché Φ y forma cerrada está hecho y validado contra enumeración
exhaustiva. Los tres checks que Francisco dictó en sesión están como tests con
sus nombres: columnas que suman uno, columna del pool entero indicadora, y ley de
soporte. La demo reproducible existe. El falsificador existe y produjo la curva.
El barrido de literatura de la tarea 7 existe en `docs/notes/2026-07-27-revision-QGT.md`.

La divergencia es de ubicación, no de contenido: el plan decía extender
`laminar_inference.py` y el código acabó en `laminar_tables.py`. No importa para
el resultado, pero conviene saberlo al buscar.

La demo sí cambió de guion. El plan tenía tres actos: el pizarrón de 11 personas
con conteo 10 y subprueba de 5, el tensor de bolsillo, y los tiempos de G=5 y
G=10. La demo actual muestra el tensor, la relación con la caché y la reutilización
al dividir un pool. El ejemplo del pizarrón, que es el que Francisco trajo a la
sesión, y la tabla de costos de G=10, que responde su pregunta sobre escala, ya
no están. Si va a haber sesión, valdría la pena recuperarlos.

## Dos tareas de Rama B que siguen sin hacerse y hoy valen más que entonces

La primera es B10', la curva fina de anidamiento en p entre 0.5 y 0.9 con el
óptimo por programación dinámica al lado, para contestar si el óptimo anida donde
el greedy no. Hoy vale más porque E4 y B1 dejaron establecido que en ese régimen
el greedy iguala al estático y el óptimo está estrictamente arriba. Falta saber
si la diferencia es precisamente que el óptimo sí anida. Si la respuesta es que
sí, se cierra el argumento completo: la jerarquía sirve, el greedy no la usa, y
por eso pierde.

La segunda es B11', que el plan describía como la comparación que nadie ha
corrido y es genuinamente informativa: creencias exactas con acciones laminares
contra creencias por producto de marginales con acciones libres. Separa el valor
de calcular bien del valor de poder elegir cualquier pool. Sigue sin correrse, y
encaja con la brecha de independencia ya medida.

## Lo que la dirección de hoy trajo y el plan no podía anticipar

Tres cosas, todas posteriores. El objetivo de valor de planificación y su colapso
por la ley de esperanzas totales, que salió de la sesión de Lowell House. La
submodularidad adaptativa como marco y el contraejemplo que la refuta, que salió
de la charla en Boston College con la profesora de allá. Y la revisión de novedad
contra los dos papers propios, que fijó qué parte del trabajo es contribución y
qué parte es cimiento.

## Lo que el plan descartó y sigue bien descartado

Las tablas incrementales padre a hijos: se enunció la herencia de la caché como
observación y no se construyó maquinaria, con el argumento de que noventa
kilobytes no ameritan optimización prematura. Branch and bound, por la objeción
de que las cotas no podan diferencias del cinco por ciento; hoy sabemos que la
cota está entre 0.70 y 0.94, así que la objeción se sostiene. Y el pipeline de
n=40, declarado fuera del horizonte, que es exactamente donde sigue.
