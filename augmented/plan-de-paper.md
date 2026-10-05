# El paper que ya existe, y las cuatro cosas que lo bloquean

La respuesta corta es que sí hay paper, pero no es el que estábamos persiguiendo.
No es un paper de garantía 1 - 1/e. Es un paper de separación, estructura y
tratabilidad, y sus piezas ya están demostradas y auditadas.

## La espina dorsal

El ancla es la separación: en el régimen homogéneo con probabilidad de estar sano
por debajo de 1/2, el mejor plan binario fijo vale B*u*q por cota de union,
mientras k pools disjuntos de tamaño G con busqueda binaria guiada por conteo
valen al menos u por (1 - (1-q)^(kG)). El conteo le gana estrictamente al binario
justo donde el binario se rinde.

De ahi cuelga la pregunta de si con conteos se puede computar algo, y ahi estan
los dos resultados que le dan cuerpo al paper. Por un lado la inferencia exacta es
#P-dura en general, por reduccion desde Exact Cover. Por otro, sobre historias
laminares es exacta y rapida: los atomos residuales particionan el soporte
probado, los conteos de pools y de atomos se determinan mutuamente, y los
marginales salen por polinomios simetricos elementales, con error de 2e-16 contra
fuerza bruta a n=12 y n=6000 procesado en 13 ms. Esa dicotomia es la justificacion
honesta de por que se restringe a laminar, y hoy esta escrita como si fuera una
conveniencia.

La tercera pieza es el lema de monotonia por refinamiento de canal, demostrado por
garbling determinista por etapas: si Q' refina a Q entonces U_Q <= U_Q', y en
particular el binario queda por debajo del conteo en toda la jerarquia. Eso
convierte el eje binario-aumentado de una intuicion en un teorema.

La cuarta es la brecha de independencia, que es el argumento cuantitativo de por
que no se puede usar el producto de marginales y hay que cargar con el posterior
exacto: distancia en variacion total de 0.275 en promedio y hasta 0.60 sobre un
nodo ya observado, y exactamente cero sobre pools disjuntos.

La quinta es la parte de politicas: la Proposicion B, que el rollout replanificado
domina al greedy dentro de la misma biblioteca laminar, con demostracion y con el
atlas exacto de 2592 instancias donde se cumple en el 100% de los casos. El mismo
atlas da el costo de exigir laminaridad, que es 0.7% en promedio y 7.2% en el peor
caso, e igual a cero en el 53% de las instancias.

## Las cuatro cosas que lo bloquean

La primera es barata y es la mas importante. El audit marca como pendiente la
politica dinamica binaria optima sobre la familia de separacion. Sin ella la
separacion cambia dos cosas a la vez, estatico a dinamico y binario a conteo, y no
se puede atribuir la ganancia al conteo. La maquinaria ya existe:
`classical_solver.solve_classical_dynamic` calcula ese optimo exacto hasta n=14 y
ya esta cableada en `hierarchy_experiment.py`. Nadie la corrio sobre la familia de
separacion. Es una tarde.

La segunda es una decision, no trabajo. La separacion esta calificada por una
inconsistencia de convencion de recompensa: bajo hard clearing estricto hace falta
una prueba extra, asi que el presupuesto es k + log2(G) + 1 y la cobertura optima
es 2^(B-2) y no 2^(B-1). El 0.966u que circula usa la convencion de credito
deductivo; bajo clearing estricto el ancla da 0.815u contra 0.6u. Sigue separando,
pero el numero publicable es otro. Hay que elegir una convencion y recalcular.

La tercera es la que puede hundir un envio. El sesgo del Gibbs swap-only estaba
demostrado y se corrigio con Metropolis-Hastings sobre caminos alternantes con
correccion de camino espejo, pero la irreducibilidad de esa cadena no esta
demostrada en todas las fibras. Se paso de un sesgo demostrado a una correccion no
demostrada. Encima `tests_gibbs_validity.py` es un script con `__main__` y sin
funciones `test_`, asi que pytest no lo colecta y la suite verde no lo cubre. O se
demuestra la irreducibilidad, o toda afirmacion del paper se cerca a inferencia
exacta y laminar y el Gibbs baja a apendice con la limitacion declarada.

La cuarta es higiene de reproducibilidad: `arbol_vs_miopia.csv` y
`greedy_vs_static_greedy.csv` no tienen generador versionado, y en el pipeline de
n=40 el rollout esta inerte, con medias identicas al greedy hasta el ultimo digito
y ambos 11% por debajo del control plano. Nada de n=40 es reportable asi.

## Lo que hay que dejar fuera

RL es deuda, no apendice: hay dos modelos PPO guardados y cero metricas en el
repo, ninguna comparacion contra el DP ni contra el greedy, y la rama clasica esta
marcada como abandonada.

El 84.5% del canal de tres niveles es una sola instancia elegida dentro de un
barrido de ocho instancias disenadas con n<=6. El lema de monotonia si es
publicable; ese numero es apendice como maximo y sin lenguaje causal.

Los certificados U_pen son un caso aparte. El teorema es real y esta demostrado:
la penalizacion es una diferencia de martingala bajo la filtracion natural, asi que
la cota es valida para cualquier V-hat y V-hat solo mueve la tension. Pero la
calidad numerica es modesta, con el cociente entre optimo y cota entre 0.70 y 0.94
a n=4, la version penalizada exacta solo corre a n<=6, U_cell solo cubre
instancias homogeneas, y el gap insignia sigue en 2.086 contra un objetivo
declarado de 1. Como seccion de metodo dentro de este paper distrae. Como paper
propio, si el gap baja, es una contribucion aparte.

## La incomodidad que un referee va a encontrar

Con perfiles homogeneos y prevalencia por encima de 0.4 el greedy nunca anida: la
fraccion de decisiones en territorio virgen es 1.000 exacta. O sea que el laminar
dinamico degenera en un diseno estatico precisamente en el regimen que motiva el
paper. No conviene esconderlo. Es la razon por la que hace falta planificacion, y
se combina bien con el resultado negativo de que el objetivo de bienestar no es
adaptive submodular: juntos dicen que el paso goloso no puede llegar solo y que la
ruta directa a una garantia esta cerrada. Eso es una seccion final honesta y es
tambien la agenda del siguiente paper.
