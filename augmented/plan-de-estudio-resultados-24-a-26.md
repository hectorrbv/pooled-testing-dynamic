# Plan de estudio: resultados de los notebooks 24 a 26

Objetivo: poder explicar cada resultado sin abrir el notebook, reconstruir cada
número clave a mano, y contar la historia completa en cinco minutos frente a un
pizarrón. El plan asume cuatro sesiones de estudio de una a dos horas antes de
la sesión del 25 de agosto. Cada sesión termina con un criterio de dominio: si
no se cumple, la sesión no está cerrada.

Convención de estudio: q es la probabilidad de estar sano, el régimen de
interés es q < 0.5 (prevalencia alta), y `r` es el número de **infectados**
del pool. En los estados, `z_i = 1` significa “i está infectada”; por tanto una
prueba limpia tiene `r = 0` y es la única que acredita bajo hard clearing. Si
alguna celda heredada llama `R` al número de sanos, traducir antes de razonar:
`R_sanos = |T| − r`. El notebook 26 queda alineado a esta convención.

## Guía complementaria: notas de Vlad

Las notas en `../../notas vlad/` no sustituyen los notebooks: dan la intuición
y el lenguaje con que explicarlos. Úsalas en este orden.

1. **Ahora, para la sesión 3:** `Week 8 (18 to 25 August 2026).md`, §§1–4.
   “Conocido ≠ cobrado”, el menú de reentrada con uno o dos tests restantes,
   los dos mundos 1/2–1/2 y la diferencia entre el score V y S0. Es el mapa
   conceptual de los ejercicios de costo local y no-reentrada.
2. **Antes del simulacro:** `Week 6 Boston.md`, “La historia completa en cinco
   actos”, Hechos 1–2 y “De S0 y V a la extraíble”. Aporta el arco: V abre el
   pool, pero no paga por cosecharlo.
3. **Segunda pasada técnica:** `Week 4 Boston - Focus on Laminar.md`. Léelo
   para soporte de subpruebas, posteriores condicionados y la frontera
   laminar/no laminar; no es requisito para terminar el micro-caso del par.
4. **Después del arco 24–26:** `The threat.md` (átomos y laminaridad) y `The
   frontier.md` (fibra, Gibbs y caminos alternantes). Son la continuación de
   inferencia y escalabilidad, no un prerrequisito de V/C.

## Registro de avance — persona B (24 de agosto)

“Persona B” se usa aquí como diagnóstico operativo, no como una categoría
formal del repositorio: ya reconstruyes los números centrales con apoyo corto,
pero estás convirtiendo definiciones del modelo en decisiones del árbol.

| Tramo | Estado observado | Siguiente evidencia de dominio |
|---|---|---|
| Posterior del par con `r = 1` | Hecho: obtuviste 1/2 y entendiste la renormalización. | Decir que los dos mundos supervivientes pesan 0.21 y suman 0.42. |
| Score mágico V | Hecho: distingues suma de probabilidades de multiplicación y detectaste el retest degenerado. | Contrastar en una frase V (utilidad localizada) contra S0 (cobro inmediato). |
| Costo local del par | En curso: ya están las probabilidades 0.09, 0.42 y 0.49 y las ramas extremas. | Justificar `1` frente a `2` subpruebas en `r=1`, usando “solo r=0 acredita”; cerrar la media 1.5 y 3q(1−q). |
| No-reentrada, V/C y α | Pendiente de reconstrucción guiada. | Completar los ejercicios 4–5 y después §§6–8. |

Tu posición global es, por tanto, **a mitad de la sesión 3 de 4**: el
andamiaje probabilístico está firme; el próximo salto es usar la definición de
utilidad para explicar el árbol, antes de pasar a cocientes y α.

## El arco completo, primero

Los tres notebooks cuentan una sola historia y conviene tenerla antes de
entrar al detalle. El score que se propuso en sesión, V(T), se colapsa: vale lo
mismo que sumar priors, así que no distingue pruebas informativas de pruebas
inútiles (24, 25 §6). Cortar el árbol en el presupuesto arregla el colapso pero
cuesta lo mismo que resolver el problema (25 §7). Medir el costo simulando
greedy global degenera en nuestro régimen: greedy nunca agrupa y el costo es
contar personas (25 §8). La corrección dictada el 18 de agosto — greedy local,
dentro del conjunto y después de simular el conteo — produce un costo que sí
discrimina (26 §2). El contraejemplo de no-reentrada muestra qué debe corregir
cualquier regla: el score sin costo puntúa utilidad viva que sus acciones nunca
cobran (26 §4). Y el barrido de la familia V/C^α cierra el ciclo: la
devaluación por costo deshace el atasco solo si α es lo bastante grande, ningún
α domina en todos los regímenes, y el alcance del costo es una dimensión más
que hay que declarar (26 §6–§8).

Criterio previo: escribir este arco de memoria, en seis frases propias, antes
de la primera sesión. Si falta un eslabón, ahí está el hueco.

## Sesión 1 — Notebook 24: el caso de sesión

El notebook tiene cuatro actos. El acto 1 demuestra el colapso: promediar la
utilidad localizada sobre los conteos posibles devuelve las priors,
V(T) = Σ u_i·q_i, para todo pool. La consecuencia importa más que la fórmula:
el objetivo elige el pool máximo pero nunca premia volver a cobrar. El acto 2
es el caso concreto: cinco personas con q = 0.3, tres pruebas, pools de a lo
más 2. El miope hace tres individuales y vale 3q = 0.90, igual que el
estático; el óptimo abre el par, reentra tras conteo 1 y vale
q(3q² − 3q + 4) = 1.011, un 12.3% más. El acto 3 generaliza: en toda la franja
p ∈ [0.5, 0.9] el miope coincide dígito por dígito con el estático, el óptimo
queda 11 a 20% arriba, y un solo paso de anticipación es óptimo exacto desde
p = 2 − √2 ≈ 0.586. El acto 4 refuta la submodularidad adaptativa con un par
de historiales: la ganancia de probar {a} pasa de 0.05 (sin historia) a 1/3
(tras conteo 2 en el trío), casi siete veces más al saber más.

Ejercicios de reconstrucción, a mano y sin mirar:

1. Con conteo 1 en un par homogéneo, deducir P(cada miembro sano) = 1/2 y
   verificar que no depende de q.
2. Derivar q(3q² − 3q + 4) desde el árbol de tres ramas (r = 0, 1, 2 del par),
   con la probabilidad y el valor cobrado de cada rama, y evaluarla en q = 0.3.
3. Resolver de dónde sale el umbral 2 − √2 (igualar el valor del árbol con 3q).
4. Rehacer el contraejemplo del acto 4: por qué el conteo 2 en {a,b,c} sube
   P(a sana) de 0.05 a 1/3, y por qué un solo par de historiales basta para
   refutar un para-todo.

Criterio de dominio: explicar por qué el colapso de V(T) y la no-reentrada del
notebook 26 son el mismo defecto visto en dos momentos distintos (el score no
ve el costo de cobrar lo que promete).

## Sesión 2 — Notebook 25: resultados cerrados y peticiones

La parte I es el terreno firme; conviene poder citarla sin dudar. La escalera
exacta (§1): diez personas, q = 0.2, tres pruebas; contar sin adaptarse vale
casi lo mismo que adaptarse sin contar, y las dos perillas juntas valen más que
la suma de sus partes. La regla de certificación decide el baseline (§2), la
prohibición de cruces tiene un precio medible (§3), y el oráculo de rollout con
el acid test son la maquinaria de verificación (§5, gate G5).

La parte II es el material que responde las peticiones y donde nace el
notebook 26. Del §7: el valor realizable bajo presupuesto satura (con
q = 0.15 y b = 3, cinco personas llegan a 0.5426, por encima de 3 × 0.15 =
0.45), así que con presupuesto ajustado agrupar sí paga incluso en prevalencia
alta; la razón es el conteo 1, que deja a cada miembro del par en 1/2. Del §8:
el costo medido con greedy global degenera porque el argmax de q^k·k es k = 1
para todo q < 0.5; el costo colapsa al número de personas y no discrimina nada.
Del §9: con q = 0.7 el grupo conviene pero puede no caber, y una regla que
compare solo valores se equivoca en un sentido y una que compare solo costos en
el otro.

Ejercicios:

1. Probar que el argmax de q^k·k es k = 1 cuando q < 1/2 (el cociente entre
   términos consecutivos es q(k+1)/k, menor que 1 en ese régimen).
2. Explicar la diferencia entre las tres cantidades del §7: V(T) sin
   presupuesto, valor realizable con b pruebas, y el óptimo del problema.
3. Reconstruir por qué 0.5426 > 0.45 con las cuentas del conteo 1.

Criterio de dominio: responder sin pausa la pregunta "¿por qué no usan el valor
realizable exacto como score?" (porque calcularlo equivale a resolver el
problema: es la misma moraleja del menú valor-por-presupuesto).

## Sesión 3 — Notebook 26: costo local y no-reentrada

Aquí vive el trabajo propio de la semana y hay que dominar cada número. La
medición corregida (§2): fijar T, aplicar la prueba, simular el conteo, correr
greedy restringido a T condicionado a lo observado, contar subpruebas y
promediar. En un par el resultado exacto es 3q(1−q); la cuenta a mano es el
mejor ejercicio del notebook: el conteo 1 ocurre con probabilidad 2q(1−q) y
greedy usa 1.5 subpruebas en promedio (la mitad de las veces acredita a la
primera; la otra mitad deduce al sano pero debe acreditarlo con una prueba
más, porque la deducción informa y no acredita). Con q = 0.3 y cuatro
personas, el costo local da 2.09 subpruebas donde la medición global daba 4.0
plano.

El contraejemplo de no-reentrada (§4), con sus números exactos: par AB con
conteo 1, q = 0.3 de sano, utilidad 1. El score de presupuesto mágico puntúa
la reentrada individual en 1/2, el par virgen en 3/5, y el retest idéntico en
1. El orden del score es exactamente el inverso del orden de lo cobrado: el
argmax (retest) cobra cero para siempre y la peor puntuada cobra 1.17 en tres
pruebas. Nota de instancia que hay que saber defender: los dictados de la
sesión solo cuadran con probabilidad de estar sano 0.3; el acta dice
"probabilidad de actividad 0.3", y con esa otra lectura el par virgen valdría
1.4 y el orden no se reproduce.

El flag de alcance (§5): con costo restringido a T, el cociente V/C con α = 1
ya reordena bien el menú (la reentrada gana con 0.50 contra 0.40 del retest);
con alcance amplio la reentrada carga el cierre del vecino y el retest vuelve
a ganar. Las ocho esperanzas están verificadas contra cuentas a mano.

Ejercicios:

1. Derivar 3q(1−q) completo, incluyendo por qué la deducción no acredita.
2. Reconstruir los tres scores 1/2, 3/5, 1 desde el posterior de ocho estados.
3. Rehacer la tabla de V/C bajo los dos alcances y señalar dónde se invierte
   el argmax.

Criterio de dominio: explicar por qué el costo local de la reentrada individual
es cero subpruebas con alcance T solo y 0.5 con alcance amplio, sin consultar
la tabla.

## Sesión 4 — Notebook 26, §6 a §8: el barrido de α, y el simulacro final

Las reglas del barrido primero: menú laminar de pools de tamaño a lo más G,
score V/C^α con la tijera C ≤ b, costo local del notebook 26, y la regla de
no-parálisis (si la tijera vacía el menú, se toma el mejor cociente sin
filtro). El resultado fino del §6: en el estado del contraejemplo el atasco
persiste hasta α* = ln 2 / ln 2.5 ≈ 0.756 — el retest (V = 1, C = 2.5) empata
a la reentrada (V = 1/2, C = 1) cuando 2.5^α = 2 — así que de los tres α de la
sesión solo 1 y 3/2 deshacen la no-reentrada ahí. La malla del §7: 72
instancias exactas; en prevalencia alta α = 3/2 empata o gana y en q = 0.7 se
invierte (α = 3/2 sobre-castiga y deja de agrupar, cae a 0.72 del óptimo); la
familia supera a S0 solo en 7 de 72 instancias. El §8: el alcance amplio
rescata a α = 0 en prevalencia alta, empeora a los α chicos en q = 0.7, y hace
que la tijera pueda vaciar el menú; el flag y α no se pueden congelar por
separado.

Ejercicios:

1. Derivar α* desde la igualdad de scores y decir qué cambia si C del retest
   fuera 3 en vez de 2.5.
2. Explicar por qué la familia casi nunca supera a S0 en la malla, y qué
   pregunta deja eso para el rollout (el premio puede estar en el paso
   siguiente, no en el score de un paso).
3. Explicar la parálisis de la tijera con alcance amplio: por qué hasta un
   singleton puede "no caber".

El simulacro final cierra el plan: contar el arco completo en el pizarrón en
cinco minutos, sin notas, y responder tres preguntas probables. Cuáles son:
por qué la deducción no acredita y qué cambiaría si acreditara; si reentrar
tras un conteo intermedio y el ejemplo canónico de pool grande más búsqueda
binaria son el mismo mecanismo o dos; y qué se necesitaría para congelar un α
antes del atlas (la pregunta 14, que el barrido deja abierta a propósito
porque su estatuto es diagnóstico y la adopción pasa por G4a/G4b).

## Los números que hay que saber de memoria

| Cantidad | Valor | Dónde |
|---|---|---|
| Miope y estático en el caso de sesión | 3q = 0.90 | 24, acto 2 |
| Óptimo del caso, y su fórmula | q(3q² − 3q + 4) = 1.011 | 24, acto 2 |
| Umbral del paso de anticipación | p = 2 − √2 ≈ 0.586 | 24, acto 3 |
| Salto de ganancia del acto 4 | 0.05 → 1/3 (casi 7×) | 24, acto 4 |
| Posterior tras conteo 1 en un par | 1/2, independiente de q | 24, 25, 26 |
| Costo local exacto de un par fresco | 3q(1−q) | 26 §2 |
| Scores del contraejemplo | 1/2, 3/5, 1 | 26 §4 |
| Umbral de α en el contraejemplo | ln 2 / ln 2.5 ≈ 0.756 | 26 §6 |
| Instancias donde la familia gana a S0 | 7 de 72 | 26 §7 |

El plan queda cumplido cuando el simulacro sale limpio y los nueve números de
la tabla salen sin consultar nada.
