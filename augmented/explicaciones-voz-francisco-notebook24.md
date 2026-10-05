# Los cuatro actos del notebook 24, contados como los contaría Francisco

Guion de ensayo en la voz de Francisco, para llegar a la sesión con cada
ejemplo ya "escuchado" en el registro en que él los propone: ejemplo-juguete
primero, la cuenta después, mensaje por mensaje. No es transcript de nadie; es
material de preparación interno.

---

## Acto 1. La V(T) que propusimos, y dónde se colapsa

Imagínense un grupo de 4 personas, todas idénticas: utilidad u y probabilidad
q = 0.15 de estar sanas.

La idea era valorar la prueba por la utilidad localizada, ¿no? Si el conteo
sale R, sabemos que hay 4−R sanas ahí adentro, cada una con posterior
(4−R)/4, así que V(R) = u*(4−R). Con conteo 3, V = u: localizamos una persona
sana completita.

Y el valor de la prueba sería el promedio sobre los conteos: V(T) = promedio
de V(R).

Pero sabemos lo siguiente:

- el promedio de la posterior es la prior — la famosa ley de esperanzas
  totales
- así que V(T) = suma de u*q sobre el grupo, y ya. Para nuestro grupo de 4:
  0.6u, salga lo que salga la prueba.

Comparando las dos expresiones: el grupo de tamaño g vale g*u*q, así que el
objetivo siempre dice "agarra el grupo más grande". Eso sí resuelve el
bootstrapping — da el primer paso grande que greedy nunca da — y hasta ahí
vamos bien.

Lo delicado viene después: si ya probamos el grupo y queremos subdividirlo
para cobrar, el promedio tampoco se mueve. Es una martingala: el objetivo
jamás premia regresar. Encuentra y no cosecha.

La moraleja, me imagino, es que el potencial tiene que ser realizable bajo el
budget: contar solo lo que de verdad podemos extraer con las pruebas que
quedan. Y eso apunta a dos arreglos — la utilidad extraíble, o el descuento
por las log(G) pruebas del binary search — que es justo lo que queremos
discutir.

## Acto 2. Cinco personas, tres pruebas, y el par que reentra

Imagínense 5 personas idénticas, u = 1, probabilidad q = 0.3 de estar sanas —
o sea actividad alta, p = 0.7 —, budget de 3 pruebas y grupos de a lo más 2.

Sabemos lo siguiente:

- greedy hace puras pruebas individuales: cada una cobra q = 0.3, total
  3*q = 0.9. Y como no reacciona a nada de lo que observa, es idéntico al
  diseño estático — ahí el teorema de p > 1/2 aplica tal cual.
- el mejor árbol hace otra cosa: abre un par, aunque el par de entrada pague
  poquito (2q^2 = 0.18).

¿Y por qué conviene? Por lo que pasa con conteo 1: ahí sabemos que
exactamente uno de los dos está sano, cada uno con posterior 1/2. Una prueba
adentro del par vale 0.5, contra 0.3 de una persona fresca. El conteo
concentró la probabilidad — la prueba se pagó con información, no con
utilidad.

El árbol completo es un ejercicio de contar: conteo 0 (prob 0.09) cobras 2 de
golpe; conteo 1 (prob 0.42) reentras y cobras; conteo 2 (prob 0.49) el par
murió y te vas con gente fresca. Sumando las ramas: q*(3q^2−3q+4) = 1.011,
contra 0.9 de greedy. Un 12% más con el mismo budget y la misma información.

O sea: con greedy y con el estático la intuición era correcta — coinciden
dígito por dígito. La sorpresa es que Opt no coincide: la ganancia es del
conteo con reentrada, que el teorema estático no cubre. Está verificado
exacto en n=5, y ese contraste ya es un pedacito de paper.

## Acto 3. No es un ejemplo: es todo el régimen, con umbral exacto

Y esto no es un punto afortunado. Hagan de cuenta que barremos p de 0.5 a
0.9, todos idénticos: en TODO ese rango greedy = estático exacto, y el óptimo
queda entre 11% y 20% arriba.

Lo bonito: una política de un solo paso de anticipación — mira un paso
adelante para elegir la primera prueba y luego sigue greedy — vale
q*(3q^2−3q+4), y desde p ~ 0.586 esa política ya ES el óptimo exacto.

¿De dónde sale el 0.586? De un solo nodo del árbol. Tras el conteo 1 en el
par quedan dos planes:

- reentrar al par: vale 1 + q/2
- cruzar — probar un miembro del par junto con una persona fresca: vale
  (q^2 + 3q + 1)/2

Comparando las dos expresiones, la resta es (q^2 + 2q − 1)/2, que se anula en
q = raíz(2) − 1. O sea p = 2 − raíz(2) ~ 0.586. Todo el umbral es la raíz de
un empate; eso con un pizarrón se ve rápidamente.

Y fíjense en el detalle: bajo el umbral la jugada buena CRUZA el par — se
sale de lo laminar. Usen eso como intuición de qué pierde exactamente la
restricción laminar, y en qué régimen no pierde nada.

## Acto 4. La ganancia que crece al saber más

La última pieza es la de la garantía. La esperanza era que el objetivo fuera
adaptive submodular — rendimientos decrecientes — para heredar la famosa
garantía 1 − 1/e del greedy, la de Golovin y Krause.

Imagínense 4 personas con probabilidad q = 0.05 de estar sanas. Probar a la
persona a de entrada gana 0.05u.

Ahora hagan de cuenta que primero probamos el trío a-b-c y salió conteo 2:
exactamente una sana, cada una con posterior 1/3. La MISMA prueba de a ahora
gana u/3 — como siete veces más.

La ganancia creció al saber más, que es exactamente lo contrario de los
rendimientos decrecientes. Y no es un accidente numérico: es el mismo
mecanismo del acto 2 con otro sombrero — la prueba grande paga porque vuelve
valiosas las que siguen. Complementariedad, no submodularidad.

Enumerando todo en n=4 quedan violaciones genuinas, que no vienen de la
contabilidad de acreditados. Así que la ruta directa a 1 − 1/e está cerrada,
con testigo.

Para ser completamente honesto, eso no mata la garantía — mata ESA garantía.
Quedan vivas las versiones relajadas: policy improvement (que el objetivo
nuevo nunca sea peor que greedy), regret acotado contra el rollout, o
garantía por régimen. Y documentar la obstrucción con su testigo, eso ya
justifica su existencia en el paper.
