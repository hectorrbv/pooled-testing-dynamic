# Qué experimentos quedan, y cuáles valen la pena

La lista de la sesión se reordena bastante ahora que sabemos tres cosas que no se
sabían entonces: que V(T) tal como se enunció se colapsa, que el premio real es
la submodularidad adaptativa, y que el barrido ya contiene un segundo ejemplo de
separación sin que nadie lo haya leído.

Los primeros tres experimentos son una cadena. Cada uno decide si el siguiente
tiene sentido, así que conviene correrlos en orden y no en paralelo.

## E1. Confirmar numéricamente que V(T) se colapsa

Calcular V(T) con el tensor sobre unos cientos de pools y comprobar que da
exactamente la suma de u_i * q_i con las priors, para todo T. El argumento es la
ley de esperanzas totales y no debería fallar, pero conviene tenerlo como
resultado corrido y no como afirmación en un documento.

Cuesta media hora. Decide si la tarea que pidió Francisco se implementa tal cual
o se replantea. Es barato y bloquea todo lo demás, así que va primero aunque el
resultado esté cantado.

## E2. Cuál objetivo de un paso sí recupera el ejemplo canónico

Este es el experimento central y es la tarea de la sesión, hecha bien. Definir
los candidatos de objetivo por paso y ver cuál escoge la prueba grande cuando
debe:

- utilidad inmediata, que es el greedy actual y sabemos que falla;
- V lineal, que es la propuesta original y por E1 sabemos que degenera;
- el promedio sobre R del mejor u_i * q_i(R) del grupo;
- utilidad localizada descontada por las log(g) pruebas que cuesta extraerla;
- el peor caso dentro del grupo;
- el valor de continuación truncado a un paso, que es lo que ya hace el rollout.

Dos evaluaciones para cada uno. Primero, en el ejemplo canónico con la fórmula
cerrada, mirando si prefiere el pool grande sobre el individual y a partir de qué
tamaño. Segundo, en el barrido exacto de n hasta 6 que ya existe, midiendo qué
fracción del óptimo dinámico captura cada objetivo. La maquinaria de
`ExactPolicyEvaluator` ya soporta esto: es cambiar el criterio de selección y
reusar el evaluador.

Cuesta unos días. Decide con qué objetivo se trabaja el resto del proyecto. Si
ninguno recupera el ejemplo, eso también es información y hay que rediseñar antes
de seguir.

## E3. Buscar contraejemplos a la submodularidad adaptativa

El que más valor tiene por unidad de esfuerzo, y el que yo correría antes de
invertir un mes en demostrar nada.

Para cada objetivo que sobreviva E2, verificar por fuerza bruta en poblaciones de
4 a 8 personas si es adaptive monotone y adaptive submodular, en el sentido de
Golovin y Krause: que la ganancia marginal esperada de un pool, condicionada al
historial, nunca crezca cuando el historial se hace más informativo. Enumerar
todos los pares de historiales anidados y todos los pools, y quedarse con las
violaciones.

Conviene además correrlo sobre la función de valor óptima misma, no solo sobre
los candidatos. Si ni siquiera el valor óptimo de continuación es adaptive
submodular en instancias chicas, entonces la ruta directa está muerta y hay que
ir por un sustituto submodular a la Chen et al. desde el primer día, sin perder
tiempo.

Cuesta unos días y es todo enumeración exacta sobre lo que ya está construido.
Decide si el resultado que se persigue existe. Dado que el valor de información
no es submodular en general, hay una probabilidad seria de que salga negativo, y
descubrirlo en tres días en vez de en tres meses es la mejor inversión de la
lista.

## E4. Convertir la prevalencia alta homogénea en el segundo ejemplo de separación

Este ya está pagado: los datos existen y solo hay que leerlos bien. Con perfiles
homogéneos y p mayor o igual a 0.5, greedy laminar iguala exactamente al greedy
estático en las 432 instancias, y sin embargo el óptimo del árbol está
estrictamente arriba, con greedy cayendo de 0.985 a 0.944 conforme p va de 0.5 a
0.9, y casos individuales de 0.804. El rollout se queda en 0.994 o mejor.

Falta aislar la instancia más limpia y chica, extraer el árbol óptimo y el árbol
de greedy lado a lado, y explicar en una frase qué ve el óptimo que greedy no ve.
Si eso se sostiene, es una familia entera de separación en vez de un ejemplo
suelto, y además cae justo en el régimen que la charla identifica como el corazón
del proyecto.

Cuesta uno o dos días. Vale mucho porque la historia entera se apoya hoy en un
solo ejemplo, y porque el paper dinámico previo solo tiene evidencia empírica en
este punto.

## E5. El teorema del umbral, que casi seguro ya es de ustedes

Corrección sobre lo que escribí antes. No aparece en la literatura externa un
teorema citable que fije en 0.5 el umbral, pero en la charla de Boston College se
dice de forma explícita que está demostrado en el paper que el grupo estaba
enviando: agrupar no sirve de nada cuando la probabilidad de estar sano baja de
0.5, en el setting de pooled testing binario. O sea que el experimento no
descubre un teorema libre; lo reverifica. Antes de darle cualquier estatus hay
que confirmar con Francisco si la versión publicada ya cubre utilidades
heterogéneas y presupuesto, que es la única parte que podría no estar.

El experimento es barato: barrer exhaustivamente con la maquinaria exacta si el
diseño estático óptimo es siempre puras pruebas individuales cuando todos los q_i
están por debajo de 0.5, con utilidades heterogéneas y presupuestos distintos, y
buscar contraejemplos. Si no aparecen, hay un teorema que enunciar con nombre en
vez de mencionarlo de pasada como se ha venido haciendo.

Cuesta un día. Puede que ya esté demostrado en el trabajo del grupo, y en ese
caso lo que falta es solo darle estatus de teorema en el paper.

## E6 y E7. Higiene que bloquea

El rollout está inerte en la corrida de n=40: da media, error estándar, mediana y
tasa de cero idénticos al greedy hasta el último dígito. Mientras eso no se
entienda, ningún número de escala se puede reportar. Y los dos CSV que
sostuvieron la discusión de la sesión, `arbol_vs_miopia` y
`greedy_vs_static_greedy`, no tienen generador versionado, así que los dos
resultados que más se citaron no son reproducibles.

Ninguno de los dos produce conocimiento nuevo, pero ambos bloquean cosas que sí.

## Lo que yo dejaría fuera

La política de ordenar ramas por valor extraíble se descartó en la misma sesión
donde se propuso, y por buena razón: presupone que ya se hizo la primera prueba
grande, que es exactamente el problema que no se sabe resolver.

Ampliar el atlas a más celdas no contesta ninguna pregunta abierta. El atlas ya
tiene 2592 instancias y el hallazgo de E4 llevaba ahí semanas sin que nadie lo
leyera. El cuello de botella es de lectura, no de cómputo.

Y la tubería de n=40 con partículas y MILP no está lista para sostener
conclusiones: la política laminar pierde 11% contra el control más simple y el
rollout no está activo. Arreglarla es E6; usarla para concluir algo es prematuro.
