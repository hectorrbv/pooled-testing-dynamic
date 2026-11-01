# Pendientes y correcciones tras la sesión de Lowell House

## El encargo explícito

La única tarea que quedó nombrada como tarea es calcular el valor de planificación
de una prueba. Para un pool $T$ que devuelve conteo $R$, se define

$$V(R) = \sum_{i \in T} u_i \, q_i(R), \qquad q_i(R) = P(i \text{ sano} \mid \text{count}(T)=R),$$

y el valor de la prueba es $V(T) = \mathbb{E}_R[V(R)]$. La hipótesis es que
maximizar $V(T)$ en lugar de la utilidad inmediata recupera el comportamiento
planificador: cuando $R=1$ en un pool grande, $B$ se dispara, y esa cola compensa
el bajo rendimiento inmediato de la prueba grande. El pedido concreto es calcular
$V(T)$ de forma exacta en el ejemplo de baja prevalencia y alta utilidad, como
función del tamaño del grupo, y ver si reproduce la decisión del planificador.

El material para hacerlo ya existe: las posteriores individuales $q_i(R)$ salen
directamente del tensor de subpools, y `subset_pmf_cache` da la distribución de
$R$. No hay en el repo ninguna función que calcule $B$ ni $V$; hay que escribirla.

## Otras tres tareas que quedaron abiertas

La segunda es la heurística de utilidad descontada: premiar al greedy no solo por
las personas sanas que confirma, sino por las que localiza dentro de un grupo, con
un descuento. La justificación es que identificar un sano dentro de un pool de $G$
garantiza extraerlo en $\log G$ pruebas más. Quedó sin definir la forma del
descuento; se mencionaron dos candidatos, el peor caso dentro del grupo y el
promedio ponderado por posterior, sin decidir.

La tercera es la política de ramas: mantener la partición laminar actual, asignarle
a cada componente su valor extraíble, ordenar por ese valor y explorar primero la
rama de valor máximo. La objeción a esta idea la puso el propio Francisco en la
sesión: es un problema de bootstrapping, porque presupone que ya se hizo una prueba
grande inteligente.

La cuarta es buscar más ejemplos de separación entre lo dinámico y lo estático, y
en particular ejemplos donde greedy quede por debajo. Esto ya no hace falta
buscarlo: los datos que hay lo contienen, según se explica abajo.

## Lo que se afirmó en la sesión y los datos no sostienen

Sobre el régimen de tasas altas. En la sesión se dijo que greedy laminar tuvo su
mejor desempeño en tasas altas, y la objeción fue que ahí no debería haber ninguna
ganancia. Buena parte del desacuerdo es de nomenclatura: hay dos líneas base que se
llaman estático, `V_static_binary` y `V_static_greedy`, y difieren en el 31% de las
instancias del barrido. Contra la binaria, prevalencia alta sí es el mejor régimen,
con greedy ganando en 96.0% de las instancias. Contra el greedy estático, que es la
comparación justa, no hay ninguna ganancia. Los datos de
`greedy_vs_static_greedy.csv` dan la razón a la objeción en el caso homogéneo: para $p \ge 0.5$ el cociente entre greedy laminar y greedy
estático es exactamente 1.0000 en las 432 instancias, sin una sola excepción. La
ventaja real está en el régimen intermedio, $p$ entre 0.25 y 0.45, donde el
cociente medio va de 1.018 a 1.044. La ganancia en tasas altas solo aparece en el
modo heterogéneo, donde la mezcla bimodal mantiene individuos por debajo de 0.5, y
ahí vale 1.4% a 1.7% en promedio.

Sobre que todo coincide con el óptimo en tasas altas. Esta parte es falsa y es el
hallazgo más útil de la revisión. Greedy laminar iguala a greedy estático, pero
ninguno de los dos alcanza el óptimo: en el caso homogéneo el cociente entre greedy
y el mejor árbol laminar cae de 0.985 en $p=0.5$ a 0.944 en $p=0.9$, con casos
individuales de 0.804. El rollout, en cambio, se queda en 0.994 a 0.997 en todo ese
rango. El régimen de prevalencia alta y perfiles homogéneos es entonces un segundo
ejemplo de separación, con la propiedad que se estaba buscando: la planificación
paga, greedy no captura nada de ese excedente y una política de un paso de
anticipación captura casi todo. Es exactamente la pregunta que se dejó abierta al
final de la sesión, y la respuesta ya está en los datos.

Sobre que greedy laminar y greedy con la heurística de independencia son
comparables. En el modo heterogéneo con prevalencia baja, greedy laminar pierde
contra greedy estático en el 54% al 64% de las instancias, con un peor caso de
0.749. La afirmación de que no hubo mucha diferencia solo describe bien el
promedio, no la dispersión.

Sobre que con presupuesto 2 greedy siempre fue óptimo. Lo que vale 1.0 en todas
las filas de `homogeneous_b2.csv` es el cociente entre el mejor árbol laminar y el
óptimo irrestricto, que es una afirmación sobre la clase de diseños, no sobre
greedy. Greedy con $B=2$ alcanza el óptimo laminar en 78.9% de las instancias
homogéneas y en 52.3% de las heterogéneas, con brechas de hasta 25%.

Sobre la descomposición un cuarto y tres cuartos. La descomposición que existe en
los datos no separa independencia de miopía, sino elección de árbol de miopía:
`costo_arbol` es el valor del árbol balanceado sobre el mejor árbol y
`costo_miopia` es el valor de greedy sobre el óptimo dentro de ese árbol. La
partición de la pérdida es 39% árbol y 61% miopía en el caso homogéneo, y se
invierte a 69% y 31% en el heterogéneo. No hay un reparto estable de un cuarto y
tres cuartos.

## Dos cosas que la sesión no tocó y conviene revisar

El comportamiento de anidamiento, medido en `nesting.csv`, confirma la
preocupación de fondo de forma más nítida de lo que se dijo. Con perfiles
homogéneos y $p \ge 0.4$, greedy nunca anida: la fracción de decisiones en
territorio virgen es 1.000 exacta. El laminar dinámico degenera en un diseño
estático en todo ese régimen, y por eso los cocientes de arriba valen 1 exactamente.
Por debajo de $p=0.4$ sí anida, alrededor de un tercio de las decisiones.

En la corrida de $N=40$, `laminar_greedy` y `laminar_rollout` producen medias,
errores estándar, medianas y tasas de cero idénticos hasta el último dígito. El
rollout no está haciendo nada a esa escala. Además ambos quedan 11% por debajo de
`flat_independence`, de modo que el rango de 2% a 10% que se mencionó en la sesión
no describe esa corrida. Antes de reportar cualquier número de $N=40$ hay que
entender por qué el rollout está inerte.

Por último, ni `arbol_vs_miopia.csv` ni `greedy_vs_static_greedy.csv` tienen un
script generador en el repo. Los dos resultados que sostienen la discusión de la
sesión no son reproducibles hasta que ese generador se escriba y se versione.
