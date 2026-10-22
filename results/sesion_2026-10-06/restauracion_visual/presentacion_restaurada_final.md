## 1. Dos preguntas: agrupar y exigir laminaridad

Primero: ¿cuánto mejora agrupar frente a usar solo individuales?
Después: ¿cuánto pierde el óptimo laminar frente al óptimo general?

Laminar: en cada rama, los grupos son disjuntos o uno contiene al otro. AB → A es laminar; AB → AC no lo es. La clase general permite ambos. Las dos clases pueden adaptarse a los resultados.

$n=6$, $u_i=1$, estados independientes. $q$ es la probabilidad de estar sano; $p=1-q$. $B$ limita las pruebas por rama y $G$ el tamaño del grupo. Observamos el conteo de infectados $R$ y acreditamos a quienes sabemos sanos con certeza: soft clearing.

## 2. El par permite dedicar una prueba a C

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$, $u_i=1$.

![Probar AB resuelve a ambos con probabilidad 58% y permite probar C. Si R=1, probabilidad 42%, se prueba A para identificar al sano de AB.](attachment:par_libera_prueba.png)

Recuperamos toda la utilidad de A y B: 0.6 en promedio. Además, probamos C en el 58% de los casos:

$$0.6+0.58(0.3)=0.774.$$

Dos individuales obtienen 0.6. Esta política laminar mejora un 29% y alcanza el óptimo en este caso.

## 3. Agrupar mejora a individuales incluso con pocos sanos

Óptimo laminar frente a individuales. 228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

![Mejora del óptimo laminar frente a individuales. Se señala el 29% del ejemplo anterior. El contorno delimita las mejoras; los tonos claros indican mejoras pequeñas.](attachment:laminar_vs_individuales.png)

En esta malla:

- $B=1$: mejora solo para $q>0.5$.
- $B=2,3,4,5$: mejora en todos los casos.
- $B=6=n$: empate; podemos probar a todos.

La celda marcada corresponde al árbol anterior. Dentro del contorno hay mejora, incluso con tonos muy claros; fuera hay empate.

## 4. Un mismo caso, dos comparaciones

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

Utilidad esperada:

- Solo individuales: 2.
- Óptimo laminar: 2.5.
- Óptimo general: 3.

Pasar de individuales a laminar gana 0.5. Exigir laminaridad frente al general pierde 0.5.

Ahora medimos la segunda diferencia: $0.5/3\approx16.7\%$ del valor general.

## 5. Exigir laminaridad cuesta con presupuestos intermedios

Óptimo laminar frente al general. Los mismos 228 casos.

![Pérdida por exigir laminaridad. B=4, G=3, q=0.5 pierde 16.7%: general 3 frente a laminar 2.5. Las filas B=1, B=2 y B=6 son empates.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.

Con dos pruebas, agrupar ya mejora a individuales y el óptimo laminar coincide con el general. Con cuatro, la restricción laminar sí cuesta en todos los casos probados.

## 6. ¿Cómo conservar utilidad al imponer laminaridad?

Próximo experimento: seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar compatible con lo ya probado.

Compararla con el óptimo laminar desde el inicio. Así medimos cuánto cuesta conservar esas primeras decisiones. El experimento sigue pendiente.

Para discutir: ¿conviene conservar ese comienzo o transformar las decisiones desde el inicio?

[Guion oral](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Estudio y discusión del paper](28_notas_de_estudio.ipynb).