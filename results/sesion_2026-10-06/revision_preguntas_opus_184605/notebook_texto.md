## 2. resolver el par y alcanzar a C

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$, $u_i=1$.

![Probar AB resuelve a ambos con probabilidad 58% y permite probar C. Si R=1, probabilidad 42%, se prueba A para identificar al sano de AB.](attachment:par_libera_prueba.png)

Con R=0 o 2, AB queda resuelto y probamos C. Con R=1, probamos A para identificar al sano de AB.

Recuperamos toda la utilidad de A y B: 0.6 en promedio. Además, probamos C en el 58% de los casos:

$$0.6+0.58(0.3)=0.774.$$

Dos individuales obtienen 0.6. Esta política laminar mejora un 29% y alcanza el óptimo en este caso.

Para cualquier $0<q<1$, este mismo plan gana $q[q^2+(1-q)^2]>0$ frente a dos individuales.

## 3. Por tanto usar solo singletons deja de ser óptimo

En los 152 casos con $2\le B<n$, el óptimo laminar supera a individuales. La mejora aparece para todas las probabilidades probadas.

Óptimo laminar frente a individuales. 228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

Mejora (%) = $100(\text{laminar}-\text{individuales})/\text{individuales}$.

![Mejora del óptimo laminar frente a individuales. Se señala el 29% del ejemplo anterior. El contorno delimita las mejoras; los tonos claros indican mejoras pequeñas.](attachment:laminar_vs_individuales.png)

En esta malla:

- $B=1$: mejora solo para $q>0.5$.
- $B=2,3,4,5$: mejora en todos los casos.
- $B=6=n$: empate; podemos probar a todos.

Dentro del contorno hay mejora; fuera hay empate. El 29% corresponde al árbol anterior.



## 4. Los árboles explican 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

Individuales obtiene 2; el óptimo laminar, 2.5. Ahora lo comparamos con el general, que obtiene 3.

![Dos árboles óptimos empiezan con AB. Sus tres resultados tienen probabilidades 1/4, 1/2 y 1/4. Se resumen las continuaciones con su valor final esperado. En R=1, general prueba ACD y vale 3; laminar prueba A y vale 2.25.](attachment:separacion_arbol_resumen.png)

Los números inferiores son valores finales esperados de cada rama; «…» resume las pruebas restantes.

La mayor parte de la brecha está en R=1: la general prueba ACD y la laminar prueba A. Esa rama aporta $\tfrac12(3-2.25)=0.375$ de la diferencia total de 0.5.

## 5. Una realización: 4 sanos acreditados frente a 2

Mismo setup. Seguimos las dos políticas en una población donde solo B y D están infectados.

Las hojas indican los sanos acreditados y la utilidad obtenida.

![Sobre la misma población, general sigue AB, ACD, BCE, AEF con resultados 1,1,1,0 y acredita A,C,E,F. Laminar sigue AB,A,CD,C con resultados 1,0,1,0 y acredita A,C. Ambas usan cuatro pruebas.](attachment:separacion_ruta_comparada.png)

ACD no es laminar respecto de AB. Al combinar AB=1, ACD=1 y BCE=1, la general deduce que C está sano sin probarlo individualmente. Después, AEF=0 acredita a A, E y F.

La laminar dedica pruebas a A y C y termina sin probar E ni F. Esta realización muestra el mecanismo; 3 y 2.5 son los promedios sobre todas las poblaciones.

## 6. Exigir laminaridad cuesta con presupuestos intermedios

Óptimo laminar frente al general. Los mismos 228 casos.

Pérdida (%) = $100(\text{general}-\text{laminar})/\text{general}$.

![Pérdida por exigir laminaridad. B=4, G=3, q=0.5 pierde 16.7%: general 3 frente a laminar 2.5. Las filas B=1, B=2 y B=6 son empates.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.

Dentro del contorno hay pérdida, incluso con tonos muy claros; fuera hay empate.

Con dos pruebas, agrupar ya mejora a individuales y el óptimo laminar coincide con el general. Con cuatro, la restricción laminar sí cuesta en todos los casos probados.