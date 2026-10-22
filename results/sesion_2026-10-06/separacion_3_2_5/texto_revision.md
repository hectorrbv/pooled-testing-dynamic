## 4. Los árboles explican 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

Individuales obtiene 2; el óptimo laminar, 2.5. Ahora lo comparamos con el general, que obtiene 3.

![Dos árboles óptimos empiezan con AB. Sus tres resultados tienen probabilidades 1/4, 1/2 y 1/4. Se resumen las continuaciones con su valor final esperado. En R=1, general prueba ACD y vale 3; laminar prueba A y vale 2.25.](attachment:separacion_arbol_resumen.png)

Si AB devuelve un infectado, la general prueba ACD y la laminar prueba A. Esa rama aporta $\tfrac12(3-2.25)=0.375$ de la diferencia total de 0.5.

## 5. Una realización: 4 sanos acreditados frente a 2

Mismo setup. Seguimos las dos políticas en una población donde solo B y D están infectados.

![Sobre la misma población, general sigue AB, ACD, BCE, AEF con resultados 1,1,1,0 y acredita A,C,E,F. Laminar sigue AB,A,CD,C con resultados 1,0,1,0 y acredita A,C. Ambas usan cuatro pruebas.](attachment:separacion_ruta_comparada.png)

ACD combina A con personas nuevas: no es laminar respecto de AB. La general incorpora a E y F mientras resuelve a las anteriores; en esta ruta la laminar termina sin probarlos.

Esta realización explica una diferencia de decisiones. Los promedios de las políticas completas siguen siendo 3 y 2.5.