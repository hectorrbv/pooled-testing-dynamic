"""Two explicit comparisons, with a mechanism before the experimental maps."""


def add_presentation(add):
    add(r'''
## 1. Dos preguntas: agrupar y exigir laminaridad

Primero: ¿cuánto mejora agrupar frente a usar solo individuales?
Después: ¿cuánto pierde el óptimo laminar frente al óptimo general?

Laminar: en cada rama, los grupos son disjuntos o uno contiene al otro. AB → A es laminar; AB → AC no lo es. La clase general permite ambos. Las dos clases pueden adaptarse a los resultados.

$n=6$, $u_i=1$, estados independientes. $q$ es la probabilidad de estar sano; $p=1-q$. $B$ limita las pruebas por rama y $G$ el tamaño del grupo. Observamos el conteo de infectados $R$ y acreditamos a quienes sabemos sanos con certeza: soft clearing.
''', 60, cell_id='sesion28-pregunta', notes='''
Apertura: Francisco, retomamos tu pregunta sobre las individuales. Primero muestro por qué un par puede mejorarlas; luego medimos cuánto cuesta exigir laminaridad.
Todas las instancias principales comienzan sin pruebas previas y usan n=6, utilidad1 y probabilidades iguales e independientes. Óptimo significa el mejor valor esperado dentro de cada clase. Individuales está incluido en laminar, y laminar en general. No comparamos tres heurísticas. General no significa que toda política de esa clase use grupos no laminares.
Transición: empecemos con dos pruebas y un par.
''')

    add(r'''
## 2. El par permite dedicar una prueba a C

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$, $u_i=1$.

![Probar AB resuelve a ambos con probabilidad 58% y permite probar C. Si R=1, probabilidad 42%, se prueba A para identificar al sano de AB.](attachment:par_libera_prueba.png)

Recuperamos toda la utilidad de A y B: 0.6 en promedio. Además, probamos C en el 58% de los casos:

$$0.6+0.58(0.3)=0.774.$$

Dos individuales obtienen 0.6. Esta política laminar mejora un 29% y alcanza el óptimo en este caso.
''', 120, 'par_libera_prueba', cell_id='sesion28-par', notes='''
Frase oral: recuperamos la misma información de A y B; la ganancia viene de que algunas ramas permiten usar la segunda prueba en C.
R=0 tiene probabilidad .09: acreditamos2 en AB y C añade .3 de media; total condicionado2.3. R=1 tiene probabilidad .42: probar A identifica al único sano de AB, cualquiera que sea su resultado; total1. R=2 tiene probabilidad .49: AB no aporta sanos, C añade .3. Por eso .09*2.3+.42*1+.49*.3=.774. Agrupamos R=0/2 en el dibujo porque la siguiente decisión es C en ambos; no tienen la misma utilidad de AB. No sobra una prueba al terminar la política: se dedica a C, y cada rama usa B=2. D,E,F quedan sin probar.
Aquí la construcción alcanza el óptimo laminar y general. El mapa siguiente calcula óptimos; no evalúa este mismo árbol en todas sus celdas. El mecanismo es el intercambio del caso de dos pruebas. La prueba general heterogénea y su revisión humana no quedan cerradas por este gráfico.
Pregunta si ayuda: ¿este intercambio es el fenómeno que te parecía sorprendente con dos pruebas y una persona adicional?
Transición: veamos en qué regímenes el mejor laminar supera a usar solo individuales.
''')

    add(r'''
## 3. Agrupar mejora a individuales incluso con pocos sanos

Óptimo laminar frente a individuales. 228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

![Mejora del óptimo laminar frente a individuales. Se señala el 29% del ejemplo anterior. El contorno delimita las mejoras; los tonos claros indican mejoras pequeñas.](attachment:laminar_vs_individuales.png)

En esta malla:

- $B=1$: mejora solo para $q>0.5$.
- $B=2,3,4,5$: mejora en todos los casos.
- $B=6=n$: empate; podemos probar a todos.

La celda marcada corresponde al árbol anterior. Dentro del contorno hay mejora, incluso con tonos muy claros; fuera hay empate.
''', 100, 'laminar_vs_individuales', cell_id='sesion28-agrupar', notes='''
Frase oral: con dos pruebas ya hay ventaja incluso cuando hay pocos sanos, y basta una política laminar para conseguirla.
Señalar el29% y volver al .774 frente a .6. Después recorrer B=1, B=2..5 y B=6. Individuales vale Bq en estas instancias; la escala azul mide100*(laminar/individuales−1). 19 probabilidades ×6 presupuestos ×2 cotas=228;170 mejoras y58empates. B=2..5 contiene152casos, todos con mejora estricta. El contorno delimita celdas calculadas, no interpola umbrales. No concluir que cualquier grupo mejora ni que la primera acción óptima siempre deba ser un grupo.
La rama R=1 permite inferir al sano al probar a su compañero; esta acreditación necesita soft clearing. El mapa no compara con el modelo binario del paper.
Transición: agrupar ya aporta dentro de laminar. La siguiente pregunta es qué perdemos al quedarnos en esa clase.
''')

    add(r'''
## 4. Un mismo caso, dos comparaciones

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

Utilidad esperada:

- Solo individuales: 2.
- Óptimo laminar: 2.5.
- Óptimo general: 3.

Pasar de individuales a laminar gana 0.5. Exigir laminaridad frente al general pierde 0.5.

Ahora medimos la segunda diferencia: $0.5/3\approx16.7\%$ del valor general.
''', 45, cell_id='sesion28-caso', notes='''
Frase oral: en una misma instancia vemos las dos comparaciones. Ya mostramos la ganancia frente a individuales. Ahora la referencia será el óptimo general.
Los tres valores promedian con idénticos datos y recursos. La primera mejora relativa es .5/2=25%; la segunda pérdida relativa es .5/3=16.666...%. No comparar porcentajes con denominadores distintos. El mapa azul ya contiene esta instancia; el siguiente la destaca.
General3 alcanza nq=3, la cota de toda la utilidad disponible; laminar2.5 coincide entre dos solvers. Las políticas se evaluaron en64perfiles. Aún no hemos extraído un árbol breve que explique el mecanismo del óptimo n6. El ejemplo n4 de las consultas es otra instancia, y debe anunciarse si se usa.
Transición: el segundo mapa mantiene los228casos, pero cambia la comparación.
''')

    add(r'''
## 5. Exigir laminaridad cuesta con presupuestos intermedios

Óptimo laminar frente al general. Los mismos 228 casos.

![Pérdida por exigir laminaridad. B=4, G=3, q=0.5 pierde 16.7%: general 3 frente a laminar 2.5. Las filas B=1, B=2 y B=6 son empates.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.

Con dos pruebas, agrupar ya mejora a individuales y el óptimo laminar coincide con el general. Con cuatro, la restricción laminar sí cuesta en todos los casos probados.
''', 110, 'general_vs_laminar', cell_id='sesion28-mapa', notes='''
Frase oral: el blanco de B=2 no contradice el mapa anterior. Agrupar sí ayudaba; permitir toda la clase general no añade valor en esos casos. Con B=4, sí lo añade.
Señalar16.7% antes de recorrer las filas. El tono cálido mide pérdida respecto al general y tiene otra escala que el azul:100*(1−laminar/general). Blanco es empate, no ausencia de datos. El contorno incluye pérdidas muy tenues. B3separa34/38; B4yB5separan38/38cadauno. B6=npermite resolver a todos. El empateB2 es observado en esta malla homogénea desde el inicio, no para cualquier estado condicionado.
El máximo observado es1/6 enB4G3q=.5. No convertir el rendimiento5/6 en garantía. Este caso sí impone un techo sobre cualquier garantía universal c: toda salida laminar admisible vale como máximo2.5 y conservar c del general3 exige3c<=2.5, luego c<=5/6. Aleatorizar políticas laminares no supera el óptimo. No sabemos si ese techo es alcanzable; puede haber casos peores fuera de la malla.
Transición: ahora necesitamos estudiar una transformación concreta.
''')

    add(r'''
## 6. ¿Cómo conservar utilidad al imponer laminaridad?

Próximo experimento: seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar compatible con lo ya probado.

Compararla con el óptimo laminar desde el inicio. Así medimos cuánto cuesta conservar esas primeras decisiones. El experimento sigue pendiente.

Para discutir: ¿conviene conservar ese comienzo o transformar las decisiones desde el inicio?

[Guion oral](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Estudio y discusión del paper](28_notas_de_estudio.ipynb).
''', 75, cell_id='sesion28-direccion-prueba', notes='''
Cierre: el par explica una ventaja frente a individuales; los mapas separan esa ganancia del costo de exigir laminaridad. El siguiente paso es medir cuánto conserva una transformación concreta.
El mapa no evalúa la transformación. Antes de ejecutar los ocho casos acordados, fijar laminaridad tras reducir ancestros y preferir acciones laminares entre acciones generales óptimas. La continuación debe ser compatible con toda la historia conservada. Comparar con el óptimo laminar desde la raíz mide la pérdida por conservar el prefijo; comparar después con una heurística separaría su pérdida adicional.
El paper se discute aparte. Su destino es estático, con observación binaria y hard clearing; el nuestro puede seguir siendo dinámico, con conteos y soft clearing. Dos perfiles independientes no prueban la desigualdad de utilidad para imponer laminaridad. Memo de transferencia y experimento de conversión siguen pendientes. Pregunta: ¿qué propiedad tendría que conservar una transformación para adaptar tu argumento de §5?
''')
