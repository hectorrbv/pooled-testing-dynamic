"""Two explicit comparisons, with a mechanism before the experimental maps."""


def add_presentation(add):
    add(r'''
## 1. La pregunta de Francisco: ¿cuándo conviene agrupar?

En la malla, las individuales dejan de ser óptimas siempre que $2\le B<n$. Basta una política laminar para mejorarlas.

El par explica la ganancia y el mapa muestra los regímenes. Después medimos cuánto cuesta exigir laminaridad.

Laminar: en cada rama, los grupos son disjuntos o uno contiene al otro. AB → A es laminar; AB → AC no lo es. La clase general permite ambos. Las dos clases pueden adaptarse a los resultados.

$n=6$, $u_i=1$, estados independientes. $q$ es la probabilidad de estar sano; $p=1-q$. $B$ limita las pruebas por rama y $G$ el tamaño del grupo. Observamos el conteo de infectados $R$ y acreditamos a quienes sabemos sanos con certeza: soft clearing.
''', 40, cell_id='sesion28-pregunta', notes='''
Apertura: Francisco, la sesión pasada te sorprendía que con dos pruebas y una población mayor las individuales pudieran dejar de ser óptimas para cualquier probabilidad. Traemos el mecanismo del par y un mapa donde la mejora aparece en todos los casos con 2<=B<n. Ese es el resultado principal; después muestro el costo de exigir laminaridad.
Fuente de la pregunta: acta del29-sep, D4,19:18–20:20. La malla es evidencia finita. El caso homogéneo conB2 está demostrado; la extensión general por intercambio está derivada y verificada, pendiente de cierre escrito y revisión humana según el plan del5-oct.
Todas las instancias principales comienzan sin pruebas previas y usan n=6, utilidad1 y probabilidades iguales e independientes. Óptimo significa el mejor valor esperado dentro de cada clase. Individuales está incluido en laminar, y laminar en general. No comparamos tres heurísticas. General no significa que toda política de esa clase use grupos no laminares.
Transición: empecemos con dos pruebas y un par.
''')

    add(r'''
## 2. El mecanismo: resolver el par y alcanzar a C

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$, $u_i=1$.

![Probar AB resuelve a ambos con probabilidad 58% y permite probar C. Si R=1, probabilidad 42%, se prueba A para identificar al sano de AB.](attachment:par_libera_prueba.png)

Con R=0 o 2, AB queda resuelto y probamos C. Con R=1, probamos A para identificar al sano de AB.

Recuperamos toda la utilidad de A y B: 0.6 en promedio. Además, probamos C en el 58% de los casos:

$$0.6+0.58(0.3)=0.774.$$

Dos individuales obtienen 0.6. Esta política laminar mejora un 29% y alcanza el óptimo en este caso.

Para cualquier $0<q<1$, este mismo plan gana $q[q^2+(1-q)^2]>0$ frente a dos individuales.
''', 90, 'par_libera_prueba', cell_id='sesion28-par', notes='''
Frase oral: recuperamos la misma información de A y B; la ganancia viene de que algunas ramas permiten usar la segunda prueba en C.
R=0 tiene probabilidad .09: acreditamos2 en AB y C añade .3 de media; total condicionado2.3. R=1 tiene probabilidad .42: probar A identifica al único sano de AB, cualquiera que sea su resultado; total1. R=2 tiene probabilidad .49: AB no aporta sanos, C añade .3. Por eso .09*2.3+.42*1+.49*.3=.774. Agrupamos R=0/2 en el dibujo porque la siguiente decisión es C en ambos; no tienen la misma utilidad de AB. No sobra una prueba al terminar la política: se dedica a C, y cada rama usa B=2. D,E,F quedan sin probar.
Aquí la construcción alcanza el óptimo laminar y general. El mapa siguiente calcula óptimos; no evalúa este mismo árbol en todas sus celdas. El mecanismo es el intercambio del caso de dos pruebas. La prueba general heterogénea y su revisión humana no quedan cerradas por este gráfico.
El resultado no depende de q=.3. Con B2, n>=3, G>=2, utilidades1 y probabilidad común q interior, la ganancia del plan es q*(q²+p²)>0. El plan no tiene por qué ser óptimo para otros q: basta que mejore a dos individuales.
Transición: este es el mecanismo que responde a tu pregunta. El mapa muestra su alcance experimental y los dos extremos del presupuesto.
''')

    add(r'''
## 3. Resultado principal: usar solo singletons deja de ser óptimo

En los 152 casos con $2\le B<n$, el óptimo laminar supera a individuales. La mejora aparece para todas las probabilidades probadas.

Óptimo laminar frente a individuales. 228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

Mejora (%) = $100(\text{laminar}-\text{individuales})/\text{individuales}$.

![Mejora del óptimo laminar frente a individuales. Se señala el 29% del ejemplo anterior. El contorno delimita las mejoras; los tonos claros indican mejoras pequeñas.](attachment:laminar_vs_individuales.png)

En esta malla:

- $B=1$: mejora solo para $q>0.5$.
- $B=2,3,4,5$: mejora en todos los casos.
- $B=6=n$: empate; podemos probar a todos.

Dentro del contorno hay mejora; fuera hay empate. El 29% corresponde al árbol anterior.

El caso homogéneo con $B=2$ está demostrado. La extensión general por intercambio está derivada y verificada en casos pequeños; falta cerrar la prueba por escrito y revisarla.
''', 120, 'laminar_vs_individuales', cell_id='sesion28-agrupar', notes='''
Frase oral: esta es la respuesta a la pregunta de la sesión anterior. En nuestra malla, si hay al menos dos pruebas pero no alcanzan para todos, usar solo individuales pierde para todas las probabilidades probadas. El par explica por qué; el mapa localiza las mejoras y los empates.
Señalar el29% y volver al .774 frente a .6. Después recorrer B=1, B=2..5 y B=6. Individuales vale Bq en estas instancias; la escala azul mide100*(laminar/individuales−1). 19 probabilidades ×6 presupuestos ×2 cotas=228;170 mejoras y58empates. B=2..5 contiene152casos, todos con mejora estricta. El contorno delimita celdas calculadas, no interpola umbrales. No concluir que cualquier grupo mejora ni que la primera acción óptima siempre deba ser un grupo.
La rama R=1 permite inferir al sano al probar a su compañero; esta acreditación necesita soft clearing. El mapa no compara con el modelo binario del paper.
Si pide la extensión: partir de los B mejores individuos según q_i*u_i, sustituir dos pruebas individuales por un par más su continuación y conservar las otras B−2. En R=1 recuperamos toda la utilidad del par; en R=0/2 probamos a una persona c fuera de los B originales. La ganancia es (q_a*q_b+p_a*p_b)*q_c*u_c>0. Requiere estados independientes, 0<q_i<1, u_i>0, 2<=B<n, G>=2 y soft clearing; empieza sin historial. La construcción es laminar. Derivación y verificación no equivalen a revisión humana concluida. No presentar como pendiente de inventar el argumento de intercambio, ni como cerrado el enunciado general.
Pregunta para Francisco: ¿este enunciado recoge el fenómeno que te sorprendía y conviene usarlo como resultado de separación entre modelos?
Transición: esa es nuestra respuesta sobre individuales. Como segunda cuestión, medimos cuánto se pierde al exigir laminaridad.
''')

    add(r'''
## 4. Los árboles explican 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

Individuales obtiene 2; el óptimo laminar, 2.5. Ahora lo comparamos con el general, que obtiene 3.

![Dos árboles óptimos empiezan con AB. Sus tres resultados tienen probabilidades 1/4, 1/2 y 1/4. Se resumen las continuaciones con su valor final esperado. En R=1, general prueba ACD y vale 3; laminar prueba A y vale 2.25.](attachment:separacion_arbol_resumen.png)

Los números inferiores son valores finales esperados de cada rama; «…» resume las pruebas restantes.

La mayor parte de la brecha está en R=1: la general prueba ACD y la laminar prueba A. Esa rama aporta $\tfrac12(3-2.25)=0.375$ de la diferencia total de 0.5.
''', 90, 'separacion_arbol_resumen', cell_id='sesion28-caso', notes='''
Frase oral: los dos óptimos empiezan con AB. La mayor parte de la diferencia aparece cuando AB es mixto: uno prueba A; el otro puede usar ACD.
Las hojas dibujadas resumen subárboles: los valores son finales y condicionados a cada resultado de AB, incluyendo utilidad ya acreditada. General: .25*4+.5*3+.25*2=3. Laminar: .25*3.75+.5*2.25+.25*1.75=2.5. Las ramas0 y2 aportan .0625 cada una; la rama1 aporta .375, tres cuartas partes de la brecha. No se omiten del cálculo las ramas no resaltadas.
Mostramos una política óptima de cada clase, no una única política necesariamente. En ramas0/2 puede haber desempates entre primeras pruebas; no atribuir toda la pérdida a C frente a CD. La rama1 permite una separación clara: ACD no es laminar con AB. La comparación anterior es2 frente a2.5; aquí2.5 frente a3. Sus porcentajes son25% y16.7%, con diferentes denominadores.
Ambos árboles se cotejaron en64perfiles; todos los valores de nodos se reconstruyeron independientemente, y el óptimo laminar coincide con el solver de átomos. Los árboles completos están en consultas.
Transición: sigamos ahora las dos políticas sobre una misma población para ver qué información consiguen.
''')

    add(r'''
## 5. Una realización: 4 sanos acreditados frente a 2

Mismo setup. Seguimos las dos políticas en una población donde solo B y D están infectados.

Las hojas indican los sanos acreditados y la utilidad obtenida.

![Sobre la misma población, general sigue AB, ACD, BCE, AEF con resultados 1,1,1,0 y acredita A,C,E,F. Laminar sigue AB,A,CD,C con resultados 1,0,1,0 y acredita A,C. Ambas usan cuatro pruebas.](attachment:separacion_ruta_comparada.png)

ACD no es laminar respecto de AB. Al combinar AB=1, ACD=1 y BCE=1, la general deduce que C está sano sin probarlo individualmente. Después, AEF=0 acredita a A, E y F.

La laminar dedica pruebas a A y C y termina sin probar E ni F. Esta realización muestra el mecanismo; 3 y 2.5 son los promedios sobre todas las poblaciones.
''', 90, 'separacion_ruta_comparada', cell_id='sesion28-ruta', notes='''
Frase oral: aquí ambas gastan cuatro pruebas. La laminar usa A para resolver AB, y C para resolver CD; E y F quedan fuera. La general puede combinarlos con las personas anteriores y acaba identificando a los cuatro sanos.
El perfil verdadero B,D infectados no se entrega a las políticas: determina los resultados al seguir cada árbol. La primera prueba coincide. En la segunda, ACD es incompatible con AB porque comparten A y ninguno contiene al otro. La general continúa BCE y AEF. De los tres primeros conteos1,1,1 ya se deduce C sano: si C fuera infectado, el segundo obliga A=D=0 y el tercero B=E=0, contradiciendo AB=1. El último conteo AEF=0 acredita A,E,F. La laminar conoce A sano y B infectado tras A=0; después CD=1 y C=0 acredita C e infiere D infectado. E,F siguen sin observarse.
Esta realización tiene probabilidad1/64 bajo el prior. Su ganancia2 no es la brecha promedio. Tampoco prueba que toda política laminar cobre2 en ese perfil: muestra las dos políticas óptimas elegidas en el resumen. Las probabilidades de las ramas completas y la media se justifican en la pantalla anterior y en la evidencia de64perfiles.
Pregunta de investigación: ¿cómo conservar parte de esta información compartida al transformar la política en laminar?
Transición: el mapa siguiente muestra dónde vuelve a aparecer la separación, sobre los mismos228casos.
''')

    add(r'''
## 6. Exigir laminaridad cuesta con presupuestos intermedios

Óptimo laminar frente al general. Los mismos 228 casos.

Pérdida (%) = $100(\text{general}-\text{laminar})/\text{general}$.

![Pérdida por exigir laminaridad. B=4, G=3, q=0.5 pierde 16.7%: general 3 frente a laminar 2.5. Las filas B=1, B=2 y B=6 son empates.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.

Dentro del contorno hay pérdida, incluso con tonos muy claros; fuera hay empate.

¿Por qué puede importar el presupuesto? Retomemos $n=6$, $G=3$, $q=0.5$, después de $AB=1$. Comparamos la misma prueba ACD con $b=1$ o $b=3$ pruebas restantes, es decir, $B=2$ o $B=4$.

![La misma prueba ACD con una o tres pruebas restantes. Sin continuación, sus resultados uno y dos no acreditan a nadie; con continuación, conducen a BCE y permiten obtener más utilidad.](attachment:horizonte_AB1.png)

Los números inferiores son utilidades finales esperadas condicionadas a cada resultado; «…» resume las pruebas restantes.

Con $b=1$, A garantiza 1 y ACD obtiene 0.5 en promedio. Con $b=3$, A y su mejor continuación obtienen 2.25; ACD y la suya, 3. ACD no es laminar respecto de AB.

La información parcial de ACD gana valor cuando quedan pruebas para aprovecharla. Este árbol explica el cambio en el caso destacado; los demás casos del mapa pueden tener mecanismos distintos.
''', 130, ['general_vs_laminar','horizonte_AB1'], cell_id='sesion28-mapa', notes='''
Frase oral: el blanco de B=2 no contradice el mapa anterior. Agrupar sí ayudaba; permitir toda la clase general no añade valor en esos casos. Con B=4, sí lo añade.
Señalar16.7% antes de recorrer las filas. El tono cálido mide pérdida respecto al general y tiene otra escala que el azul:100*(1−laminar/general). Blanco es empate, no ausencia de datos. El contorno incluye pérdidas muy tenues. B3separa34/38; B4yB5separan38/38cadauno. B6=npermite resolver a todos. El empateB2 es observado en esta malla homogénea desde el inicio, no para cualquier estado condicionado.
El máximo observado es1/6 enB4G3q=.5. No convertir el rendimiento5/6 en garantía. Este caso sí impone un techo sobre cualquier garantía universal c: toda salida laminar admisible vale como máximo2.5 y conservar c del general3 exige3c<=2.5, luego c<=5/6. Aleatorizar políticas laminares no supera el óptimo. No sabemos si ese techo es alcanzable; puede haber casos peores fuera de la malla.
El árbol nuevo fija el MISMO historial AB=1, n6,G3,q=.5 y cambia solo las pruebas restantes. No son dos resultados diferentes del primer test. Las cuatro ramas de ACD tienen probabilidades1/8,3/8,3/8,1/8 en ambos paneles. Con b1, sus utilidades son3,0,0,1: E[U]=.5. En R3 sabemos B sano por AB=1, por eso cobra1. En R1 yR2 nadie queda identificado sano. Probar A, en cambio, acredita1 seguro y es óptimo para ambas clases con b1.
Con b3, las ramas de ACD continúan E,BCE,BCE,E y sus valores finales son4,10/3,8/3,2: promedio3. Si forzamos A, incluso permitiendo después pruebas generales, la mejor continuación da9/4; el óptimo laminar también da9/4. Es una inversión entre esas dos alternativas, no solo un cambio de desempate. Las ramas intermedias de ACD requieren continuación para convertir información parcial en utilidad; la celda5 mostró una realización de BCE. El árbol ilustra una causa de la brecha en el caso destacado, no una prueba de que explique toda la malla. El b indica pruebas disponibles antes de ACD, así que quedan0 o2 después de ella.
Los seis árboles forzados (A con continuación general, A con continuación laminar y ACD general para cada horizonte) se evaluaron sobre los32perfiles compatibles con AB=1; todos sus nodos se auditaron. Los valores B4 coinciden con la evidencia anterior.
Frase oral: la restricción elimina ACD en ambos casos. Con la última prueba no la echamos de menos porque A es mejor; con tres pruebas sí, porque podemos aprovechar lo aprendido después.
Transición: una transformación tendría que conservar también las pruebas posteriores que dan valor a esta información.
''')

    add(r'''
## 7. ¿Cómo conservar utilidad al imponer laminaridad?

Próximo experimento: seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar compatible con lo ya probado.

Compararla con el óptimo laminar desde el inicio. Así medimos cuánto cuesta conservar esas primeras decisiones. El experimento sigue pendiente.

Para discutir: ¿conviene conservar ese comienzo o transformar las decisiones desde el inicio?

[Guion oral](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Estudio y discusión del paper](28_notas_de_estudio.ipynb).
''', 60, cell_id='sesion28-direccion-prueba', notes='''
Cierre: el par explica una ventaja frente a individuales; los mapas separan esa ganancia del costo de exigir laminaridad. El siguiente paso es medir cuánto conserva una transformación concreta.
El mapa no evalúa la transformación. Antes de ejecutar los ocho casos acordados, fijar laminaridad tras reducir ancestros y preferir acciones laminares entre acciones generales óptimas. La continuación debe ser compatible con toda la historia conservada. Comparar con el óptimo laminar desde la raíz mide la pérdida por conservar el prefijo; comparar después con una heurística separaría su pérdida adicional.
El paper se discute aparte. Su destino es estático, con observación binaria y hard clearing; el nuestro puede seguir siendo dinámico, con conteos y soft clearing. Dos perfiles independientes no prueban la desigualdad de utilidad para imponer laminaridad. Memo de transferencia y experimento de conversión siguen pendientes. Pregunta: ¿qué propiedad tendría que conservar una transformación para adaptar tu argumento de §5?
''')
