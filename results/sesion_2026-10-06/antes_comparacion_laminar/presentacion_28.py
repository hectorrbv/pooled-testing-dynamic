"""The four presentation screens; longer derivations live in the study notebook."""


def add_presentation(add):
    add(r'''
## 1. ¿Cuándo conviene agrupar?

$n=6$, $B=1,\ldots,6$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$: 228 casos exactos.

En todo el notebook: $q$ es la probabilidad de estar sano, $p=1-q$, estados independientes y $u_i=1$. Cada prueba devuelve el conteo de infectados $R$; usamos soft clearing: acreditamos a quien sabemos sano con certeza.

Comparamos el óptimo que permite grupos y cruces con usar solo individuales, cuyo valor es $Bq$. El color muestra la mejora porcentual: 50% significa cobrar 1.5 veces esa referencia.

![Ganancia sobre individuales: con una prueba depende de q; con dos a cinco hay mejora en toda la malla; con seis hay empate.](attachment:agrupar_vs_individuales.png)

- $B=1$: agrupar mejora si $q>0.5$; con $q\leq0.5$ no mejora.
- $B=2,\ldots,5$: mejora en los 152 casos, también con pocos sanos.
- $B=6=n$: empate; las individuales ya resuelven a todos.
''', 120, 'agrupar_vs_individuales', cell_id='sesion28-agrupar', notes='''
Frase oral: con una sola prueba importa que el grupo entero salga sano; con dos o más también podemos aprovechar la información de un grupo mixto.
Son 19 probabilidades × 6 presupuestos × 2 cotas G = 228 casos; 170 mejoras y 58 empates. El blanco es cero, no falta de datos. Se muestra 100(V_general*/(Bq)−1); el otro mapa usará otro denominador. Las probabilidades son iguales para las seis personas y no hay pruebas previas. Cada política elegida se evaluó por filtrado independiente de los 64 estados; los óptimos proceden del Bellman y B=n del certificado de utilidad total. No es Monte Carlo ni una comparación de π_L con π_ratio. Tampoco dice que toda primera acción óptima tenga tamaño mayor que 1: puede empezar individual y agrupar después. En q=.5, B=1 un par también empata con una individual. En q=.3, B=2, G=2 el valor es .774 frente a .6, mejora 29%, que se explica en el siguiente bloque. El máximo 170.75% ocurre con q=.95, G=3, B=1 o 2; no usar ese extremo como afirmación general.
Pregunta: cuánto de la ganancia persiste al variar las probabilidades entre personas manteniendo su media.
''')

    add(r'''
## 2. El par deja libre una prueba en el 58% de los casos

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$.

Dos individuales, A y B, dan $0.3+0.3=0.6$ de utilidad esperada. Probando primero AB también resolvemos a ambos con un máximo de dos pruebas:

![Tras probar AB, los conteos cero o dos resuelven a ambos y permiten probar C; con conteo uno, probar A identifica al sano de AB.](attachment:par_libera_prueba.png)

La oportunidad adicional aparece cuando AB ya está resuelto: $0.09+0.49=0.58$. Entonces podemos probar C, que está sano con probabilidad 0.3.

$$E[U]=\underbrace{0.6}_{\text{A y B}}+\underbrace{0.58\times0.3}_{\text{oportunidad de C}}=0.774.$$

Es el 29% de mejora marcado en el mapa. La ganancia viene de usar la segunda prueba donde todavía aporta información.
''', 105, 'par_libera_prueba', cell_id='sesion28-par', notes='''
Frase oral: recuperamos toda la información de A y B y, el 58% de las veces, además podemos probar C.
R0 ocurre .09, R2 ocurre .49; ambos resuelven a A y B en una prueba. R1 ocurre .42, y una individual identifica al sano del par por inferencia. Las tres personas restantes no se usan en esta política de dos pruebas. Se mantiene n=6 para explicar exactamente una celda del heatmap: G2, B2, q.3. El resultado .774 coincide con el óptimo guardado. El .58 multiplica al .3 de C: probabilidad de llegar a la oportunidad por utilidad esperada de aprovecharla. El .6 es la contribución esperada de A y B en todas las ramas, no una cantidad cobrada en cada rama. Las utilidades totales condicionadas son 2.3, 1, .3 para R0, R1, R2, lo que también da .774. Si R2, no se acredita a nadie de AB, pero se libera la prueba para C. Este argumento usa conteos y soft clearing. El mismo intercambio sirve como candidato de prueba general; no significa que todo pool ni toda apertura agrupada sea óptima.
''')

    add(r'''
## 3. Agrupar y permitir cruces son preguntas distintas

$n=4$, $B\in\{2,3,4\}$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$: 114 casos exactos.

Una política laminar puede probar AB y después A o CD. Probar AB y después AC crea un cruce: los grupos comparten A y ninguno contiene al otro.

Comparamos dos óptimos dinámicos: con y sin esa restricción. El eje vertical mide cuánto pierde el laminar respecto al general, en porcentaje.

![Con n=4, la pérdida por prohibir cruces es cero para B=2 y B=4; con B=3 llega al 12.5 por ciento.](attachment:mapa_precio_laminar.png)

Con $B=2$ hay empate en esta malla. Con $B=3$ los cruces pueden ayudar. Con $B=4=n$, probar a todos elimina la diferencia.

El máximo observado es 12.5%: $q=0.5$, $B=3$, $G=3$. Ambos métodos permiten agrupar; el empate de este mapa no significa que las individuales sean óptimas.
''', 110, 'mapa_precio_laminar', cell_id='sesion28-mapa', notes='''
Frase oral: primero medimos el beneficio de agrupar; ahora medimos qué parte perdemos si solo permitimos grupos laminares.
Esta es otra malla, n4 en vez de n6, conservada para no perder la línea Q2. El porcentaje es 100(1−V_laminar*/V_general*), no la mejora relativa al individual del bloque 1. Hay 81 empates y 33 separaciones; ambas clases pueden adaptar. n4, B2, G2, q.3: individual .6, laminar .774, general .774. Empatar las últimas dos no anula la ventaja sobre individuales. La pérdida máxima 12.5% procede de 1.75/2, explicada en el bloque 4. Cada política se verificó sobre 16 perfiles, laminar contra solver de átomos y general en auditoría independiente. No implica garantía universal 87.5% ni empate universal con B2. El control B4=n es especial: todos pueden ser probados individualmente; con n6, B4 la conclusión no se deduce.
Pregunta: en qué presupuestos y tamaños aparece la mayor pérdida al aumentar n.
''')

    add(r'''
## 4. La restricción y la regla pierden cosas distintas

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $p_i=0.5$.

- 2.00: el plan con cruces ABC → ABD → ACD identifica a todos mediante sus tres conteos.
- 1.75: el mejor valor posible si exigimos laminaridad, aun adaptando las pruebas.
- 1.50: una conversión simple que prueba ABC, luego D y, si ABC sigue mixto, A.

Aquí perdemos 0.25 por la restricción y otros 0.25 por cómo elegimos las pruebas. Conservar ABC y continuar con la regla del cociente alcanza 1.75 en este piloto. Esa regla compara utilidad esperada por prueba esperada y vuelve a decidir tras cada resultado.

La siguiente pregunta: ¿cuánto conservamos si seguimos la política general y cambiamos justo antes de su primer cruce? Compararemos la mejor continuación laminar con la regla del cociente. Ese experimento sigue pendiente.

[Guion para presentarlo](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Árboles, tareas y estudio de §5](28_notas_de_estudio.ipynb).
''', 115, cell_id='sesion28-cierre', notes='''
Frase oral: una mala conversión puede costar más que exigir laminaridad.
2 es el óptimo general porque las tres pruebas distinguen los 16 estados y alcanzan toda la utilidad esperada 4q. La realización 1,1,2 y la comprobación de los 16 perfiles están en las notas. 1.75 es el óptimo laminar verificado; 1.5 la regla concreta ABC, D, A si hace falta. El orden importa: después de ABC mixto, probar A permite que la última sea D si ABC ya quedó resuelto o B si queda un par mixto. Ese plan da 1.75, y coincide con el piloto de cociente. No decir que usar cociente alcanza siempre el óptimo ni que este caso ya ejecuta el experimento de primer cruce. La propuesta completa compara general, cambio antes del primer cruce a continuación laminar óptima, cambio a cociente, óptimo laminar desde la raíz y cociente desde la raíz. Así se separan pérdida de clase, compromiso con prefijo y heurística. Empezar con los 8 casos acordados tras fijar definición y desempate. Respecto a Nick: queremos transformar dinámicas generales en dinámicas laminares; su destino es estático y permite cruces. Estudiar la técnica no transfiere automáticamente el factor 2.
''')
