# augmented/notebooks/28_pooling_y_precio_laminar.ipynb

## 1. ¿Cuánto cuesta exigir laminaridad?

General: cualquier grupo de hasta $G$ personas.
Laminar: en cada rama, los grupos son disjuntos o uno contiene al otro.

AB seguido de A es laminar; AB seguido de AC no lo es.

Comparamos el mejor valor esperado de cada clase. Ambas adaptan las pruebas y tienen los mismos datos y presupuesto $B$.

$n=6$, $u_i=1$, estados independientes; $q$ es la probabilidad de estar sano y $p=1-q$. Observamos el conteo de infectados $R$ y acreditamos a quienes sabemos sanos con certeza: soft clearing.

## 2. Con los mismos recursos: 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

- Óptimo general: 3 sanos acreditados en promedio.
- Óptimo laminar: 2.5.

La restricción cuesta una sexta parte del valor general: aproximadamente 16.7%.

## 3. La pérdida aparece con presupuestos intermedios

228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

Color: pérdida porcentual respecto al general. Blanco: empate. El contorno delimita los casos con pérdida.

![Pérdida por laminaridad: se destaca el caso B=4, G=3, q=0.5 de la pantalla anterior.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.

## 4. ¿Qué podemos garantizar al imponer laminaridad?

El caso de 3 frente a 2.5 descarta una garantía universal superior a $5/6$ del valor general. Todavía falta demostrar cuánto podemos garantizar.

Propuesta pendiente: seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar. Compararla con el óptimo laminar desde el inicio.

¿Conviene conservar esas primeras decisiones o modificarlas antes?

[Guion oral](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Consultas](28_notas_de_estudio.ipynb).

# augmented/notebooks/28_notas_de_estudio.ipynb

# Notebook 28 · material de estudio

[Volver a las cuatro pantallas de presentación](28_pooling_y_precio_laminar.ipynb).

La presentación principal mantiene una sola comparación: óptimo general frente a óptimo laminar. Estas consultas responden preguntas diferentes y se abren por separado:

- Agrupar frente a individuales: el mapa y el ejemplo del par.
- Evaluar una regla concreta: el piloto de conversión y su continuación.
- Estudiar §5 de Nick: planes estáticos, políticas dinámicas y la demostración.
- Revisar tareas, cálculos y ejemplos adicionales.

Salvo en las consultas del paper de Nick: conteos exactos, soft clearing, estados independientes, q = probabilidad de estar sano, p = 1−q y utilidad 1 por persona.


## Otra pregunta: laminar frente a individuales

$n=6$, $B=1,\ldots,6$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$: 228 casos exactos.

En este experimento: $q$ es la probabilidad de estar sano, $p=1-q$, estados independientes y $u_i=1$. Cada prueba devuelve el conteo de infectados $R$; usamos soft clearing: acreditamos a quien sabemos sano con certeza.

Laminar significa que los grupos son disjuntos o uno contiene al otro. Comparamos el óptimo laminar con usar solo individuales, cuyo valor es $Bq$. El color muestra la mejora porcentual: 50% significa cobrar 1.5 veces esa referencia.

![Ganancia del óptimo laminar sobre individuales: con una prueba depende de q; con dos a cinco hay mejora en toda la malla; con seis hay empate.](attachment:laminar_vs_individuales.png)

- $B=1$: agrupar mejora si $q>0.5$; con $q\leq0.5$ no mejora.
- $B=2,\ldots,5$: mejora en los 152 casos, también con pocos sanos.
- $B=6=n$: empate; las individuales ya resuelven a todos.

## Por qué el par mejora a las individuales

$n=6$, $B=2$, $G=2$, $q_i=0.3$, $p_i=0.7$.

Dos individuales, A y B, dan $0.3+0.3=0.6$ de utilidad esperada. Probando primero AB también resolvemos a ambos con un máximo de dos pruebas:

![Tras probar AB, los conteos cero o dos resuelven a ambos y permiten probar C; con conteo uno, probar A identifica al sano de AB.](attachment:par_libera_prueba.png)

La oportunidad adicional aparece cuando AB ya está resuelto: $0.09+0.49=0.58$. Entonces podemos probar C, que está sano con probabilidad 0.3.

$$E[U]=\underbrace{0.6}_{\text{A y B}}+\underbrace{0.58\times0.3}_{\text{oportunidad de C}}=0.774.$$

Es el 29% de mejora del mapa de laminar frente a individuales. Esta política es laminar y aquí también alcanza el óptimo general.

## Otra pregunta: cuánto pierde una regla concreta

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $p_i=0.5$.

- 2.00: el plan no laminar ABC → ABD → ACD identifica a todos mediante sus tres conteos.
- 1.75: el mejor valor posible si exigimos laminaridad, aun adaptando las pruebas.
- 1.50: una conversión simple que prueba ABC, luego D y, si ABC sigue mixto, A.

Aquí perdemos 0.25 por la restricción y otros 0.25 por cómo elegimos las pruebas. Conservar ABC y continuar con la regla del cociente alcanza 1.75 en este piloto. Esa regla compara utilidad esperada por prueba esperada y vuelve a decidir tras cada resultado.

La siguiente pregunta: ¿cuánto conservamos si seguimos la política general y cambiamos justo antes de perder laminaridad? Compararemos la mejor continuación laminar con la regla del cociente. Ese experimento sigue pendiente.

[Guion para presentarlo](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Árboles, tareas y estudio de §5](28_notas_de_estudio.ipynb).

## Para trabajar. Mis tareas para la sesión del 6 de octubre

Según el [plan del 5–6 de octubre](../../docs/notes/2026-10-05-plan-mon-tue.md), en este orden:

1. §5 del paper: preparar una página con los pasos de la demostración. Para cada paso: sirve tal cual, necesita otra idea o falla con conteos; explicar por qué en una frase.
2. Conversión laminar: trabajar los ocho casos existentes. Fijar qué pruebas respetan la laminaridad y cómo desempatar; cambiar antes de ejecutar la primera prueba incompatible. Comparar continuaciones general, laminar óptima y $\pi_{ratio}$, ponderadas por la probabilidad de llegar, y el óptimo laminar desde el inicio. Separar las pérdidas por laminaridad, por conservar el prefijo y por la heurística.
3. Revisión con Vladimir: leer la proposición sobre individuales, ejecutar `pytest augmented/tests_brecha_convencion.py` y cerrar la revisión pendiente de la Parte 2; registrar acuerdo u objeciones.

Lunes 5: revisión antes de las 20:00. Martes 6: reunir comentarios entre 12:00 y 13:00, ensayo a las 16:00 y sesión a las 19:00. Horas de Praga.

## Para estudiar el paper. La simulación pertenece a la demostración

Nick: prueba binaria, hard clearing. Negativo significa que todos están sanos; positivo, que hay al menos un infectado.

![La población ficticia selecciona un camino. Sus grupos se recortan aleatoriamente y se evalúan en otra población independiente. Es una construcción matemática del paper.](attachment:nick_construccion.png)

La meta de [§5](https://arxiv.org/html/2601.22419v2#S5) es demostrar que existe un plan fijo con al menos la mitad del valor de una política dinámica. Si el promedio de los planes construidos alcanza esa mitad, alguno de ellos también la alcanza.

El recorte retiene cada aparición de una persona con probabilidad $1/2$. Esas monedas son independientes entre grupos, incluso para la misma persona. Este proceso no es cómo deciden $\pi_L$ y $\pi_{ratio}$.

## Para estudiar §5.4. Las dos orientaciones pesan lo mismo

Una persona, $q=0.3$ sano y $p=0.7$ infectado. Sorteamos su estado dos veces, independientemente y con el mismo prior.

![Cuatro resultados posibles de dos sorteos: sano-sano 0.09, sano-infectado 0.21, infectado-sano 0.21 e infectado-infectado 0.49. Al condicionar en que difieren, las dos orientaciones tienen probabilidad un medio.](attachment:nick_dos_perfiles.png)

Aunque $q\ne1/2$, si sabemos que los dos estados difieren, cada orientación tiene probabilidad $1/2$. Con varias personas independientes, ocurre lo mismo en cada coordenada donde los perfiles difieren. [§5.4](https://arxiv.org/html/2601.22419v2#S5.SS4).

Podemos reutilizar esta identidad probabilística. Todavía necesitamos otra desigualdad que controle cuánta utilidad pierde nuestra transformación laminar.

## Consulta. El mapa anterior con cuatro personas

$n=4$, $B\in\{2,3,4\}$, $G\in\{2,3\}$, $q=0.05,0.10,\ldots,0.95$, $u_i=1$.

![Pérdida relativa por exigir laminaridad en la malla anterior de 114 casos con n=4.](attachment:mapa_precio_laminar.png)

Con $B=2$ hay empate en esta malla; con $B=3$ la pérdida alcanza 12.5%; con $B=4=n$ todos pueden probarse individualmente. El empate con cuatro pruebas se debe aquí al tamaño de población. En la malla nueva de $n=6$, cuatro pruebas sí pueden separar ambos óptimos.

## Consulta. Tres conteos positivos pueden identificar a los sanos

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $u_i=1$.

![Las pruebas ABC, ABD y ACD devuelven 1, 1 y 2 infectados. Juntas fuerzan que A y B sean sanos y C y D infectados.](attachment:conteos_identifican.png)

El dibujo muestra una realización. Los tres conteos distinguen los 16 perfiles posibles: el plan completo obtiene 2 de utilidad esperada. El óptimo laminar, aun adaptándose, obtiene 1.75.

El aporte de estas pruebas no laminares es combinar información para deducir identidades, incluso sin una prueba negativa.

## Consulta. Fijar un camino puede perder valor antes de imponer laminaridad

$n=4$, $B=2$, $G=2$, $q_i=0.3$, $u_i=1$.

![Tras probar AB, se prueba C si R es cero o dos, y A si R es uno. Las probabilidades de las ramas son 0.09, 0.42 y 0.49; sus utilidades totales esperadas son 2.30, 1 y 0.30.](attachment:adaptacion_y_muestreo.png)

La política del árbol vale 0.774; el mejor plan fijo laminar, 0.600.

Un camino simulado elige AB → A (42%) o AB → C (58%). Al aplicar esas listas a otra población independiente, valen 0.60 y 0.48:

$$0.42(0.60)+0.58(0.48)=0.5304.$$

Dentro de la clase laminar, fijar las decisiones cuesta 0.174. Este muestreo pierde otros 0.0696.

La política inicial ya es laminar: aquí se pierde valor sin modificar su laminaridad. Es un control con conteos, sin el recorte aleatorio de Nick.

## Consulta. Replanear cambia dónde gastamos la última prueba

$n=4$, $B=3$, $G=3$, $q_i=0.5$, $u_i=1$.
$\pi_{ratio}$ compara planes locales de hasta tres pruebas, limitados por el presupuesto, por $E[U]/E[T]$. Ejecuta una prueba y vuelve a decidir.

Ambas reglas conservan ABC. Quitar repetidos elimina a quienes ya aparecieron: ABC → ABD → ACD queda en ABC → D. La prueba libre se dedica al cobro inmediato.

Si ABC ya está resuelto, ambas reglas prueban D.

![Tras ABC mixto, quitar repetidos lleva a probar D primero. Replanear prueba A y dedica la última prueba a D solo si ABC ya quedó resuelto; en otro caso refina con B.](attachment:orden_continuacion.png)

Las políticas completas valen 1.50 y 1.75. Frente al plan no laminar de valor 2, la primera pierde 0.50: 0.25 por la clase laminar y 0.25 por la regla. Replanear recupera aquí la segunda parte.

Propuesta: seguir la política general con resultados reales y cambiar antes de ejecutar la primera prueba incompatible con la laminaridad. Comparar continuación laminar óptima, cociente y cociente desde el inicio.

## Consulta. Con una sola prueba, agrupar puede perder

$n=2$, $B=1$, $G=2$, $q_A=0.9$, $q_B=0.1$, $u_A=u_B=1$.
Conteos y soft clearing; estados independientes.

![Probar A obtiene 0.90 y probar AB obtiene 0.18.](attachment:una_prueba.png)

La individual A obtiene 0.90; el grupo AB, $2(0.9)(0.1)=0.18$.
Refuta que siempre convenga agrupar dos o más personas. No resuelve la afirmación restringida a $B\geq2$.

## Consulta. Dos parámetros distintos

$\lambda$ es el precio por prueba que elegimos en $\pi_L$: $E[U]-\lambda E[T]$.

$\Lambda$ es una cantidad determinada por la instancia del paper:

$$\Lambda(J,G)=\max_{1\leq|A|\leq G}\sum_{i\in A}\prod_{j\in A\setminus\{i\}}q_j.$$

Los singletons dan 1, por lo que $\Lambda\geq1$. Si todos los $q_i\leq1/2$, entonces $\Lambda=1$ y su modelo binario admite un óptimo individual. [§3.2](https://arxiv.org/html/2601.22419v2#S3.SS2).

No son el mismo parámetro. En el ejemplo del [notebook 27](27_sesion_francisco.ipynb), con $\lambda=0.30$ y $0.31$ usábamos $q_i=0.20$, $p_i=0.80$, y ambos abrían el mismo trío.

## Consulta. Qué sigue abierto después del contraejemplo

Con conteos y soft clearing, $B\geq2$, $G\geq2$, $n>B$, priors independientes $0<q_i<1$ y utilidades positivas, una política con un par mejora la mejor política formada solo por individuales.

Ordena por $u_iq_i$. Sustituye las dos primeras individuales por probar el par y, solo si es mixto, probar a uno de sus miembros. Recuperas la misma información y utilidad de ambos. Si el conteo es cero o dos, ahorras una prueba y la dedicas a la persona $B+1$.

La mejora es

$$[q_1q_2+(1-q_1)(1-q_2)]u_{B+1}q_{B+1}>0.$$

La política construida es laminar y mejora el mejor plan de individuales. No demuestra que abrir cualquier grupo sea óptimo, ni que toda primera acción óptima deba agrupar.

En el plan, el caso general sigue como derivación verificada: falta cerrar la nota escrita y la revisión de Héctor.

## Consulta. Objetivo y materiales

Queremos una transformación $T$ que, dada una política general $\pi$, produzca políticas laminares admisibles y satisfaga

$$E_\omega[V(T(\pi,\omega))]\geq cV(\pi).$$

$\omega$ es azar independiente de la población real; $V$ ya promedia sobre la salud. La constante $c>0$ debería ser independiente de $n,B,G$ para obtener la garantía deseada. Cada salida respeta el presupuesto y el tamaño de grupo. La política transformada puede seguir siendo adaptativa.

Si se demostrara para toda $\pi$, implicaría $V^*_{laminar}\geq cV^*_{general}$. Desde historial vacío, con priors independientes de soporte completo, ya hay una garantía básica de $1/G$; el argumento está en la nota enlazada abajo. No se aplica directamente a un historial condicionado. Una constante independiente también de $G$ sigue abierta para políticas dinámicas laminares; para planes estáticos laminares no existe tal constante universal.

La revisión también separa fijar un camino e imponer laminaridad: una política ya laminar de valor 0.774 baja a 0.5304 al reproducir un camino virtual; la mejor estática laminar vale 0.60. Por eso el próximo control conserva las decisiones tomadas con resultados reales y compara continuación óptima con $\pi_{ratio}$.

[Revisión con Opus y dirección de prueba](../../docs/notes/2026-10-04-revision-y-direccion-prueba.md).

[Guion de la presentación y orden de estudio](../../docs/notes/2026-10-06-guion-notebook-28.md).

[Guía y próximos pasos](../../docs/notes/2026-10-04-preparacion-siguiente-sesion.md) · [Evidencia exacta](../../results/sesion_2026-10-04/evidencia_28.json).

Los mapas se calculan con `python -m augmented.experimento_mapa_laminar_28`, `python -m augmented.experimento_agrupar_individuales` y `python -m augmented.experimento_tres_referencias_28`. Para reconstruir ambos notebooks desde los datos guardados: `python augmented/notebooks/build_28_sesion.py`.