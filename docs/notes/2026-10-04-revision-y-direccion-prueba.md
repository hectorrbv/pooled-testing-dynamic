# Dirección de investigación: qué conservar del paper y qué demostrar

La prioridad es separar la pérdida de adaptación, la pérdida por exigir laminaridad y la pérdida de la regla que construye la continuación. El notebook 28 ya demuestra una separación entre clases; todavía no proporciona una aproximación general.

Revisión realizada con Claude Opus (`claude-opus-5-5`): una lectura crítica del notebook y §5, seguida de una revisión específica de las proposiciones 1/G, bloques residuales y obstrucción estática. Los [comentarios originales](../../results/sesion_2026-10-04/revision_opus/revision.md) y la [segunda revisión](../../results/sesion_2026-10-04/revision_opus/revision_lemas.md) quedan separados de las conclusiones contrastadas de esta nota. La revisión por un modelo no sustituye una demostración.

Convenciones: q es probabilidad de salud; pruebas de conteo exacto, soft clearing, utilidades no negativas, independencia y 0<q_i<1. Salvo indicación contraria, historial inicial vacío. B es un límite en cada trayectoria y G limita el tamaño de cada prueba. Laminaridad significa que los grupos de una trayectoria son disjuntos o anidados.

## 1. Un resultado elemental que sí podemos justificar

Sea w_i=u_iq_i y sea T_k la suma de los min(k,n) valores w_i más grandes. Entonces

\[
V^*_{general}\leq T_{BG}\leq G T_B\leq G V^*_{laminar}.
\]

Por tanto, probar individualmente a las mejores B personas ya garantiza 1/G del óptimo general. Esta es una cota básica, sin afirmación de novedad, y no implica que las individuales sean óptimas.

Prueba. Para acreditar a i es necesario haberlo incluido en alguna prueba: sin esa observación, independencia y soporte completo mantienen ambas posibilidades para su estado. El evento «i aparece alguna vez» es independiente de su propio estado. En efecto, fijando todos los demás estados y el azar de la política, cambiar solo el estado de i no cambia ninguna decisión anterior a su primera aparición. Sea a_i la probabilidad de que aparezca. La utilidad esperada es como máximo sum_i w_i a_i. Cada trayectoria contiene a lo sumo BG personas distintas, de modo que 0<=a_i<=1 y sum_i a_i<=BG. Maximizar esa suma ponderada da T_BG. El orden de los pesos implica T_BG<=G T_B; las mejores B individuales alcanzan T_B y forman una política laminar.

Alcance. Esto sirve para G fijo. Una constante independiente también de G exige otro argumento. No aplicar esta prueba directamente a un historial ya condicionado: la independencia inicial usada para la primera aparición puede haberse perdido.

Con q_i=q, u_i=1 y B<=n hay otra cota, propuesta por Opus y comprobada aquí:

\[
V^*_{general}\leq\frac{B\log_2(G+1)}{\log_2(1/q)},\qquad
\frac{V^*_{laminar}}{V^*_{general}}
\geq\frac{q\log_2(1/q)}{\log_2(G+1)}.
\]

Prueba. Una historia h que acredita k personas implica que esas k están sanas, de modo que P(h)<=q^k. Promediar -log_2 P(h) da H(h)>=E[k]log_2(1/q). Una política determinista obtiene a lo sumo log_2(G+1) bits por prueba y, con un máximo de B pruebas, H(h)<=B log_2(G+1). Las acciones y la parada dependen del historial, por lo que no agregan información; se pueden rellenar las trayectorias cortas con símbolos deterministas. Condicionar en la semilla extiende el argumento a políticas aleatorias. Comparar con B individuales, de valor Bq, da el cociente. Para q=1/2 resulta 1/[2 log_2(G+1)]. Se puede tomar el máximo de esta garantía y 1/G; no siempre mejora a la primera.

No extrapolar esta última fórmula a utilidades arbitrarias ni a probabilidades heterogéneas. Tampoco es una prueba de que esa dependencia logarítmica sea necesaria para el óptimo dinámico laminar.

## 2. Un límite concreto: la salida no puede ser siempre estática

Con q_i=1/2 y u_i=1, cualquier plan estático laminar de B pruebas tiene utilidad esperada a lo sumo B/2. La misma cota vale si elegimos al azar ese plan antes de observar a la población real.

Prueba. En una familia laminar fija, restar a cada grupo sus hijos máximos produce a lo sumo B bloques residuales disjuntos con conteo conocido. Un bloque de tamaño m acredita a sus integrantes exactamente cuando su conteo es cero. Su contribución esperada es m/2^m<=1/2. Se suman como máximo B contribuciones. Fuera de los grupos probados no se acredita a nadie. Promediar sobre planes independientes de la población conserva la cota.

El resultado clásico de [Cantor y Mills (1966)](https://doi.org/10.4153/CJM-1966-007-2) proporciona familias de O(n/log n) grupos cuyos conteos distinguen todos los perfiles binarios, permitiendo G=n. Esos planes con cruces identifican a todos y obtienen n/2 con el prior anterior. Para B=O(n/log n),

\[
\frac{V^*_{estático\ laminar}}{V^*_{general}}
\leq\frac{B}{n}=O(1/\log n)\longrightarrow0.
\]

Consecuencia: no puede existir una conversión universal que conserve una fracción constante y termine siempre en planes estáticos laminares. Esto también descarta distribuciones independientes de tales planes. No refuta una garantía para políticas dinámicas laminares: sus bloques dependen de los resultados reales y el cálculo para una partición fija deja de ser aplicable.

Es una obstrucción para una versión más restrictiva de nuestra meta, no una prueba de que la meta dinámica sea falsa. Tampoco prueba nada contra §5 del paper de Nick, cuyo modelo y clase de salida son distintos.

Para atacar la meta dinámica, Opus propone estudiar si, con q_i=1/2 y u_i=1, existe K absoluto con V*_{dinámico laminar}<=KB. Si se demostrara, las mismas matrices refutarían una constante universal también en la clase dinámica. Esta desigualdad sigue siendo una conjetura: no tenemos una prueba ni un contraejemplo. No basta una cota de peor caso para identificar a todos; aquí necesitamos una cota de utilidad esperada con presupuesto fijo.

## 3. Qué tomar de §5

En [§5.4 del paper](https://arxiv.org/html/2601.22419v2#S5.SS4), dos perfiles independientes con el mismo prior producto se condicionan en sus pares no ordenados, coordenada por coordenada. Donde discrepan, las dos orientaciones tienen igual probabilidad; quedan perfiles complementarios uniformes sobre un subcubo. Esta identidad probabilística también vale con nuestras pruebas de conteo: no depende de cómo observemos los estados.

Lo pendiente es la desigualdad que se quiere promediar mediante esa identidad. El lema del paper usa pruebas binarias, hard clearing y oportunidades de acreditar a una persona mediante una prueba negativa. No justifica una desigualdad análoga para una conversión laminar con soft clearing. Además, el recorte independiente puede conservar cruces.

Una formulación suficiente, todavía por demostrar para una transformación concreta, sería comparar en cada subcubo la suma de probabilidades de acreditar a cada persona bajo la política original con la suma correspondiente a la transformada evaluada sobre el perfil complementario. Eso permitiría reutilizar la simetrización para priors heterogéneos. Una desigualdad por persona y por subcubo es más fuerte que una garantía agregada para una instancia; puede fallar aunque una regla siga funcionando bien en promedio.

Como objeto propio del modelo, conviene usar los bloques residuales con conteo conocido. Tras deduplicar pruebas repetidas, sus conteos se obtienen restando conteos de hijos a los de padres; los priors condicionados factorizan entre bloques, aunque no son uniformes si los q_i difieren. Con soporte completo, un bloque de conteo cero acredita a todos; si su conteo iguala su tamaño, todos están infectados; un bloque mixto no acredita a nadie por sí solo. Una nueva prueba laminar solo corta parcialmente un bloque; los demás los contiene completos o los evita. Esto ya sustenta `laminar_inference.py`: no se presenta como hallazgo nuevo. Puede servir para formular un argumento sobre cuánta utilidad pierde cada refinamiento permitido.

## 4. La comprobación nueva cambia el próximo experimento

Con n=4, B=2, G=2, q_i=0.3 y u_i=1, consideremos:

- Probar AB.
- Si R=1, probar A y acreditar al sano de AB.
- Si R=0 o R=2, probar C.

Esta política ya es laminar y vale 0.774. Un camino virtual elige el calendario AB,A con probabilidad 0.42 o AB,C con probabilidad 0.58. Evaluados sobre otra población independiente, esos calendarios valen 0.60 y 0.48. Su promedio es

\[
0.42(0.60)+0.58(0.48)=0.5304.
\]

Se perdió utilidad antes de intentar eliminar cruces. La mejor política estática laminar aquí vale 0.60, alcanzado por dos individuales: cada bloque residual de tamaño m aporta m(0.3)^m<=0.3 y hay a lo sumo dos bloques. Por tanto, 0.774-0.60=0.174 es la brecha entre esas clases; 0.60-0.5304=0.0696 es pérdida adicional del procedimiento de muestreo. No llamar «brecha de adaptación» al cociente 0.774/0.5304. Conservar el prefijo de un camino simulado no es el control más limpio para estudiar el precio de laminaridad.

Se comprobaron exactamente los seis casos previos con historial vacío, sin Monte Carlo. El resultado depende de qué política óptima se transforme: en el caso de cuatro personas, el árbol óptimo dinámico guardado vale 2, pero reproducir un camino virtual da 53/32. En cambio, el calendario fijo ABC,ABD,ACD también vale 2 y no cambia al muestrear caminos. Los desempates entre políticas óptimas sí importan para este procedimiento.

## 5. Próximo experimento recomendado

Seguir la política fuente con resultados reales mientras sus acciones sigan siendo laminares. Justo cuando proponga el primer cruce, reemplazar su continuación. Esta regla conserva íntegra una política fuente que ya es laminar.

Primero usar la mejor continuación laminar exacta disponible en ese historial. Después repetir con π_ratio. Medir:

1. Óptimo general frente al óptimo laminar desde el inicio: pérdida de clase.
2. Óptimo laminar desde el inicio frente a conservar el prefijo real y continuar de forma laminar óptima: pérdida por comprometer ese prefijo.
3. Continuación óptima frente a π_ratio tras el mismo prefijo: pérdida de la heurística.

Mantener el muestreo de caminos como control separado y añadir π_ratio desde el inicio: así sabremos si conservar parte de la política fuente aporta algo. No usar el muestreo como prueba del precio de laminaridad. Empezar por los seis casos ya guardados; conservar aparte los historiales condicionados. No está implementada todavía esta comparación de continuaciones tras el primer cruce.

En paralelo, como comprobación de la meta teórica, localizar una matriz detectora explícita algo mayor que el ejemplo de cuatro personas, verificar que sus firmas son distintas y calcular el óptimo dinámico laminar con el estado por tamaños y conteos. Opus recordó una familia con n=12 y B=7 como siguiente tamaño, pero esa parametrización concreta no se ha verificado en la fuente; no la presentamos como un experimento listo. Una caída en el cociente sería evidencia para investigar, no una refutación asintótica.

## Para discutir con Francisco

«Ya sabemos que los cruces pueden aportar: 2 frente a 1.75. También sabemos que fijar un camino puede perder utilidad incluso si no hay cruces. Quiero separar esas dos cosas».

«Tenemos una garantía elemental de 1/G. ¿Buscamos mejorarla para G pequeño, o una constante que tampoco dependa de G?».

«La versión estática laminar no puede dar una constante universal cuando G crece. Propongo conservar la adaptación y estudiar qué se pierde al llegar al primer cruce».

«De §5 reutilizaría la simetrización. El trabajo nuevo sería una desigualdad para la transformación con conteos, no asumir que sobrevive el factor dos».

Precisión sobre la revisión: la frase final de Opus «la mitad de las veces» para el ahorro del par no vale en general. La probabilidad exacta es q_1q_2+(1-q_1)(1-q_2). La mejora estricta sobre las individuales sí vale bajo las condiciones de la consulta del notebook, y la política que la obtiene es laminar.

## Verificación y alcance

[Comprobaciones exactas](../../results/sesion_2026-10-04/revision_opus/verificacion_direccion.json) y [script reproducible](../../results/sesion_2026-10-04/revision_opus/verifica_direccion.py). Además de los seis casos, se revisaron las 276 familias laminares de hasta tres pruebas sobre cuatro personas: 2332 historias compatibles y 2460 posibles refinamientos legales. Las pruebas matemáticas anteriores son los argumentos generales; la enumeración es una comprobación finita adicional.
