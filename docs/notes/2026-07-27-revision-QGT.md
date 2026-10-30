# Estado del arte: conteo agrupado adaptativo, políticas laminares e inferencia condicional

_Corte de búsqueda: 27 de julio de 2026._

## Dictamen ejecutivo

La crítica es correcta en lo esencial: el proyecto estaba usando lenguaje de
novedad para piezas probabilísticas y algorítmicas que ya pertenecen a
literaturas maduras. El choque más importante no es sólo con “binary
splitting”, sino con un antecedente mucho más directo:

- Wang, Zhao y Chuah estudian **quantitative group testing adaptativo con
  planes nested**, definen “nested” mediante exactamente la condición de que
  dos pools sean disjuntos o uno contenga al otro, y obtienen el plan nested
  óptimo en forma cerrada para el criterio minimax de identificación completa
  [Wang et al. 2018].
- Han, Rajan, Frazier y Jedynak estudian **group testing bayesiano bajo
  observaciones de suma**, con prior conjunto, horizonte fijo, política
  adaptativa greedy y una garantía de aproximación para pérdida entrópica
  [Han et al. 2017].
- Tarlow et al. estudian **inferencia exacta en modelos recursivos de
  cardinalidad**, incluyendo restricciones cardinales organizadas como árbol,
  mediante belief propagation exacto [Tarlow et al. 2012].
- Condicionar Bernoullis independientes a una suma fija es la distribución
  **conditional Bernoulli**, también llamada conditional Poisson o rejective
  sampling según el área. Su ley, marginales, muestreo y dependencia negativa
  tienen literatura extensa [Chen--Liu 1997; Borcea--Brändén--Liggett 2009;
  Adamczak--Polaczyk 2023].

Por tanto, ni el “tensor”, ni la factorización elemental de átomos residuales,
ni el signo de las covarianzas deben presentarse como resultados nuevos. El
Lema A es, a lo sumo, una **especialización útil y una exposición autocontenida**
de hechos conocidos.

Lo que sí parece quedar abierto —con la cautela de que una búsqueda no prueba
novedad— es la intersección exacta de estos cuatro ingredientes:

1. horizonte y presupuesto de tests fijos;
2. feedback de conteo exacto;
3. prior heterogéneo y utilidades individuales;
4. objetivo terminal de bienestar por individuos certificados como sanos.

Dentro de esa intersección, los blancos defendibles son:

- caracterizar cuándo una política óptima de bienestar vuelve a testar dentro
  de un pool observado;
- acotar o refutar la pérdida de restringirse a políticas laminares,
  \(V^{\mathcal L}/V^*\);
- demostrar el caso homogéneo con \(B=2\), si realmente es cierto;
- cuantificar separadamente error de creencias, restricción de acciones y
  miopía;
- estudiar robustez ante prior mal especificado o dependiente.

La recomendación editorial es tajante: **“inferencia laminar exacta” debe ser
infraestructura y Related Work, no el claim central**.

## 1. Dónde cae exactamente el modelo del proyecto

El nombre estándar del canal es **quantitative group testing (QGT)**,
**additive group testing** o **group testing under sum observations**. Para un
estado binario \(Z\in\{0,1\}^n\), una consulta \(A\subseteq[n]\) devuelve

\[
Y_A=\sum_{i\in A} Z_i.
\]

El modelo del proyecto no coincide por completo con el QGT clásico porque no
busca recuperar todo \(Z\) minimizando tests. Su objetivo es maximizar, con
presupuesto fijo, la utilidad de las personas que quedan certificadas como
sanas. Ésta es la distinción que debe cargar la novedad.

| Trabajo/línea | Canal | Prior | Adaptividad | Objetivo | Relación con el proyecto |
|---|---|---|---|---|---|
| QGT combinatorio clásico | conteo exacto | exactamente \(d\) defectuosos | sí/no | identificar todo \(Z\), peor caso | mismo canal, objetivo distinto |
| Wang--Zhao--Chuah (2018) | conteo exacto | \(|Z|=d\) | sí, nested | mínimo número de tests, minimax | choque directo en “laminar/nested” |
| Han--Rajan--Frazier--Jedynak (2017) | conteo exacto | prior bayesiano conjunto | sí | minimizar entropía posterior | choque directo en Bayes + suma + greedy |
| SQGT | conteo cuantizado | usualmente sparse/combinatorio | mayormente no adaptativo; extensiones por etapas | recuperación | aproxima un canal clínico realista |
| Finster et al. (2023) | binario limpio/no limpio | Bernoulli heterogéneo | estático | bienestar de sanos certificados | ancestro directo del objetivo |
| Lopez et al. (2026) | binario limpio/no limpio | Bernoulli heterogéneo | sí | bienestar de sanos certificados | ancestro directo dinámico |
| Este proyecto | conteo exacto | Bernoulli heterogéneo producto | sí | bienestar con \(B,G\) fijos | combinación específica no hallada |

“No hallada” significa que no apareció una coincidencia directa en la búsqueda
por QGT, sum observations, nested QGT, Bayesian adaptive group testing y
welfare-maximizing pooled testing. No equivale a una certificación de novedad.

## 2. QGT adaptativo y planes nested: el antecedente que hay que citar primero

### 2.1 Wang, Zhao y Chuah (2018)

El trabajo más cercano a la rama laminar es
[_Optimal Nested Test Plan for Combinatorial Quantitative Group Testing_](https://doi.org/10.1109/TSP.2017.2780053).
Su definición exige que para cualesquiera dos pools probados \(A,B\), se cumpla
\(A\cap B\in\{\varnothing,A,B\}\). Es la definición de familia laminar usada
en el repositorio.

El artículo:

- usa respuestas de conteo exacto;
- considera planes adaptativos;
- explota la descomposición del problema en dos subproblemas independientes
  después de observar el conteo de un subconjunto;
- caracteriza en forma cerrada el plan nested óptimo;
- prueba optimalidad en orden frente a todos los planes cuando \(d\) es fijo;
- observa que, si \(d\) es desconocido, el primer test nested óptimo prueba a
  toda la población y así revela \(d\).

El antecedente conceptual llega al menos a Aigner y Schughart (1985), quienes
ya habían determinado el número de tests del plan nested óptimo, aunque no el
plan cerrado. Por tanto, decir que “descubrimos que los conteos laminares se
prestan a una recursión sobre un árbol” sería indefendible.

La diferencia salvadora es el criterio. Wang et al. resuelven identificación
completa, prior uniforme sobre conjuntos de tamaño \(d\) y peor caso. Aquí se
optimiza bienestar esperado, con horizonte corto, pool máximo, probabilidades
y utilidades heterogéneas, y no se exige clasificar a todos.

### 2.2 La literatura nested binaria es todavía más antigua

La revisión de Malinovsky y Albert,
[_Nested Group Testing Procedures for Screening_](https://doi.org/10.1002/9781118445112.stat08363),
traza procedimientos jerárquicos y nested bajo modelos probabilísticos hasta
Sobel y Groll (1959). Reporta programación dinámica para el procedimiento
nested óptimo y una larga secuencia de mejoras.

Esto importa por dos razones:

1. “Nested” no es un nombre nuevo para una restricción inventada por el
   proyecto; es terminología establecida.
2. La justificación “restringimos a nested porque permite DP” también es
   clásica. La contribución tendría que ser una garantía para el nuevo
   objetivo o una caracterización de régimen, no la observación de
   tratabilidad.

### 2.3 Dorfman y binary splitting no son sinónimos

La crítica acierta al conectar la idea con procedimientos jerárquicos, pero
mezcla dos algoritmos distintos:

- **Dorfman** es un procedimiento de dos etapas: se prueba un grupo y, si sale
  positivo, típicamente se prueba individualmente a sus miembros.
- **Generalized binary splitting**, asociado a Hwang, busca defectuosos
  recursivamente mediante subdivisiones y es plenamente adaptativo.

Ambos generan estructura nested, pero no conviene escribir que binary
splitting es “literalmente Dorfman generalizado”. El lenguaje seguro es:
“las políticas laminares pertenecen a la familia histórica de procedimientos
jerárquicos/nested, que incluye a Dorfman por etapas y a variantes de binary
splitting”.

## 3. Bayes + conteos + adaptividad ya existe

Han et al.,
[_Bayesian Group Testing Under Sum Observations: A Parallelizable Two-Approximation for Entropy Loss_](https://doi.org/10.1109/TIT.2016.2628784),
consideran consultas que devuelven el número de objetos en el conjunto, un
prior bayesiano conjunto y un número fijo de preguntas. Proponen:

- una política diádica no adaptativa;
- optimalidad de esa política dentro de su clase;
- una aproximación factor 2 frente al óptimo adaptativo para pérdida de
  entropía;
- un greedy adaptativo que maximiza la reducción entrópica a un paso.

Éste es el antecedente que impide describir “posterior exacto + elegir el
siguiente pool con utilidad” como un paradigma nuevo. La diferencia vuelve a
ser la función objetivo: entropía/identificación en Han et al. frente a
bienestar de clearance en este proyecto.

La literatura reciente tampoco está quieta:

- Soleymani y Javidi, [_A Non-Adaptive Algorithm for the Quantitative Group
  Testing Problem_](https://proceedings.mlr.press/v247/soleymani24a.html)
  (COLT 2024), reducen la brecha entre cotas informacionales y algoritmos
  eficientes no adaptativos.
- Soleymani y Javidi, “Quantitative Group Testing with Tunable Adaptation”
  (ISIT 2024, DOI
  [10.1109/ISIT57864.2024.10619221](https://doi.org/10.1109/ISIT57864.2024.10619221)),
  estudian explícitamente el intercambio entre número de etapas y ganancia de
  adaptividad.
- Soleymani y Javidi,
  [_Learning to Ask: Decision Transformers for Adaptive Quantitative Group
  Testing_](https://arxiv.org/abs/2509.01723) (2025), formulan QGT adaptativo
  como recuperación de vectores enteros y usan offline RL; reportan bajar, en
  promedio, de la cota informacional no adaptativa.
- Tan, Pascual Cobo y Venkataramanan,
  [_Quantitative Group Testing and Pooled Data in the Linear Regime with
  Sublinear Tests_](https://arxiv.org/abs/2408.00385) (2024), dan un algoritmo
  AMP eficiente para el régimen lineal no adaptativo.

Estos trabajos tienen objetivos de recuperación y escalamiento asintótico, no
el bienestar del proyecto. Deben aparecer para situar el canal y evitar la
impresión de que el área terminó en los papers de pooled testing binario.

## 4. El tensor de un subpool es conditional Bernoulli elemental

Sean \(X_i\sim\mathrm{Bernoulli}(p_i)\) independientes sobre un átomo \(D\), y
condiciónese en \(\sum_{i\in D}X_i=c\). La ley resultante se conoce como
**conditional Bernoulli**; en muestreo también aparece como **conditional
Poisson** o **rejective sampling**. Chen y Liu sistematizan la Poisson-binomial
y conditional Bernoulli en
[_Statistical Applications of the Poisson-Binomial and Conditional Bernoulli
Distributions_](https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n44/j7n44.htm)
(1997).

Para \(S\subseteq D\), si \(f_A(k)\) denota la PMF Poisson-binomial del
subconjunto \(A\), entonces

\[
\Pr\!\left(\sum_{i\in S}X_i=s\;\middle|\;
                 \sum_{i\in D}X_i=c\right)
=\frac{f_S(s)f_{D\setminus S}(c-s)}{f_D(c)}.
\]

Ésta es exactamente la fila del tensor que el proyecto necesita. Si todos los
\(p_i=p\), los pesos se cancelan y queda

\[
\Pr\!\left(\sum_{i\in S}X_i=s\;\middle|\;
                 \sum_{i\in D}X_i=c\right)
=\frac{\binom{|S|}{s}\binom{|D|-|S|}{c-s}}
       {\binom{|D|}{c}},
\]

la distribución hipergeométrica. Implementarla y probarla es correcto;
presentarla como un lema probabilístico nuevo no lo es.

La literatura algorítmica va además más lejos que una tabla explícita. Heng,
Jacob y Ju estudian muestreo MCMC de conditional Bernoulli en
[_A Simple Markov Chain for Independent Bernoulli Variables Conditioned on
Their Sum_](https://arxiv.org/abs/2012.03103), y Adamczak y Polaczyk estudian
concentración específica para esta misma ley en
[_Concentration Inequalities for Some Negatively Dependent Binary Random
Variables_](https://doi.org/10.30757/ALEA.v20-48).

## 5. Asociación negativa: el hecho es conocido, pero la atribución de la crítica es demasiado rápida

Joag-Dev y Proschan introducen y desarrollan asociación negativa en
[_Negative Association of Random Variables with Applications_](https://doi.org/10.1214/aos/1176346079)
(1983). Entre sus ejemplos están la multivariada hipergeométrica y leyes
invariantes bajo permutación. Eso cubre limpiamente el caso homogéneo, donde
condicionar a la suma produce un subconjunto uniforme de tamaño fijo.

Pero con \(p_i\) heterogéneos la ley conditional Bernoulli **no es
intercambiable**. Por eso la frase “las variables intercambiables condicionadas
a su suma son NA, punto” no justifica por sí sola el caso general que usa el
repositorio.

El resultado general sigue siendo conocido. Una ruta moderna y segura es:

1. una medida Bernoulli producto tiene polinomio generador estable;
2. su componente homogénea de grado \(c\), que corresponde a condicionar en la
   suma \(c\), conserva la propiedad relevante;
3. las medidas strongly Rayleigh implican asociación negativa.

La referencia estándar es Borcea, Brändén y Liggett,
[_Negative Dependence and the Geometry of Polynomials_](https://doi.org/10.1090/S0894-0347-08-00618-8)
(JAMS 2009). La literatura de rejective sampling también usa explícitamente la
asociación negativa de estos indicadores; véase Bertail y Clémençon,
[_Sharp Exponential Inequalities in Survey Sampling: Conditional Poisson
Sampling Schemes_](https://arxiv.org/abs/1610.03776).

Conclusión editorial: el signo no es nuevo, pero para priors heterogéneos es
mejor citar conditional Poisson/strongly Rayleigh que atribuir todo sin matiz
a la hipergeométrica intercambiable de 1983.

## 6. El “Lema A” frente a modelos gráficos de cardinalidad

### 6.1 Qué hace el lema

Para una familia laminar observada, sea \(C(A)\) el conjunto de hijos
inmediatos de \(A\) y defínase el residuo

\[
D_A=A\setminus\bigcup_{C\in C(A)} C,
\qquad
c(D_A)=c(A)-\sum_{C\in C(A)}c(C).
\]

Los residuos no vacíos son disjuntos. Bajo un prior producto, la densidad
posterior restringida por las igualdades de conteo puede escribirse como
producto de factores, uno por átomo. Por ello:

- los átomos son independientes entre sí después de condicionar;
- dentro de cada átomo hay una conditional Bernoulli;
- la PMF de un pool candidato es la convolución de las contribuciones de los
  átomos intersectados y de los individuos aún no observados.

La demostración es correcta. Sin embargo, es una factorización directa de un
producto sujeto a restricciones sobre bloques disjuntos.

### 6.2 Qué ya estaba en la literatura

Tarlow et al.,
[_Fast Exact Inference for Recursive Cardinality Models_](https://arxiv.org/abs/1210.4899)
(UAI 2012), consideran potenciales que dependen del número de variables
binarias activas y luego potenciales cardinales recursivos organizados en
árbol. Dan marginalización y muestreo exactos mediante una representación
tree-structured y belief propagation, con costo \(O(D\log^2D)\) para su
construcción general.

El historial laminar del proyecto es una especialización particularmente
simple: los factores son igualdades duras de conteo, el prior aporta
potenciales unarios Bernoulli y las diferencias padre-hijo fijan directamente
los conteos residuales. El lenguaje honesto es:

> “Especializamos inferencia exacta para potenciales cardinales recursivos al
> historial de conteos laminar del problema de bienestar, obteniendo una
> implementación por átomos conditional-Bernoulli y PMFs Poisson-binomial.”

No es lenguaje honesto:

> “Descubrimos que la inferencia posterior exacta es tratable en familias
> laminares.”

### 6.3 Veredicto componente por componente

| Componente del Lema A | Estado |
|---|---|
| restar conteos padre-hijo | identidad elemental / folclor |
| residuos laminares disjuntos | propiedad básica de familias laminares |
| factorización entre residuos | corolario inmediato del prior producto |
| ley dentro del átomo | conditional Bernoulli conocida |
| PMF de un subpool | fórmula condicional Poisson-binomial conocida |
| convolución entre átomos | sum-product estándar |
| implementación integrada al objetivo de bienestar | contribución de ingeniería útil |
| usar la representación para comparar políticas | aplicación potencialmente publicable |

No se encontró una fuente que enuncie exactamente el mismo paquete con la
notación “átomos residuales de un historial adaptativo de bienestar”. Eso no
convierte el paquete en teorema nuevo cuando cada paso es una especialización
inmediata de resultados más generales.

## 7. Rollout: la Proposición B también es estándar

La desigualdad “un rollout exacto que incluye la acción de la política base no
puede rendir peor que esa política base” es la propiedad clásica de mejora de
política del rollout. Bertsekas, Tsitsiklis y Wu la estudian en
[_Rollout Algorithms for Combinatorial Optimization_](https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2020/03/rollout.pdf)
(1997).

Aquí sí hay valor en verificar cuidadosamente las hipótesis —misma clase de
acciones, evaluación exacta, estado posterior suficiente y cierre laminar—,
pero la desigualdad abstracta no debe venderse como contribución teórica. El
claim propio puede ser computacional: cuánto recupera el rollout en este
problema y qué parte del gap queda sin explicar.

## 8. Qué parte de la crítica del “greedy nunca anida” no está demostrada

La asociación negativa dentro de un pool con conteo fijo no implica que un
subpool sea menos atractivo que territorio virgen. La comparación correcta es
contra el prior, y depende de si el conteo observado fue alto o bajo respecto
a lo esperado.

En el caso homogéneo, si \(|A|=a\), se observó \(c(A)=r\), y se considera un
subpool \(S\subseteq A\) de tamaño \(t\), entonces

\[
\Pr(S\text{ limpio}\mid c(A)=r)
=\frac{\binom{a-r}{t}}{\binom{a}{t}}.
\]

Un pool virgen de tamaño \(t\) está limpio con probabilidad \((1-p)^t\). Si
\(r/a<p\), el pool observado resultó **más sano de lo esperado**, y sus
subpools pueden ser mucho más atractivos que un pool virgen, aunque \(r>0\).
Por ejemplo, con \(a=10,r=1,p=0.6,t=2\), la probabilidad anidada es
\(\binom{9}{2}/\binom{10}{2}=0.8\), mientras la virgen es \(0.4^2=0.16\).

Así que el falsificador empírico propuesto es muy bueno, pero su resultado no
se deduce del signo de la dependencia. La hipótesis refinada es:

> el greedy anida cuando un conteo previo identifica un bloque cuya tasa
> realizada aparente es suficientemente menor que la tasa de riesgo de las
> alternativas vírgenes, ponderada además por utilidades y tamaños.

Esto también explica por qué la ganancia puede aparecer en prevalencia alta:
un conteo inusualmente bajo crea un bloque posterior muy valioso.

## 9. Priors dependientes y mala especificación: la crítica es plenamente válida

El prior producto no es inocuo en una enfermedad contagiosa. La literatura ha
estudiado correlación por hogares, geografía o redes sociales:

- Lendle, Hudgens y Qaqish,
  [_Group Testing for Case Identification with Correlated
  Responses_](https://doi.org/10.1111/j.1541-0420.2011.01674.x)
  (_Biometrics_, 2012), modelan respuestas correlacionadas dentro de clusters
  y muestran que incorporar la correlación puede reducir tests.
- Lin et al.,
  [_Positively Correlated Samples Save Pooled Testing Costs_](https://doi.org/10.1109/TNSE.2021.3081759)
  (2021), prueban ahorro adicional bajo correlación positiva y proponen formar
  pools usando un grafo social; reportan reducciones aproximadas de 20--35%
  frente a agrupamiento aleatorio en sus experimentos.
- Best, Malinovsky y Albert,
  [_The Efficient Design of Nested Group Testing Algorithms for Disease
  Identification in Clustered Data_](https://doi.org/10.1080/02664763.2022.2071419)
  (2023), incorporan modelos beta-binomial para datos agrupados al diseñar
  procedimientos nested.

También hay literatura directamente sobre mismatch en diseño adaptativo.
Fan et al.,
[_Adaptive Group Testing with Mismatched Models_](https://arxiv.org/abs/2110.02265)
(2021), formulan el problema como Bayesian optimal experimental design y
estudian cómo parámetros de test mal especificados deterioran complejidad de
muestra y entropía posterior.

Aunque esas fuentes no usan la misma utilidad terminal, establecen que la
robustez no es una objeción periférica. El paper debe separar:

1. **error en las probabilidades marginales** \(p_i\);
2. **dependencia omitida** entre individuos;
3. **mismatch del canal**: sensibilidad, especificidad, dilución o error del
   conteo;
4. **shift de población** entre entrenamiento y despliegue.

El experimento mínimo recomendado es una matriz cruzada:

| Política | Creencia usada al decidir | Mundo generador |
|---|---|---|
| exacta laminar | prior correcto | prior correcto |
| exacta laminar | prior perturbado | prior correcto |
| Nick/independencia | prior correcto | prior correcto |
| Nick/independencia | prior mejor calibrado | prior dependiente/perturbado |

Debe reportarse welfare, regret frente a un oráculo que conoce el mundo, tasa
de anidamiento y calibración de \(P(R_t=0\mid H)\). Comparar sólo “posterior
exacto vs aproximado” bajo el mismo prior sintético favorece artificialmente a
la exactitud algebraica.

## 10. El canal de conteo exacto tampoco es una descripción automática de qPCR

SQGT modela un sumador seguido de un cuantizador. Emad y Milenkovic introducen
el marco en
[_Semi-Quantitative Group Testing_](https://arxiv.org/abs/1202.2887).
Nambiar et al.,
[_Semi-Quantitative Group Testing for Efficient and Accurate qPCR Screening of
Pathogens with a Wide Range of Loads_](https://arxiv.org/abs/2307.16352),
desarrollan un esquema adaptativo de dos etapas con umbrales de Ct y reportan
24% menos tests que GT binario manteniendo una tasa de falsos negativos
despreciable en su evaluación.

En qPCR, Ct depende de carga viral, dilución y ruido; no observa literalmente
el número de infectados salvo bajo un modelo adicional. Si el paper usa
motivación clínica, debe presentar tres canales:

- ideal: conteo exacto;
- semicuantitativo: intervalo/bin del conteo;
- realista: distribución de Ct condicionada en cargas y composición del pool.

Blackwell sí permite ordenar canales cuando uno es un garbling conocido de
otro, pero la ganancia de welfare del canal ideal es un upper bound de valor,
no evidencia de que el laboratorio entregue ese canal.

## 11. Mapa de novedad: qué retirar, qué conservar y qué atacar

### Retirar como claims de novedad

- “Descubrimos la distribución del subpool condicionada al conteo.”
- “Probamos asociación negativa inducida por el conteo.”
- “Descubrimos inferencia exacta para historiales nested/laminares.”
- “Probamos que rollout domina a su greedy base.”
- “Binary splitting muestra por primera vez que anidar sirve con conteos.”

### Conservar como infraestructura o lemas auxiliares citados

- fórmula conditional-Bernoulli/Poisson-binomial;
- descomposición en átomos residuales;
- implementación exacta y validada contra enumeración;
- cache y consultas de PMF por demanda;
- policy improvement como sanity check de implementación.

### Posibles contribuciones propias

1. **Frontera de anidamiento para bienestar.** Condiciones en
   \((p,u,B,G,H)\) bajo las cuales el óptimo o el greedy re-testa un
   descendiente.
2. **Precio de laminaridad.** Cota \(V^{\mathcal L}\geq\alpha V^*\) bajo
   hipótesis claras, o familia adversaria que descarte una cota constante.
3. **Caso homogéneo \(B=2\).** Demostrar o refutar \(V^{\mathcal L}=V^*\).
4. **Descomposición causal del regret algorítmico.** Restricción de acciones,
   aproximación posterior y miopía, variando uno a la vez.
5. **Robustez decision-theoretic.** Cuándo una política posterior-exacta bajo
   modelo equivocado pierde contra una heurística menos exacta y mejor
   calibrada.
6. **Valor de resolución bajo bienestar.** Curva binario \(\to\) bins SQGT
   \(\to\) conteo exacto con el mismo presupuesto y canal de ruido.

## 12. Blanco teórico inmediato: \(B=2\) homogéneo

Éste sigue siendo el objetivo más limpio porque la literatura nested de QGT no
resuelve el criterio de bienestar y la malla del repositorio sugiere igualdad.

Después del primer pool \(A\), por simetría cualquier segundo pool \(S\) queda
descrito por

\[
x=|S\cap A|,\qquad y=|S\setminus A|.
\]

Un cruce no laminar corresponde a \(0<x<|A|\) y \(y>0\). Condicionado en
\(c(A)=r\), su probabilidad de ser limpio es

\[
\frac{\binom{|A|-r}{x}}{\binom{|A|}{x}}(1-p)^y.
\]

La conjetura \(B=2\) puede atacarse mostrando que, para cada \(r\), maximizar la
ganancia esperada sobre \((x,y)\) alcanza un extremo compatible con laminaridad:
\(x=0\), \(y=0\), o inclusión completa cuando sea factible. La herramienta
natural parece ser log-concavidad/razones consecutivas de la hipergeométrica,
no asociación negativa en abstracto.

Antes de intentar la prueba completa conviene:

1. simbolizar la recompensa incremental exacta incluyendo personas ya
   certificadas;
2. verificar por fuerza bruta el claim de extremo para todos
   \(a,r,G\) pequeños, no sólo el valor total del DP;
3. buscar un contraejemplo directamente en las coordenadas \((a,r,x,y)\);
4. sólo después formular el lema de frontera.

Esta ruta produce un resultado estructural específico del objetivo del paper,
en vez de reprobar hechos generales de QGT.

## 13. Lenguaje seguro para la próxima conversación con Francisco

Versión breve:

> Revisamos la literatura y la inferencia no es la novedad: el tensor es
> conditional Bernoulli, la dependencia negativa es conocida y los planes
> nested de QGT existen desde hace décadas, con una caracterización cerrada en
> Wang--Zhao--Chuah. También hay inferencia exacta general para potenciales
> cardinales recursivos. Lo que no encontramos resuelto es el objetivo de
> bienestar con conteos, priors/utilidades heterogéneos y presupuesto corto.
> Reposicionamos la inferencia como infraestructura y enfocamos la teoría en
> cuándo el óptimo anida y cuánto cuesta restringirse a laminar.

Versión de contribución tentativa para un draft:

> Building on quantitative group testing under sum observations, nested test
> plans, and exact inference for recursive cardinality models, we study a
> distinct finite-budget objective: maximizing the expected utility of
> individuals certified as healthy. We use a specialized conditional-Bernoulli
> message-passing implementation for laminar histories, and investigate the
> structural and welfare consequences of restricting adaptive policies to
> nested tests.

No usar “first”, “novel” o “new theorem” hasta cerrar una búsqueda de citas
hacia atrás y hacia adelante alrededor de Wang et al., Han et al. y Tarlow et
al., y hasta tener un resultado propio que no sea una especialización
inmediata.

## 14. Bibliografía núcleo

1. S. X. Chen and J. S. Liu. “Statistical Applications of the
   Poisson-Binomial and Conditional Bernoulli Distributions.” _Statistica
   Sinica_ 7 (1997), 875--892.
   [Fuente](https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n44/j7n44.htm).
2. K. Joag-Dev and F. Proschan. “Negative Association of Random Variables
   with Applications.” _Annals of Statistics_ 11(1), 1983.
   [DOI](https://doi.org/10.1214/aos/1176346079).
3. J. Borcea, P. Brändén, and T. M. Liggett. “Negative Dependence and the
   Geometry of Polynomials.” _JAMS_ 22(2), 2009.
   [DOI](https://doi.org/10.1090/S0894-0347-08-00618-8).
4. D. Tarlow, K. Swersky, R. S. Zemel, R. P. Adams, and B. J. Frey. “Fast
   Exact Inference for Recursive Cardinality Models.” UAI 2012.
   [Preprint](https://arxiv.org/abs/1210.4899).
5. W. Han, P. Rajan, P. I. Frazier, and B. M. Jedynak. “Bayesian Group
   Testing Under Sum Observations: A Parallelizable Two-Approximation for
   Entropy Loss.” _IEEE TIT_ 63(2), 2017.
   [DOI](https://doi.org/10.1109/TIT.2016.2628784).
6. C. Wang, Q. Zhao, and C.-N. Chuah. “Optimal Nested Test Plan for
   Combinatorial Quantitative Group Testing.” _IEEE TSP_ 66(4), 2018.
   [DOI](https://doi.org/10.1109/TSP.2017.2780053),
   [preprint](https://arxiv.org/abs/1407.2283).
7. M. Aldridge, O. Johnson, and J. Scarlett. “Group Testing: An Information
   Theory Perspective.” _Foundations and Trends in Communications and
   Information Theory_ 15(3--4), 2019.
   [Preprint](https://arxiv.org/abs/1902.06002).
8. A. Emad and O. Milenkovic. “Semi-Quantitative Group Testing.” 2012.
   [Preprint](https://arxiv.org/abs/1202.2887).
9. Y. Malinovsky and P. S. Albert. “Nested Group Testing Procedures for
   Screening.” _Wiley StatsRef_, 2021.
   [DOI](https://doi.org/10.1002/9781118445112.stat08363).
10. M. Cuturi, O. Teboul, Q. Berthet, A. Doucet, and J.-P. Vert. “Noisy
    Adaptive Group Testing Using Bayesian Sequential Experimental Design.”
    2020. [Preprint](https://arxiv.org/abs/2004.12508).
11. M. Fan, B.-J. Yoon, F. J. Alexander, E. R. Dougherty, and X. Qian.
    “Adaptive Group Testing with Mismatched Models.” 2021.
    [Preprint](https://arxiv.org/abs/2110.02265).
12. S. Finster, M. González Amador, E. Lock, F. Marmolejo-Cossío, E. Micha,
    and A. D. Procaccia. “Welfare-Maximizing Pooled Testing.” EC 2023.
    [Preprint](https://arxiv.org/abs/2206.10660).
13. N. Lopez, F. Marmolejo-Cossío, J. R. Tello Ayala, and D. C. Parkes.
    “Dynamic Welfare-Maximizing Pooled Testing.” 2026.
    [Preprint](https://arxiv.org/abs/2601.22419).
14. A. Nambiar et al. “Semi-Quantitative Group Testing for Efficient and
    Accurate qPCR Screening of Pathogens with a Wide Range of Loads.” 2023.
    [Preprint](https://arxiv.org/abs/2307.16352).
15. M. Soleymani and T. Javidi. “A Non-Adaptive Algorithm for the
    Quantitative Group Testing Problem.” COLT 2024.
    [PMLR](https://proceedings.mlr.press/v247/soleymani24a.html).
16. M. Soleymani and T. Javidi. “Learning to Ask: Decision Transformers for
    Adaptive Quantitative Group Testing.” 2025.
    [Preprint](https://arxiv.org/abs/2509.01723).
17. S. D. Lendle, M. G. Hudgens, and B. F. Qaqish. “Group Testing for Case
    Identification with Correlated Responses.” _Biometrics_ 68(2), 2012.
    [DOI](https://doi.org/10.1111/j.1541-0420.2011.01674.x).
18. Y.-J. Lin, C.-H. Yu, T.-H. Liu, C.-S. Chang, and W.-T. Chen.
    “Positively Correlated Samples Save Pooled Testing Costs.” _IEEE
    Transactions on Network Science and Engineering_ 8(3), 2021.
    [DOI](https://doi.org/10.1109/TNSE.2021.3081759).
19. A. F. Best, Y. Malinovsky, and P. S. Albert. “The Efficient Design of
    Nested Group Testing Algorithms for Disease Identification in Clustered
    Data.” _Journal of Applied Statistics_ 50(10), 2023.
    [DOI](https://doi.org/10.1080/02664763.2022.2071419).

## 15. Próximo paso bibliográfico antes de afirmar novedad

Esta nota cubre el primer barrido y ya basta para corregir el posicionamiento.
Para una revisión publicable falta una búsqueda de citación en dos direcciones:

1. referencias y trabajos citantes de Wang--Zhao--Chuah (nested QGT);
2. referencias y trabajos citantes de Han--Rajan--Frazier--Jedynak (Bayesian
   sum observations);
3. trabajos citantes de Tarlow et al. que usen igualdades cardinales duras;
4. búsqueda por “probabilistic quantitative group testing”, “H-type group
   testing”, “spring-scale coin weighing”, “nested additive queries” y
   “Bayesian cardinality potentials”;
5. búsqueda separada por objetivos parciales: certification, clearance,
   selective classification y value of information bajo presupuesto.

Hasta completar esa red de citas, el estándar correcto es “we are not aware
of prior work combining...”, seguido de la lista precisa de ejes, nunca “this
is the first adaptive quantitative group testing method”.
