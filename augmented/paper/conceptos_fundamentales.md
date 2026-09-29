# Conceptos fundamentales del proyecto

Glosario en capas para dominar las partes 2–5 del proyecto (inferencia, algoritmos,
resolución y certificados). Cada concepto tiene tres niveles:

- 🟢 **La idea en llano** — una analogía o una frase, sin matemáticas.
- 🔵 **La versión precisa** — la definición real, con la notación del paper.
- 🟣 **Dónde vive en nuestro proyecto** — el archivo, el número o el resultado concreto.

Léelo en orden la primera vez: los conceptos se construyen unos sobre otros.
Después úsalo como diccionario.

---

## Bloque A — Inferencia: saber quién está limpio (capa 2)

### A1. Posterior y perfiles consistentes

🟢 Antes de medir, cada persona tiene una probabilidad de estar activa (el prior).
Después de cada measurement, esas probabilidades cambian: eso es el posterior. Un
"perfil consistente" es una hipótesis del mundo (quién está activo y quién no)
que no contradice ningún resultado observado.

🔵 El perfil latente es un vector Z ∈ {0,1}^n. La historia h es la lista de pares
(grupo t, conteo r). El posterior es P(Z | h) ∝ 1[cada measurement de h se cumple
bajo Z] · P(Z bajo el prior). Los perfiles consistentes son el soporte de esa
distribución: {Z : |t ∩ Z| = r para todo (t, r) ∈ h}.

🟣 `bayesian.py` (`bayesian_update_by_counting` calcula las marginales
P(Z_i = 0 | h) enumerando perfiles consistentes). El ejemplo canónico de
deducción cruzada: measurements {0,1}→1 y {1,2}→0 implican que la persona 0 está
activa sin haberla testeado sola.

### A2. #P-hardness (por qué la inferencia exacta es intratable)

🟢 Hay problemas donde *encontrar* una solución es fácil pero *contar cuántas
soluciones hay* es brutalmente difícil. Calcular el posterior exige exactamente
eso: contar todos los mundos consistentes con lo observado. Nuestro Proposición 1
dice que ese conteo es tan difícil como los problemas de conteo más difíciles
que se conocen.

🔵 #P es la clase de problemas de contar soluciones de problemas NP. Calcular la
marginal exacta P(X_i = 0 | Ax = r) es #P-hard, por reducción de #Exact Cover:
cada conjunto del sistema es una variable, cada elemento del universo es un
measurement con conteo 1, y con priors uniformes la constante de normalización
cuenta exactamente los exact covers.

🟣 Es el resultado teórico central del paper (Proposición 1). La ironía que
define el proyecto: las mismas deducciones cruzadas que hacen valioso el conteo
(A1) son las que vuelven intratable calcularlo en general. Por eso existen los
regímenes tratables (A3–A5) y el sampler (A6+).

### A3. Grupos disjuntos → la posterior factoriza

🟢 Si los grupos nunca comparten personas, cada grupo es un mundo aparte: lo que
aprendes de un grupo no dice nada de los otros. Entonces puedes resolver cada
grupo por separado y multiplicar. "Factorizar" = partir un problema grande en
problemas chicos independientes.

🔵 Si los soportes de los measurements son disjuntos, P(Z | h) = ∏_g P(Z_g | h_g):
la distribución conjunta es el producto de las distribuciones por grupo. Cada
factor se calcula por enumeración sobre |grupo| ≤ G personas — costo 2^G, no 2^n.

🟣 Primer régimen tratable del paper. Es también la razón profunda de que el
sampler descomponga en componentes conexas (A6): la factorización aplica entre
componentes aunque dentro de cada una haya traslape.

### A4. Estructuras laminares → programación dinámica

🟢 "Laminar" = los grupos están anidados como muñecas rusas o son disjuntos, pero
nunca se traslapan a medias. Esa jerarquía forma un árbol, y los árboles se
recorren de las hojas a la raíz resolviendo cada nodo con las respuestas de sus
hijos. Eso es programación dinámica: resolver subproblemas chicos una vez,
guardar la respuesta, y combinar.

🔵 Una familia de conjuntos es laminar si para todo par A, B se cumple A ⊆ B,
B ⊆ A, o A ∩ B = ∅. El árbol de inclusión permite calcular la normalización por
DP: en cada nodo se lleva la distribución del conteo de activos de su subárbol,
combinando hijos por convolución. Costo polinomial en n.

🟣 Segundo régimen tratable (Proposición del paper sobre groups anidados). En la
práctica los diseños jerárquicos de testeo (pool grande → subpools) son laminares.

### A5. Treewidth acotado → junction tree

🟢 El treewidth mide qué tan "parecido a un árbol" es el enredo de traslapes
entre grupos. Traslapes en cadena o en jerarquía: casi-árbol, fácil. Traslapes
todos-con-todos: nada de árbol, difícil. Si el enredo es casi-árbol, hay un
algoritmo estándar que lo explota.

🔵 Se construye el grafo donde los individuos son vértices y cada measurement une a
sus miembros en un clique. Si ese grafo tiene treewidth w, el junction tree
calcula marginales exactas en tiempo O(n · 2^w): exponencial solo en el ancho,
no en n.

🟣 Tercer régimen tratable. Es la generalización que contiene a A3 (w ≤ G−1 con
grupos disjuntos) y a A4. Fuera de estos tres regímenes: muestreo (A6).

### A6. Muestreo MCMC y el sampler de Gibbs

🟢 Cuando no puedes calcular una distribución, puedes *pasearte* por ella: un
caminante da pasos aleatorios entre mundos consistentes, diseñados para que a la
larga visite cada mundo con la frecuencia correcta. Promediar lo que ve el
caminante ≈ calcular el posterior. Eso es MCMC (Markov Chain Monte Carlo);
"Gibbs" es la variante donde cada paso cambia una parte del mundo a la vez.

🔵 Se construye una cadena de Markov sobre los perfiles consistentes cuya
distribución estacionaria es P(Z | h). Dos requisitos: **irreducibilidad** (desde
cualquier perfil consistente se puede llegar a cualquier otro — si no, la cadena
queda atrapada en una isla) y **balance** (visitar cada estado con la
probabilidad correcta, ver A8). Las marginales se estiman como frecuencias de
la trayectoria.

🟣 `gibbs_analysis.py`. Es la capa de inferencia para el caso general (traslape
arbitrario, n grande). Todo lo que se construye encima —greedy a escala,
certificados, la demo de flota— confía en que este sampler sea correcto. Por eso
las tres correcciones (A10) importan tanto.

### A7. Movimientos de camino alternante (por qué los pasos obvios fallan)

🟢 El caminante no puede dar cualquier paso: cada paso debe aterrizar en un mundo
consistente. Cambiar a una sola persona casi siempre rompe algún conteo (si un
grupo midió 2 activos, quitar uno deja 1 ≠ 2). Intercambiar dos personas del
mismo grupo respeta ese grupo, pero nunca cambia el número total de activos: el
caminante queda atrapado en el "piso" del edificio donde empezó. El movimiento
que falta es una cadena de ajustes coordinados —+1 aquí, −1 allá, +1 más allá—
que respeta todos los grupos a la vez pero sí cambia el total: el camino
alternante. Es el movimiento que conecta los pisos.

🔵 Un movimiento válido es un vector d ∈ {−1, 0, +1}^n con A·d = 0 restringido a
factibilidad 0/1 (no bajar de 0 ni pasar de 1). Los caminos alternantes son los
vectores del núcleo de A con soporte en un camino del grafo de measurements
compartidos; a diferencia de los swaps (que también están en el núcleo pero
conservan Σd = 0), los alternantes tienen Σd ≠ 0 y cambian el conteo total.

🟣 El ejemplo del paper: con {0,1}=1, {1,2}=1 y prior 0.15, la posterior exacta
es (0.15, 0.85, 0.15), pero un sampler de solo swaps devuelve (0,1,0) o (1,0,1)
según la semilla — converge con toda confianza a la respuesta equivocada. El
movimiento (+1,−1,+1) es el que le falta. Este fue el defecto de
**irreducibilidad** (primera corrección).

### A8. Balance detallado, Metropolis y el factor de Hastings

🟢 Imagina el paseo como flujo de agua entre estados: en equilibrio, el flujo de
A hacia B debe igualar el de B hacia A. Eso es balance detallado. La receta
Metropolis lo garantiza *si propones ida y vuelta con la misma facilidad*. Pero
si tu mecanismo propone "subir" más fácilmente que "bajar", tienes que
compensar la asimetría con un factor de corrección — el factor de Hastings. Si
no lo pones, la cadena se pasea sin problema y converge estable... a una
distribución sesgada hacia donde era más fácil proponer.

🔵 Balance detallado: π(x)·P(x→y) = π(y)·P(y→x). Metropolis acepta la propuesta
con min(1, π(y)/π(x)) y es correcto solo con propuesta simétrica
(q(x→y) = q(y→x)). Metropolis–Hastings acepta con
min(1, [π(y)·q(y→x)] / [π(x)·q(x→y)]) — el cociente q(y→x)/q(x→y) es el factor
de Hastings, y corrige propuestas asimétricas.

🟣 Este es exactamente el bug de la tercera corrección: la propuesta por caminos
alternantes es asimétrica (el número de caminos proponibles desde x no es igual
al número desde y) y la aceptación era Metropolis puro. La corrección calcula el
cociente por el "camino espejo" (commit 308e7ff). Nota el patrón peligroso:
**la cadena convergía, estable en todas las semillas — a la posterior
equivocada.** La estabilidad no es evidencia de corrección.

### A9. Distancia TV y mixing time (cómo se mide "correcto" y "rápido")

🟢 TV (variación total) responde "¿qué tan lejos están dos distribuciones?": es
el máximo error que puedes cometer al calcular la probabilidad de cualquier
evento usando la distribución equivocada. TV = 0 significa idénticas;
TV = 0.067 significa que alguna probabilidad está mal por 6.7 puntos. El mixing
time responde "¿cuántos pasos necesita el caminante para olvidar dónde empezó?"
— cuántos pasos hasta que su distribución esté a TV chica de la meta.

🔵 TV(μ, ν) = ½ Σ_x |μ(x) − ν(x)| = max_A |μ(A) − ν(A)|. El mixing time es
t_mix(ε) = min{t : max_x TV(P^t(x,·), π) ≤ ε}, típicamente con ε = 1/4.

🟣 Medir en TV fue indicación explícita de Francisco. El bug de A8 producía
TV 0.067 en el contraejemplo mínimo; tras la corrección, la matriz de transición
exacta da TV 0.000000 en las cinco topologías auditadas. El mixing time del
Gibbs corregido como función del traslape K es la pregunta abierta #3 de la
sesión — el puente entre la línea D1 y la capa de inferencia de los certificados.

### A10. Las tres correcciones del Gibbs (la historia completa en tres líneas)

1. **Irreducibilidad** (junio): swaps solos dejan la cadena atrapada en un nivel
   de conteo → se añadieron los caminos alternantes (A7).
2. **Descomposición** (junio): la posterior factoriza entre componentes conexas;
   cada componente chica se resuelve exacta por enumeración — el sampler solo
   corre donde es inevitable.
3. **Hastings** (julio, commit 308e7ff): la propuesta alternante es asimétrica y
   faltaba el factor de corrección (A8) → factor por camino espejo, verificado
   con la matriz de transición exacta. Suite 79/79.

🟣 `paper/correcciones_gibbs.md`. La moraleja metodológica: dos de los tres bugs
producían un sampler *estable y convincente*. La única prueba aceptada fue
enumerar la matriz de transición exacta y comparar contra la posterior exacta.

---

## Bloque B — Algoritmos: jugar bien la partida (capa 3)

### B1. El óptimo exacto por DP y por qué muere en n≈14

🟢 El óptimo se calcula como el ajedrez por fuerza bruta: "si mido este grupo y
sale 0 hago esto, si sale 1 hago aquello..." — el árbol completo de jugadas y
respuestas, eligiendo en cada nodo la rama de mayor valor esperado. El problema:
cada measurement de un grupo de tamaño |t| ramifica en |t|+1 resultados posibles (no
2 como el binario), y el estado debe recordar qué perfiles siguen siendo
posibles. El árbol explota.

🔵 DP sobre el estado (paso t, conjunto de perfiles consistentes, conjunto ya
acreditado). El espacio de estados crece exponencialmente en n y B; el measurement
augmented ramifica en |t|+1 resultados contra 2 del binario. Factible hasta
n ≈ 14.

🟣 `solver.py` (`solve_optimal_dapts`). Es la **referencia de verdad** en todos
los experimentos: cada afirmación empírica del proyecto se valida contra este
óptimo donde es computable. Su límite de escala es la razón de existir de los
certificados (Bloque D).

### B2. El greedy miope y la aproximación de independencia

🟢 El greedy juega sin pensar en el futuro: en cada paso elige el grupo que
maximiza la ganancia *inmediata* esperada ("miope" = solo ve la jugada actual).
Y para puntuar rápido comete un segundo pecado: trata a las personas como
independientes (multiplica sus probabilidades individuales), cuando la historia
ya las correlacionó — si dos personas compartieron un grupo con conteo 1, saber
de una informa de la otra, y el producto de marginales ignora eso.

🔵 En cada paso elige argmax_t [∏_{i∈t}(1 − p̃_i)] · [Σ_{i∈t} u_i], donde p̃_i
son las marginales posteriores actuales. El scoring exacto reemplazaría el
producto por P(r = 0 | h) sobre perfiles consistentes — implementado en
`exact_greedy_myopic_expected_utility`. Observación útil: la elección miope
coincide entre binario y augmented (solo r = 0 da utilidad inmediata); el conteo
paga únicamente vía mejores posteriores futuros.

🟣 `greedy.py`. Es el algoritmo que de verdad corre a escala (n=50+), y es
empíricamente excelente: 0.94–0.98 del óptimo donde el óptimo es checkeable.

### B3. La descomposición del hueco (de qué está hecho el 5%)

🟢 El greedy pierde ~5–6% contra el óptimo. ¿Por qué pierde? Dos causas
separables: la miopía (no mirar el futuro) y la independencia (puntuar con el
producto). Se midió cada una por separado, y la miopía es unas tres cuartas
partes del hueco.

🔵 Comparando greedy estándar vs greedy con scoring exacto vs óptimo (n=5–7):
hueco total ~5–6%, del cual miopía pura ~4.2–4.5 pp e independencia ~1.0–1.6 pp
(y la parte de independencia crece con n).

🟣 `paper/lineas_research_francisco.md` §1. Lectura operativa: la palanca grande
es atacar la miopía (lookahead), la chica y barata es el scoring exacto (ya
implementado).

### B4. Lookahead: la "ley del colapso" era el cableado (errata) ⭐

🟢 "Lookahead de un paso" = antes de decidir, simular una jugada hacia adelante
("si mido esto y sale r, ¿qué haría después y cuánto valdría?"). El hallazgo
corregido: bien cableado (inferencia exacta), un paso de anticipación recupera
~90% del hueco miope a TODO horizonte medido. El "colapso con el horizonte"
(99/40/16) que este bloque celebraba era la degradación del cableado legacy
(updates secuenciales + Poisson-Binomial) componiéndose con la profundidad —
no una propiedad de la anticipación.

🔵 Re-medido 17-jul-2026 con ambos cableados sobre 30 instancias idénticas
(n=6, G=4): recuperación legacy 100/92/42/38% en B=1..4 (reproduce la forma
del colapso publicado); recuperación exacta 100/100/89/93%. El residuo ~10%
en B≥3 es la miopía de segundo orden real. No citar 99/40/16 salvo como
errata.

🟣 `paper/lineas_research_francisco.md` §2 (errata),
`experiments_lookahead_exact.py`, `data/lookahead_law_rewired.csv`. La
pregunta teórica se desplaza: de "¿qué profundidad d(B)?" a "¿cuánta calidad
de inferencia necesita el lookahead para conservar su valor?".

### B5. El hallazgo del horizonte

🟢 El valor del conteo (sobre el binario) no es constante: crece con el
presupuesto B y con la población n. Razón: el conteo paga vía mejores
posteriores *futuros*; con más futuro por delante, más veces cobra.

🔵 U_DA − U_D crece con la escala: +0.63% en n=3, +3.97% en n=5, +5.07% en n=7
(200 instancias por configuración). El régimen B chico da el hueco menor porque
casi no hay pasos futuros que aprovechen la información extra.

🟣 Experimentos del paper (`experiments.py`, `horizon_experiment.py`). Es el eje
B del mapa de Francisco, y el complemento estático de la ley B4: el beneficio
del conteo y el costo de la miopía crecen *juntos* con B.

---

## Bloque C — Resolución: cuánta información da cada measurement (capa 4)

### C1. El canal de resolución (cap)

🟢 Entre "el measurement solo dice sí/no" y "el measurement dice el conteo exacto" hay una
perilla: ¿cuántos niveles distingue la respuesta? Un canal de 3 niveles dice
"0, 1, o 2-o-más". La pregunta: ¿cuánto del valor del conteo exacto sobrevive
si la respuesta es más burda? (Motivación real: los instrumentos dan lecturas
cuantitativas burdas, no conteos perfectos.)

🔵 El canal con tope `cap` reporta min(r, cap): cap=1 es el binario, cap=∞ es el
conteo exacto, cap intermedio es la escalera de resolución. Para cada cap se
recalcula el óptimo (`solve_optimal_dapts(cap)`) y se traza valor vs cap.

🟣 `experiments_resolution.py`, `data/resolution_curve.csv`,
`figures/resolution_curve.png`. **El resultado: el canal de 3 niveles captura
84.5% del valor del conteo completo.** Es la dirección D2 de Francisco, y en el
mapa certificado (D6) tiene versión con garantía: en B=3, el canal de 3 niveles
certifica exactamente lo mismo que el conteo exacto (0.85 vs 0.79 del binario).

---

## Bloque D — Certificados: garantizar sin conocer el óptimo (capa 5)

### D1. El problema de certificación y el cociente

🟢 A n=50 nadie puede calcular el óptimo (B1). Entonces, ¿cómo demuestras que tu
estrategia es buena? Emparedándola: tu estrategia da un piso ("logro 19.5") y
necesitas un techo sobre el óptimo ("nadie puede lograr más de X"). El cociente
piso/techo es la garantía: "estoy al menos al piso/techo del óptimo". El techo
tiene que ser computable sin conocer el óptimo — esa es toda la dificultad.

🔵 Certificado = greedy / U_bound, con U_bound ≥ OPT demostrable. Como
greedy ≤ OPT ≤ U_bound, el cociente subestima la calidad real: greedy/OPT
(real) ≥ greedy/U_bound (certificado). Apretar el certificado = bajar U_bound
hacia OPT.

🟣 `certificates.py`, `data/certificates_small_n.csv`. El dato que define el
programa: greedy/OPT real ≈ 0.98, pero lo certificado ≈ 0.7 en n chico y 0.58
en n=50. **El cuello de botella es la demostración, no el algoritmo.**

### D2. U_PI: la cota hindsight (información perfecta)

🟢 El techo más simple: imagina un adversario que hace trampa — conoce el futuro
(sabe exactamente quién está activo). Con ese conocimiento, limpia a las mejores
personas limpias que le caben en el presupuesto. Nadie honesto puede superar al
tramposo, así que su valor es un techo válido. Problema: el tramposo es
*demasiado* bueno, y el techo queda muy alto — certifica poco.

🔵 U_PI = E_Z[suma de las top B·G utilidades de individuos limpios bajo Z].
Domina a toda política adaptada (la política óptima es una función de la
historia; el adversario PI optimiza conociendo Z directamente). Exacta por
enumeración en n chico, Monte Carlo a cualquier escala. Se afloja cuando B·G se
acerca al número esperado de limpios: el tramposo limpia a casi todos y la cota
tiende a U_max, ignorando lo difícil que es *deducir* sin conocer Z.

🟣 `certificates.py::u_pi_exact`, `u_pi_mc`. A n=50, B=G=5: U_PI = 46.9 contra
greedy = 19.5 → certifica solo 58%. Validada por construcción: U_PI ≥ OPT en
las 106 instancias exactas.

### D3. Relajación de información con penalización (Brown–Smith–Sun) ⭐

🟢 La idea que arregla al tramposo: déjalo ver el futuro, pero **cóbrale multa
por usarlo**. Si la multa está bien diseñada, a un jugador honesto no le cuesta
nada (en promedio), pero al tramposo le come exactamente la ventaja que le daba
la trampa. El techo baja y el certificado aprieta. Diseñar la multa correcta es
el arte — y aplicar esta técnica a este problema es el hueco que reclamamos
como propio.

🔵 (Brown, Smith & Sun 2010: *information relaxations and duality*.) La multa
del paso t es π_t = V̂(h + (a, r_obs)) − E_{r∼P(·|h,a)}[V̂(h + (a, r))]: lo que
V̂ dice que ganaste viendo el resultado real, menos lo que esperabas ganar sin
verlo. Bajo la filtración natural, π_t es una diferencia de martingala con media
cero → para toda política honesta la multa esperada es 0 → 
U_pen = E_Z[max_política (welfare − Σ π_t)] ≥ OPT **para cualquier V̂**. La
validez es gratis; el apriete depende de qué tan buena sea V̂.

🟣 `certificates.py::u_pen_exact`. El problema interno (el max del adversario
con multa) se resuelve exacto por DP sobre historias — restringirlo invalidaría
la cota — así que hoy solo corre en n ≤ ~6. El paso pendiente a escala es la
penalización *descomponible* (WS1.stretch).

### D4. V̂ (la función de valor aproximada) y las variantes

🟢 La multa se calcula con una "regla de tasación" V̂ que estima cuánto vale
cada situación del juego. Cualquier regla da un techo válido (D3); una regla
*inteligente* da un techo apretado. Tenemos tres reglas: la trivial (todo vale
0 → recupera el tramposo sin multa), la simple (suma de utilidades por
probabilidad de estar limpio), y la sofisticada (lo que ganaría el greedy
jugando desde aquí).

🔵 `v_hat="zero"` → π ≡ 0, recupera U_PI. `v_hat="umax"` → potencial
V̂(h) = Σ u_i · P(Z_i = 0 | h), estático. `v_hat="greedy"` → valor-a-futuro del
greedy miope con el presupuesto restante, dependiente del tiempo. Además la
multa se escala por c ∈ {0.5, 1, 2} y se toma min sobre los agregados (el min
por-perfil no sería cota válida).

🟣 El hallazgo empírico sorprendente: **la V̂ simple ("umax") certifica mejor
que la sofisticada ("greedy")**, porque el adversario interno explota el error
de independencia de la V̂ sofisticada — el independence gap (B2/B3) atacando al
certificado desde dentro. Es la pregunta abierta #1 de la sesión con Francisco.

### D5. Por qué la penalización se apaga: holgura × horizonte ⭐

🟢 La multa con regla miope funciona en unas configuraciones y no en otras. La
explicación vigente ya NO es el eco con el lookahead (ese eco murió dos veces:
el apagado no es función del horizonte solo, y la propia "ley" del lookahead
resultó artefacto de cableado — ver B4). El fenómeno del lado del certificado
es la **holgura**: la interacción holgura × horizonte (hallazgo 07-jul-2026).
Con B·G ceñido a n la penalización miope muerde incluso en B=3, y en escasez
(B·G &lt; n) muerde más que nunca; con holgura sobrada no hay nada que cobrar.

🔵 En B=2 U_pen mejora sobre U_PI (0.63 → 0.68, 0.70 → 0.73, 0.79 → 0.84); en
B=3 con holgura, sin cambio — pero eso es la holgura, no la profundidad. La
pregunta de diseño de la V̂ sigue abierta, ahora sin el falso apoyo del
"análogo dual de la ley del lookahead".

🟣 Tabla de `masterplan_una_pagina.md` §2, datos en
`data/certificates_small_n.csv`; hallazgo de holgura del 07-jul-2026.

### D6. El mapa certificado (la figura que une todo)

🟢 Una sola figura con las perillas de Francisco (presupuesto B, resolución cap)
donde cada punto lleva dos números: cuánto del valor *logra de verdad* el greedy
(solo computable en n chico) y cuánto se puede *garantizar* a cualquier escala.
La banda entre ambos números ES el programa de investigación: valor que existe
pero aún no sabemos certificar.

🔵 n=5, G=3, 12 instancias, prevalencia 0.25–0.65. Hallazgos: la fracción
certificada crece con el horizonte (0.58 en B=1 → 0.85 en B=3), y en B=3 el
canal de 3 niveles certifica igual que el conteo completo (0.85 vs 0.79 del
binario) — la versión certificada del 84.5% de C1.

🟣 `figures/certified_map.png`, `data/certified_map.csv`. Autoexamen: si puedes
narrar esta figura sin ayuda —qué es cada eje, qué son los dos números, qué
significa la banda— dominas el proyecto completo.

### D7. La demo de flota (la traducción a industria)

🟢 El mismo objeto matemático en lenguaje de agentes de IA: los "individuos" son
componentes de una flota, los "grupos" son evals por lotes, "activo" = defectuoso,
y el certificado dice "estos componentes quedan garantizados limpios y la
asignación del presupuesto fue al menos X% de la óptima incalculable".

🔵 Re-cableada 17-jul-2026 (inferencia exacta por componentes + selector por
frecuencia conjunta muestral; la configuración se imprime en la cabecera, sin
dependencias de entorno silenciosas). Números reproducibles, 300 sims:
régimen saturado B=10, G=5 → motor 188.3±1.8 vs U_PI exacta 231.3 =
**81% certificado** (aleatorio 46%); régimen de escasez B=6, G=5 (cap 30 &lt;
n=50, el titular: ahí la cota no colapsa a U_max) → motor 142.7±2.1 vs U_PI
202.0±0.02 = **71% certificado** (aleatorio 35%). Corrida ilustrativa: 38/50
certificados limpios, cero falsos limpios.

🟣 `demo_fleet_certification.py`. Se enseña solo si la conversación va a
industria. Los viejos 31/78% eran irreproducibles (Mosek expirado cayendo en
silencio a un heurístico); citarlos solo como historia del fix. La semilla
teórica seria (conteos con ruido: el grader se equivoca) es la cuarta
perilla — pregunta abierta #2.

---

## El hilo de una frase por bloque

- **A**: contar hace valiosa la información pero intratable la inferencia; hay
  tres islas tratables y, para el resto, un sampler que costó tres correcciones
  dejar exacto.
- **B**: el greedy es casi óptimo; lo que pierde es ~¾ miopía y ~¼ propagación
  de marginales (el scoring conjunto casi no pesa), y un paso de anticipación
  bien cableado recupera ~90% del hueco a todo horizonte medido.
- **C**: un canal burdo de 3 niveles captura 84.5% del valor del conteo exacto.
- **D**: como el óptimo es incalculable, se certifica con un techo; el techo
  tramposo da 58%, la multa de Brown–Smith–Sun lo aprieta, y la multa muerde
  cuando el presupuesto es ceñido (holgura × horizonte), no según la
  profundidad de la V̂.
