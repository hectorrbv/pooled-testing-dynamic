# Lema A — construcción (Persona A)

*Documento de trabajo del plan semanal (lunes, cimientos). Se construye parte
por parte en sesión guiada; cada parte se cierra solo cuando la demostración
sobrevive la lupa de revisor. La versión LaTeX final para el paquete de
Francisco se traducirá de aquí. Solucionario de referencia (no consultado
durante la construcción): `lemma_A_laminar_inference.tex`.*

*Ejemplo ancla (notebook 22, celda 13): pools {0,…,7}→4, {0,1,2,3}→1,
{0,1}→1, {4,5}→2. Átomos: {6,7} con 1, {2,3} con 0, {0,1} con 1, {4,5} con 2.*

---

## Vocabulario de trabajo

- **Familia laminar**: cualesquiera dos pools son ajenos o están encajados
  (uno dentro del otro).
- **Hijos de un pool A**: los pools de la familia directamente adentro de A —
  contenidos en A sin ningún pool intermedio. (Los nietos no son hijos.)
- **Átomo residual D_A**: los sueltos de A — las personas de A que no están
  en ningún hijo de A. `D_A = A ∖ ⋃ hijos`.
- **Conteo residual c(D_A)**: deducido, no medido: `c(D_A) = c(A) − Σ c(hijos)`.
  Solo se restan los hijos directos (los conteos de los nietos ya vienen
  incluidos en los de los hijos; restarlos dos veces es doble conteo).
- **Torre de i**: los pools que contienen a la persona i, que por laminaridad
  están totalmente encajados entre sí.

---

## Parte (i) — Los átomos parten a la población testeada ✅

*Versión definitiva redactada por Persona A (22-jul), con revisión de
sutilezas en sesión guiada.*

**Conceptos.**

*Átomo:* Un átomo es la región donde viven los individuos "sobrantes" de un
pool, es decir, aquellos que no pertenecen a ningún hijo de ese pool.

*Hijos:* Un hijo de un pool es un subpool de la familia directamente
contenido dentro del pool padre en un solo nivel. Por ejemplo: los pools
{0,…,7}, {0,1,2,3}, {0,1}. El pool {0,1,2,3} es hijo del {0,…,7}, mientras
{0,1} es hijo de {0,1,2,3}, no así de {0,…,7}.

**Enunciado.** En una historia laminar, ningún par de átomos distintos
comparte a una persona (la intersección de cualquier par de átomos
distintos es vacía) y la unión de los átomos es toda la población testeada:
cada persona cae entonces en exactamente un átomo: el de su pool más chico.

**Observación.** Varios pools pueden compartir un individuo solo si están
encajados entre sí (contenidos los unos en los otros); si no se contiene un
pool en el otro, entonces no puede haber traslape de un individuo testeado
que respete la laminaridad. La laminaridad solo permite dos opciones:
ajenos o encajados.

**Demostración.**

*Al menos un átomo* (nos describe cómo cada individuo testeado debe ser
parte de al menos un átomo): Sea i un sujeto testeado. Los pools que
contienen a i forman una colección no vacía (i fue testeado: algún pool lo
contiene) y están anidados en cadena (una torre), porque, por la
observación anterior, al compartir al sujeto i deben estar encajados entre
sí.

Sea A_i el pool más chico de esa torre. Entonces i está suelto en A_i: si
i perteneciera a algún hijo de A_i, ese hijo sería un pool de la familia,
más chico que A_i, que contiene a i — eso contradice que A_i es el pool más
chico de la torre. Por lo tanto i cae en el átomo de A_i.

*A lo más un átomo* (nos describe la restricción de cómo un individuo i
solo puede ser parte de un átomo): Supongamos que el sujeto i cae también
en el átomo de otro pool B diferente a A_i. Como B contiene al individuo i
(el átomo de B vive dentro de B), B está en la torre de i, así que B y A_i
están encajados. B no puede estar estrictamente contenido en A_i porque A_i
es el piso más bajo de la torre; sabemos que B y A_i no son el mismo pool,
entonces A_i está estrictamente contenido dentro de B.

Consideremos los pools de la familia estrictamente dentro de B que
contengan a A_i (existe al menos uno: A_i mismo). Todos los candidatos
contienen a A_i, así que comparten gente (al menos al individuo i) y, por
laminaridad, están encajados: la colección es una cadena y tiene un único
máximo, que se llamará C. C es un hijo de B: si otro pool de la familia
estrictamente dentro de B contuviera estrictamente a C, también contendría
a A_i y C no habría sido el máximo. Ese hijo C contiene al individuo i
porque contiene a A_i. Entonces i está cubierto por un hijo de B: no está
suelto en B, es decir, no cae en el átomo de B, lo cual contradice la
suposición. ∎

**Observación.** La laminaridad se usó exactamente una vez — "ajenos o
encajados" — y es la que fabrica la torre. Sin ella hay personas con dos
pools mínimos incomparables (la persona 1 con {0,1} y {1,2}) y los átomos
se traslapan: la partición muere.

**Moralejas de revisor de esta parte:**
- Cada frase debe sobrevivir a los ejemplos que ya están sobre la mesa
  ("A_i es hijo de B" moría contra la persona 0).
- Una prueba por contradicción cierra nombrando la contradicción.
- La justificación va antes o junto a la afirmación, no después.
- "Hijo" es una relación exacta (directamente adentro, sin intermediario),
  no un sinónimo de "más chico".

**Resumen en llano (validado al cierre del lunes):**
1. El átomo de un pool es su sobrante: los miembros que quedan sueltos al
   quitar todos sus hijos. Cada pool tiene exactamente **un** átomo (un
   conjunto, posiblemente vacío); las personas son sus miembros. Su conteo
   no se mide: se deduce — lo que reportó el pool menos lo que sus hijos ya
   explican vive forzosamente en los sueltos.
2. Cada persona testeada cae en exactamente un átomo porque sus pools forman
   una torre y **solo en el piso más bajo estás destapado; en cualquier piso
   más arriba, el piso de abajo te tapa**.
3. "Estar en el átomo" es siempre relativo a un pool: dentro de un pool A
   cada miembro está o tapado (dentro de un hijo de A) o suelto (en el átomo
   de A) — mutuamente excluyente. Que tu átomo resulte ser hijo de un pool
   más grande es hablar de otro pool, no de A.
4. Compartir personas entre pools es legal si están encajados (la persona 0
   vive en tres); el crimen es el traslape a medias — ahí la persona
   compartida tiene dos pisos más bajos empatados, cae en dos átomos, y la
   partición muere. Ese traslape es exactamente el fenómeno que hace #P-hard
   la inferencia general.

---

## Parte (ii) — Los conteos de nodos y átomos se determinan mutuamente

*Redacción cerrada (22-jul), en palabras de Persona A. Demostraciones de (a)
y (c) pendientes: inducción estructural, próxima sesión.*

**Pieza 1 (hijos disjuntos).** Dos hijos distintos C₁, C₂ de un mismo pool A
no comparten a ninguna persona.

*Demostración.* Supongamos que existiera una persona j en C₁ ∩ C₂. Como C₁ y
C₂ comparten a j, ajenos no son; por laminaridad entonces están encajados,
digamos C₁ dentro de C₂ — y la contención es estricta porque son distintos.
Pero ser hijo de A significa ser máximo entre los pools de la familia
estrictamente contenidos en A. Entonces C₂ es un pool de la familia que está
estrictamente dentro de A y contiene estrictamente a C₁ — lo cual contradice
que C₁ sea máximo, es decir, que sea hijo de A. Por lo tanto j no puede
existir. ∎

**Pieza 2 (identidad de contabilidad).** Para todo pool A de una historia
laminar y para todo mundo z: los activos de A en z son los activos de su
átomo D_A en z más la suma de los activos de cada hijo de A en z.

*Demostración.* El pool A se parte exactamente en su átomo y sus hijos.
Nadie de A queda fuera de esa lista: quien no está en ningún hijo está, por
definición de átomo, en D_A. Nadie aparece dos veces: los hijos entre sí no
comparten ninguna persona (Pieza 1), y el átomo no comparte gente con ningún
hijo, por la definición misma de átomo. Al contar activos sobre partes que
cubren todo sin traslaparse, los conteos se suman. ∎

**Enunciado.** Sea una historia laminar, con conteos reportados c(A) y
conteos residuales c(D_A). Escribimos sub(A) para el conjunto de pools de la
familia contenidos en A, incluido A mismo (el subárbol de A).

(a) Para todo mundo z: z cumple todas las restricciones de los pools
[r(A, z) = c(A) para cada pool A] si y solo si cumple todas las
restricciones de los átomos [r(D_A, z) = c(D_A) para cada pool A].

(b) Para todo pool A: `c(D_A) = c(A) − Σ_{C hijo de A} c(C)`.

(c) Para todo pool A: `c(A) = Σ_{B ∈ sub(A)} c(D_B)`.

**Moralejas de revisor de esta parte:**
- Dónde se para el "si y solo si" es exactamente qué estás reclamando:
  (pools) ⟺ (átomos) es un teorema; "válido ⟺ ambos" era un círculo.
- Un enunciado afirma, no argumenta: los porqués viven en la demostración.
- Cada palabra debe estar previamente definida o cuantificada ("válido" no
  existía; sub(A) se declara antes de usarse).
- Disciplina de tipos: tras un "=" solo vive una expresión del mismo tipo
  que el lado izquierdo — nunca una oración.
- No mezclar pisos: los conjuntos se parten (personas); los conteos se
  suman (números). La identidad de contabilidad conecta los dos pisos.
- A veces el paso no requiere demostración sino reconocer una definición
  (la cobertura del átomo era el diccionario, no un teorema).
- En la (c), el lado derecho solo puede contener residuales: esa es la
  promesa "los átomos solos reconstruyen". Con reportados a la derecha es
  la (b) disfrazada.
- Prueba con hipótesis de más huele mal: la cobertura no usa laminaridad;
  invocar armas que no disparan delata que no sabes cuál disparó.

**Prueba de fuego numérica (ancla):** c({0,…,7}) = 4 = 1+0+1+2 — los
residuales de TODO el subárbol, hasta el fondo; c({0,1,2,3}) = 1 = 0+1;
en una hoja, c({4,5}) = 2 = su propio residual.

---

## Parte (iii) — El posterior factoriza entre átomos ⬜

*(Pendiente. Prior producto + condicionar a conteos sobre bloques disjuntos.)*

---

## Coherencia y casos degenerados ⬜

*(Pendiente. Los `ValueError` del código como hipótesis del enunciado.)*

---

## Corolarios ⬜

*(Pendiente. Complejidad dado el bosque padre-hijos; distribución exacta de
R_t para pools compatibles.)*
