# Revisión crítica: notebook 28, §5 y precio de laminaridad

**Resumen.** Las afirmaciones del notebook son correctas en su alcance. El problema principal está en el objetivo. Una constante universal $c$ independiente de $G$ probablemente es falsa en nuestro modelo **[conjetura con argumento fuerte]**. Conviene intentar refutarla antes de buscar una transformación. Abajo hay dos resultados demostrados y un experimento que separa las causas de pérdida.

Etiquetas: **[D]** demostrado aquí · **[B]** bosquejo completo · **[C]** conjetura · **[R]** referencia por verificar · **[E]** experimento · **[H]** heurística.

---

## 1. Revisión de afirmaciones

**Contraejemplo con B=1 [D].** Es correcto: probar A vale 0.9 y probar AB vale $2(0.9)(0.1)=0.18$. Depende de la heterogeneidad: con $q_A=q_B=0.9$, AB vale 1.62 y supera a A. Conviene decirlo para que no parezca que agrupar es malo en general.

**Mejora del par con B≥2 [D].** La identidad $[q_1q_2+(1-q_1)(1-q_2)]\,u_{B+1}q_{B+1}$ es exacta. Las restricciones están bien puestas (B≥2, G≥2, n>B, $0<q_i<1$, $u_i>0$). Faltan tres precisiones:

- **No hace falta ordenar.** Funciona con cualquier par del plan individual. El orden por $u_iq_i$ solo hace falta para identificar el mejor plan individual.
- **La política nueva es laminar.** Por tanto la afirmación es más fuerte: $OPT_{lam}>OPT_{indiv}$ estrictamente. Sobran los «seis ejemplos»; la identidad es exacta.
- **Depende de los conteos.** Con resultado binario, el par positivo seguido de una individual positiva deja al otro indeterminado. Esto contrasta con lo que el notebook cita de §3.2 del paper: en el modelo binario, $q_i\le 1/2$ implica que lo individual es óptimo. En nuestro modelo eso falla siempre que B≥2. No pude verificar §3.2, porque no está en el texto que pasaste. Sí verifiqué el álgebra: para $|A|=k\ge2$, $\sum_i\prod_{j\ne i}q_j\le k2^{1-k}\le1$.

**Ejemplo de cuatro personas [D].** Con $s=A+B+C+D$ se tiene $c_1=s-D$, $c_2=s-C$ y $c_3=s-B$. Por tanto $c_1+c_2+c_3=2s+A$: A es la paridad de la suma y de ahí salen $s$ y luego B, C, D. Hay identificación total con valor $2=\sum q_iu_i$, que es la cota absoluta. El 1.75 laminar lo tomo como verificado por los dos solvers.

**Piloto [E, alcance sobreestimado].** El 1.75 de «ABC + π_ratio» casi no mide la conversión: de la política fuente solo se conserva una prueba. Falta el control «π_ratio sin fuente». La frase «replanear recupera toda la pérdida evitable» vale solo para esta instancia, con una fuente estática.

**Falta definir laminaridad con precisión.** ¿Se cuentan los agentes ya determinados? Por ejemplo, tras AB=1 y A, probar BC cruza literalmente a AB, pero equivale a probar C. El Lema A de abajo resuelve esto: ambas convenciones dan el mismo óptimo **[B]**.

**Recorte aleatorio, «1/16» [D].** Es correcto. Pero no es el problema principal. La falla de fondo es que la desigualdad por agente de §5 no sobrevive a los conteos (ver §2).

---

## 2. Qué se reutiliza de §5 y qué falla

**Se reutiliza tal cual:**

- **Linealidad por agente.** $V=\sum_i u_i\,P(i\text{ acreditado})$ también vale con soft clearing.
- **Simetrización (Teorema 5.4).** Solo usa el prior producto: condicionado a $U$, $Z$ es uniforme en un subcubo y $Z'$ es su compañero. Sirve para cualquier desigualdad por agente que se demuestre en subcubos uniformes.
- **Desaleatorización.** Si cada realización está en la clase objetivo, el promedio acota el óptimo de la clase.
- **Caso 1** (focal fijo infectado). Es trivial.

**Falla en nuestro modelo:**

- **La dicotomía consulta/oferta.** Con conteos se puede acreditar a $i$ en una prueba que no lo contiene. Ejemplo: $\{i,j\}=1$ y luego $\{j\}=1$. Esto ocurre incluso en políticas laminares. Con cruces el abono es global: en el ejemplo ABC/ABD/ACD se acredita a A sin que ninguna prueba dé cero.
- **Lema 5.2.** Su paso de consulta parte $R$ en $R_0$ y $R_+$, y usa que $R_+$ es creciente. Una consulta de conteo tiene $|A|+1$ hijos. Los conjuntos de nivel $\{|z_A|=c\}$ con $0<c<|A|$ no son crecientes ni decrecientes, así que la inducción no arranca.
- **Caso 3, ecuación (24).** Bajo $x^H$ y $x^I$ las trayectorias se separan en la *primera prueba que contiene a $i$*, no en la primera que lo acredita. El acoplamiento se pierde.
- **El destino.** El paper llega a asignaciones estáticas que pueden cruzarse, no a políticas laminares.

**Desconocido.** No sé si «dinámico ≤ 2·estático» vale con conteos. No debe usarse.

**Primer lema nuevo razonable:** el **Lema A** de §5, más la **Conjetura L** de §3, que decide la pregunta 3.

---

## 3. ¿Es plausible una constante universal? Probablemente no

**El mecanismo [H].** Una prueba binaria da a lo más 1 bit. Una prueba de conteo sobre $G$ personas da hasta $\log_2(G+1)$ bits. Los diseños cruzados aprovechan ese contenido por inferencia lineal. Los laminares, según el Lema A, solo acreditan bloques cuyo conteo residual es cero.

**Familia candidata [R].** En coin-weighing y matrices detectoras (Shapiro–Fine 1960; Erdős–Rényi 1963; Söderberg–Shapiro 1963; Lindström 1964–65; Cantor–Mills 1966), recuerdo dos hechos. Primero, bastan $\sim 2n/\log_2 n$ pesadas no adaptativas para identificar $n$ monedas. Segundo, existe una construcción con $n=k2^{k-1}$ monedas y $B=2^k-1$ pesadas. Para $k=2$ da $n=4$ y $B=3$, que coincide con nuestro ejemplo. Todo esto está pendiente de verificar en las fuentes.

**Conjetura L [C].** Para $q\equiv1/2$, $u\equiv1$ y $G$ sin cota, $OPT_{lam}\le K\cdot B$ con $K$ absoluta.

Si L vale y la familia anterior existe, entonces $OPT_{gen}=n/2$, y por tanto
$$\frac{OPT_{lam}}{OPT_{gen}}\le\frac{2K(2^k-1)}{k2^{k-1}}\approx\frac{4K}{k}\to0.$$

**Indicios a favor de L [H].**

- Con B=1 el máximo laminar es 1/2.
- Con B=2 calculé a mano 1.25 (par más refinamiento; un trío da 1.125). Hay que confirmarlo con el DP.
- Un átomo de tamaño $s$ es cero con probabilidad $2^{-s}$ a priori. Para acreditar bloques grandes hace falta crear antes átomos de baja densidad, y eso es exponencialmente raro.

**El Lema B (abajo) muestra lo contrario para G acotado.** Con $q$ homogéneo y G acotado, la razón sí está acotada por abajo. Cualquier refutación necesita $G\to\infty$; en la familia de arriba, $G\gtrsim B\log B$. La formulación correcta parece ser **$c(G)=\Theta(1/\log G)$ [C]**, independiente de $n$ y $B$.

**Recomendación: intentar refutar primero.** Es barato y decide qué teorema perseguir.

---

## 4. El experimento propuesto y uno mejor

**Muestrear una trayectoria virtual es un artefacto del destino estático.** El paper muestrea porque una política estática no puede adaptarse. Nuestra clase laminar sí puede. Hay una conversión determinista legítima: seguir π con los resultados reales de las pruebas y, ante el primer cruce, continuar laminarmente. Esto usa resultados observados, no el estado oculto. El muestreo puede seguir siendo útil como técnica de prueba, pero en el experimento debe medirse como una pérdida separada.

**Experimento [E].** Fijar todo antes de mirar resultados, incluida la regla de desempate entre π óptimas.

**Instancias:**
1. El caso de cuatro personas, como control.
2. Los seis casos guardados con $n\le6$.
3. El caso $k=3$ de la familia de §3: $n=12$, $B=7$, $q=1/2$, con $G$ igual al mayor grupo de la matriz.
   - Verificar la identificación por fuerza bruta: hay $2^{12}$ perfiles, o equivalentemente hay que comprobar que no existe un vector no nulo en $\{-1,0,1\}^{12}$ en el núcleo.
   - Calcular $L^*$ con el DP sobre átomos que justifica el Lema A.

**Cantidades por instancia:** $G^*$, $L^*$, $V_{real+opt}$, $V_{virt+opt}$, $V_{real+ratio}$, $V_{virt+ratio}$ y $V_{ratio\ solo}$. Aquí «opt» significa continuación laminar óptima por DP desde la historia.

**Descomposición de la pérdida:**

| Pérdida | Fórmula |
|---|---|
| Clase | $G^*-L^*$ |
| Comprometerse con el prefijo de π | $L^*-V_{real+opt}$ |
| Muestreo (puede ser negativa) | $V_{real+opt}-V_{virt+opt}$ |
| Heurística | $V_{\cdot+opt}-V_{\cdot+ratio}$ |
| Aporte de la fuente | $V_{real+ratio}-V_{ratio\ solo}$ |

Registrar también la probabilidad de que ocurra un cruce y en qué paso.

**La instancia 3 es la más informativa.** Si $L^*/6$ ya es claramente menor que 1.75/2 = 0.875, hay evidencia de que la razón cae con $k$. Sería evidencia, no una prueba.

---

## 5. Resultados demostrables

**Lema A (estructura laminar) [D].** Sea $F$ la familia laminar probada en una trayectoria. Definimos los átomos $\alpha_S=S\setminus\bigcup\{T\in F:T\subsetneq S\}$, más el resto no probado.

- (a) Toda política laminar equivale a otra, con el mismo valor y sin más pruebas en ninguna trayectoria, en la que cada prueba es un subconjunto $W_t$ de un átomo vigente.
- (b) $i$ queda acreditado si y solo si su átomo final tiene conteo 0. Por tanto, el abono ocurre en una única prueba: la que crea ese átomo cero.

*Prueba de (a).* Sea $P$ el menor conjunto probado que contiene a $S_t$. Cada hijo maximal $C$ de $P$ cumple $C\subseteq S_t$ o $C\cap S_t=\varnothing$. Que $C\supsetneq S_t$ contradice la minimalidad de $P$. Por tanto $S_t=(\bigsqcup C)\sqcup W_t$ con $W_t\subseteq\alpha_P$, y $\mathrm{cnt}(W_t)=\mathrm{cnt}(S_t)-\sum\mathrm{cnt}(C)$: la información es la misma. Las particiones en átomos solo se refinan, así que cada $W_t$ está contenido en un $W_s$ anterior o es disjunto de él. La familia de los $W$ es laminar y además $|W_t|\le G$.

*Prueba de (b).* El conocimiento final es el vector de conteos de los átomos. El posterior es el producto, sobre átomos, del prior condicionado a su suma. Si $0<c<m$ y $0<q<1$, cada miembro tiene probabilidad positiva de ambos estados. ∎

Quitar los agentes ya determinados preserva la laminaridad. Por eso las dos convenciones de §1 dan el mismo óptimo **[B]**. Para $q$ homogéneo, (a) reduce el óptimo laminar a un DP sobre multiconjuntos de pares (tamaño, conteo).

**Lema B (cota entrópica y precio acotado con G fijo) [D].** Sea $q_i\equiv q$ y $u\equiv1$.

$$OPT_{gen}\le\frac{B\log_2(G+1)}{\log_2(1/q)},\qquad
\frac{OPT_{lam}}{OPT_{gen}}\ge\frac{q\log_2(1/q)}{\log_2(G+1)}\quad(n\ge B).$$

*Prueba.* Sea $h$ la transcripción completa y $C(h)$ los acreditados. El evento $h$ implica $X_{C(h)}=0$, así que $P(h)\le q^{|C(h)|}$. Por tanto $E|C|\log_2(1/q)\le H(h)$. Por la regla de la cadena, $H(h)\le\sum_t H(\mathrm{cnt}_t\mid\text{pasado})\le B\log_2(G+1)$; la transcripción es libre de prefijos porque π es determinista. Del lado laminar, B singletons dan $Bq$. ∎

Con $q=1/2$ la razón es al menos $1/(2\log_2(G+1))$. Es débil en los extremos de $q$. Para cerrar la brecha haría falta división binaria laminar, que creo que da del orden de $1/\log(1/q)$ por prueba **[H]**.

**Discusión priorizada:**
1. **Refutar la constante universal.** Verificar la construcción $k=3$ en las fuentes [R], calcular $L^*$ con el DP del Lema A y probar la Conjetura L, por ejemplo con una función potencial sobre la dinámica de átomos.
2. **Reformular el objetivo como $c(G)$.** Probar $OPT_{lam}\ge c\,OPT_{gen}/\log_2(G+1)$ para priors producto heterogéneos y utilidades arbitrarias. El Lema B lo da solo para $q$ homogéneo.
3. **Aprovechar §5 donde sí encaja.** Usar simetrización y desaleatorización. Rehacer la reducción por agente sobre árboles de conteo, con eventos de abono por átomo cero, que incluyen los abonos por complemento.

---

## Tres ideas para Francisco

1. **Por qué cruzar ayuda.** Con conteos, combinar grupos que se cruzan permite deducir que alguien está sano sin que ningún grupo suyo haya dado cero. Con grupos anidados solo se declara sano a un bloque entero que da cero. Esa diferencia es exactamente lo que cuesta la laminaridad.
2. **Cuándo el costo está acotado.** Una prueba responde un número entre 0 y G, así que transmite poca información si los grupos son pequeños. Con grupos pequeños el costo está acotado; lo probamos con probabilidades iguales. Con grupos enormes, los diseños tipo «pesar monedas» parecen ganar cada vez más. Sospechamos que no existe una constante fija y que la pérdida crece lentamente con G. ¿Le interesa el régimen de grupos pequeños?
3. **Con dos o más pruebas, un par siempre mejora a lo individual.** En nuestro modelo, con dos o más pruebas, conviene incluir al menos un par: la mitad de las veces el conteo resuelve a dos personas con una sola prueba. En el modelo binario del paper, según su §3.2 (no verificado aquí), lo individual puede ser óptimo. Es una diferencia de fondo entre ambos modelos.
