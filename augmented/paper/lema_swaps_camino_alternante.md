# Los swaps como caso mínimo del kernel de caminos alternantes

*Formalización del lema "swaps ⊂ caminos alternantes", con la obstrucción
estructural que hace insuficiente a cualquier kernel que preserve el conteo,
y el contexto bibliográfico (bases de Markov). Preparado para discusión con
Francisco, 14 de julio de 2026.*

*Evidencia ejecutable: `augmented/bayesian.py::_propose_alternating_move`,
flag de ablación `count_preserving_only` (bayesian.py), pruebas en
`augmented/tests_correctness_fixes.py`, registro adversarial en
`augmented/mcmc_adversarial_instances.py`.*

---

## 1. Marco y notación

Población $[n] = \{1,\dots,n\}$, perfil latente $z \in \{0,1\}^n$ con prior
producto $\pi_0(z) = \prod_i p_i^{z_i}(1-p_i)^{1-z_i}$. Una historia de
mediciones exactas consiste en grupos $S_1,\dots,S_m \subseteq [n]$ con
conteos observados $r_t = \sum_{i \in S_t} z_i$. Sea $A \in \{0,1\}^{m\times n}$
la matriz de incidencia, $A_{ti} = \mathbf 1[i \in S_t]$. El soporte de la
posterior es la **fibra**

$$\mathcal F_r \;=\; \{\, z \in \{0,1\}^n : A z = r \,\},
\qquad \pi(z) \;=\; \pi_0(z \mid \mathcal F_r).$$

Muestrear $\pi$ es una instancia del problema de Diaconis–Sturmfels [1]:
muestrear una distribución condicionada al valor de un estadístico lineal
$Az$, caminando sobre la fibra con movimientos del núcleo entero de $A$.

**Movimientos.** Un *movimiento* es un vector
$\delta \in \ker_{\mathbb Z}(A) \cap \{-1,0,+1\}^n$, $\delta \neq 0$; es
*aplicable* en $z$ si $z + \delta \in \{0,1\}^n$ (es decir, $\delta_i = +1$
solo donde $z_i = 0$ y $\delta_i = -1$ solo donde $z_i = 1$). Todo movimiento
aplicable preserva la fibra: $A(z+\delta) = Az = r$. Trabajamos dentro de una
**componente conexa** del grafo de co-aparición en mediciones (la posterior
factoriza entre componentes y el sampler opera por componente); dentro de una
componente todo agente pertenece a al menos una medición.

**Firma de conteo.** Para un movimiento $\delta$ definimos
$\sigma(\delta) := \sum_i \delta_i = \#\{i:\delta_i = +1\} - \#\{i:\delta_i=-1\}$,
el cambio neto en el número total de activos.

## 2. Los dos generadores

**Swap.** Un *swap* es el movimiento $\delta^{a\to b} = e_b - e_a$ (desactivar
$a$, activar $b$). Es un elemento del núcleo si y solo si

$$A(e_b - e_a) = 0 \iff A e_a = A e_b,$$

es decir, si $a$ y $b$ pertenecen **exactamente a las mismas mediciones**
("gemelos de medición"). El generador previo del sampler (commit `44f7e5f`)
proponía swaps dentro de un grupo y descartaba los inválidos; su conjunto de
movimientos válidos es exactamente el conjunto de swaps entre gemelos.

**Camino alternante.** El generador actual
(`_propose_alternating_move`) construye movimientos por reparación aleatoria:
elige un agente inicial $a_0$ y lo voltea ($\delta_{a_0} = \pm 1$ según su
estado); cada medición desbalanceada por un flip se repara volteando en
sentido contrario a un compañero elegible de esa medición, propagando el
desbalance hasta que toda medición queda balanceada (éxito: $A\delta = 0$ por
construcción, aplicado atómicamente tras la aceptación) o se agota el
presupuesto de pasos (rechazo). Un camino de $L$ flips con signos alternados
tiene $\sigma(\delta) = 0$ si $L$ es par y $\sigma(\delta) = \pm 1$ si $L$ es
impar. La propuesta es asimétrica, de modo que la aceptación lleva el factor
de Hastings [7] calculado por el camino espejo (commit `308e7ff`; verificado
contra la matriz de transición exacta de la cadena, TV $0.067 \to 0.000000$
en las cinco topologías auditadas).

## 3. El lema

**Lema 1 (los swaps son los caminos alternantes de longitud 2).**
Dentro de una componente conexa:

1. Los movimientos con soporte de tamaño 2 son exactamente los swaps válidos:
   $\delta = e_b - e_a$ con $Ae_a = Ae_b$.
2. Todo swap válido y aplicable en $z$ es propuesto por el procedimiento de
   camino alternante con probabilidad positiva: es el camino que termina en su
   primer paso de reparación.
3. En consecuencia, $\{\text{swaps válidos}\} = \{\text{movimientos de camino
   alternante con } |\mathrm{supp}(\delta)| = 2\} \subsetneq \{\text{movimientos
   de camino alternante}\}$, y la inclusión es estricta en cuanto el núcleo
   contenga un vector aplicable con tres o más flips (p. ej. el $(+1,-1,+1)$
   de la instancia canónica del §5).

*Demostración.*
(1) Sea $\delta = \sigma_a e_a + \sigma_b e_b$ con $\sigma_a,\sigma_b \in \{\pm 1\}$
y $A\delta = 0$. Si $\sigma_a = \sigma_b$, entonces $A e_a + A e_b = 0$; como
$A$ tiene entradas no negativas, ambas columnas son cero — imposible en una
componente conexa, donde todo agente tiene grado $\geq 1$. Luego
$\sigma_a = -\sigma_b$, y $A\delta = \pm(Ae_b - Ae_a) = 0$ equivale a
$Ae_a = Ae_b$. La aplicabilidad en $z$ fija la orientación: $z_a = 1$,
$z_b = 0$ para $\delta = e_b - e_a$.

(2) Sea $\delta = e_b - e_a$ aplicable en $z$, con $Ae_a = Ae_b$. Con
probabilidad $\geq 1/n$ el procedimiento arranca en $a_0 = a$ y fija
$\delta_a = -1$; toda medición $t \ni a$ queda con desbalance $-1$. El paso de
reparación toma una medición pendiente $\hat t \ni a$ y busca un compañero
elegible con dirección $+1$; el conjunto elegible contiene a $b$ (está en
$S_{\hat t}$ por ser gemelo, está inactivo y fuera del camino), así que con
probabilidad $\geq 1/|S_{\hat t}|$ el procedimiento voltea a $b$. Esto suma
$+1$ al desbalance de toda medición que contiene a $b$; como
$t \ni b \iff t \ni a$, **todos** los desbalances se cancelan simultáneamente
y el bucle termina con el movimiento $\delta = e_b - e_a$. Toda elección tiene
probabilidad positiva, luego $q(z \to z+\delta) > 0$.

(3) Inmediato de (1) y (2). $\blacksquare$

**Corolario 1.1 (jerarquía del generador viejo).** Los movimientos de sitio
único válidos ($\delta = \pm e_a$ con $Ae_a = 0$) son los caminos de longitud
1 y solo existen para agentes fuera de toda medición — inexistentes dentro de
una componente. Así, dentro de una componente, todo movimiento válido del
generador viejo es un camino alternante; el recíproco es falso.

## 4. La obstrucción: graduación por conteo

La razón por la que *ningún* kernel construido con swaps (ni con cualquier
otro conjunto de movimientos con $\sigma \equiv 0$) puede ser correcto en
general no es de eficiencia sino estructural.

**Proposición 2 (reducibilidad de los kernels que preservan el conteo).**
La fibra se gradúa por el conteo total:
$\mathcal F_r = \bigsqcup_c \mathcal F_r^c$ con
$\mathcal F_r^c = \{z \in \mathcal F_r : |z| = c\}$, $|z| = \sum_i z_i$.

1. Un movimiento cruza niveles si y solo si $\sigma(\delta) \neq 0$. Los swaps
   tienen $\sigma = 0$.
2. Si el generador de la cadena satisface $\sigma \equiv 0$, el nivel inicial
   $c_0 = |z_0|$ es invariante: toda distribución límite $\mu$ está soportada
   en $\mathcal F_r^{c_0}$, y por tanto

   $$\mathrm{TV}(\mu, \pi) \;\geq\; 1 - \pi(\mathcal F_r^{c_0}),$$

   con igualdad $\mu = \pi(\cdot \mid \mathcal F_r^{c_0})$ cuando la cadena es
   ergódica dentro del nivel. Como las marginales son esperanzas de funciones
   con valores en $[0,1]$, el error marginal obedece
   $\max_i |\mathbb E_\mu z_i - \mathbb E_\pi z_i| \leq \mathrm{TV}(\mu,\pi)$,
   y la cota inferior de TV es un sesgo que **ninguna cantidad de iteraciones
   reduce**.

*Demostración.* (1) es la definición de $\sigma$. (2) La invariancia del nivel
es inducción sobre los pasos. Para la cota: tomando el evento
$B = \mathcal F_r^{c_0}$, $\mathrm{TV}(\mu,\pi) \geq \mu(B) - \pi(B) =
1 - \pi(\mathcal F_r^{c_0})$. Para la igualdad con
$\mu = \pi(\cdot \mid \mathcal F_r^{c_0})$: la diferencia $\mu - \pi$ es
positiva exactamente en $\mathcal F_r^{c_0}$, con masa positiva total
$\sum_{z \in \mathcal F^{c_0}} \pi(z)\big(\tfrac{1}{\pi(\mathcal F^{c_0})} - 1\big)
= 1 - \pi(\mathcal F^{c_0})$. $\blacksquare$

**Lectura.** El defecto del kernel swap-only no es mixing lento: la cadena
converge de forma perfectamente estable a la distribución equivocada, con un
sesgo en TV igual a la masa de los niveles que no puede visitar. La
estabilidad entre semillas no es evidencia de corrección — dos de los tres
bugs del Gibbs produjeron samplers estables y convincentes
(`paper/correcciones_gibbs.md`).

## 5. Tres instancias que separan los kernels

**(a) Cadena impar $n=3$ (canónica).** Mediciones $\{1,2\}=1$, $\{2,3\}=1$,
prior $p_i = 0.15$. La fibra tiene dos perfiles: $(0,1,0)$ en el nivel 1 y
$(1,0,1)$ en el nivel 2, con $\pi(\mathcal F^1) = 0.85$ y
$\pi(\mathcal F^2) = 0.15$; marginales exactas $(0.15,\, 0.85,\, 0.15)$. No
hay gemelos de medición (las columnas de 1, 2, 3 difieren), así que **no
existe ningún swap válido**: el kernel swap-only reporta su perfil semilla.
Por la Proposición 2, el sesgo en TV es $0.15$ o $0.85$ según el nivel de la
semilla — exactamente los errores marginales medidos en
`test_swap_only_kernel_is_stuck_in_one_count_level` (el umbral `err > 0.10`
de esa prueba está calibrado justo debajo de
$\min_c (1 - \pi(\mathcal F^c)) = 0.15$). El movimiento que falta es el camino
alternante impar $(+1,-1,+1)$, con $\sigma = +1$.

**(b) Cadena par $n=4$.** Mediciones $\{1,2\}=1$, $\{2,3\}=1$, $\{3,4\}=1$.
La fibra es $\{(0,1,0,1),\, (1,0,1,0)\}$ — **ambos perfiles en el nivel 2**.
Tampoco hay gemelos, así que el kernel swap-only sigue atascado en su semilla
*aun dentro de un solo nivel*. El camino alternante par
$(+1,-1,+1,-1)$ conecta ambos perfiles y tiene $\sigma = 0$: lo propone
incluso el kernel de ablación `count_preserving_only`. Moraleja doble: (i) el
conjunto "movimientos alternantes con $\sigma = 0$" es **estrictamente más
rico** que el conjunto de swaps (este movimiento no se factoriza en swaps
válidos); (ii) por eso la ablación es un sustituto *generoso* del generador
viejo — solo puede ser mejor que los swaps literales — y aun así queda
atrapada en toda fibra multinivel. El fracaso medido de la ablación aísla la
preservación del conteo como la causa, no los detalles de implementación del
generador viejo.

**(c) Cadena impar $n=7$** (`cadena_impar_n7` del registro adversarial).
Mediciones $\{i,i+1\}=1$ para $i=1,\dots,6$. La fibra son los dos perfiles
complementarios $(1,0,1,0,1,0,1)$ (nivel 4) y $(0,1,0,1,0,1,0)$ (nivel 3), y
el **único** movimiento que los conecta voltea a los siete agentes a la vez:
$\delta = \pm(+1,-1,+1,-1,+1,-1,+1)$, $\sigma = \pm 1$, soporte completo. La
familia de cadenas impares crecientes muestra que el grado de los movimientos
necesarios **no está acotado uniformemente** sobre la clase de matrices de
incidencia que producen los historiales de medición: ningún conjunto de
movimientos con soporte acotado conecta todas estas fibras. (Esto explica
además por qué el presupuesto de pasos del generador debe escalar con el
tamaño de la componente; en la implementación, `max_steps = 6|comp| + 12`.)

## 6. Contexto bibliográfico

**Bases de Markov (el marco general).** Diaconis y Sturmfels [1] formalizaron
el problema: un conjunto de movimientos $\mathcal M \subseteq \ker_{\mathbb Z}(A)$
conecta *todas* las fibras $\{z \geq 0 : Az = r\}$ si y solo si es una
**base de Markov**, y estas se calculan como bases de Gröbner del ideal tórico
$I_A$. Para una $A$ *fija*, una base de Markov finita siempre existe; la
pregunta relevante aquí es de grado y de uniformidad sobre la clase de
instancias. En nuestro lenguaje: la pregunta abierta del proyecto es si la
familia de movimientos que el generador alternante propone con probabilidad
positiva contiene una base de Markov para toda $A$ de historial (véase §7).

**Donde los swaps sí bastan — y por qué aquí no.** El resultado clásico de
Ryser [2] (véase también el survey de Brualdi [3]) dice que las matrices 0–1
con márgenes de fila y columna fijos están conectadas por *interchanges*
$2 \times 2$ — swaps. Análogamente, Dobra [4] mostró que los modelos
log-lineales *descomponibles* admiten bases de Markov de grado 2 (movimientos
tipo swap). La diferencia estructural con nuestro problema es exactamente la
Proposición 2: en esos escenarios **las restricciones fijan el conteo total**
(los márgenes suman el total), de modo que preservar el conteo no es un
defecto sino una tautología. En nuestras fibras $Az = r$ con grupos
solapados, el conteo total **no** queda determinado por $r$ (la instancia
canónica tiene niveles 1 y 2), así que cualquier kernel que preserve el conteo
es genéricamente reducible. La intuición "los swaps bastan", importada de esos
modelos, falla aquí por una razón identificable y demostrable.

**Cotas 0–1.** La literatura de bases de Markov advierte que las cotas de
celda (aquí $z_i \leq 1$) pueden invalidar bases del problema sin cotas y
requerir movimientos adicionales (Aoki–Hara–Takemura [5], caps. sobre tablas
acotadas). Nuestro caso hereda esa dificultad: los movimientos deben ser
aplicables *sin salir de* $\{0,1\}^n$, que es precisamente lo que la
construcción por reparación garantiza paso a paso.

**MCMC sobre fibras y la corrección de la propuesta.** El uso de cadenas
sobre fibras condicionadas para inferencia exacta viene de Besag y Clifford
[6]. La necesidad del cociente de propuestas cuando el mecanismo es asimétrico
es Hastings [7]; en nuestro sampler la asimetría del camino alternante exige
el factor espejo, sin el cual la cadena — irreducible y estable — converge a
una estacionaria sesgada (TV $0.067$ en el contraejemplo del audit
2026-07-06). Panorama general del área en Drton–Sturmfels–Sullivant [8].

## 7. Alcance: lo que este documento no afirma

El Lema 1 y la Proposición 2 son resultados completos: los swaps son el caso
de longitud 2 del kernel alternante, y ningún kernel que preserve el conteo
puede ser ergódico en fibras multinivel. Lo que **queda abierto** es el
recíproco fuerte: que el kernel alternante completo sea irreducible en *toda*
fibra alcanzable por historiales de medición (equivalentemente, que contenga
una base de Markov para esa clase de $A$). El estado actual es empírico:
verificación por matriz de transición exacta (TV $0.000000$) en cinco
topologías y en las familias adversariales del registro
(`mcmc_adversarial_instances.py`, $n \leq 16$ con enumeración exacta como
árbitro). La cadena impar del §5(c) delimita la forma que puede tener un
teorema general: necesitará movimientos de soporte no acotado, lo que excluye
las técnicas de grado acotado de los casos clásicos [2, 4].

## Referencias

[1] P. Diaconis, B. Sturmfels. *Algebraic algorithms for sampling from
conditional distributions.* Annals of Statistics 26(1):363–397, 1998.

[2] H. J. Ryser. *Combinatorial properties of matrices of zeros and ones.*
Canadian Journal of Mathematics 9:371–377, 1957.

[3] R. A. Brualdi. *Matrices of zeros and ones with fixed row and column sum
vectors.* Linear Algebra and its Applications 33:159–231, 1980.

[4] A. Dobra. *Markov bases for decomposable graphical models.* Bernoulli
9(6):1093–1108, 2003.

[5] S. Aoki, H. Hara, A. Takemura. *Markov Bases in Algebraic Statistics.*
Springer Series in Statistics, 2012.

[6] J. Besag, P. Clifford. *Generalized Monte Carlo significance tests.*
Biometrika 76(4):633–642, 1989.

[7] W. K. Hastings. *Monte Carlo sampling methods using Markov chains and
their applications.* Biometrika 57(1):97–109, 1970.

[8] M. Drton, B. Sturmfels, S. Sullivant. *Lectures on Algebraic Statistics.*
Oberwolfach Seminars 39, Birkhäuser, 2009.
