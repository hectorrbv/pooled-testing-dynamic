# Proposición B — mejora de política dentro de una biblioteca laminar

## Enunciado

Fijemos una biblioteca finita de pools \(\mathcal L\), un presupuesto finito
\(B\) y el mismo modelo de recompensa para dos políticas. Para un estado de
información \(s\) y \(b\in\{0,\ldots,B\}\) pruebas restantes, sea
\(\pi^{\mathrm g}\) la política greedy y sea

\[
V_b^{\mathrm g}(s)
=
\mathbb E^{\pi^{\mathrm g}}
\!\left[\sum_{j=0}^{b-1}r(S_j,A_j,S_{j+1})\,\middle|\,S_0=s\right]
\]

su valor verdadero. Definamos su función de continuación

\[
Q_b^{\mathrm g}(s,a)
=
\mathbb E\!\left[
  r(s,a,S')+V_{b-1}^{\mathrm g}(S')
  \mid s,a
\right],
\qquad a\in\mathcal A_{\mathcal L}(s),
\]

donde \(\mathcal A_{\mathcal L}(s)\) es el conjunto de acciones admisibles de
la biblioteca en \(s\). El rollout replanificado elige en cada estado

\[
\pi_b^{\mathrm r}(s)
\in
\operatorname*{arg\,max}_{a\in\mathcal A_{\mathcal L}(s)}
Q_b^{\mathrm g}(s,a),
\]

pero, después de ejecutar esa acción, vuelve a aplicar la misma regla rollout
en el estado siguiente.

**Proposición B.** Supongamos que:

1. todas las esperanzas que definen \(Q_b^{\mathrm g}\) se calculan con la
   distribución condicional verdadera;
2. greedy y rollout usan el mismo espacio de estados, recompensas y
   transiciones;
3. la acción \(\pi_b^{\mathrm g}(s)\) pertenece al conjunto de candidatas
   \(\mathcal A_{\mathcal L}(s)\) que maximiza el rollout, para todo estado
   alcanzable y todo \(b\).

Entonces, para todo estado alcanzable \(s\) y todo presupuesto restante
\(b\),

\[
V_b^{\mathrm r}(s)\ge V_b^{\mathrm g}(s).
\]

En particular, desde el estado inicial \(s_0\), el welfare esperado del
rollout dentro de \(\mathcal L\) domina al del greedy dentro de esa misma
biblioteca.

## Demostración por inducción hacia atrás

La prueba separa tres desigualdades que a veces se confunden en una sola.

**Caso base.** Con \(b=0\) no quedan pruebas. Ambas políticas reciben cero
recompensa adicional, luego

\[
V_0^{\mathrm r}(s)=0=V_0^{\mathrm g}(s).
\]

**Paso inductivo.** Supongamos que
\(V_{b-1}^{\mathrm r}(s')\ge V_{b-1}^{\mathrm g}(s')\) para todo estado
sucesor \(s'\). Escribamos
\(a_{\mathrm r}=\pi_b^{\mathrm r}(s)\) y
\(a_{\mathrm g}=\pi_b^{\mathrm g}(s)\). Entonces

\[
\begin{aligned}
V_b^{\mathrm r}(s)
&=
\mathbb E\!\left[
  r(s,a_{\mathrm r},S')+V_{b-1}^{\mathrm r}(S')
  \mid s,a_{\mathrm r}
\right] \\
&\ge
\mathbb E\!\left[
  r(s,a_{\mathrm r},S')+V_{b-1}^{\mathrm g}(S')
  \mid s,a_{\mathrm r}
\right] \\
&=Q_b^{\mathrm g}(s,a_{\mathrm r}) \\
&\ge Q_b^{\mathrm g}(s,a_{\mathrm g}) \\
&=V_b^{\mathrm g}(s).
\end{aligned}
\]

La primera desigualdad usa la hipótesis inductiva dentro de la esperanza. La
segunda usa que rollout maximiza \(Q_b^{\mathrm g}\) y que la propia acción
greedy está entre las candidatas. La última igualdad es la ecuación de valor
de la política greedy: tomar su acción y continuar con ella misma. Esto cierra
la inducción. \(\square\)

## Por qué la hipótesis de exactitud tiene contenido

La prueba no necesita que greedy sea óptimo, ni que \(\mathcal L\) aproxime
bien al conjunto de todos los pools. Sí necesita que el orden de las acciones
por \(Q_b^{\mathrm g}\) sea el orden bajo la distribución posterior verdadera.
Éste es precisamente el punto en que entra el Lema A.

Si \(\mathcal L\) es una biblioteca laminar fijada antes de observar las
respuestas, cualquier subfamilia ya observada también es laminar. El Lema A
descompone su historial en átomos residuales disjuntos, convierte los conteos
de los nodos en conteos de átomos y factoriza el posterior entre átomos. Los
mensajes Poisson--binomial dentro de cada átomo permiten obtener exactamente
la ley de \(R_t\) para cada pool compatible \(t\in\mathcal L\). Por tanto, las
probabilidades de las ramas y las recompensas esperadas usadas en
\(Q_b^{\mathrm g}\) son las verdaderas.

No basta con conocer marginales individuales exactas y multiplicarlas. Dentro
de un átomo condicionado a un conteo, los indicadores individuales están
correlacionados. La expectativa exacta debe conservar la distribución
condicional del átomo; reemplazarla por el producto de sus marginales cambia
la ley de \(R_t\) y rompe la cadena de igualdades de la demostración.

Fuera de una biblioteca laminar, el Lema A ya no entrega esa factorización.
El notebook 21, §7, documenta el síntoma empírico en \(N=50\), \(B=5\): el
rollout basado en independencia predice una mejora en el régimen aumentado,
pero para \(G=5\) esa mejora no se materializa y es estadísticamente
indistinguible de cero. No contradice la Proposición B: allí se está
maximizando una aproximación \(\widehat Q\), no el \(Q\) verdadero supuesto en
el enunciado.

## Qué garantiza y qué no

La proposición compara dos políticas dentro de la **misma** biblioteca:

\[
V^{\mathrm g,\mathcal L}
\le
V^{\mathrm r,\mathcal L}
\le
V^{\mathcal L}
\le
V^*.
\]

La primera desigualdad es la Proposición B. La segunda sólo dice que el
rollout es una política factible de \(\mathcal L\). La tercera viene de la
inclusión de espacios de acciones. Ninguna de ellas da por sí sola una cota
de \(V^{\mathrm r,\mathcal L}/V^*\); para eso hace falta la parte de calidad de
la biblioteca, es decir, la Conjetura C o una cota por régimen.

Tampoco se conserva la garantía si rollout elimina acciones que greedy sí
puede tomar. En particular, si greedy incluye la opción de no hacer nada o un
pool de desempate, esa misma opción debe aparecer en el argmax de rollout.

## Cuantificación del fallo aproximado

La exactitud no es una formalidad. Si sólo sabemos que

\[
|\widehat Q_b(s,a)-Q_b^{\mathrm g}(s,a)|\le\varepsilon
\quad\text{para toda acción candidata},
\]

y rollout elige maximizando \(\widehat Q\), entonces únicamente podemos
deducir

\[
Q_b^{\mathrm g}(s,a_{\mathrm r})
\ge
Q_b^{\mathrm g}(s,a_{\mathrm g})-2\varepsilon.
\]

El factor dos aparece porque la acción elegida puede estar sobrevalorada en
\(\varepsilon\) y la acción greedy subvalorada en \(\varepsilon\). Repitiendo
el argumento hacia atrás, una cota uniforme permite una pérdida acumulada de
hasta \(2b\varepsilon\). Así se entiende el resultado del notebook 21 §7: si
el premio real del lookahead es pequeño, un error de independencia de la
misma escala puede borrarlo por completo.

## Correspondencia con el notebook 22, §7

En `greedy_and_rollout_values`:

- `base(k, remaining, cleared)` es \(V_{B-k}^{\mathrm g}(s)\);
- `q_base` es \(Q_{B-k}^{\mathrm g}(s,a)\);
- el `max` sobre `scored` implementa la acción \(a_{\mathrm r}\);
- la llamada recursiva a `rollout`, no a `base`, implementa el rollout
  replanificado cuya dominancia requiere la inducción anterior;
- `remaining` conserva la distribución conjunta exacta por perfiles, por lo
  que el experimento de esa sección sí satisface la hipótesis de exactitud.

Si se aplicara policy improvement sólo en la raíz y luego se siguiera greedy,
la dominancia en \(s_0\) sería inmediata por el argmax y no requeriría
inducción. La inducción aparece porque la política del notebook vuelve a
mejorar la acción en cada estado sucesor.
