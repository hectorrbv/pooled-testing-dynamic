# Direcciones y propuesta para la sesión del 3 de noviembre

Para Héctor y A. Palabras simples, ecuaciones completas. Fuentes: notebooks 25 a 30, los dos loops del 17–19 de octubre y el notebook 31 (`augmented/notebooks/31_tasa_laminar.ipynb`, hecho hoy). Notación: $n$ personas, cada una sana con probabilidad $q$, utilidad 1; $B$ pruebas; grupos de a lo sumo $G$; conteos exactos y soft clearing.

## 1. Por qué se siente estancado

- **Los notebooks 25 a 28 ajustan reglas en casos chicos.** Costo local, no-reentrada, $\pi_L$, el $\lambda$ del 0.9307, el Mapa 4. Cada uno arregla una instancia y abre otra. No dejan un enunciado.
- **Los notebooks 29 y 30 verifican los pasos de Francisco.** El paso de partir ($\le2$), los diseños detectores, el techo celda por celda. Todo es correcto, pero el resultado es de él; nosotros lo comprobamos.
- **Los loops buscaron donde lo asintótico no se ve.** El loop 1 se estancó en bloques sin cruces ($n=16$, $B\le8$). El loop 2 empató con rollout en $n=50$.
- **La semana se va en piezas administrativas.** WhatsApps, firmas, tablas de congresos.

En resumen: tenemos muchos números verificados y ningún resultado propio con enunciado. Lo que sigue busca cambiar eso.

## 2. Lo nuevo de hoy: la tasa laminar exacta

**El resultado.** Con $q$ homogéneo y población grande, el óptimo laminar acredita, por prueba, exactamente
$$c^*(G,q)=\max_{\text{estrategias de un bloque}}\ \frac{\mathbb E[\text{acreditados}]}{\mathbb E[\text{pruebas}]} .$$
Una estrategia de un bloque prueba un grupo nuevo de $M\le G$ personas y después parte el átomo que resulta, o lo abandona. Para todo $n$ y $B$:
$$V_{\rm lam}(n,B,G)\le c^*(G,q)\,B,\qquad \frac{V_{\rm lam}}{B}\to c^*(G,q)\ \text{ si } n\ge BG .$$

**Por qué.** Es el método del potencial de Francisco con el mejor potencial posible. Sea $W(m,r)=\max\{\text{parar},\ \mathbb E[W(\text{partes})]-c^*\}$, con "parar" igual a $m$ si $r=0$ y a $0$ si no. Entonces una prueba nueva sube $\sum W$ a lo sumo $c^*$ en promedio, partir también, y $W(m,0)=m$ domina lo acreditado. Su $\phi=(m-2r)^2/m$ es un potencial válido con constante 2 en $q=\tfrac12$. El óptimo da la constante exacta. La cota inferior repite la mejor estrategia de bloque con gente nueva.

**Los números** ($q=\tfrac12$, exactos hasta $G=20$):

| $G$ | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 96 |
|---|---|---|---|---|---|---|---|---|
| $c^*(G)$ | 1/2 | 2/3 | 9/13 | 14/19 | 0.778 | 0.813 | 0.838 | 0.869 |
| techo $\log_2(G+1)$ | 1 | 1.58 | 2 | 2.32 | 3.17 | 4.09 | 5.04 | 6.60 |

**Segunda vía.** En 18 casos exactos ($G\le4$, $B\le12$, $n=BG$), el óptimo laminar nunca pasa de $c^*(G)B$ y se acerca a esa línea al crecer $B$. Los 32 casos de la malla también quedan abajo.

**La regla óptima.** Abrir el grupo más grande posible. Si un átomo tiene más de la mitad infectada, abandonarlo; si no, probar una parte de alrededor de la mitad. Es la regla de cociente $E[U]/E[T]$ que pidió Francisco el 22 de septiembre, y es óptima con $q$ homogéneo.

**Otros $q$.** La tasa se calcula igual. Con $q=0.9$: 4.21 por prueba en $G=16$ y 4.81 en $G=48$. En todos los $q$ la curva se aplana; cambia el nivel. El "2" de Francisco solo vale en $q=\tfrac12$.

**General contra laminar.** Söderberg–Shapiro acredita 0.833 por prueba con $G=7$; el laminar, 0.768. El detector de 4 no gana. El diseño del loop 1 ($n=16$, $B=4$, $G=8$) acredita 3.378 contra el laminar exacto 2.865.

**El techo sin el $+1$.** Hoja por hoja, $m(h)\le\log_2(1/P(h))$. Al promediar:
$$\mathbb E[\text{acreditados}]\le H(\text{historial})\le B\log_2(G+1),\qquad\text{y con } q \text{ general: } \le\min\Big\{BG,\ \frac{B\log_2(G+1)}{\log_2(1/q)}\Big\}.$$

Estado: derivación del soporte IA, comprobada numéricamente; falta la prueba escrita por A o B.

## 3. Direcciones, de la más prometedora a la menos

1. **El teorema de la tasa laminar.** Ya calculado y verificado. Falta escribirlo: forma normal por átomos, potencial $W$, renovación. Dos páginas. Da la mitad laminar del paper con constante exacta, la sección 4 (un algoritmo laminar óptimo con $q$ homogéneo) y responde la pregunta (36). Pregunta abierta: ¿$c^*(G,\tfrac12)\to1$ cuando $G\to\infty$? En $G=96$ vale 0.869 y sigue subiendo.
2. **El techo por entropía.** Una línea mejora el lema de A: sin el $+1$ y para todo $q$. Es la mitad superior del precio en cualquier $q$.
3. **La curva de separación con diseños reales.** Implementar la construcción de Lindström ($k=3$: 12 personas con 7 pruebas, verificable por enumeración) y medir su tasa por prueba contra $c^*(G)$. Reorientar el loop 1 a esa meta: maximizar la *tasa* de un diseño de bloque, con cruces, en lugar de un valor con $(n,B)$ fijo. Eso dice desde qué $G$ el general gana de verdad y con qué pendiente.
4. **¿$q=\tfrac12$ es el peor caso? (pregunta 35).** El techo no lo decide: la cota del precio alcanza su máximo en $q\approx0.75$, justo donde el techo cambia de régimen. Hace falta el lado general con $q\ne\tfrac12$: diseños de bloque evaluados exactamente para $q$ general contra $c^*(G,q)$.
5. **Q1 con personas distintas.** Generalizar la regla de cociente a $q_i$ heterogéneos: cada bloque candidato tiene su cociente; se prueba el de mayor cociente. Medirla en la batería del loop 2 contra el óptimo laminar y contra rollout. Si iguala a rollout con menos cómputo, es el algoritmo de la sección 4 en general.
6. **Después del primer paper.** Las tres extensiones de Francisco. Con pruebas imperfectas los átomos dejan de existir; el método del potencial tendría que cambiar.

## 4. Propuesta concreta para el martes

**El mensaje (2 minutos).** "Tu método del potencial, con el potencial óptimo, da la constante exacta del lado laminar, y la estrategia óptima es una regla de cociente como la que pediste en septiembre."

**El enunciado que proponemos** (para el Overleaf, sección 3):

> **Teorema.** Con $q$ homogéneo, grupos de a lo sumo $G$ y conteos exactos, toda política laminar cumple $V_{\rm lam}\le c^*(G,q)\,B$, donde $c^*$ es la mejor tasa de un bloque; con $n\ge BG$ la cota se alcanza salvo un término $O(G)$. En $q=\tfrac12$, $c^*(G)\le2$. Toda política cumple $V\le B\log_2(G+1)$. Con las matrices detectoras, $V_{\rm gen}\ge\big(\tfrac14-o(1)\big)B\log_2G$. Luego el precio de la laminaridad en $q=\tfrac12$ es $\Theta(\log G)$.

**El mapa del paper** (las cuatro secciones del 27-oct):

| Sección | Resultado | Dueño | Estado |
|---|---|---|---|
| 1. Formulación | modelo, tres clases | equipo | escrito en partes |
| 2. Separación factor $G$ | aumentado contra no aumentado | Francisco y equipo | enunciada (fila 2026-10-06) |
| 3. Separación $\log G$ | tasa laminar $c^*$ + techo + Lindström | equipo (tasa, techo), Francisco (construcción) | tasa y techo verificados; construcción pendiente |
| 4. Algoritmos laminares | regla de cociente; óptima con $q$ homogéneo; medida con $q_i$ distintos | equipo | óptima demostrada en lo homogéneo (por escribir); lo heterogéneo, por medir |

**Tres preguntas para Francisco.**
- ¿Crees que $c^*(G,\tfrac12)\to1$? Si es así, el precio sería $\log_2G/4$ o algo así, con constantes explícitas.
- ¿Qué construcción de Lindström con tope $G$ nos das, para dibujar la curva general con números exactos?
- ¿La sección 4 puede ser esta regla de cociente, demostrada en lo homogéneo y medida en lo heterogéneo?

**Lo que nos comprometemos a traer la semana siguiente.** La prueba escrita del teorema de la tasa (Héctor y A); la construcción de Lindström $k=3$ verificada; el loop 1 reorientado a tasas.

## 5. Qué no decir

- No "demostrado": el teorema de la tasa está calculado y verificado; la prueba escrita falta.
- No "$c^*\to1$": es una conjetura; los valores con $G>20$ son de punto flotante.
- La línea de Lindström en la figura es asintótica y de literatura, por cotejar.
- No "$q=\tfrac12$ es el peor caso": el techo no lo decide.
- La base de la semana sigue primero: el WhatsApp y la firma de la nota.
