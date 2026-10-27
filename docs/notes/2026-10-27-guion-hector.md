# Guion de Héctor para la sesión del 27 de octubre (19:00 Praga)

Parte del equipo que lleva Héctor; el guion congelado de A va en `2026-10-27-guion-sesion.md`. Material: notebook 29 (base recortada) y notebook 30 (`augmented/notebooks/30_experimentos_pendientes.ipynb`). Todo número citado está en el repo con script o CSV.

## Refresco de la sesión anterior (20 de octubre)

Francisco trajo, de palabra, dos mitades: toda política laminar vale $O(B)$ por una función potencial $\phi(m,r)=(m-2r)^2/m$ sumada sobre átomos (una prueba virgen la sube 1 en promedio; partir un átomo la sube una constante; telescopio), y una construcción general con matrices detectoras (Lindström) que identifica a todos y obtiene $\Omega(B\log G)$. Conjetura que $\log G$ es el factor óptimo. Su ítem: formalizar el potencial y subirlo al Overleaf; mandar los números de Lindström. El equipo mostró los dos ejemplos binario > laminar, la cadena del techo $B\log_2(G+1)$ de palabra y la medición $n=15$ (0.828). Quedó pendiente de nuestro lado el escrito de los contraejemplos y el párrafo del techo.

## Orden y minutos

1. **Lo que se mandó por escrito (1 min).** Los dos contraejemplos con árboles y valores; la tabla $q=\tfrac12$; el párrafo del techo. Se dice qué salió y qué no; nada se presenta como cerrado si no se envió.

2. **El paso de partir, verificado exacto (3 min).** Notebook 30, sección 1. Enumeración en racionales, 638 tripletas con $m\le12$: el salto esperado es $4r(m-r)/(m(m-1))$, no depende del tamaño de la prueba, es a lo sumo $m/(m-1)$ y toca 2 solo en $m=2$. La prueba virgen sube exactamente 1. Con el telescopio, $V_{\rm lam}\le2B$, si el potencial domina la utilidad. Pregunta: ¿la constante 2 es la que él tiene, o la mejora?

3. **Diseños que identifican a todos (3 min).** Sección 2. Diez diseños con $n\le16$, todos distinguen los $2^n$ perfiles, general $=n/2$. El óptimo laminar exacto conserva entre 0.88 y 0.96 de $n/2$ en $n\le8$ y 0.828 en Söderberg–Shapiro ($n=15$, $B=9$, $G=7$). Pocos puntos, sin extrapolar. Pregunta: ¿qué sucesión $(n,B,G)$ usa su construcción, para correrla exacta hasta donde alcance?

4. **El techo, comprobado celda por celda (2 min).** Sección 3. En 24 celdas de la malla con árbol óptimo general exacto: hoja por hoja, despejar $m$ personas exige probabilidad $\le2^{-m}$; en promedio ninguna celda supera $\log_2K$, y la cota del lema es $\log_2K+1$. Pregunta: ¿el $+1$ del promedio se quita con su argumento o es el precio de pasar de "siempre" a "en promedio"?

5. **Qué dejó la búsqueda de diseños (2 min).** Sección 5 (del notebook 29). En $n=16$, $B=4$, el DP por bloques de hasta seis personas sube de 0.625 a 0.844 sanos por prueba, y la curva se aplana porque nunca cruza bloques. Es lo que su argumento predice: sin cruces no hay crecimiento. Lo que sigue es un piloto con cruces entre bloques, solo con su familia $(n,B,G)$ en la mano.

6. **Cierre (1 min).** Pedir los números de Lindström y su escrito del potencial para cotejar; proponer que el Overleaf tenga modelo, lema de información (equipo), potencial [Francisco], construcción [Francisco], ejemplos y mediciones.

## Qué no se dice

- No "demostrado" para $V_{\rm lam}\le2B$: la identidad del split está verificada; la dominación del potencial sobre la utilidad es de Francisco y no está escrita.
- No "el factor es $\log G$": la medición $n\le16$ no distingue el orden. Ambos valores quedan lejos de sus techos ($2B=18$; $B\log_2 8=27$ en $n=15$).
- No "la regla del loop Q1 gana al rollout": en $n=50$ no se distingue.
- No "autoresearch encontró": se dice cuántos experimentos corrieron y qué cambió.
- Los diseños de búsqueda aleatoria no son mínimos.

## Si pregunta por la nota "individuales"

El barrido con semilla propia (20261026, 1 774 instancias) cerró en cero fallas; falta la lectura de la prueba y la firma. Se dice así.
