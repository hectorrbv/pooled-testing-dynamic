# Guion de Héctor para la sesión del 27 de octubre (19:00 Praga)

Palabras simples, ecuaciones completas. Material: notebook 29 (el recortado) y notebook 30. Cada número que digas está en el repo.

## Las palabras y los símbolos

- $n$ personas. Cada una está sana con probabilidad $q=\tfrac12$ y vale $u=1$ punto si la acreditamos sana.
- Una **prueba** junta un grupo de a lo sumo $G$ personas y devuelve cuántas están infectadas. Tenemos $B$ pruebas.
- **Laminar:** dos grupos que usemos deben ser separados, o uno dentro del otro. No pueden cruzarse a medias. **General:** sin esa regla.
- **Acreditar** o **despejar:** decir con certeza que una persona está sana. $V$ es el número esperado de acreditados; $V_{\rm lam}$ con la regla laminar, $V_{\rm gen}$ sin ella.
- **Historial:** la lista de resultados vistos. $K$ es cuántos historiales distintos puede producir una estrategia.
- **Átomo:** un pedazo de $m$ personas del que sabemos cuántas están infectadas, $r$, pero no cuáles.
- **Potencial:** el marcador de Francisco, $\phi(m,r)=(m-2r)^2/m$ por átomo, sumado sobre átomos. Sube cuando un átomo queda casi todo sano o casi todo infectado. Un átomo con $r=0$ aporta $m$: justo lo acreditado.
- **Prueba virgen:** probar personas nuevas. **Partir un átomo:** probar una parte de un átomo conocido.
- **Diseño detector:** una lista fija de pruebas cuyos resultados, juntos, dicen el estado de cada persona. Acredita $n/2$ en promedio.

## Qué pasó en la sesión anterior (20 de octubre)

Francisco dijo, de palabra y sin escribirlo, dos cosas.

Primera: con la regla laminar, $V_{\rm lam}\le C\cdot B$ para una constante $C$. El argumento es el marcador. Una prueba virgen lo sube $1$ en promedio. Partir un átomo lo sube a lo sumo una constante. Después de $B$ pruebas el marcador vale a lo sumo $C\cdot B$. Y el marcador es siempre $\ge$ lo acreditado. Entonces lo acreditado también es $\le C\cdot B$.

Segunda: sin la regla, con diseños detectores, $V_{\rm gen}\ge c\cdot B\log_2 G$. Así que el laminar pierde un factor $\log_2 G$. Él cree que ese es el peor factor posible.

Pendiente de él: escribir la prueba del marcador y mandar los números de los diseños detectores. Pendiente de nosotros: mandar los dos ejemplos por escrito y el párrafo del techo.

## Qué decir hoy, en orden

**1. Qué mandamos por escrito (1 minuto).** Decir qué se envió y qué no. Lo que no se envió no se presenta como terminado.

**2. Revisamos su paso de partir el átomo (3 minutos).** Notebook 30, sección 1. Calculamos exacto cuánto sube el marcador al partir un átomo de $m$ personas con $r$ infectadas, probando una parte de $k$ personas. Si en esa parte salen $X$ infectadas, el salto es
$$\Delta=\phi(k,X)+\phi(m-k,\,r-X)-\phi(m,r),$$
y su promedio es
$$\mathbb E[\Delta]=\frac{4\,r\,(m-r)}{m\,(m-1)}\le\frac{m}{m-1}\le2 .$$
Lo comprobamos en los 638 casos con $m\le12$. No depende de $k$. Llega a $2$ solo con $m=2$; con átomos grandes se acerca a $1$. La prueba virgen sube exactamente $1$: si $R$ es el número de infectados en un grupo nuevo de $M$ personas, $\mathbb E[(M-2R)^2/M]=1$. Si su marcador domina lo acreditado, entonces $V_{\rm lam}\le 2B$.

Pregunta para él: ¿tu constante también es $2$, o tienes una mejor?

**3. Diseños que identifican a todos (3 minutos).** Sección 2. Probamos diez diseños detectores con $n\le16$. Con cada uno se sabe el estado de todos, así que $V_{\rm gen}=n/2$. Con las mismas $(n,B,G)$ y la regla laminar, el mejor valor exacto queda abajo: $V_{\rm lam}/(n/2)$ está entre $0.88$ y $0.96$ con $n\le8$, y vale $0.828$ con el diseño de $n=15$, $B=9$, $G=7$. Son pocos puntos; no sacamos conclusiones sobre el crecimiento.

Pregunta para él: ¿con qué sucesión $(n,B,G)$ va tu construcción? Con eso la corremos exacta hasta donde alcance la computadora.

**4. El techo, comprobado caso por caso (2 minutos).** Sección 3. Nuestra parte es el techo para cualquier estrategia, laminar o no:
$$\mathbb E[\text{acreditados}]\le\log_2K+1\le B\log_2(G+1)+1 .$$
La idea: para acreditar a $m$ personas, el historial tiene que ser raro, con probabilidad $\le 2^{-m}$. Lo comprobamos en 24 casos pequeños con la mejor estrategia general exacta. En ninguno se rompe; de hecho ninguno pasa de $\log_2K$.

Pregunta para él: el $+1$ aparece al pasar de "siempre" a "en promedio". ¿Con tu argumento se puede quitar?

**5. Qué dejó la búsqueda de diseños (2 minutos).** Sección 5. Con $n=16$ y $B=4$, un programa buscó diseños. Lo mejor que encontró trabaja por bloques de hasta $6$ personas y acredita $0.84$ sanos por prueba, contra $0.63$ del diseño inicial. Al permitir $G$ más grande no mejora, porque nunca cruza bloques. Es justo lo que dice su argumento: sin cruces no hay crecimiento.

**6. Cierre (1 minuto).** Pedirle sus números de los diseños detectores y su escrito del marcador. Proponer el orden del documento compartido: modelo, techo (nosotros), marcador (él), construcción (él), ejemplos y mediciones.

## Qué no decir

- No decir "demostrado" para $V_{\rm lam}\le2B$. Lo comprobado es el paso de partir. Que el marcador domina lo acreditado es de Francisco y no está escrito.
- No decir "el factor es $\log_2 G$". Con $n\le16$ no se ve el crecimiento; en $n=15$ los dos valores quedan lejos de sus techos ($2B=18$ y $B\log_2 8=27$).
- No decir que la regla nueva de búsqueda le gana al rollout. Con $n=50$ empatan.
- No decir "la inteligencia artificial encontró". Decir cuántos experimentos corrieron y qué cambió.
- Los diseños que salieron de búsqueda aleatoria no son los más pequeños posibles.

## Si pregunta por la nota "individuales nunca óptimas"

Corrimos el barrido con nuestra propia semilla: 1 774 casos, cero fallas. Falta leer la prueba y firmar. Se dice así.
