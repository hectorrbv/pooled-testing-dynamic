1. Correcciones a las preguntas preliminares

- A ya está resuelta en el texto. En 5.4 se emparejan dos perfiles completos antes de correr la política: Z elige el camino y Z' lo evalúa. Un historial como AB=1 no define un subcubo. Es solo la rama que siguen algunos perfiles del subcubo; en 5.1, una consulta. Nunca se condiciona en el historial. Lo que no se transfiere está en el lema 5.2: este parte en dos conjuntos crecientes (z_A=0 y el resto). Con conteo, el conjunto «AB=1» no es creciente: si le agregas un infectado, pasa a 2.
- B es buena idea, pero tiene una trampa. B se acredita porque A está infectado. En el paper, más infectados nunca ayudan a acreditar (R creciente, f decreciente); con conteos, sí. Además, en laminar el certificado ya se conoce: un bloque residual con conteo cero (AB menos A). Lo usamos en `laminar_inference.py`; no es hallazgo nuevo.
- C: 3 vs 2.5 solo da c≤5/6. Lo que podría impedir una constante independiente de G es otra cosa, todavía conjetura (ver pregunta 3). El experimento de cambio sigue sin ejecutarse.
- Sobre subcubos: la malla con q=.5 no basta. El lema 5.3 debe valer en todo subcubo, es decir, con cada q_j en {0, ½, 1}, para cualquier política y en la forma «camino elegido con z, evaluado con su complementario». Además, la simetrización solo sirve si la construcción usa un perfil virtual. Una política que corre sobre el perfil real, como la de seguir y cambiar, no la necesita.

2. Preguntas orales (las tres primeras, si solo hay 10 min)

1) «Para reutilizar 5.4, ¿hay que probar nuestra desigualdad en cada subcubo, con cada persona sana fija, infectada fija o con probabilidad ½, y para cualquier política? Además, con conteos un infectado fijo solo suma una constante al conteo, en vez de arruinar la prueba. ¿Se simplifica 5.3 y lo difícil queda en el árbol?»
Para qué: fijar qué hay que demostrar y no usar q=.5 como evidencia. Aclaración de lectura, con un poco de investigación.

2) «Con AB=1 y A=1 acreditamos a B justo porque A está infectado; en tu lema, más infectados nunca ayudan. ¿Ves algo que sustituya esa monotonía? Nuestra candidata para la "oferta" en laminar es un bloque residual con conteo cero.»
Para qué: ubicar dónde se rompe la prueba con soft clearing. Esa pregunta (23) sigue sin respuesta. Investigación.

3) «Tenemos 1/G elemental y un caso que limita a 5/6. ¿Te interesa una constante que dependa de G o solo una independiente de G? Si con q=½ una laminar dinámica acreditara como mucho K personas por prueba, las familias con pruebas no laminares que identifican a todos con pocas pruebas descartarían la segunda.»
Para qué: decidir el objetivo antes de invertir en la prueba. Investigación.

4) «En el caso 3 vs 2.5, C se deduce de AB=1, ACD=1 y BCE=1 aunque está en dos pruebas positivas, y la pérdida aparece justo desde B=3. ¿Te parece natural una clase intermedia: laminar más a lo sumo una prueba no laminar por trayectoria?»
Para qué: convertir la hipótesis de Héctor en algo medible. Que empiece en B=3 es coincidencia sugerente, no evidencia. Investigación.

5) «¿Nos explicas con el ejemplo del apéndice B de dónde sale el 2 en 5.2, la cuenta |R_p|≥(2^k−1)|R_z|?»
Para qué: Héctor aún no domina 5.2, y esa cuenta (cuántos perfiles continúan contra cuántos se detienen) es la que habría que rehacer con conteos. Aclaración; Francisco ofreció explicarla.

6) «Confirmo mi lectura: el subcubo se arma con Z y Z' antes de correr la política, y Z' es independiente del camino. ¿Correcto?»
Para qué: cerrar A en 30 segundos y que Héctor pueda explicar el papel de Z y Z'. Aclaración.

3. Siguiente paso

No repetir mapas de calor. Ejecutar dos comparaciones nuevas, cada una con una pregunta de sí o no anotada antes de correrla:

- El experimento de seguir y cambiar, ya diseñado y sin ejecutar, primero en n6, B4, G3, q=.5 y después en los seis casos guardados. Hay que reportar por separado tres pérdidas: la de clase, la de comprometer el prefijo y la de la heurística. Pregunta: ¿cambiar al primer paso no laminar recupera más de 2.5?
- El óptimo de la clase «laminar más una prueba no laminar» en los 114 casos con B entre 3 y 5. Pregunta: ¿cierra la mayor parte de la brecha?

La primera mide una pérdida que ya sospechamos; la segunda pone a prueba la hipótesis nueva de Héctor. En paralelo, sin código: escribir en media página qué desigualdad haría falta en cada subcubo, solo si la construcción elegida usa un perfil virtual.
