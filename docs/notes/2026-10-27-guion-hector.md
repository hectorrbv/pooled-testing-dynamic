# Guion de Héctor para la sesión del 27 de octubre (19:00 Praga)

Versión en palabras simples. Material: notebook 29 (el recortado) y notebook 30. Cada número que digas está en el repo.

## Las palabras que vamos a usar

- **Persona sana o infectada.** Cada persona está sana con probabilidad un medio. Vale 1 punto si la acreditamos sana.
- **Prueba.** Juntamos a varias personas en un grupo y el resultado dice cuántas están infectadas. Un grupo tiene a lo sumo G personas. Tenemos B pruebas.
- **Laminar.** Regla sobre los grupos: dos grupos que usemos deben ser separados, o uno debe estar dentro del otro. No pueden cruzarse a medias.
- **General.** Sin esa regla. Los grupos pueden cruzarse como quieran.
- **Acreditar / despejar.** Decir con certeza que una persona está sana. Eso es lo que cuenta.
- **Historial.** La lista de resultados que hemos visto hasta ahora.
- **Átomo.** Un pedazo de personas del que ya sabemos cuántas están infectadas, pero no cuáles.
- **Potencial.** Un marcador que Francisco inventó. Mide qué tan cerca estamos de acreditar gente. Sube cuando un átomo queda casi todo sano o casi todo infectado.
- **Prueba virgen.** Probar un grupo de personas nuevas, de las que no sabemos nada.
- **Partir un átomo.** Probar solo una parte de un átomo que ya conocíamos.
- **Diseño detector.** Una lista fija de pruebas que, mirando todos sus resultados juntos, permite saber el estado de cada persona.

## Qué pasó en la sesión anterior (20 de octubre)

Francisco dijo, de palabra y sin escribirlo, dos cosas.

Primera: con la regla laminar, lo que se puede acreditar crece como el número de pruebas B, multiplicado por una constante. Su argumento es el marcador: cada prueba sube el marcador poco (una prueba virgen lo sube 1 en promedio; partir un átomo lo sube como mucho una constante). Después de B pruebas el marcador vale como mucho una constante por B. Y el marcador siempre es mayor o igual que lo acreditado. Entonces lo acreditado también es como mucho una constante por B.

Segunda: sin la regla laminar, con diseños detectores, se acredita mucho más: del orden de B por el logaritmo de G. Así que el laminar pierde un factor logaritmo de G. Él cree que ese es el peor factor posible.

Lo que quedó pendiente de él: escribir la prueba del marcador y mandar los números de los diseños detectores. Lo que quedó pendiente de nosotros: mandar los dos ejemplos por escrito y el párrafo del techo.

## Qué decir hoy, en orden

**1. Qué mandamos por escrito (1 minuto).** Decir qué se envió y qué no. Lo que no se envió no se presenta como terminado.

**2. Revisamos su paso de partir el átomo (3 minutos).** Notebook 30, sección 1. Calculamos exacto, sin redondeos, cuánto sube el marcador cuando se parte un átomo. Probamos todos los casos con átomos de hasta 12 personas: 638 casos. Resultado: sube como mucho 2, y solo llega a 2 con átomos de 2 personas. Con átomos grandes sube cerca de 1. La prueba virgen sube exactamente 1. Si su marcador de verdad domina lo acreditado, entonces lo acreditado con la regla laminar es como mucho 2 por B.

Pregunta para él: ¿tu constante también es 2, o tienes una mejor?

**3. Diseños que identifican a todos (3 minutos).** Sección 2. Probamos diez diseños detectores con hasta 16 personas. Con cualquiera de ellos se sabe el estado de todos, y se acreditan n/2 sanos en promedio (la mitad de la gente). Con las mismas pruebas y la regla laminar, lo mejor posible acredita menos: entre 88% y 96% de eso con 8 personas o menos, y 83% con el diseño de 15 personas, 9 pruebas y grupos de 7. Son pocos puntos; no sacamos conclusiones sobre el crecimiento.

Pregunta para él: ¿con qué tamaños de población, pruebas y grupo va tu construcción? Con eso la corremos exacta hasta donde alcance la computadora.

**4. El techo, comprobado caso por caso (2 minutos).** Sección 3. Nuestra parte es el techo: ninguna estrategia, laminar o no, acredita en promedio más que B por el logaritmo de G más 1, más uno. La idea: para acreditar a m personas, el historial tiene que ser muy raro, con probabilidad menor que 1 entre 2 elevado a m. Lo comprobamos en 24 casos pequeños con la mejor estrategia general exacta. En ninguno se rompe, y sobra margen.

Pregunta para él: el "más uno" aparece al pasar de "siempre" a "en promedio". ¿Con tu argumento se puede quitar?

**5. Qué dejó la búsqueda de diseños (2 minutos).** Sección 5. Con 16 personas y 4 pruebas, un programa buscó diseños. Lo mejor que encontró trabaja por bloques de hasta 6 personas y acredita 0.84 sanos por prueba, contra 0.63 del diseño inicial. Pero al permitir grupos más grandes no mejora, porque nunca cruza bloques. Eso es justo lo que dice su argumento: sin cruces no hay crecimiento.

**6. Cierre (1 minuto).** Pedirle sus números de los diseños detectores y su escrito del marcador. Proponer el orden del documento compartido: modelo, techo (nosotros), marcador (él), construcción (él), ejemplos y mediciones.

## Qué no decir

- No decir "demostrado" para el "2 por B". Lo que comprobamos es el paso de partir. Que el marcador domina lo acreditado es de Francisco y no está escrito.
- No decir "el factor es logaritmo de G". Con 16 personas o menos no se ve el crecimiento.
- No decir que la regla nueva de búsqueda le gana al rollout. Con 50 personas empatan.
- No decir "la inteligencia artificial encontró". Decir cuántos experimentos corrieron y qué cambió.
- Los diseños que salieron de búsqueda aleatoria no son los más pequeños posibles.

## Si pregunta por la nota "individuales nunca óptimas"

Corrimos el barrido con nuestra propia semilla: 1 774 casos, cero fallas. Falta leer la prueba y firmar. Se dice así.
