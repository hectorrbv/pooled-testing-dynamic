# Pizarrón, 10 min — grupo de estudiantes

Carga: con un ejemplo de media hoja se ve que adaptarse vale 9 veces más que el
mejor diseño fijo. Cero fórmulas en el pizarrón; solo dibujos y números.

## Min 0–2. Problema y giro

Dibujo: monigotes, un óvalo alrededor de tres.

- "Muchas personas, pocas pruebas, puedes probar en grupo: ¿a quién certificas sano?"
- Preguntar qué probarían primero. Van a decir individuales — guardarlo.
- Giro: la prueba cuenta cuántos sanos hay, y tú te adaptas a lo que ves.
- De palabra: el diseño fijo y el adaptativo binario ya tienen paper; conteos con
  adaptación, no.

## Min 2–4. El montaje y el diseño fijo

Escribir: 10 pruebas, 1 de cada 100 personas sana.

- Prevalencia alta: lo escaso es lo sano.
- Teorema conocido: con menos de la mitad sana, el mejor diseño fijo son puras
  pruebas individuales. Justo lo que dijeron.
- 10 × 0.01 = **0.1 personas** en promedio. Nueve de cada diez veces, manos vacías.

## Min 4–6.5. La estrategia adaptativa (corazón)

Dibujo: cuatro cajas de 64, una se prende; abajo, la cadena 64→32→16→8→4→2→1.

- Fase 1: cuatro pruebas grupales de 64 → tocas **256 personas con 4 pruebas**.
- Fase 2: si una dice que hay sanos, seis pruebas de búsqueda binaria la aíslan
  con certeza.
- Valor = probabilidad de que haya al menos un sano entre 256 = **0.92**.
- Contra 0.1. Mismo presupuesto, misma información. **9×**.

## Min 6.5–9. Los tres insights

Uno por minuto, cada uno con su dibujo mínimo.

1. Las probabilidades. La probabilidad de que las 256 estén infectadas es 7.6%;
   entre 10 personas era 90%. Lo raro individualmente es casi seguro
   colectivamente. Y mirar 25 veces más gente costó 4 pruebas, no 25: el
   presupuesto entra multiplicando en el diseño fijo y en el exponente en el
   adaptativo.
2. El algoritmo. Cuatro de las diez pruebas no certifican a nadie: compran
   información. Y la búsqueda binaria no descubre nada nuevo — ya sabías que
   había alguien; las seis pruebas se pagan solo para localizarlo. Saber que
   existe no sirve: hay que poder señalarlo. Una política golosa nunca arranca así.
3. Dónde se rompe. La estrategia encuentra una sola persona, así que está topada
   en 1. El diseño fijo crece sin tope. La separación vive donde lo sano es
   escaso; con abundancia, el fijo gana. La ventaja máxima posible aquí era 10× y
   llegamos a 9.2×.

## Min 9–10. Cierre

- Pregunta que se llevan: esta estrategia solo usó "¿hay alguien?" — un bit.
  La prueba devuelve un número. ¿Qué compra el conteo completo?
- Ahí es donde estamos trabajando.

## Números en la cabeza

10 pruebas · 1 de cada 100 · 0.1 · 4 grupos de 64 = 256 · 6 de búsqueda ·
0.92 · 9× · 7.6% · tope 10×

## Aire improvisado

- El "0.1 personas" déjalo caer como chiste: "en promedio certificas un décimo de persona".
- Las cuatro cajas dibújalas una por una; la que se prende, con color.
- La cadena 64→32→16→8→4→2→1 cuéntala con los dedos, no la escribas antes.
- Si algo se traba, lo único que hay que aterrizar es 0.1 contra 0.92.

## Si preguntan de más

- De dónde sale 0.92: probabilidad de que al menos una de 256 esté sana, con 1% cada una.
- Por qué 64: es el balance entre cubrir gente y pagar búsqueda binaria; con
  grupos de 2 ya se gana 1.7×, y satura cerca de 10×. (Slide de respaldo.)
- El ejemplo de 5 personas con solver exacto está en respaldo, por si sale el
  tema de qué hace el conteo completo.
- Umbrales y expresiones cerradas en q: trabajo en curso, sin revisar. No entrar.

## Ojo con la desigualdad final del mensaje

En el mensaje original la comparación queda como B*q < 1 − (1−q)^B, tomando
kG ≈ B. Esa desigualdad no se cumple nunca: por Bernoulli, (1−q)^B ≥ 1 − B*q,
así que el lado derecho siempre es menor o igual que el izquierdo. La separación
sí existe, pero viene precisamente de que kG es mucho mayor que B — 4 pruebas
cubren 256 personas, no 4. Vale la pena comentarlo con Francisco.
