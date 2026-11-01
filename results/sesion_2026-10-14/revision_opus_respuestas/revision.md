(a) Errores y matices

Tus cinco respuestas son correctas en lo esencial. Encuentro estos matices:

- Respuesta 1, el par:
  - La única utilidad que debe ser positiva es la de C, u_C. A y B pueden tener cualquier utilidad no negativa.
  - Los priors interiores solo sirven para dos cosas: que q_C>0 y que q_A q_B + p_A p_B > 0.
  - El intercambio necesita conteos. El binario no distingue R=1 de R=2.
  - También necesita soft clearing. Si A sale infectado, B se cobra por deducción, no por estar en un grupo negativo.
  - Con esas dos condiciones, el par cobra lo mismo que las individuales en A y B, perfil a perfil, y no solo en promedio.
  - La comparación es contra individuales con las mismas B pruebas, y C es una persona que esas individuales no prueban.
- Respuesta 2, binario y laminar:
  - La familia se acerca a 10/9, pero no lo alcanza. Di "se acerca", no "alcanza".
  - El paper da dos factores 2. Su combinación en 4 es nuestra. Verifica tres cosas: que sus hipótesis coinciden con las nuestras, que una política binaria disjunta es laminar, y que soft clearing paga al menos lo mismo que hard clearing.
  - Una búsqueda local con n=4 no dice si la brecha crece con n, B o G.
- Respuesta 4, log G:
  - La cota V_lam ≤ KB solo puede valer con q=1/2 o con q lejos de 1. Si q se acerca a 1, un grupo de G sanos cobra casi G con una sola prueba.
  - La construcción también necesita bastantes bloques, es decir n grande frente a B·log G, y redondeo de B.
  - La celda 8 dice B=O(n/log n). Eso solo vale si G ≥ n. Tu versión por bloques corrige esto.
  - El conteo en sí tiene nombre clásico: pesada de monedas (coin weighing). Verifica la referencia antes de citarla.
- Paso de información de la celda 4:
  - Francisco puede preguntar por qué "identificar a un sano fija un bit". La respuesta: después de ver los resultados, cada persona no identificada aporta como mucho un bit de incertidumbre.
  - Por eso, el número esperado de personas identificadas no supera la información obtenida.
- Respuesta 5, jerarquía:
  - La pérdida máxima de la malla, 5.26%, es justo esta celda: 162/171. Siete de los 9/64 vienen de la jerarquía.
  - Sigue siendo un solo testigo.
  - Conservar la jerarquía de G4 solo garantiza no empeorar si eliges la mejor con evaluación exacta.
  - No mezcles este 5.26% con el 21.15%. El 21.15% es laminar frente a general.
- Consultas conservadas:
  - En la celda 9, aclara que π_L y π_ratio son reglas antiguas, distintas de greedy y rollout.

(b) Respuestas sugeridas

1. Celda 1, hipótesis para ganancia estricta:
Basta con G≥2, 2≤B<n, independencia, 0<q_i<1 y u_C>0. Con conteos y soft clearing, el par cobra lo mismo que las individuales en A y B, perfil a perfil. Además prueba C cuando R=0 o R=2. La ganancia es $(q_Aq_B+p_Ap_B)q_Cu_C>0$. Son condiciones suficientes. No hace falta q<1/2.

2. Celda 2, qué limita la ventaja:
Con nuestras hipótesis, las cotas publicadas, combinadas, dan $V_{\rm bin}\le4V_{\rm lam}$. Es una garantía, no un ejemplo. La mejor familia conocida se acerca a 10/9 sin alcanzarlo. No sabemos dónde está el límite real. En este ejemplo, la ventaja viene de volver a probar A con C. No verificamos ese mecanismo en la familia de cuatro.

3. Celda 4, construcción del mismo orden:
Candidata: dividir la población en bloques de G personas. Cada bloque se resuelve con una matriz detectora de O(G/log G) pruebas. Cada bloque da G/2 sanos esperados, es decir Ω(log G) por prueba. Para una brecha Ω(log G) faltan tres piezas: justificar la matriz, demostrar $V_{\rm lam}\le KB$ en q=1/2 y tener suficientes bloques.

4. Celda 5, jerarquía antes que regla:
En este testigo, sí. En n=7, B=4, G=5, rollout pierde 9/64: 7/64 vienen de la jerarquía y 2/64 de la decisión. Una jerarquía para G=4 es legal en G=5 y alcanza 171/64. Proponemos conservarla como opción, sin unir grupos de jerarquías distintas. Un solo testigo no ordena prioridades en general.

5. Celda 7, cuánto crece la separación:
No lo sabemos. La familia de cuatro personas lleva la razón $V_{\rm bin}/V_{\rm lam}$ cerca de 10/9. La cota combinada permite hasta 4. Cada factor 2 viene de un paso distinto. Aunque cada paso fuera ajustado, el producto podría no serlo. Siguiente paso: perturbar la familia con s<1 y probar con más personas o pruebas.

6. Celda 8, cota $V_{\rm lam}\le KB$:
Todavía no. La pregunta solo tiene sentido con q_i=1/2 y u_i=1. Si q se acerca a 1, un grupo de G sanos cobra casi G con una prueba. Las individuales obligan a K≥1/2. Sin esa cota, la matriz detectora solo muestra que el general obtiene Ω(log G) por prueba. No muestra separación.

7. Celda 9, decisión de continuación:
Todavía no localizamos la rama. Conocemos el tamaño de las pérdidas, no su causa. No las atribuimos al costo esperado ni a los desempates sin verlo. Proponemos seguir el mismo historial en la política laminar óptima y en cada heurística. Después marcamos la primera decisión distinta y su pérdida esperada.

(c) Pregunta para Francisco

Con q_i=1/2 y u_i=1, ¿una política laminar dinámica obtiene como mucho una constante por prueba? Si la respuesta es sí, los bloques detectores dan separación Ω(log G). Si es no, ¿qué estructura laminar acumula más?

Recomiendo esta pregunta porque es la pieza que falta en su propia dirección. Además, no alarga el notebook.