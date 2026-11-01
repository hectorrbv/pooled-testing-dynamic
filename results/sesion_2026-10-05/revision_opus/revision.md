# Revisión del notebook 28 para la sesión con Francisco

## 1. Veredicto

El notebook está casi listo. Todas las cifras de los cuatro bloques que recalculé coinciden con los datos y el código. Antes de la sesión hay que cambiar cuatro cosas, ninguna de las cuales pide cálculos nuevos:

- **Bloque 4:** cambiar el nombre de «recortar» y explicar en una línea qué hace la operación.
- **Bloque 3:** decir de dónde sale 0.5304 y corregir la frase sobre «separar fijar de cruces».
- **Bloque 1:** poner en pantalla el denominador y que ambos óptimos pueden adaptar.
- **Nota de Nick:** corregir «el teorema no se traslada».

Además, la consulta alcance-B debe decir qué modelo usa. El pase principal no crece.

## 2. Hallazgos

**1. Bloque 4 y consulta de Nick: «recortar» nombra dos operaciones distintas. Riesgo confirmado.**
- En el bloque 4, `trim_to_disjoint` es determinista: quita de cada grupo a quien ya apareció (ABD→D; ACD queda vacío).
- En Nick, el recorte es aleatorio: una moneda de 1/2 por cada aparición de una persona en un grupo.
- El bloque 3 dice «no aplica el recorte de Nick» justo antes de que el bloque 4 hable de «recortar».
- En el acta (D7), Francisco propuso como ruta repartir personas al azar, «incluirlas o no en una prueba». Es fácil que lea el 1.50 como un resultado de esa ruta.
- **Corrección:** llamarlo «quitar repetidos» y añadir una línea de definición (ver §3).

**2. Bloque 3: el 0.174 no es solo el costo de «fijar las decisiones». Error de formulación confirmado.**
- El plan fijo con cruces AB, AC vale 0.627 (q=.3; cuento solo los casos con crédito):
  - sss: 0.027·3
  - ssi y sis: 0.063·2 cada uno
  - isi e iis: 0.147·1 cada uno
  - total: 0.081 + 0.252 + 0.294 = 0.627
- Eso es más que el mejor plan fijo laminar, 0.600. Entre planes fijos, los cruces ya valen al menos 0.027.
- Por tanto, 0.774 − 0.600 es la diferencia entre la clase adaptativa laminar y la fija laminar. La frase «sirve para separar el costo de fijar las decisiones del costo de prohibir cruces» promete más de lo que muestra.
- La pantalla no explica de dónde sale 0.5304; solo lo hacen las notas.
- La §5 del paper ya advierte que el camino sin el recorte aleatorio «need not retain any constant fraction». Hay que presentarlo como la versión con conteos de esa advertencia, no como hallazgo propio.

**3. Bloque 4: la figura y la descripción de π_ratio quedan cortas. Riesgo de interpretación.**
- La figura empieza en «Probar D / Probar A», sin la prueba ABC ni la condición R∈{1,2}. Es una rama, no la política completa, aunque el texto lo diga arriba.
- No se dice que ambas reglas conservan ABC.
- `PoliticaRatio.decide` hace esto: para cada grupo nuevo o bloque mixto y cada horizonte de 1 a 3 pruebas, resuelve exactamente el mejor plan local laminar. Luego elige el mayor E[U]/E[T] y ejecuta solo su primera prueba.
- Con B=3, el horizonte cubre todo el presupuesto, así que aquí π_ratio es casi un planificador exacto. Su 1.75 coincide con el óptimo laminar.
- No conviene sugerir que una heurística barata recupera la pérdida.
- Comprobé a mano la rama ABC mixto: A resuelve con probabilidad 1/3, y R=1 y R=2 dan 2 y 1 sanos. Los valores de las políticas completas también cuadran:
  - replanear: 1/8·3.5 + 1/8·0.5 + 3/4·5/3 = 1.75
  - quitar repetidos: 1/8·3.5 + 1/8·0.5 + 3/4·4/3 = 1.50

**4. Bloque 1: falta el denominador y que ambas clases adaptan. Riesgo de interpretación.**
- Con el denominador invertido, el mismo caso da otra cifra: 1 − 1.75/2 = 12.5%, pero 2/1.75 − 1 = 14.3%.
- El paper compara dinámico con estático, así que Francisco puede leer «laminar» como estático. Lo mismo pasa en el bloque 2 con «el óptimo laminar obtiene 1.75».
- El panel B=4 es un control que se cumple por teorema (B=n, valor nq). Se ve igual que el empate empírico de B=2.
- En B=2 y B=4, la curva G=3 queda tapada por la de G=2; parece que falta.
- **Qué está certificado y qué no:**
  - El 12.5% y los 38 casos de B=4 están certificados, porque el óptimo general alcanza la cota nq (2 = 4·0.5).
  - Los empates de B=2 dependen de un único Bellman general. El replay verifica el valor de la política, no su optimalidad.
- **Corrección:** fórmula en el texto, título del panel «B = 4 = n (control)», y una frase «ambas G valen 0».

**5. Nota de la consulta de Nick: «El teorema no se traslada a conteos/soft clearing» exagera. Error de alcance confirmado.**
- Ningún material muestra que el factor 2 falle con conteos. Lo que sabemos es que la prueba no se aplica tal cual.
- **Pasos que sí se transfieren:**
  - promediar y luego elegir el mejor plan;
  - sumar persona por persona;
  - la identidad de §5.4.
- **Pasos que no se transfieren tal cual:**
  - El árbol de consultas y ofertas: con conteos se acredita sin ninguna prueba negativa, como muestra el propio bloque 2.
  - La afirmación (24) del Caso 3: con conteos, un grupo que contiene a i da conteos distintos según i esté sano o infectado, y los caminos se separan antes de t.
- El ejemplo AB, AC solo muestra que el recorte conserva cruces. Hay uno más fuerte: con entrada laminar ABC, AB, el resultado AC, AB cruza, con probabilidad 1/8·1/4 = 1/32.

**6. Consulta alcance-B: no dice el modelo ni el estatus. Omisión confirmada.**
- El intercambio es correcto (lo verifiqué), pero solo vale con conteos y soft clearing. Según el plan y el acta, con hard clearing falla en 156 de 650 casos.
- Ambos documentos etiquetan el caso general como «derivación + verificado n≤5» hasta que haya prueba escrita y la firma de Héctor.

**7. Conflictos entre el plan y la evidencia** (`docs/notes/2026-10-05-plan-mon-tue.md`):

- **(a) La cota 1/G.** El plan dice que no está validada. Sin embargo, la nota del 4-oct da una prueba que encuentro correcta, y Francisco la llamó obvia (acta D3). La celda meta puede afirmarla con enlace a la nota, añadiendo que supone historial vacío. No aplica directamente a la continuación del primer cruce, que parte de un historial condicionado.
- **(b) Los «ocho casos».** La tabla del plan tiene 6 filas: 5 con historial vacío y 1 condicionada. `verificacion_direccion.json` tiene 6 casos con historial vacío, incluido «individuales» con G=1. Ese caso y par_B2 (laminar = general) no pueden producir un primer cruce estricto si se desempata a favor de lo laminar.
- **(c) La definición de cruce del plan.** Contaría como cruce un superconjunto (A y luego ABC), salvo que la reducción lo convierta en BC. `BellmanPerfiles` en modo laminar admite ancestros, así que hay que fijar esto antes del experimento.
- **(d) Lo que espera el plan para el martes.** El §6.3 espera de Héctor el memo y la tabla del primer cruce; el notebook trae pilotos. Hay que decirlo así en la sesión. Además, la regla del §7 «sin números fuera de la tabla» choca con los números del notebook: aclarar si solo aplica a la nota.

## 3. Frases de reemplazo

- **Bloque 1:** «Pérdida relativa = 1 − V*laminar/V*general; ambos óptimos pueden adaptar. Con B=4=n el empate es obligado: es un control.»
- **Bloque 2:** «El mejor laminar, aun adaptándose, obtiene 1.75.»
- **Bloque 3:**
  - «Adaptar da 0.774, que también es el óptimo general. La mejor lista fija laminar da 0.600. Un camino simulado fija AB,A (prob. .42, vale .60) o AB,C (prob. .58, vale .48): .42·.60 + .58·.48 = 0.5304.»
  - «La primera diferencia es entre clases; la segunda, del procedimiento. §5 ya advierte que el camino sin recorte aleatorio puede perder mucho.»
- **Bloque 4:**
  - Encabezado de la figura: «Tras ABC con R=1 o 2 (prob. 3/4)».
  - «Ambas reglas conservan ABC. Quitar repetidos: cada grupo pierde a quien ya apareció (ABD→D, ACD vacío); la prueba libre va al mayor cobro inmediato. π_ratio: el mejor plan local de 1 a 3 pruebas por grupo o bloque, con mayor E[U]/E[T]; ejecuta solo su primera prueba.»
- **Nick:** «La prueba no se aplica tal cual a conteos; no sabemos si el factor 2 sobrevive.»
- **Alcance-B:** primera línea «Conteos y soft clearing»; estatus «derivación, verificado n≤5; prueba escrita pendiente».

## 4. Qué verifiqué y qué no

**Verifiqué:**
- Leí la exportación completa del notebook. Vi con Read las siete PNG. Leí el plan, el acta, la guía visual, la nota del 4-oct, §5 del paper, `experimento_mapa_laminar_28.py`, `experimento_sesion_28.py`, `figuras_28.py`, `bellman_perfiles.py`, `politica_ratio.py`, `datos_figuras_28.json` y `verificacion_direccion.json`.
- Rehíce a mano estas cuentas:
  - 0.774, 0.600, 0.5304 y 0.627;
  - la inyectividad de ABC/ABD/ACD (diferencia (−2t,t,t,t));
  - 1.75, 1.50, 3/4 y 1/3;
  - 0.18 y 9/82/9 del caso B=1;
  - la ganancia del intercambio, Λ≤1 con q≤1/2, 1/16 y 1/32;
  - el argumento 1/G;
  - los 81 empates y 33 separaciones a partir de las pérdidas guardadas.

**No verifiqué:**
- No ejecuté código.
- No leí `mapa_laminar_28.json` (solo `datos_figuras_28.json`, que trae las mismas pérdidas) ni `bm17_toy_solver.py`.
- No certifico la optimalidad del Bellman general fuera de los casos donde alcanza nq.
- No cotejé §3.2 del paper (Λ, Thm 3.3) ni el ejemplo de λ=0.30/0.31.
- No comparé el .ipynb con su generador.
