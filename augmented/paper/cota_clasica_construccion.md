# Cota clásica q ≤ ½ — construcción (Persona A)

*Documento de trabajo del plan lun 28 → mar 29-sep (`docs/notes/2026-09-28-plan-lun-mar.md`, §3 A).
Se construye parte por parte; cada parte se cierra solo cuando la prueba sobrevive la lupa de
revisor. De aquí salen `seccion_separacion.tex` (IA, martes) y el Lemma `lem:classical` de
`main.tex`. Solucionario de referencia (no consultado durante la construcción): companion de
Francisco, Thm 7.1 (Harris 7.1 + potencial normalizado 7.3).*

*Ejemplo ancla: (q, G, B) = (0.05, 16, 7), n = 48. Lado clásico que hay que probar: 7 × 0.05 = 0.35.
Lado aumentado (ya verificado por B): 0.9147 (CBS) y 1.1233 (exacto restringido a G ≤ 8, Mapa 3).*

---

## Vocabulario de trabajo

- **Modelo clásico (binario):** un test sobre un pool T solo responde "hay al menos un infectado"
  (positivo) o "nadie" (negativo). Sin conteos. Presupuesto B, duro.
- **Indicadores de salud:** H_i = 1 si i está sana, 0 si infectada; independientes, Pr(H_i = 1) = q.
- **Negativo / positivo de un pool P:** negativo = {H_i = 1 para toda i ∈ P};
  positivo = {existe i ∈ P con H_i = 0}.
- **Acreditadas N_t:** personas que están en algún pool negativo *antes* del test t.
  (Ojo: bajo posterior-zero "acreditar" es Pr(infectada | h) = 0; que en el modelo binario eso
  coincida con "estar en un negativo" NO se asume: se prueba en la Parte 3.)
- **Rendimiento de un test:** número de personas *nuevas* que acredita: |T_t ∖ N_t| si T_t sale
  negativo, 0 si sale positivo.
- **Evento creciente / decreciente** en los H_i: A es creciente si al cambiar cualquier H_i de 0 a 1
  se sigue cumpliendo A; decreciente al revés. ("Negativo" es creciente; "positivo" es decreciente;
  "T todo sano" es creciente.)
- **Desigualdad de Harris (1960):** con H_1, …, H_n independientes, A creciente y D decreciente:
  Pr(A ∩ D) ≤ Pr(A) · Pr(D). (Equivalente: dos eventos crecientes se correlacionan ≥ 0.)
  Es la única herramienta externa; se enuncia con tus palabras antes de usarla.

---

## Enunciado a probar

**Lema (cota clásica en salud rara).** Modelo binario, q homogénea con q ≤ ½, u ≡ 1, n ≥ B.
Toda política dinámica con B tests acredita en esperanza a lo más B·q personas, y B tests
individuales sobre personas distintas acreditan exactamente B·q. Luego
OPT^D_bin(B) = OPT^S_bin(B) = B·q.

Plan de la prueba en cuatro partes (las tres del plan operativo + la suma):

1. rendimiento de un test sin historia = m·q^m;
2. m·q^m ≤ q si q ≤ ½;
3. con historia el rendimiento no sube: Pr(T todo sano | h) ≤ q^m (Harris);
4. sumar sobre los B tests y exhibir la igualdad.

---

## Ejercicio 0 — números antes de letras (30 min)

**0.a Rendimiento sin historia.** Llena m·q^m:

| q \ m | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 0.05 | | | | | |
| 0.3 | | | | | |
| 0.5 | | | | | |
| 0.6 | | | | | |

(i) ¿En qué fila y columna un grupo *empata* al singleton? (ii) ¿En qué fila el par le *gana* al
singleton, y qué le pasa al lema ahí? (iii) En el ancla (q = 0.05), ¿cuánto rinde el pool de 16
comparado con el singleton?

**0.b Harris con las manos.** n = 3, q = 0.3. Primer test {1,2}: sale positivo. Calcula
Pr(H_1 = 1 | positivo) y compáralo con q; Pr(H_1 = H_3 = 1 | positivo) y compáralo con q²;
el rendimiento esperado del segundo test si es {1,3}, {1} o {3}. ¿Quién gana? Repite con q = 0.5.

**0.c Negativos.** Mismo n = 3, q = 0.3. Primer test {2}: negativo (2 queda acreditada). Segundo
test {1,3}: ¿Pr(H_1 = H_3 = 1 | h)? ¿Cambió algo respecto a "sin historia"? ¿Por qué?

**0.d Posterior-zero.** En 0.b, tras el positivo de {1,2}, ¿alguna persona no acreditada puede tener
Pr(infectada | h) = 0? Calcula Pr(H_1 = 0 | positivo).

*(Las respuestas están en un solucionario aparte de la IA; se cotejan al terminar, no antes.)*

---

## Parte 1 — Rendimiento de un test sin historia ⬜

**Enunciado.** Sin historia, un test sobre T con |T| = m tiene rendimiento esperado m·q^m.

*Tu prueba aquí.* (Debe decir qué acredita un test y por qué solo el negativo lo hace, y de dónde
sale q^m.)

## Parte 2 — m·q^m ≤ q para todo m ≥ 1 si q ≤ ½ ⬜

**Enunciado.** Si 0 < q ≤ ½ entonces m·q^m ≤ q para todo entero m ≥ 1, con igualdad exactamente en
m = 1 y, si q = ½, también en m = 2.

*Tu prueba aquí.* (Pista de forma, no de contenido: reduce a m·q^{m−1} ≤ 1 y usa q ≤ ½. Di también
qué pasa con q > ½ y m = 2: ahí vive el umbral ½ del corolario del juego de dos tests.)

## Parte 3 — La historia no ayuda ⬜

**Enunciado.** Sea h cualquier historia (secuencia de pools con su resultado), N las personas en
algún pool negativo de h, y T ⊆ [n] ∖ N con |T| = m. Entonces Pr(T todo sano | h) ≤ q^m.

**Corolario (se prueba aquí mismo).** En el modelo binario, si i ∉ N entonces
Pr(i infectada | h) ≥ 1 − q ≥ ½ > 0. Luego bajo posterior-zero y bajo estricta se acredita
exactamente a N: la cota no depende de la convención.

*Tu prueba aquí.* Ingredientes que la lupa va a buscar: (a) separar h en negativos y positivos;
(b) qué medida queda al condicionar en los negativos (¿sigue siendo producto sobre [n] ∖ N?
¿por qué?); (c) que cada positivo, restringido a [n] ∖ N, es un evento decreciente y que la
intersección de decrecientes es decreciente; (d) que "T todo sano" es creciente; (e) Harris;
(f) por qué Pr(h) > 0 no estorba.

## Parte 4 — Suma sobre los B tests e igualdad ⬜

**Enunciado.** Para toda política, E[acreditadas] ≤ B·q; B singletons distintos dan B·q.

*Tu prueba aquí.* Ingredientes: (a) acreditadas totales = Σ_t rendimiento_t (cada persona se
cuenta una sola vez: ¿en qué test?); (b) E[rendimiento_t | h_t] ≤ q por Partes 1–3 (ojo con T_t que
incluya acreditadas, o m = 0); (c) esperanza iterada; (d) políticas que paran antes de B;
(e) la igualdad y dónde se usa n ≥ B.

---

## Números apretados (después de la prueba, 30 min)

Con el lema, el lado clásico del ancla y de la frontera deja de ser cota y pasa a ser óptimo exacto.
Llena:

| Celda | n | Clásico exacto B·q | Laminar | Factor |
|---|---|---|---|---|
| Ancla (0.05, 16, 7), CBS | 48 | | 1 − 0.95^48 = | |
| Ancla, exacto G ≤ 8 (Mapa 3) | 48 | | 1.1233 | |
| Frontera (0.05, 8, 6), exacto | 24 | | 0.8654 | |

Familia G = 2^j, B = j + 1, q = c/G: razón ≥ (1 − (1 − c/G)^G) / ((j+1)·c/G). Escribe por qué el
numerador tiende a 1 − e^{−c} y por qué el cociente crece como G / log₂ G. Calcula j = 4 y j = 8
con c = 1.

Fuentes de los valores laminares: `results/mapa_homogeneo_3.csv` (Mapa 3, B) y §16 del plan
maestro (CBS, aritmética verificada 2026-08-30).

---

## Alcance: qué NO se prueba hoy

- **q heterogénea (q_i ≤ ½) y u heterogénea:** es el Thm 7.1 completo del companion
  (óptimo = top-B por q_i·u_i). Las Partes 1–3 pasan casi sin cambio (rendimiento
  ≤ max_{i∈T} q_i·u_i · m/2^{m−1}), pero la suma de la Parte 4 se rompe: el mismo "líder" de alto
  q_i·u_i puede repetirse en varios tests si sale positivo dentro de un grupo. Ahí es donde el
  companion mete el potencial normalizado (7.3). Se anota como extensión; la separación del paper
  solo necesita el caso homogéneo.
- **El lado aumentado:** no se prueba, se cita: CBS es construcción (Prop `prop:cbs`, demostrada)
  y 1.1233 / 0.8654 son valores exactos de B (VERIFICADO).

---

## Lupa de revisor (se aplica al cerrar cada parte)

1. Cada objeto tiene nombre antes de usarse (T, m, N, h, H_i).
2. Cada frase sobrevive a q = 0.6, m = 2: ¿dónde exactamente se usó q ≤ ½? (Debe ser solo en la Parte 2.)
3. Cada frase sobrevive a 0.b y 0.c: ¿dónde se usaron los negativos, dónde los positivos?
4. Harris se enuncia con sus hipótesis (independencia, monotonía) y se verifica cada una sobre los
   eventos concretos.
5. La suma de la Parte 4 no cuenta a nadie dos veces y no supone que la política use los B tests.
6. Se dice dónde se usa n ≥ B.

---

*Versiones para Francisco (al cerrar): (V1) la prueba en palabras de A, tal cual; (V2) la versión
formal refinada por IA en `seccion_separacion.tex`, etiquetada como tal. Ambas se conservan.*
