## A — Correcta

- **Independencia de Ii y Xi.** El argumento de acoplamiento funciona. Si la ejecución con Xi=0 nunca prueba a i, ninguna salida depende de Xi, así que la ejecución con Xi=1 es idéntica. Por tanto Ii es función de (X₋ᵢ, semilla).
- **Acreditación.** Para acreditar a i hace falta Xi=0 e Ii=1, lo que da E[crédito_i] ≤ wi·E[Ii].
- **Cotas.** El máximo fraccionario con 0≤pᵢ≤1 y Σpᵢ≤BG es T_BG, y Σ_i Ii ≤ Σ|S_t| ≤ BG vale por trayectoria. T_BG ≤ G·T_B se obtiene partiendo en G bloques de B, y el min(k,n) cubre BG>n. Las individuales dan T_B y son laminares.

Observación útil: el testigo (B individuales) es **estático** laminar. Por tanto A ya da V*_estático_laminar ≥ V*_general/G. Esto importa para C.

## B — Correcta, con dos precisiones

- **Átomos.** Los A_S son disjuntos porque cada elemento pertenece a A_S del mínimo conjunto probado que lo contiene. El paso de los conteos de S a los de los átomos es triangular e invertible. La factorización vale también en políticas adaptativas, porque la verosimilitud de la historia es (probabilidad de la semilla) × indicadora de compatibilidad. Dentro de cada átomo la ley es producto de Bernoullis condicionado a la suma, que no es uniforme si los qᵢ difieren.
- **Precisión 1.** Al enunciar "solo los átomos con conteo 0 acreditan", conviene cubrir explícitamente r=|A|: todos están infectados con certeza y no hay sanos que acreditar. Con soporte completo, si 0<r<|A| cada integrante es sano en algún perfil compatible e infectado en otro.
- **Precisión 2.** Las pruebas repetidas del mismo conjunto deben deduplicarse antes de definir "hijos máximos".
- **Nueva prueba T.** Sea S\* el mínimo probado que contiene a T. Por minimalidad, cada hijo de S\* queda contenido en T o es disjunto, así que solo A_{S\*} se corta. Si ningún probado contiene a T, T es unión de raíces más parte de la zona exterior. En ambos casos la actualización es un split de un único átomo. Correcto.

## C — Correcta como está acotada, con una condición omitida importante

- **Cuenta de la cota.** Hay ≤B átomos y m/2^m ≤ 1/2, con igualdad en m=1,2, así que V ≤ B/2. La mezcla independiente de X hereda la cota.
- **Cantor–Mills.** El resultado es consistente con lo que conozco (Canad. J. Math. 18, 1966, pesadas no adaptativas con B ~ 2n/log₂n). Yo no puedo verificar el DOI. Con esas matrices se obtiene n/2.
- **Condición omitida: G.** Por A, para todo G fijo existe la constante 1/G. Lo que C prueba es que **no hay constante uniforme en G**: la instancia usa G=n. Hay que decirlo así.
- **Es una brecha de cruces.** El comparador de Cantor–Mills es él mismo estático. C separa estático-laminar de estático-con-cruces, sin que intervenga la adaptatividad.
- **Mejora en q=1/2.** Hay una cota informacional: H(X|T) ≤ Σᵢ H(Xᵢ|T) ≤ n − E|K|, y además H(X|T) ≥ n − H(T). Por tanto E|K| ≤ H(T) ≤ B·log₂(G+1). Junto con B/2 para las individuales, el cociente estático-laminar/general es ≥ 1/(2log₂(G+1)), y C es ajustada salvo constante. El parámetro relevante es log G, no n. Esto vale solo en q=1/2: con q≠1/2, H(Xᵢ|T=t) puede superar h(qᵢ) y el argumento no se traslada.

## ¿Justifica C priorizar la transformación adaptativa?

No todavía. **Falta el paso decisivo: acotar V*_dinámico_laminar en la misma instancia** (q=1/2, G=n, B~n/log n).

- Que C "no refute" el caso dinámico no implica que el caso dinámico tenga constante.
- La adaptación sí rompe m/2^m por prueba. Ejemplo: en el átomo {a,b,c} con r=1, probar {a,b} rinde 4/3. Y un átomo grande con r=1 permite búsqueda binaria muy rentable.
- Pero en q=1/2 los átomos de baja densidad son exponencialmente raros, y mi sospecha (no probada) es que el dinámico laminar también es O(B) ahí.
- Si así fuera, la meta de "constante universal" muere también para laminar dinámico. La meta honesta sería entonces un factor en G, entre 1/G y Θ(1/log G).

Antes de invertir en la transformación, haría dos cosas:
1. Calcular V*_dinámico_laminar exacto con el solver para n pequeño en esa familia de instancias, y compararlo con B/2 y con n/2.
2. Intentar una cota tipo potencial por prueba.

## Datos auxiliares

Verifiqué las cuentas:
- 0.09·2.3 + 0.49·0.3 + 0.42·1 = 0.774.
- El calendario {AB,A} vale 0.6 y {AB,C} vale 0.48.
- 0.42·0.6 + 0.58·0.48 = 0.5304.

Pero ojo: la "población virtual" es una transformación concreta y mala. El mejor plan estático laminar (dos individuales, o {AB,A}) vale 0.6. La brecha de adaptación relevante es 0.774/0.6 ≈ 1.29, no 0.774/0.5304. Conviene presentar la versión con 0.6 para que no parezca inflada.

## Lo más prudente con Francisco

1. **A** como garantía base, segura y sin pretensión de novedad.
2. **C** enunciada como "no hay constante uniforme en G para planes estáticos laminares; en q=1/2 la brecha es Θ(log G)". No como evidencia a favor de lo dinámico.
3. Plantear explícitamente como **pregunta abierta**: ¿el laminar dinámico escapa a la obstrucción? Proponer el cálculo exacto en la familia Cantor–Mills pequeña antes de comprometer la transformación adaptativa como hito hacia SODA.
4. El ejemplo n=4 sirve como ilustración del valor de adaptar, citando 0.6 como referencia estática.
