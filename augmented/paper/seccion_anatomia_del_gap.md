# Sección nueva: la anatomía del gap

Propuesta de estructura para una sección nueva de
`unified_dynamic_augmented_group_counting.tex`, que hoy va Introduction → Model
and Comparison Classes → **A Separation for Dynamic Count-Valued Testing** →
Efficient Exact Inference → Empirical Evidence → Discussion → Conclusion.

**Dónde entra.** Como sección 4, inmediatamente después de la separación y antes
de la inferencia. La sección 3 establece que lo dinámico gana; ésta responde la
pregunta que un referee hace enseguida: *gana por qué, y cuánto de esa ganancia
sobrevive a las restricciones que hacen el problema tratable.*

**Título propuesto.** *What Structure Costs: Anatomy of the Dynamic Gap.*

**Tesis de la sección, en una frase.** La ventaja de lo dinámico se descompone en
dos ingredientes ortogonales —contar y adaptarse— cuyo valor por separado es
comparable y cuya combinación supera la suma; y la restricción laminar, que es la
que vuelve tratable todo lo demás, cuesta un porcentaje medible que crece
justamente con la calidad de la política.

---

## 4.1 Descomposición exacta del gap

Escalera de cinco peldaños en n = 10, q = 0.2, por enumeración exacta del espacio
de creencias. Dos ejes ortogonales —canal de observación (binario contra conteo) y
diseño (estático contra adaptativo)— más la restricción estructural.

| B | est. binario | din. binario | est. conteos | din. laminar | din. irrestricto |
|---|---|---|---|---|---|
| 1 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| 2 | 0.400 | 0.488 | 0.408 | 0.536 | 0.536 |
| 3 | 0.600 | 0.790 | 0.800 | 0.928 | 1.000 |

Tres afirmaciones, todas [VERIFICADO n≤10]:

1. Contar sin adaptarse vale aproximadamente lo mismo que adaptarse sin contar
   (0.800 contra 0.790). Ninguno de los dos ingredientes domina al otro.
2. Juntos valen más que la suma de sus partes: 0.600 → 1.000 es +67%, contra +33%
   y +32% por separado.
3. Con B = 1 las cinco columnas coinciden, lo que fija el caso base y descarta que
   la escalera sea un artefacto de normalización.

Artefacto: `results/separacion_n10_q02.csv` con procedencia; generador
`augmented/experiments_separacion_n10.py`; anclas de regresión en
`augmented/tests_separacion_n10.py`.

## 4.2 La regla de certificación decide cuál es el baseline

Ésta es la subsección con más contenido nuevo y la que hay que escribir con más
cuidado, porque cambia el enunciado de la separación.

Dos reglas posibles para cuándo un agente cuenta como certificado sano:

- **(a) Acreditación por pool limpio** — cuenta si pertenece a algún pool cuya
  observación fue "todos sanos". Es el strict hard clearing del modelo (§5.7): una
  deducción informa pero no paga.
- **(b) Acreditación por inferencia** — cuenta si el sistema de restricciones lo
  determina sano, se haya observado un cero que lo cubra o no.

Bajo (a), las pruebas individuales **sí** son el óptimo estático para q < 1/2. La
prueba es una cota de unión de una línea: un pool de tamaño g aporta g·u·q^g, y
g·q^g ≤ q para todo g ≥ 1 cuando q < 1/2, así que ningún diseño de B pruebas
supera B·u·q.

Bajo (b) **no lo son**. Tres pruebas de conteo sobre cuatro agentes los
identifican exactamente: con incidencias 011, 101, 110 y 111, los conteos son
a+b+d, a+c+d y b+c+d, cuya suma es 2(a+b+c)+3d; la paridad determina d y las tres
ecuaciones despejan el resto. Eso da 4q = 0.800 contra 3q = 0.600. No es un
accidente de B = 3: es el problema clásico de pesar monedas, donde B pesadas
identifican del orden de B·log B objetos, así que bajo (b) el óptimo estático
crece superlinealmente en el presupuesto.

**Qué escribir.** El paper adopta (a) como convención normativa. La subsección
debe (i) declararlo, (ii) dar la cota de unión que hace de (a) un teorema, y (iii)
exhibir el diseño de cuatro agentes como la razón por la que la elección no es
inocua. Un argumento de separación medido contra B·u·q es correcto bajo (a) y
compara contra el rival equivocado bajo (b).

## 4.3 Qué cuesta la laminaridad

- **El costo, exacto.** En n = 10, q = 0.2, B = 3: laminar 0.928 contra
  irrestricto 1.000, una pérdida de 7.2%.
- **El mecanismo.** La política óptima cruza explícitamente. Abre un grupo de 3;
  si el conteo sale 1, su segunda prueba toma dos miembros de ese grupo junto con
  un agente fresco. Paga porque el conteo 1 deja un residuo de información —hay
  exactamente un sano, no se sabe cuál— y el pool cruzado lo cobra mientras tantea
  territorio nuevo: si sale conteo 0, el tercer miembro queda certificado y además
  se aprendió que el fresco está infectado.
- **Por qué aun así es la clase de trabajo correcta.** Con pruebas de conteo, todo
  grupo laminar termina con su conteo conocido exactamente, así que agregar un
  grupo entero a un pool nuevo suma una constante conocida y no informa. Las
  creencias se factorizan en urnas hipergeométricas independientes y el estado se
  reduce a un multiconjunto de pares (tamaño, conteo). El contraste operativo es
  medible: el solver laminar llega a B = 8 en segundos, el irrestricto no termina
  B = 4.

La honestidad exigida: 7.2% es una instancia. La búsqueda adversaria previa
encontró razón 0.9069 en otro punto. Evidencia finita, jamás cota.

## 4.4 Evidencia de comportamiento: dónde salen las políticas de la clase laminar

Definición formal de cruce: t cruza T si y sólo si t ∩ T ∉ {∅, t, T}. Quedan
fuera disjuntas, descendientes, ancestros y repeticiones.

Cada decisión de cada política, en cada estado alcanzable, clasificada y ponderada
por P^π(H). Barrido de 54 instancias, 872 decisiones, n ∈ {4,5,6}, B ∈ {1,2,3},
G ∈ {2,3}, prevalencias 0.05 / 0.45 / 0.90.

**Resultado principal.** La masa de decisiones cruzadas crece monótonamente con la
calidad de la política: S₀ 0.26%, rollout 2.15%, óptimo 3.67%. Cuanto mejor es la
política, más necesita cruzar. La restricción laminar muerde precisamente sobre
las políticas buenas, no sobre las malas.

**Resultado secundario.** El óptimo no maximiza Q^g: su regret local medio contra
el Q del rollout es 0.0026, mientras el rollout lo maximiza por construcción y da
0. Es esperado —Q^g evalúa con continuación golosa, no óptima— y cuantifica
exactamente cuánto le falta al rollout.

Artefactos: `results/falsificador_decisiones.csv` y su resumen por instancia.

## 4.5 Planear sin resolver: el oráculo de rollout

- Proposición B (mejora de política) ya está demostrada; aquí se verifica que el
  código sea esa política y no una variante parecida: la hipótesis de que la
  acción golosa pertenece al conjunto de candidatas se comprueba estado por
  estado, y la dominancia V^r ≥ V^g se verifica en todo estado alcanzable.
- Validación cruzada: dos evaluadores que sólo comparten el modelo —DP hacia atrás
  sobre creencias contra enumeración hacia adelante de los 2^n perfiles latentes,
  sin probabilidades dentro del bucle— coinciden a 2.2 × 10⁻¹⁶.
- Ancla: n = 5, q = 0.3, B = 3, G = 2 da goloso 0.900 y rollout 1.011, +12.3%. Y
  0.900 = 3q coincide dígito por dígito con el óptimo estático, porque el goloso
  homogéneo no reacciona a lo que observa y por tanto es un diseño fijo disfrazado.

## 4.6 Cotas certificadas donde el cálculo exacto muere

Donde la enumeración exacta ya no llega, se certifica. La familia de cotas
`_first_unroll_bound` se estrechó por una sucesión de lemas estructurales
aceptados —desenrollado con recursión a profundidad 6, cobertura parcial, raíz
compartida entre capas, tamaño de raíz compartido entre valores de limpios,
tightening por revelación, topes por partición— llevando el gap de certificación
insignia de 1.847 a 1.420.

El punto metodológico vale tanto como el número: de las palancas probadas, la
mayoría resultó **plana** y quedó registrada con el mismo rigor que las aceptadas,
varias con refutación explícita de por qué la cota no se mueve. Un certificado en
el que sólo se reportan los éxitos no es un certificado.

**Pendiente antes de escribir esta subsección:** confirmar la semántica exacta de
las columnas `gap_anchor_max`, `sep_lo_flagship` y `sep_hi_flagship` de
`results/autoresearch_laminar_cert.tsv`. Los números están respaldados, pero la
frase que los interpreta no se puede escribir sin fijar qué cociente es cada uno.

## 4.7 Limitaciones de la sección

Una subsección corta y explícita, no una nota al pie:

- La anatomía exacta vive en n ≤ 10, B ≤ 3. Es evidencia, no cota.
- El crecimiento B·log B del óptimo estático bajo la regla (b) se enuncia como
  consecuencia del problema de pesar monedas; caracterizarlo no es objetivo aquí.
- El 7.2% laminar es una instancia; la evidencia adversaria da 0.9069 en otro
  punto. Ninguno de los dos es una cota.
- Las cotas del certificador son superiores y no ajustadas; el gap 1.420 es
  precisamente la medida de cuánto falta, y se reporta como tal.

---

## Figuras

1. **La escalera** (4.1). Barras horizontales de los cinco peldaños en B = 3, con
   los dos ejes anotados. Es la figura que resume la sección entera.
2. **El diseño de cuatro agentes** (4.2). Matriz de incidencia 3 × 4 con los
   conteos al margen. Hace visible de un golpe por qué la regla importa.
3. **Masa de cruce por política** (4.4). Tres barras, S₀ / rollout / óptimo. La
   monotonía es el mensaje.
4. **Trayectoria del gap certificado** (4.6). Escalera descendente 1.847 → 1.420
   por keep aceptado, con las palancas planas marcadas.

## Orden de escritura sugerido

4.1 y 4.3 primero: son los que ya tienen artefacto, test de regresión y número
cerrado. Luego 4.5, que sólo hay que redactar porque la verificación ya corrió.
Después 4.2, que es la que necesita cuidado de enunciado. 4.4 y 4.6 al final: la
primera porque conviene ampliar el barrido antes de fijar los porcentajes, la
segunda porque depende de resolver la semántica de las columnas.
