# Héctor: tu parte del plan y sesión de estudio para Francisco

Preparado el 14 de septiembre de 2026. La reunión figura en el plan para el martes 15; esta revisión no confirma su hora. Fuente del reparto: [ajuste operativo del 11 de septiembre](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-11-plan-vie-sab-dom.md), que modifica el [plan semanal del 10](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-10-plan-semana.md).

**Actualización del 21-sep:** la sección 4 bis añade el contraejemplo de no-reentrada y el cambio de medición de costo entre los notebooks 25 y 26. El recorrido de estudio se centra ahora en esa evolución; las cuentas de acreditación de la sección 4 quedan como referencia. El contraejemplo de seis personas de la sección 5 es una etapa posterior y tiene una función distinta.

**Empieza por el panorama general:** [ruta de investigación y discusión con Francisco](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-21-mapa-ruta-investigacion-francisco.md). Reconstruye las sesiones hasta el PDF del 25 de agosto, explica sus seis direcciones y conecta Bellman, las políticas π y λ con las preguntas abiertas. Léelo antes de retomar los ejercicios de esta guía.

**Prioridad acordada el 21-sep:** [repaso centrado en contraejemplos y guion con notebooks](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-21-repaso-contraejemplos-francisco.md). El 26 esencial es la presentación principal; el nuevo 27 compara Bellman general y laminar en ocho casos. Ya hay enumeración independiente, 28 pruebas y árboles guardados. El error de cobro del complemento descrito en §7 quedó corregido y el ejemplo de seis personas se reevaluó.

## 1. Qué te toca

**Tú eres B (Héctor): conviertes la pregunta científica en experimentos comprobables y explicas qué permiten concluir.** Vladimir es A: redacta la proposición de brecha de convención y la sección de modelo. Los dos deben poder explicar el trabajo del otro; no basta con repartir código y teoría.

| Tu responsabilidad | Qué debes poder entregar o explicar |
|---|---|
| Validación y correcciones | Comparación del solver con enumeración independiente; separar victorias, empates y derrotas; etiquetar convención y clase de políticas; corregir la reserva del ancla; depurar mínimos empíricos citados. |
| Mapas homogéneos | Mapa 1: primera acción óptima y valor relativo a singletons. Mapa 2: aproximación homogénea al régimen del piloto. Mapa 3: comparar el óptimo con la construcción del ancla, o declarar la frontera computacional. |
| Verificación de la teoría de A | Rehacer el ejemplo de dos pruebas, revisar la fórmula general y objetar la simulación de posterior-zero bajo acreditación estricta. |
| Lectura del companion | Explicar Thm 8.2 y Cor 8.3: garantía 1/G, hipótesis, regla de paro y comparador correcto. Revisar la sección de modelo como lector externo. |
| Presentación propia, 10 minutos | Contraejemplo → π_L → sensibilidad a λ → evidencia del portafolio → mapas que efectivamente existan. |
| Extra | Probar una regla de λ sin rejilla. El ajuste del 11 lo deja como extra. |

**Prioridad de estudio vigente:** no-reentrada en 26 esencial §5; instancia de seis personas en §7; π_L en §8; cruces en 27 §2–4; ensayo oral. El 24 y el 25-II quedan de apoyo para submodularidad y la derivación del colapso. El estudio prepara los entregables; no equivale a haber completado todos los pendientes del plan.

## 2. Qué existe en esta copia y qué no está cerrado

Revisión local del 14-sep; no se consultaron copias privadas ni se sincronizó el remoto. Último commit local observado: `619cb6e`. El notebook 25 tenía una modificación previa del usuario y se preservó.

| Pieza | Evidencia local |
|---|---|
| Solver B-M17, Bellman por tipos, políticas de densidad y π_L | Existen. Se reprodujeron los casos pequeños indicados abajo. |
| Contraejemplo y portafolio | Hay notas y scoreboards. El scoreboard de λ conserva resultados de diferentes rejillas: hay que distinguirlos. |
| Enumeración independiente | Los nombres originales del plan no se encontraron. Actualización 21-sep: `bellman_perfiles.py` y `tests_bellman_perfiles.py` aportan la comparación independiente general/laminar y 28 pruebas; no se afirma haber ejecutado toda la matriz del plan original. |
| Mapas homogéneos 1–3 y nota de mapas | No encontrados en las rutas del plan. No presentarlos como terminados. |
| Lectura de Thm 8.2 y actas de verificación cruzada | No encontradas en las rutas de cierre. Esta guía sirve como preparación, sin firmar por nadie. |
| Corrección del ancla | `acid_test.py` todavía calcula k con la reserva estricta: k=2. |
| Gana / empata / pierde | `run_showcase` todavía agrupa los empates con las victorias mediante `>= 1 - 1e-9`. |
| Proposición y sección de modelo de A | No encontradas en las rutas previstas. El plan asigna su redacción a A y su revisión a B. |

## 3. Sesión de estudio: 100 minutos, papel y lápiz

En cada bloque: intenta responder sin mirar, compara con la explicación y vuelve a explicarlo en voz alta. Si una respuesta no sale, repite ese mecanismo antes de avanzar.

| Minutos | Trabajo | Señal de comprensión |
|---|---|---|
| 0–20 | No-reentrada, 26 esencial §5 | Reconstruyes las tres acciones y explicas por qué repetir AB es inútil aunque V lo premie. |
| 20–40 | Seis personas, 26 esencial §7 | Dibujas AEF, explicas el refinamiento de E y distingues apertura de continuación. |
| 40–55 | π_L, abandono y λ, §8 | Explicas la penalización y sus ratios frente a los dos óptimos. |
| 55–75 | Cruces, notebook 27 §2–3 | Reproduces 1.30 frente a 1.45, la ganancia inicial 0.063 y el certificado de tres tríos. |
| 75–85 | Límites y apoyo del 24 Acto 4 | Distingues fallo de un score, de una batería, de submodularidad y pérdida por laminaridad. |
| 85–100 | Ensayo y corrección | Presentas en diez minutos y usas cinco para corregir lagunas. |

Si solo tienes 45 minutos: no-reentrada (10), seis personas y π_L (15), cruces (10), ensayo (10). La sección 4 sirve para repasar esperanza condicional si hace falta; no es necesario repetir toda la comparación de convenciones.

## 4. El ejemplo que debes dominar antes de los resultados grandes

Población homogénea, n=4, q=P(sano)=0.3, p=P(infectado)=0.7, utilidad u=1, máximo G=2 personas por pool y B=2 pruebas **en cada trayectoria**. El conteo R informa infectados.

- **Strict:** cobra quien estuvo físicamente en una prueba con conteo cero.
- **Posterior-zero:** también cobra quien queda probado sano por deducción. No basta con que sea muy probable.
- **Laminar por trayectoria:** los pools de una misma historia son disjuntos o están anidados; no tienen cruces parciales. Ramas alternativas pueden elegir familias distintas.

Primero prueba AB. Esta es la cuenta completa:

| Conteo de AB | Probabilidad | Mejor continuación aquí | Valor total posterior-zero | Valor total strict |
|---|---:|---|---:|---:|
| R=0 | q²=0.09 | Cobras A y B; pruebas un virgen | 2+0.3 | 2+0.3 |
| R=1 | 2pq=0.42 | Pruebas A | 1 | 0.5 |
| R=2 | p²=0.49 | Pruebas un virgen | 0.3 | 0.3 |

En R=1, si A sale sano, cobras A. Si A sale infectado, sabes que B está sano: bajo posterior-zero cobras B; bajo strict queda sin acreditar porque ya no hay pruebas. **La deducción convierte ambas respuestas del refinamiento en cobro.**

\[
V_{pz}(\text{par primero})=0.09(2.3)+0.42(1)+0.49(0.3)=0.774=387/500.
\]

\[
V_{strict}(\text{par primero})=0.09(2.3)+0.42(0.5)+0.49(0.3)=0.564.
\]

Dos pruebas individuales valen 2q=0.600. El óptimo strict elige individuales y vale 0.600; el óptimo posterior-zero vale 0.774.

**Tres comparaciones diferentes:**

1. Cambiar la convención manteniendo «par primero» con continuación óptima en este caso: 0.774−0.564=0.210.
2. Comparar los óptimos de ambas convenciones: 0.774−0.600=0.174.
3. Medir la mejora frente a singletons: 0.774/0.600=1.29, un 29% más de utilidad esperada.

### Precisión necesaria en la fórmula general del plan

Para la política concreta «abrir par; si R=1 refinarlo; si R=0 o 2 probar un virgen», con n≥3:

\[
W_{pz}=q^2(2+q)+2pq+p^2q,
\quad W_{strict}=q^2(2+q)+pq+p^2q.
\]

Así, W_pz−W_strict=pq y W_pz−2q=q(q²+p²)>0 para 0<q<1. Esto prueba que **existe** una política posterior-zero mejor que dos singletons en todo ese intervalo.

Pero estos valores no son siempre los mejores con primera acción fijada. Para q>1/2 puede convenir abrir otro pool en vez de refinar. Comprobación exacta: n=4, G=2, B=2, q=0.9 da **3.24 en ambas convenciones** con par primero y continuación óptima; su diferencia es cero, no pq=0.09. Para q≤1/2 las continuaciones de la tabla sí son óptimas con par primero en este modelo homogéneo.

**Llevar a A:** «¿El enunciado compara una política explícita o los valores óptimos con par primero? Fijemos esa distinción y el rango de q antes de escribir la igualdad». No hay que descartar el ejemplo de q=0.3: está comprobado.

## 4 bis. El contraejemplo que conecta los notebooks 25 y 26

**Fuente:** [26 completo, §2 y §4–§6](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/26_costo_local_y_no_reentrada.ipynb), resumido en [26 esencial, «El contraejemplo completo y la reparación»](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/26_esencial.ipynb). Estas cuentas pertenecen a la **variante estricta histórica** de la Parte I. Se estudian para entender el cambio de algoritmo; el modelo vigente del proyecto sigue siendo posterior-zero.

### Antes del contraejemplo: qué cambia en el costo

El [25, Parte II, §8](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/25_resultados_y_peticiones_parte_ii.ipynb) mide el costo haciendo correr greedy desde cero. Para dos personas frescas con q=0.3, greedy hace dos individuales: costo 2.

El 26 primero prueba el par y mide las subpruebas que quedan **condicionadas al conteo**. Si R=0 o R=2, no hace falta continuar. Si R=1, bajo la regla estricta se necesitan 1 o 2 subpruebas, con igual probabilidad. Por tanto:

\[
C_{\mathrm{sub}}(\text{par fresco})
=0.09(0)+0.42(1.5)+0.49(0)=0.63,
\qquad C_{\mathrm{total}}=1+0.63=1.63.
\]

El costo ya incorpora lo que revela la prueba inicial. **1.63 es un promedio, no un máximo por trayectoria.** El filtro heurístico que compara costo esperado con presupuesto no sustituye el control del presupuesto duro.

### El estado y las tres acciones

Ahora estamos en un estado distinto: **AB ya fue probado y dio R=1**. Uno está sano y otro infectado, con probabilidad 1/2 para cada asignación. C y D son personas frescas con q=0.3; todas las utilidades son 1.

El score de «presupuesto mágico» puntúa la masa sana esperada **dentro del conjunto de la acción**:

\[
V(T)=\sum_{i\in T}u_iP(i\text{ sano}\mid h).
\]

Aquí V es un score de potencial, distinto del valor esperado de ejecutar una política completa. Para el cociente usaremos siempre C_total, incluyendo la prueba candidata:

| Acción candidata | V: masa sana esperada en T | C_total: costo esperado con continuación local | V/C_total | Cobro esperado en la próxima prueba, strict |
|---|---:|---:|---:|---:|
| Probar A: reentrar y refinar AB | 0.5 | 1 | 0.5 | 0.5 |
| Probar el par fresco CD | 0.6 | 1.63 | ≈0.3681 | 0.18 |
| Volver a probar AB entero | 1 | 2.5 | 0.4 | 0 |

**Por qué V falla:** sabe que hay una unidad sana en AB y le asigna score 1. Pero repetir AB devuelve otra vez el mismo conteo 1 con certeza: no identifica a la persona sana y no cobra nada. Si se vuelve a elegir por el mismo score y se permite ese retest, el estado informativo no cambia y la regla puede repetirlo hasta agotar las pruebas. Se llama «no-reentrada» porque la política no entra a refinar el grupo: vuelve a probarlo entero.

**Por qué aparece 2.5 en el retest:** gastas 1 en repetir AB; luego sigues teniendo el mismo par de conteo 1, que cuesta 1.5 subpruebas esperadas resolver y acreditar bajo strict. En cambio, probar A cuesta 1 si el seguimiento se restringe al conjunto T={A}; el costo de acreditar al vecino B no se carga a esa acción con este alcance local.

### Qué corrige el costo y qué queda abierto

Al dividir por el costo total, probar A obtiene 0.5 y supera al retest (0.4) y al par fresco (≈0.3681). El costo corrige **el orden de este menú**. Esto no prueba que el cociente sea óptimo en general.

La familia V/C_total^α permite graduar el descuento. En la comparación de A contra retest AB:

\[
\frac{0.5}{1^\alpha}>\frac{1}{2.5^\alpha}
\quad\Longleftrightarrow\quad
\alpha>\frac{\ln2}{\ln2.5}\approx0.7565.
\]

Esta cuenta compara scores **antes del filtro de costo**; para observar el cambio con todo este menú factible puede tomarse b≥3. Si el filtro elimina el retest, el umbral deja de describir esa decisión.

También importa qué trabajo incluyes en C_total. Si al probar A cargas el cierre del vecino B, su costo pasa a 1.5 y su score a 0.5/1.5≈0.3333; el retest conserva 0.4 y vuelve a ganar. Esta es la pregunta del 26 §5: qué costos pertenecen a la acción y cuáles al resto del estado. En el barrido de 72 instancias, ningún α domina en todos los regímenes.

### Por qué este ejemplo importa para tu presentación

1. **Explica una decisión mala con tres acciones y cuentas visibles.** Distingue utilidad disponible de utilidad que una prueba permite cobrar.
2. **Da sentido al cambio del 25 al 26.** Medir después del conteo permite valorar el trabajo restante, en lugar de ignorar la información de la primera prueba.
3. **Muestra tanto la mejora como sus límites.** El cociente arregla este ranking, pero depende de α y del alcance del costo; hace falta el barrido.
4. **Ordena los avances posteriores.** C3 busca una fórmula mejor; el ejemplo heterogéneo de seis personas examina la batería en otro régimen; π_L explora una puntuación por utilidad menos costo con abandono. Son etapas distintas.

No es necesario reabrir el debate de convenciones para explicar su relevancia. Los números de la tabla quedan etiquetados como históricos: bajo posterior-zero, por ejemplo, refinar A acredita 1 con certeza. En el menú canónico del solver actual, el retest íntegro de un átomo informativo tampoco es una acción candidata. Por eso este ejemplo diagnostica el score y el menú históricos; no es un fallo demostrado de π_L.

**Frase para Francisco:** «El score de masa sana puede preferir repetir una prueba que no aporta información, porque confunde tener utilidad dentro de un grupo con poder cobrarla. El costo local posterior al conteo corrige ese ejemplo, pero su alcance y la intensidad del descuento cambian el ranking; por eso después comparamos políticas completas».

### Ejercicio de estudio, 10–15 minutos

1. Dibuja AB con R=1 y C,D frescos. Reconstruye V de las tres acciones sin mirar la tabla.
2. Explica qué cambia en el historial al repetir AB y al probar A.
3. Calcula V/C_total y elige la mejor acción; después cambia solo el costo de A de 1 a 1.5.
4. Explica por qué arreglar ese menú no garantiza buen rendimiento en toda población.

<details>
<summary>Control de respuestas</summary>

Los V son 0.5, 0.6 y 1. Repetir AB no aporta información nueva; probar A distingue las dos configuraciones de AB. Con costos 1, 1.63 y 2.5 gana A; al cargarle costo 1.5, gana el retest. Son decisiones en un estado particular y con un alcance del costo concreto; el barrido comprueba que cambiar α puede ayudar en un régimen y perjudicar en otro.

</details>

## 5. Tu historia experimental

**Dos contraejemplos, dos funciones:** el de no-reentrada anterior cuestiona un score de masa sana en un estado de cuatro personas; el siguiente evalúa el rendimiento de una batería de políticas completas en una instancia heterogénea de seis personas. El segundo ya aparece en el 26 esencial §7 y es el que motiva la comparación posterior con π_L.

### El contraejemplo de seis personas

[Nota del contraejemplo](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-01-contraejemplo-universal-greedy.md). Instancia: p=(0.9, 0.825, 0.875, 0.8, 0.95, 0.85), u=(2,1,1,1,4,2), B=3, G=4; posterior-zero, laminar por trayectoria.

El solver da OPT_lam=1.0645140625 y abre {0,4,5}. Ese trío sale completamente sano con probabilidad 0.1·0.05·0.15=0.00075. Sin embargo, sus conteos permiten elegir continuaciones valiosas. Tras R=1, probar a 4 acredita utilidad 4 en cualquiera de las dos respuestas: o 4 está sano y vale 4, o está infectado y 0 y 5 quedan sanos por deducción, con utilidad 2+2. Ese 4 es el cobro de ese refinamiento, no todo el valor esperado del juego.

El [scoreboard histórico](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/results/contraejemplo_scoreboard.tsv) registra ratios 0.657577 para las tres implementaciones de densidad/inmediato y 0.656676 para C3. Son ratios contra OPT_lam. El título histórico «universal» se refiere a esa batería; no a todo algoritmo greedy posible. El 21-sep se corrigió el error de §7 y se reevaluó esta instancia: esos resultados se conservan. El óptimo general nuevo es 1.1104640625, frente al laminar 1.0645140625.

### Qué cambia π_L

\[
I_\lambda(C)=\sup_{\pi\text{ local}}\mathbb E[\text{utilidad acreditada}-\lambda\,\text{pruebas usadas}],
\]

incluyendo abandonar sin costo. Una exploración fallida puede consumir una sola prueba; reservar y penalizar siempre todo un horizonte puede hacerla parecer demasiado cara. El índice permite decidir si seguir o abandonar según los resultados.

La implementación usa horizonte rodante 3: ejecuta una acción y vuelve a planear. Añade no-parálisis: si ningún proyecto tiene índice positivo, busca el mejor cobro inmediato. Esos parámetros forman parte de la política; no es el Bellman global completo.

Reproducido el 14-sep con λ=0.001: W(π_L)=1.026265 y ratio=**0.964069** en el contraejemplo. Es una mejora de aproximadamente 30.65 puntos porcentuales de ratio respecto al registro de 0.657577. No significa optimalidad ni prueba de eficiencia a gran escala.

### Qué puedes decir del 0.9307

La [nota del portafolio](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-02-constante-empirica-portafolio.md) reporta unas 7,000 instancias exploradas. El [scoreboard](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/results/constante_scoreboard.tsv) conserva 12 filas, incluidas repeticiones y distintas configuraciones; no contiene esas 7,000 mediciones.

Una misma candidata pasa de 0.794040 con una rejilla gruesa a 0.930703 con una rejilla fina, λ≈1.238469. Las filas anteriores incluyen también 0.796554, 0.844360 y 0.870990. No se deben agregar como si fueran una única política con una única calibración.

**Frase defendible:** «La nota reporta un mínimo empírico cercano a 0.93 tras ajustar la rejilla. El CSV demuestra la mejora de la candidata a 0.930703; para certificar el mínimo de toda la batería debemos reevaluar sus candidatas bajo la misma especificación». No hay una garantía universal del 93%.

El portafolio se elige **antes** de observar resultados: comparas valores esperados bajo el prior y despliegas una política. Ejecutar todas y quedarte con el mejor resultado realizado cambia el presupuesto y el problema. Evaluar esos valores también tiene un costo computacional.

## 6. Mapas y garantía: qué significan

Bellman por tipos guarda «cuántos vírgenes, qué tamaños y conteos tienen los átomos, cuántas pruebas quedan». Con personas del mismo tipo no necesita guardar identidades. Eso comprime el estado; no vuelve independientes a las personas dentro de un átomo de conteo conocido.

Verificado aquí: q=0.3, B=3, G=3 produce valor 1.24938 y 77 estados para n=12, 120 y 500. En cada rama se pueden exponer como máximo BG=9 personas, y las tres poblaciones ya tienen suficientes vírgenes. No es una afirmación de costo independiente de n en toda población heterogénea; B, G y el número de tipos siguen importando.

- **Mapa 1:** acción óptima inicial y OPT_lam/(Bqu), especificando n suficiente. Guardar empates entre acciones: un único argmax puede ocultarlos.
- **Mapa 2:** q=0.90 o 0.95, G=5, n=130. Es una aproximación homogénea al régimen del piloto, no el piloto heterogéneo completo.
- **Mapa 3:** para q=0.05, G=16, B=7, la construcción posterior-zero permite k=3 y da la cota inferior 1−0.95^48≈0.9147. La reserva estricta del plan usa k=2 y da 1−0.95^32≈0.8063. Singletons valen 0.35. Estas son cotas de construcciones; no son óptimos certificados. Si OPT supera la cota, eso solo no prueba que supere el valor completo de la política cover-then-bisect: hay que evaluar esa política, que podría cobrar más de un sano.

### Lectura de Thm 8.2 y Cor 8.3

Fuente: companion local, páginas 14–15. Presentarlo como enunciado del companion en revisión, sin adjudicarlo como resultado propio ni dar por validado todo el documento.

El teorema afirma que el greedy que maximiza correctamente el **cobro inmediato total** satisface:

\[
\mathbb E W(\pi_M)\ge\frac1G\,OPT_{aug}^{dinámico}.
\]

El comparador permite políticas aumentadas sin restricción laminar. La garantía es más fuerte que compararse solo con OPT_lam, pero empeora al crecer G. Para G=4 asegura 0.25, compatible con un ratio observado de 0.6576. No asegura 0.93.

La idea de la prueba: con B pruebas de tamaño G se exponen como máximo BG personas. Bajo priors independientes, el óptimo queda acotado por la suma A_BG de los mayores q_i u_i. Hasta que una persona se prueba por primera vez, su singleton sigue siendo factible y mantiene valor q_i u_i; el greedy debe elegir algo que valga al menos eso. Distribuyendo esas personas entre B pasos, como máximo G por paso, se obtiene la cota 1/G. El argumento exige elegir el máximo correcto y no parar voluntariamente mientras hay un singleton virgen de valor positivo.

El corolario ofrece otra vía: probar los mejores min(B,n) singletons cobra A_B≥A_BG/G. Elegir ex ante entre esa política y una candidata conserva el piso 1/G con evaluación exacta. Esto no prueba que seleccionar el mejor pool heterogéneo sea barato.

En el CSV histórico `ratio_pi_M` mide W_piM/OPT_lam. Para una certificación frente al irrestricto se puede reportar W/A_BG, una cota inferior del ratio, etiquetada como tal; no llamar a OPT_lam «óptimo irrestricto».

## 7. Hallazgo de revisión que debes conocer

En [densidad_companion.py](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/densidad_companion.py), `_score_inmediato` calculaba mal cuándo el complemento queda sano. La condición correcta es **s=r**, donde s es el conteo del subconjunto probado y r el conteo del átomo: solo entonces r−s=0. El código anterior usaba el evento s=|S|. **Corregido el 21-sep**, con regresión y cotejo por enumeración heterogénea.

Caso mínimo reproducido: átomo de tres personas homogéneas, dos infectados, u=1; se prueba una persona. Con probabilidad 1/3 sale sana y se cobra 1. Con probabilidad 2/3 sale infectada y el complemento todavía contiene un infectado: se cobra 0. Cobro esperado correcto: **1/3**. El score antiguo devolvía **5/3**; el score corregido y Bellman a un paso devuelven 1/3.

**Alcance de la corrección:** la instancia de seis personas se reevaluó y mantiene los valores citados. Las búsquedas históricas completas no se repitieron. π_L usa su propio cálculo de recompensas. La nueva validación está en `tests_bellman_perfiles.py`.

## 8. Preguntas para discutir

**Para Vladimir, al verificar su proposición:**

1. ¿La igualdad pq compara políticas explícitas o continuaciones óptimas, y para qué q?
2. ¿D(h) incluye solo sanos deducidos todavía no acreditados físicamente, evitando doble conteo?
3. ¿Las pruebas adicionales conservan laminaridad? Agrupar sanos deducidos de átomos distintos puede cruzar pools antiguos; la construcción debe justificar su legalidad o probar por piezas compatibles.
4. ¿El sobrecosto se suma a lo largo de una trayectoria y luego se maximiza entre trayectorias? No se pueden mezclar personas de ramas alternativas ni sustituir el presupuesto duro por su esperanza.
5. En el ancla, ¿estamos comparando cotas, valores completos de políticas o valores óptimos?

**Para Francisco, preguntas propuestas para tu parte:**

- «El ejemplo pequeño muestra que deducir sanos cambia qué conviene probar. ¿Organizamos el primer resultado alrededor de esa brecha de convención?»
- «π_L mejora el contraejemplo, pero depende del precio de las pruebas. ¿Qué regla de λ aceptaríamos como especificación del algoritmo: una estimación computable o una relajación dual?»
- «¿El objetivo inmediato es aproximar OPT_lam o también comparar laminar contra irrestricto? ¿Qué parámetros puede contener el factor?»
- «Para la proposición de simulación, ¿cómo preservamos laminaridad al acreditar por pruebas físicas los sanos deducidos?»
- «Corregimos el cobro del complemento y reevaluamos la instancia de seis personas. ¿Qué alcance de verificación adicional necesita la comparación con las políticas del companion?».

El plan también reserva preguntas de equipo sobre el reparto con el otro paper y el estatuto del companion; A puede conducirlas.

## 9. Guion de tus diez minutos

**El guion vigente está en [repaso de contraejemplos](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-21-repaso-contraejemplos-francisco.md).** Lo que sigue conserva el guion previo del 14-sep como referencia; sus prioridades fueron sustituidas por no-reentrada, seis personas y precio de laminaridad.

**0:00–1:00. Pregunta y modelo.** «Me concentré en evaluar reglas de decisión bajo conteos aumentados, crédito por deducción y presupuesto duro. El benchmark computable es el óptimo laminar por trayectoria».

**1:00–3:00. Mecanismo.** Dibuja AB con R=1. «Probar A paga en ambas respuestas bajo posterior-zero. En el juego de dos pruebas obtenemos 0.774 frente a 0.600 de singletons. Manteniendo par primero, strict vale 0.564».

**3:00–5:00. Contraejemplo y π_L.** «La batería implementada registró un caso de seis personas con mejor ratio 0.6576. El óptimo abre un trío casi nunca limpio porque el conteo permite deducciones valiosas. π_L, con abandono y λ=0.001, alcanza 0.9641 en ese caso, que reproduje».

**5:00–7:00. Alcance.** «La evidencia histórica de 0.93 es empírica y depende de la rejilla. El CSV conserva configuraciones distintas. El error del score inmediato ya se corrigió; la instancia de seis personas mantiene los resultados, pero no repetimos toda la búsqueda histórica».

**7:00–8:30. Escala y pendientes.** «La compresión homogénea por tipos hace tratables poblaciones grandes cuando el horizonte es pequeño. Los mapas previstos aún no están en esta copia; la cuenta del ancla es una cota realizable, no un óptimo medido».

**8:30–10:00. Decisión solicitada.** «Propongo cerrar convención y validación, precisar la proposición y fijar cómo elegimos λ. ¿Qué comparador y qué forma de garantía priorizamos?»

## 10. Autoevaluación sin mirar

1. Si AB tiene un infectado y A sale infectado, ¿por qué B cobra en un modelo y no en el otro?
2. ¿Por qué 0.210 y 0.174 no son la misma brecha?
3. ¿El trío óptimo necesita salir limpio para que probarlo haya valido la pena?
4. ¿Abandonar localmente una exploración permite exceder B en alguna rama?
5. ¿Qué cambia al afinar λ y por qué afecta al significado del portafolio?
6. ¿0.93 observado implica 0.93 para toda instancia? ¿Refuta 0.6576 una garantía 1/G con G=4?
7. ¿Por qué n=120 y n=500 pueden dar el mismo valor y número de estados?
8. Si OPT_lam supera 0.9147, ¿ya probaste que cover-then-bisect es subóptimo?

<details>
<summary>Respuestas breves, para consultar después del intento</summary>

1. El conteo restante es cero; posterior-zero acredita por certeza, strict exige prueba física de cero.
2. 0.210 compara par primero entre convenciones; 0.174 compara óptimos o posterior-zero contra singletons en este caso.
3. No: el conteo puede habilitar refinamientos y deducciones útiles.
4. No: abandonar libera pruebas no gastadas; B sigue siendo un máximo por trayectoria.
5. Cambia la política seleccionada; una garantía debe fijar la regla de calibración y su costo.
6. No y no: un mínimo observado no es un teorema; 1/G=0.25.
7. Son homogéneas y ambas tienen suficientes vírgenes para el máximo BG de exposición.
8. No: 0.9147 es una cota inferior de la construcción; falta su valor completo.

</details>

## Reproducción y material para abrir

Las comprobaciones de esta preparación están en [el script de estudio](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-14-verificar-estudio.py) y [su salida](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-14-verificacion-estudio.json). Se ejecutan con:

```bash
python3 /Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/docs/notes/2026-09-14-verificar-estudio.py
```

Son verificaciones puntuales: cuentas racionales del ejemplo, comparación con Bellman, reproducción de π_L, compresión homogénea y caso del score. No sustituyen un enumerador independiente de todo el espacio de políticas ni repiten la búsqueda de 7,000 instancias.

Para el ensayo, abre la sección 7 del [notebook 26](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/26_esencial.ipynb), que contiene el árbol del contraejemplo. Las secciones anteriores incluyen material histórico: comprobar siempre la convención antes de reutilizar sus frases.
