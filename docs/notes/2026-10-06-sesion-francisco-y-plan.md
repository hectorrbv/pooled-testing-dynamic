# Sesión del 6 de octubre: conclusiones y plan de trabajo

Plan redactado el 11 de octubre de 2026 a partir de la transcripción aportada por Héctor y contrastado con los resultados locales. La próxima reunión no queda fechada en esta transcripción. Las prioridades y el reparto que siguen son propuestas del equipo, salvo los compromisos identificados expresamente.

[Grabación de la sesión](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ) · [Plan anterior, 5–6 de octubre](2026-10-05-plan-mon-tue.md) · [Plan maestro](../plans/2026-08-01-plan-maestro-politicas-laminares-planificacion.md).

La dirección que deja la reunión es comparar tres clases: binario dinámico, aumentado laminar y aumentado general. Ya hay ejemplos donde contar aporta y donde exigir laminaridad cuesta. Ahora interesa determinar cuánto pueden crecer esas diferencias y encontrar un ejemplo donde la flexibilidad del binario compense su menor información. Los algoritmos eficientes siguen siendo la tercera parte del programa.

## 1. Qué quedó acordado y qué sigue abierto

- Francisco presentó dos comparaciones: aumentado general entre una y G veces el binario; aumentado laminar entre un cuarto y G veces el binario. La cota de un cuarto combina dos factores dos conocidos. [12:26–15:31](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=746), [30:01–33:39](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=1801).
- Propuso buscar un ejemplo donde el óptimo binario supere al aumentado laminar, intentando que la diferencia sea grande. No presentó ese ejemplo ni demostró que exista. [35:58](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=2158), [57:03](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=3423).
- Señaló q=1/2 y las matrices detectoras como ruta para separar aumentado general de aumentado laminar. La dificultad pendiente es limitar lo que puede obtener cualquier política dinámica laminar. [38:12–45:33](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=2292), [54:03–57:03](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=3243).
- Héctor presentó el mecanismo del par y los mapas del notebook 28. Al cerrar se comprometió a enviar el notebook y la redacción del ejemplo por WhatsApp. Este es el encargo personal explícito. El envío no está acreditado por los archivos revisados. [46:39–53:59](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=2799), [58:49–59:03](https://fathom.video/share/37Fd8GHvUKHue6YKjKESwCBYGuVysWyJ?timestamp=3529).

La transcripción propone búsquedas para el equipo; no permite asignar todo el trabajo nuevo exclusivamente a Francisco, como hace el resumen automático.

## 2. Correcciones necesarias al resumen automático

q es probabilidad de salud; p=1−q, de infección. Por tanto, q→0 significa muchos infectados. Laminar permite grupos disjuntos o anidados dentro de cada trayectoria; no significa que todos sean disjuntos.

Usamos V_bin para el óptimo binario dinámico con hard clearing, V_lam para el aumentado dinámico laminar con soft clearing y V_aug para el aumentado dinámico general con soft clearing. Los tres comparten población, priors, utilidades, B y G. Son utilidades esperadas totales, no scores locales. Bajo independencia inicial y 0<q_i<1:

\[
V_{bin}\le V_{aug}\le G V_{bin},\qquad
\tfrac14 V_{bin}\le V_{lam}\le V_{aug}.
\]

El aumentado general puede imitar una política binaria. La cota de un cuarto se refiere al laminar y no significa que ya exista un ejemplo que alcance esa pérdida. Para la cadena del factor cuatro, Francisco usa binario dinámico ≤ 2 × binario estático con solapamientos ≤ 4 × binario estático disjunto ≤ 4 × aumentado laminar. El [paper de Nic, introducción y §5](https://arxiv.org/html/2601.22419v2), documenta las dos comparaciones binarias; falta dejar la cadena incorporada a nuestra nota con sus convenciones.

El posible peor caso cerca de q=1/2 corresponde a aumentado general frente a aumentado laminar. Sigue siendo una hipótesis apoyada por ejemplos finitos. No es un resultado universal sobre aumentado frente a binario: la separación grande de esa otra comparación aparece en el régimen de pocos sanos.

Las matrices detectoras deben distinguir perfiles binarios mediante AZ con sumas enteras, no módulo dos. No se exige que A sea invertible sobre todos los vectores reales. La escala mencionada al final fue n/log n, no log n; antes de utilizarla hay que comprobar el teorema y el tamaño máximo de sus grupos. El enlace del paper que Francisco compartió no aparece en la transcripción.

## 3. Lo que ya tenemos para reutilizar

- [Notebook 28](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb): presentación del par, malla y separación entre clases. Conservar las ediciones de Héctor; preparar cualquier extensión después de obtener un resultado nuevo.
- Ejemplo n=6, B=2, G=2, q=0.3, u_i=1: individuales 0.6; política del par 0.774. Probar AB; con R=1 probar A e inferir al sano de AB; con R=0 o R=2 probar C. La utilidad es 2q+[q²+(1−q)²]q. Es una política laminar.
- Ejemplo n=6, B=4, G=3, q=1/2, u_i=1: individuales 2, óptimo laminar 2.5, óptimo aumentado general 3. [Árboles y auditoría independiente](../../results/sesion_2026-10-06/separacion_3_2_5/evidencia.json). Es una separación exacta finita; no demuestra una brecha asintótica.
- [Nota teórica del 4 de octubre](2026-10-04-revision-y-direccion-prueba.md): argumento de primera aparición para V_aug≤T_BG≤G T_B, obstrucción para planes estáticos laminares y propuesta de estudiar el caso dinámico. Esta reunión refuerza esa ruta; no todo lo discutido constituye un resultado nuevo.
- [Proposición de la brecha de convenciones](../../augmented/paper/proposicion_brecha_convencion.tex): contiene el caso de dos pruebas. El plan anterior distingue ese caso probado de la extensión general pendiente de revisión humana; esta reunión no acredita que esa revisión haya terminado.

## 4. Lista de tareas, en orden

### P0. Cerrar la entrega prometida

- [ ] Héctor: preparar el notebook 28 y una nota breve con setup, árbol del par y cálculo 0.774 frente a 0.6.
- [ ] Incluir el enunciado general con sus hipótesis: independencia, 0<q_i<1, u_i>0, G≥2 y 2≤B<n. Separar el caso ya probado de lo que aún necesita revisión.
- [ ] Revisar con Vladimir la demostración general antes de etiquetarla como cerrada. El intercambio propuesto recupera toda la utilidad de dos individuales y añade (q_a q_b+p_a p_b)q_c u_c>0 al aprovechar una persona adicional.
- [ ] Héctor: enviar o confirmar el envío a Francisco y registrar qué versión recibió. La transcripción contiene una promesa de envío, no evidencia de que se haya realizado.

Cierre: paquete legible y envío registrado. No requiere ampliar la malla ni rediseñar todas las figuras.

### P1. Escribir correctamente las cotas presentadas

Reparto propuesto: Vladimir redacta; Héctor revisa hipótesis y ejemplos.

- [ ] Una página con las tres clases, las dos cadenas de desigualdades y las referencias exactas de los factores dos.
- [ ] Completar el paso adaptativo de V_aug≤T_BG: antes de incluir a una persona por primera vez, su estado no influye en la decisión de incluirla. No basta contar BG personas por trayectoria para justificar una esperanza con selección adaptativa.
- [ ] Precisar en qué límite se alcanza G. Para la estrategia que cubre k grupos de tamaño G y reserva ℓ=⌈log₂G⌉ pruebas para localizar un sano, B=k+ℓ y n=kG. Su garantía es al menos 1−(1−q)^(kG). Dividida por el óptimo binario Bq, al tomar q→0 da kG/(k+ℓ), no G para k fijo. Después, k→∞ aproxima G. Es una derivación que debe quedar escrita y revisada.

Cierre: cada desigualdad tiene argumento y supuestos; el límite no omite el costo de localizar al sano. La cota elemental también da V_lam≥V_aug/G; mejorar su dependencia de G sigue abierto.

### P2. Buscar binario mejor que aumentado laminar

Reparto propuesto: Héctor prepara el experimento; Vladimir revisa los casos candidatos.

- [ ] Disponer de un Bellman binario exacto o reutilizar uno validado. La referencia debe ser el óptimo con historial completo, no el greedy binario existente. Cambiar observación y acreditación de manera explícita.
- [ ] Empezar con n=4,5, 2≤B<n, 2≤G≤n, u_i=1 y q∈{0.55,0.65,0.75,0.85,0.95}. Ampliar a n=6 solo si el costo de cómputo lo permite.
- [ ] Si no aparece separación homogénea, añadir casos heterogéneos de q_i y u_i. Registrar la cobertura y los casos que excedan los límites de cómputo.
- [ ] Para un candidato V_bin>V_lam, evaluar independientemente la política binaria sobre todos los perfiles y verificar el óptimo laminar. Extraer el menor árbol que explique la ventaja.

Por qué buscar ahí: cuando todos los q_i≤1/2, el óptimo binario son las mejores individuales, según el [Teorema 3.3(i)](https://arxiv.org/html/2601.22419v2#S3.SS2). El laminar puede reproducirlas; por tanto, en ese régimen es imposible V_bin>V_lam. Esta exclusión es una consecuencia nuestra de un resultado del paper, no una tarea enunciada en la reunión.

Cierre: un contraejemplo exacto con mecanismo, o un informe delimitado de búsqueda sin hallazgo. No encontrarlo no demuestra dominancia universal.

### P3. Convertir q=1/2 en una ruta de prueba

Reparto propuesto: Vladimir comprueba literatura y argumento; Héctor verifica ejemplos y medidas.

- [ ] Recuperar el título o enlace exacto del paper de matrices detectoras compartido por Francisco. La nota del 4 de octubre cita Cantor–Mills como antecedente, pero no sabemos si es el mismo.
- [ ] Verificar una matriz pequeña: número de filas B, máximo de unos por fila G y firmas AZ diferentes para todos los perfiles. Comparar con el óptimo dinámico laminar bajo exactamente esos B y G.
- [ ] Usar q_i=1/2 y u_i=1. Si el diseño identifica todos los perfiles, alcanza utilidad esperada n/2. Comparar ese diseño concreto con V_lam, sin llamarlo óptimo general salvo que alcance esta cota de toda la utilidad disponible.
- [ ] Intentar demostrar o refutar una afirmación precisa: ¿existe una constante K independiente de n, B y G tal que V_lam≤K B en este régimen?
- [ ] Explorar como argumento una cantidad asociada a los bloques residuales cuyo valor esperado inicial y cambio por prueba permitan acotar toda la utilidad acreditada. Es una idea de prueba, no un lema disponible.

Si se prueba V_lam≤K B y hay diseños admisibles con B=O(n/log n), entonces V_lam/V_aug=O(1/log n): se descartaría una constante universal independiente de G. Hay que permitir el crecimiento de G que requiera la construcción. Si el objetivo es G fijo, la pregunta cambia.

La dificultad es la utilidad esperada con presupuesto fijo. Una cota de peor caso para identificar a toda la población, o una cota válida solo para planes estáticos laminares, no resuelve el caso dinámico. Una separación tampoco demuestra por sí sola que sea el peor caso posible.

Cierre inicial: una construcción verificada y una formulación concreta del argumento laminar que falta. No se exige cerrar toda la teoría antes de la próxima reunión.

### P4. Volver a las heurísticas con la comparación correcta

- [ ] Después de P0 y del primer avance en P2/P3, medir π_L y π_ratio frente a V_lam en los casos relevantes.
- [ ] Separar la pérdida de la restricción, V_aug−V_lam, de la pérdida de la heurística, V_lam−V_π. Mantener parámetros fijados de antemano para evaluar generalización.

Esta línea sigue vigente, pero la reunión no pidió otra clasificación extensa de heurísticas. El experimento anterior de cambiar a una continuación laminar en la primera acción incompatible queda como diagnóstico secundario; no desaparece ni se presenta como terminado.

## 5. Tu orden de trabajo y estudio

1. Primera sesión: reconstruir el árbol del par y cerrar el paquete pendiente. Debes poder explicar en un minuto de dónde sale 0.774 y qué hipótesis hacen estricta la mejora.
2. Segunda sesión: dibujar las tres clases y reproducir los argumentos del factor cuatro y de G. Debes distinguir una cota garantizada, un ejemplo que separa y una afirmación de peor caso.
3. Tercera sesión: aprender matrices detectoras con una matriz pequeña y decidir qué experimento ejecutará P2 y cuál P3. La pregunta es por qué varias ecuaciones de conteo permiten inferencias que la estructura laminar restringe.

La lectura de §5 del paper de Nic y los comentarios prometidos siguen pendientes si no se han entregado. Ahora conviene estudiarla como una herramienta posible para una desigualdad concreta, después de aclarar si buscamos una garantía para G fijo o independiente de G. La reducción a subcubos no resuelve por sí sola el nuevo argumento.

Para la próxima discusión, llevar como máximo tres piezas: el resultado escrito del par; una búsqueda exacta binario–laminar con alcance explícito; y una matriz o argumento que haga precisa la hipótesis en q=1/2. Las preguntas útiles son: «¿Buscamos una constante independiente de G?», «¿La cota V_lam≤K B capturaría tu intuición?» y «Si el binario gana, ¿qué decisión concreta logra aprovechar que el laminar no puede?». La fecha de entrega se fija cuando se confirme la próxima reunión.

## 6. Estado de este despacho

Se revisaron la transcripción, el plan previo, la nota teórica y la evidencia guardada del notebook. Se consultó el paper de Nic para contrastar los resultados binarios. Este documento crea el plan; no ejecuta las nuevas búsquedas, no acredita envíos, no modifica el notebook y no cierra demostraciones pendientes. Los responsables propuestos no sustituyen un acuerdo con Vladimir o Francisco.
