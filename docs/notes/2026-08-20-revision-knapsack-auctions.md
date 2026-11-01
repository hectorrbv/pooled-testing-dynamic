# Revisión C-M1 — "Knapsack Auctions" (encargo post-sesión de Francisco, 2026-08-20)

**Encargo:** mensaje de Francisco al grupo tras la sesión del 2026-08-18, con insistencia en revisarla. Contexto de sesión: el colapso valor/costo como "knapsack optimization", con las subastas como el ejemplo más cercano que conoce [18:31–19:13 del acta].

**Estado:** identificación y mapeo preliminar por soporte IA. **Fuente primaria pendiente de lectura; A valida antes de incorporar cualquier claim.** Nada de esta nota es citable en el paper propio (§25).

## Identificación

- **Referencia:** Gagan Aggarwal, Jason D. Hartline, *Knapsack Auctions*, ACM-SIAM Symposium on Discrete Algorithms (SODA), 2006.
- **Qué hace (del abstract, no verificado contra el PDF):** problema de knapsack teórico-de-juegos motivado por venta de anuncios; cada agente tiene **valuación privada** por colocar su objeto en la mochila y cada objeto un **tamaño público**; diseño de subastas *truthful*; objetivo: fracción constante del profit de un algoritmo de pricing óptimo omnisciente; aproximación de factor constante en el caso especial de capacidad ilimitada.

## Mapeo al problema propio (hipótesis de trabajo, formato §21)

| Eje | Coincidencia | Diferencia | Acción |
|---|---|---|---|
| Objeto bidimensional | (valor, tamaño) bajo capacidad ≈ nuestro par (V(T), C(T)) bajo presupuesto b — candidata F (§14.8) | su tamaño es público y determinista; nuestro C(T) es una **esperanza medida** (greedy local post-prueba, estocástica vía posterior) | al leer: ¿qué se rompe si el tamaño es aleatorio/estimado? |
| Colapso 2D→1D | razones valor/tamaño ("bang per buck") como heurística canónica de knapsack; sin respuesta única — coincide con la directriz de sesión | los exponentes 1/2 y 1.5 dictados en sesión **no se atribuyen a este paper** hasta verificar dónde aparecen | extraer qué razón usa el paper y con qué garantía; rastrear la fuente de los exponentes fraccionarios |
| Benchmark | comparar contra un óptimo omnisciente ≈ nuestra disciplina rollout/óptimo como vara (§15, §24) | su benchmark es profit de pricing; el nuestro, bienestar esperado | ¿el estilo de análisis (aproximación contra benchmark restringido) sirve para la pregunta 5 de §20? |
| Capa estratégica | — | valuaciones privadas + truthfulness **no existen** en nuestro problema (no hay agentes que declaren) | descartar explícitamente la maquinaria de incentivos al citar; el préstamo es solo la estructura knapsack |
| Selección | knapsack sobre candidatos = una de las tres reglas de la candidata F | su selección es one-shot; la nuestra secuencial-adaptativa (conteos que llegan) | ¿existe versión adaptativa/estocástica en la literatura derivada? (stochastic knapsack — a barrer) |

## Acciones

1. Conseguir el PDF y leer (C-M1 → validación de A).
2. Verificar qué exponente de la razón aparece con garantía y en qué régimen; documentar la procedencia de α ∈ {1/2, 3/2}.
3. Alimentar el diseño del barrido 23.6 con las variantes que el paper justifique.
4. Barrido corto de vecinos: *stochastic knapsack* y *adaptive stochastic knapsack* (Dean–Goemans–Vondrák) como puente entre knapsack estático y nuestro caso adaptativo — candidatos para §21, misma disciplina.
5. Integrar la fila resultante a la tabla general de literatura cuando exista `docs/notes/2026-08-XX-revision-QGT.md`.
