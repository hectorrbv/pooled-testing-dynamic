# Respuestas para discutir en el notebook 29

Se revisaron las siete preguntas «Para discutir» de la presentación y de sus consultas conservadas, en dos rondas con Claude Opus (`claude-opus-5-5`). Se añadieron respuestas breves y se sincronizó el generador. Se conserva la eliminación de la línea de fecha que Héctor hizo en la introducción. Las figuras no cambian.

Las respuestas distinguen lo establecido de las propuestas:

- El intercambio del par gana estrictamente bajo las hipótesis suficientes indicadas. No exige q<1/2.
- La familia binaria se aproxima a 10/9; no prueba un máximo. La cota combinada 4 tampoco es un ejemplo alcanzado.
- Las matrices detectoras por bloques son una ruta propuesta: todavía hay que justificar diseños compatibles con G y la cota uniforme de utilidad laminar. No se presenta la separación de orden log G como demostrada.
- El testigo de siete personas favorece comparar jerarquías completas antes de complicar rollout. No justifica unir sus grupos en una biblioteca que podría dejar de ser laminar ni da una prioridad universal.

## Corrección del control homogéneo anterior

La pregunta anterior presuponía que el fallo estaba en una continuación. Las trazas muestran que ambas reglas abren ABCD, mientras un óptimo laminar abre AB. Se cambió a «¿qué decisión explica la pérdida?».

Para n=5,B=3,G=4,q=.95,u=1, el óptimo laminar vale 4.6462125. Forzar ABCD y continuar óptimamente da 4.56914375. La apertura pierde 0.07706875; las continuaciones añaden solo 0.000095 para π_L y 0.00045125 para π_ratio. No se atribuye la elección de apertura a un score, horizonte o desempate sin examinar ese paso.

Bellman condicionado y enumeración directa de las dos pruebas restantes coinciden. El cálculo se reproduce con `python -m augmented.diagnostico_control_29`; [resultado completo](../../results/sesion_2026-10-14/diagnostico_control_29.json). También se corrigió el redondeo: las pérdidas totales son aproximadamente 0.07716 y 0.07752, respectivamente.

La revisión de Opus sirvió para discutir hipótesis y redacción; la comprobación numérica procede del código y las trazas. Durante la segunda ronda se corrigió además una afirmación demasiado fuerte del revisor: no es necesario G≥n para obtener O(n/log n) pruebas mediante bloques; la condición asintótica relevante en ese argumento es log G=Θ(log n). El notebook mantiene la formulación condicional O(G/log G) por bloque, sin reclamar una construcción ya verificada.
