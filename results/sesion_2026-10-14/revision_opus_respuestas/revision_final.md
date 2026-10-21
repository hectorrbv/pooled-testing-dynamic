No veo fallos matemáticos. Confirmo las cifras y tus dos correcciones.

Matiz 1, bloques de tamaño G. Tienes razón y mi frase era demasiado fuerte. Con bloques de G personas, el costo es (n/G)·O(G/log G) = O(n/log G). Si log G = Θ(log n), por ejemplo G = n^c, el orden es O(n/log n) con G < n. Tu formulación condicionada es correcta.

Matiz 2, familia de cuatro. Acepto el mecanismo certificado. BD y AD se cruzan en D sin que uno contenga al otro, así que no son laminares. Un límite de alcance: el mecanismo explica esta familia, pero no dice cuánto puede crecer la brecha.

Descomposición del control. Todas las cifras son correctas:

- 371697/80000 − 731063/160000 = 12331/160000 = 0.07706875.
- Para π_L: 731063/160000 − 3655239/800000 = 76/800000 = 19/200000 = 0.000095.
- Para π_ratio: 731063/160000 − 1827477/400000 = 361/800000 = 0.00045125.
- La probabilidad de R=3 es 4p³q = 19/40000. La de R=2 es 6p²q² = 1083/80000.
- En R=2: (1083/80000)·(33/20 − 97/60) = (1083/80000)·(1/30) = 361/800000.
- En R=3: (19/40000)·(1/5) = 19/200000.
- Comprobé R=3 a mano. Probar A da 1/4·1.95 + 3/4·0.95 = 6/5. Probar E primero también da 6/5. AB da 1.

Cada pérdida de continuación coincide exactamente con la de una sola rama. Por eso, las demás desviaciones, si existen, no cuestan nada. Puedes decirlo así en el notebook.

Objeciones de alcance:

1. Mi sugerencia: no escribas por qué las heurísticas abren ABCD. Por ejemplo, no lo atribuyas al horizonte de tres pruebas ni al criterio de costo. Todavía no lo verificaste. Tu respuesta propuesta ya respeta esto.
2. Las pérdidas totales, 0.07716 y 0.07752, coinciden con el 0.0772 del notebook. Las dos diferencias de continuación son muy pequeñas. No permiten elegir una regla ganadora.
3. En la nueva respuesta sobre jerarquía, "comparar jerarquías completas" exige evaluar cada candidata de forma exacta antes de la primera prueba. Con n ≤ 7 es posible. Para una n más grande no está garantizado. Si quieres, añade "en esta escala".
4. "Pérdida ya inevitable al abrir ABCD" es correcto si significa "aun con continuación óptima". Tu respuesta lo dice de forma explícita. Mantén esa condición junto a la palabra "inevitable".