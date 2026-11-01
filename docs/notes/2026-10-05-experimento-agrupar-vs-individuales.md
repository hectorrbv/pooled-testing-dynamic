# Experimento: cuándo conviene agrupar frente a probar individuos

La pregunta elegida por Héctor es cuánto mejora la utilidad al permitir grupos, frente a la mejor política formada solo por pruebas individuales. La comparación se hace entre políticas completas: que una política empiece con una individual no significa que nunca agrupe después.

## Primera malla, ejecutada

n=6, B=1,...,6, G∈{2,3}, q=.05,.10,...,.95, utilidad 1 por persona. Probabilidades iguales e independientes, historial vacío, conteos exactos y soft clearing. Son 6×2×19=228 casos.

La referencia individual vale I=Bq. El otro valor, V, es el óptimo dinámico que permite cualquier grupo de hasta G personas, incluidos los grupos de una sola persona. Permite cruces. El color muestra 100(V/I−1): 0% significa empate y 50% significa obtener una vez y media la utilidad de las individuales. Se guardan también la ganancia absoluta y ambos valores.

El [heatmap general frente a individuales](../../results/sesion_2026-10-06/figures/agrupar_vs_individuales.png) tiene q en columnas, B en filas y un panel por G, con la misma escala. El 6 de octubre se añadió el óptimo laminar a los mismos 228 casos. Para mantener una sola pregunta en la presentación, el [notebook 28](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) se centra en general frente a laminar. La comparación frente a individuales y el ejemplo del par se conservan en [consultas separadas](../../augmented/notebooks/28_notas_de_estudio.ipynb). El [archivo unificado](../../results/sesion_2026-10-06/tres_referencias_28.json) conserva los tres valores y explicita ambos denominadores. Su mayor pérdida laminar observada es 16.67%, en n=6, B=4, G=3, q=.5: individuales=2, laminar=2.5, general=3.

## Qué mirar

- B=1: el óptimo es max_k kq^k. Para G=2 o3, las individuales empatan con el óptimo cuando q≤.5; agrupar mejora cuando q>.5. En q=.5 un par también empata con una individual.
- 2≤B<6: permitir grupos mejora en los 152 casos de la malla. El argumento de sustituir dos individuales por un par explica por qué esperamos mejora estricta bajo estas hipótesis; la malla mide su magnitud.
- B=n=6: las individuales ya identifican a todos, de modo que hay empate en las 38 celdas de control.

En total hay 170 mejoras y 58 empates. Son resultados de esta malla homogénea; no una evaluación de una heurística eficiente ni un mapa de todas las instancias heterogéneas.

Las aperturas óptimas por tamaño se guardan conservando empates, aprovechando la simetría del prior homogéneo. No se interpreta un único desempate del solver como una preferencia estricta. Para B=n no se enumeraron esas aperturas: se usó el certificado de las seis individuales.

## Verificación y siguiente paso

Los 228 valores se evaluaron por filtrado independiente de los 64 estados de salud. El caso B=n usa la cota de utilidad total y su política que la alcanza. Se cotejaron la fórmula de B=1, las desigualdades I≤V≤nq, la mejora estricta para presupuestos interiores y la monotonía en B y G. Evaluar la política confirma su valor; el resto de los óptimos se obtiene del Bellman exacto existente.

Para continuar, elegir tres celdas con mecanismos distintos y dibujar sus árboles: una prueba con q bajo, dos pruebas con q bajo y dos pruebas con q alto. Después variar la heterogeneidad manteniendo la media q: comparar, por ejemplo, seis probabilidades .3 con tres .1 y tres .5. En esa segunda fase la referencia individual debe seleccionar los mayores u_iq_i, en lugar de usar B por la media. Las utilidades permanecen en 1 para estudiar una variación a la vez.

[Datos exactos](../../results/diseno_mapa_2026-10-05/agrupar_vs_individuales.json) · [Experimento](../../augmented/experimento_agrupar_individuales.py) · [Figura](../../augmented/notebooks/figura_agrupar_individuales.py).

Reproducir desde la raíz con `python -m augmented.experimento_agrupar_individuales` y `python augmented/notebooks/figura_agrupar_individuales.py`.
