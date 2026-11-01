# Notebook 28: mapa, mecanismo y dirección de investigación

Actualización del 6 de octubre: el notebook incorpora la malla de 228 casos sobre agrupar frente a individuales. El [guion vigente](2026-10-06-guion-notebook-28.md) describe sus cuatro bloques; los árboles extensos y el estudio de §5 están en [otro notebook](../../augmented/notebooks/28_notas_de_estudio.ipynb). Lo que sigue documenta la versión anterior.

[Notebook principal](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb). Cuatro pantallas, unos ocho minutos. Las tareas de Héctor, la explicación visual del paper, el contraejemplo con B=1 y las consultas matemáticas quedan después, fuera del pase principal. El notebook 27 conserva el mapa de heurísticas; este estudia el costo de imponer laminaridad.

## Qué decir en cada pantalla

1. «La diferencia depende del presupuesto». En los 114 setups con n=4, los óptimos general y laminar coinciden cuando B=2. Con B=3 aparecen diferencias; el máximo es 12.5%, en q=.5 y G=3. Con B=4=n desaparecen: cuatro pruebas individuales identifican a todos y ambos óptimos valen 4q. La pregunta es cómo cambia esta ventana de presupuestos al aumentar n. El máximo observado y el empate con B=2 son evidencia finita; el empate con B>=n se justifica porque las individuales alcanzan toda la utilidad disponible.

2. «Los conteos juntos revelan identidades». ABC=1, ABD=1 y ACD=2 obligan a que A y B sean sanos. Ninguna prueba tiene conteo cero. El dibujo ilustra una realización; la comprobación de los 16 perfiles justifica que el plan completo identifica a todos y obtiene 2, frente a 1.75 del óptimo laminar. La pregunta para la prueba es qué parte de estas deducciones sobrevive al eliminar cruces.

3. «Fijar un camino introduce otra pérdida». Una política que ya es laminar vale .774. El mejor plan fijo laminar vale .6. Reproducir un camino elegido en una población independiente vale .5304. La primera diferencia mide la adaptación dentro de la clase laminar; la segunda es una pérdida adicional de ese procedimiento de muestreo. Es un control con conteos y sin el recorte aleatorio de Nick, no una evaluación de su teorema.

4. «El orden de las continuaciones importa». Ambas reglas conservan ABC. Quitar repetidos deja ABC,D; tras ABC mixto, probar D primero deja una sola prueba dentro de ABC. Refinar A primero permite dedicar la última prueba a D si ABC ya quedó resuelto. Las dos políticas completas valen 1.50 y 1.75. El siguiente experimento debe separar el costo de exigir laminaridad, el de conservar un prefijo y el de escoger una continuación heurística.

## Próximo experimento, todavía pendiente

Seguir la política general con los resultados reales hasta que proponga el primer cruce. Cambiar antes de ejecutar ese cruce. A partir del mismo historial, comparar la mejor continuación laminar con la regla del cociente. Comparar además con el óptimo laminar desde el inicio para separar la pérdida de conservar ese prefijo. Incluir el cociente desde el principio para comprobar si conservar el prefijo aporta algo. Seguir el plan del 5–6 de octubre: comenzar con los ocho casos existentes (seis desde historial vacío y dos condicionados). Antes de ejecutar, fijar qué cuenta como cruce tras reducir ancestros y preferir acciones laminares al desempatar entre acciones óptimas.

Conservar el muestreo de caminos como control separado. El origen del calendario y la clase laminar son dos restricciones distintas.

## Orden para estudiarlo

La presentación abre con el mapa porque ayuda a discutir resultados. Para aprender, podemos empezar por las consultas del paper y avanzar una idea por turno:

1. Distinguir una regla que responde a los resultados de una lista de pruebas ya fijada. En el paper las pruebas son binarias y el crédito es hard clearing.
2. Entender que la población ficticia se introduce en la demostración para construir esa lista. Nuestras políticas no deciden siguiendo una población ficticia.
3. Entender que una garantía en promedio implica la existencia de algún plan al menos tan bueno como el promedio. Esto todavía no demuestra la garantía.
4. Estudiar el recorte aleatorio de cada aparición persona-prueba. La construcción completa incluye ese recorte; muestrear el camino sin recortarlo no es suficiente para invocar el teorema.
5. Usar la consulta gráfica de una persona con q=.3: los dos sorteos diferentes tienen probabilidad .21 cada uno. Condicionado a que difieren, cada orientación pesa 1/2. No cambiamos el prior original.
6. Extender esa igualdad a varias personas y distinguir la identidad probabilística reutilizable de la desigualdad de utilidad que falta demostrar para nuestra transformación.

Después, estudiar el árbol de .774 y la continuación tras ABC. Volver al mapa al final, cuando cada diferencia de valor tenga un mecanismo reconocible.

## Evidencia y límites

El nuevo mapa usa n=4, u_i=1, 19 probabilidades homogéneas de salud entre .05 y .95, B en {2,3,4} y G en {2,3}: 19×3×2=114 casos. Historial inicial vacío, priors independientes, conteos exactos y soft clearing. Se calcularon ambos óptimos con fracciones exactas y se evaluaron las 228 políticas por filtrado independiente de los 16 perfiles. Los 114 valores laminares también coinciden exactamente con el solver de átomos. Hay 81 empates y 33 separaciones. Los 38 casos con B=4 alcanzan 4q en ambos solvers. No se hizo una búsqueda exhaustiva sobre poblaciones heterogéneas ni sobre n mayor.

Las figuras del árbol y del cambio de orden usan los datos exactos anteriores. La probabilidad de ABC mixto es 3/4; condicionado a esa rama, A resuelve ABC con probabilidad 1/3. Los valores 1.50 y 1.75 corresponden a las políticas completas, no solo a la rama dibujada.

[Datos del mapa](../../results/sesion_2026-10-05/mapa_laminar_28.json) · [Datos de figuras](../../results/sesion_2026-10-05/datos_figuras_28.json) · [Experimento](../../augmented/experimento_mapa_laminar_28.py) · [Generador](../../augmented/notebooks/build_28_sesion.py) · [Figuras](../../augmented/notebooks/figuras_28.py).

Reproducir desde la raíz: `python -m augmented.experimento_mapa_laminar_28`, seguido de `python augmented/notebooks/build_28_sesion.py`.

La revisión teórica, sus demostraciones y sus conjeturas siguen en [la nota del 4 de octubre](2026-10-04-revision-y-direccion-prueba.md). El notebook se puede abrir sin ejecutar código.

Revisión del 5 de octubre: [hallazgos de Opus y correcciones aplicadas](2026-10-05-revision-opus-notebook-28.md).
