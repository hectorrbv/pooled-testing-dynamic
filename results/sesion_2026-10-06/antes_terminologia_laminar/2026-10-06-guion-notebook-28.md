# Notebook 28 · un solo hilo para presentar

[Presentación](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb) · [Consultas separadas](../../augmented/notebooks/28_notas_de_estudio.ipynb).

La pregunta del pase principal es: ¿cuánto cuesta exigir laminaridad? Durante toda la presentación comparamos óptimo general con óptimo laminar. La población, las probabilidades, las utilidades, el tipo de observación y el presupuesto son los mismos para ambos. Seis minutos más discusión.

## 1. Fijar la comparación

“Buscamos el mejor resultado en dos clases de políticas. La general permite cualquier grupo de hasta G personas; la laminar exige que los grupos sean disjuntos o uno contenga al otro. Ambas pueden agrupar y adaptar las pruebas. Solo cambia esa restricción.”

AB seguido de AC es un cruce: comparten A y ninguno contiene al otro. AB seguido de A o de CD es compatible con laminaridad.

Aquí ‘óptimo’ significa el mayor valor esperado entre todas las políticas de esa clase. El óptimo general no puede valer menos porque también puede elegir una política laminar.

Transición: “Ahora que fijamos qué estamos comparando, veamos dónde cuesta esa restricción.”

## 2. Mostrar el mapa

“Calculamos 228 casos con n=6. El blanco significa que los dos óptimos empatan. En esta malla coinciden con una, dos o seis pruebas. Las separaciones aparecen con tres, cuatro o cinco; el peor caso observado pierde 16.67%.”

Recorrer las filas en ese orden. La pérdida siempre usa la misma referencia: (general−laminar)/general. Hay 118 empates y 110 separaciones. G=3 y las probabilidades de salud intermedias producen las mayores pérdidas. B=6=n permite resolver a todos y explica el empate de esa fila. No convertir los empates observados con B=2 en un teorema universal.

Son 19 probabilidades × 6 presupuestos × 2 tamaños máximos. Todas las personas tienen el mismo q, los estados son independientes, cada sano vale 1, las pruebas devuelven conteos y usamos soft clearing.

Transición: “Tomemos una celda para expresar esa pérdida en unidades de utilidad.”

## 3. Interpretar un caso del mismo mapa

“Con n=6, B=4, G=3 y q=.5, el óptimo general obtiene 3 y el laminar 2.5. Exigir laminaridad cuesta media unidad: .5/3 = 16.67%. Incluso eligiendo la mejor política laminar, esa diferencia permanece.”

Los dos valores ya incluyen las mejores decisiones posteriores a cada resultado. No estamos midiendo una mala elección de parámetros ni una heurística particular. El general alcanza toda la utilidad esperada disponible, nq=3.

No añadir aquí la referencia individual 2, la cuenta del par .774 ni los valores 1.50/1.75 del piloto n=4: responden otras preguntas y romperían el hilo. Si Francisco pide un mecanismo de inferencia, el ejemplo de tres conteos está en las consultas; aclarar que es una instancia distinta n=4. El mapa por sí solo cuantifica la pérdida, no explica todas las decisiones del caso n=6.

Transición: “Conocer la pérdida nos da una referencia. El siguiente objetivo es construir una política que conserve valor al imponer la restricción.”

## 4. Proponer una construcción

“Queremos seguir la política general y cambiar antes de su primer cruce a la mejor continuación laminar compatible. La compararemos con el óptimo laminar desde el inicio. Así medimos cuánto cuesta haber conservado las decisiones iniciales de la general.”

La comparación de óptimos está terminada. El experimento de cambiar antes del primer cruce sigue pendiente. Antes de ejecutarlo hay que fijar la definición del cruce tras reducir ancestros y el desempate entre acciones óptimas; empezaremos con los ocho casos acordados.

Pregunta para Francisco: “¿Es razonable conservar el prefijo y cambiar en el primer cruce, o deberíamos modificar decisiones anteriores?”

Después se podrá comparar una heurística como el cociente con la continuación laminar óptima. Esa comparación mide otra pérdida y se abrirá como una pregunta separada.

## Temas que se abren por separado

- Agrupar frente a individuales: el mapa de 170 mejoras y 58 empates, y el ejemplo del par que obtiene .774 frente a .6. Su pregunta es por qué agrupar ayuda.
- Una regla frente al mejor laminar: el piloto n=4 con valores 2, 1.75 y 1.50. Su pregunta es cuánto pierde una conversión concreta además del costo de la clase.
- La sección 5 de Nick: su comparación es dinámica frente a estática, con observación binaria y hard clearing. El destino estático permite cruces. La identidad de dos perfiles independientes es reutilizable; la desigualdad que controle nuestra transformación aún requiere trabajo.

Estos temas se conservan en el notebook de consulta, con encabezados distintos. La presentación principal no necesita explicarlos para sostener su comparación.

## Evidencia y reproducción

La malla principal compara los óptimos general y laminar en las mismas 228 instancias n=6. Para B<6, los 190 óptimos laminares coinciden entre Bellman por perfiles y solver de átomos; las 228 políticas laminares se evaluaron sobre los 64 estados y se verificó que no cruzaran. Para B=6, las individuales alcanzan la cota de utilidad total. Los generales proceden de la malla exacta ya calculada y verificada. No se relanzaron experimentos para este cambio de narrativa.

[Datos de los tres valores](../../results/sesion_2026-10-06/tres_referencias_28.json) · [Malla anterior n=4](../../results/sesion_2026-10-05/mapa_laminar_28.json) · [Pilotos](../../results/sesion_2026-10-04/evidencia_28.json).

Reconstruir ambos notebooks desde los datos guardados: `python augmented/notebooks/build_28_sesion.py`. El pase principal está en `presentacion_28.py`; las consultas de las otras comparaciones, en `contexto_28.py`; las demás notas y tareas, en el generador.
