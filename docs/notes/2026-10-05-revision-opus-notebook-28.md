# Revisión del notebook 28 con Opus — 5 de octubre

Se revisaron los cuatro bloques de presentación, las siete consultas, las siete figuras, las tareas del plan y los argumentos que conectan con el paper. Claude Opus (`claude-opus-5-5`) leyó los textos y las imágenes; no ejecutó código. Su [informe original](../../results/sesion_2026-10-05/revision_opus/revision.md) se conserva separado de las correcciones aplicadas.

## Correcciones aplicadas

1. El mapa define la pérdida como (general − laminar)/general y aclara que ambas clases pueden adaptar. El panel B=4=n es un control: probar a todos alcanza 4q. Se indican las dos curvas coincidentes para que ninguna parezca ausente.
2. El bloque 3 muestra cómo se obtiene 0.5304: el camino simulado elige AB,A con probabilidad .42 o AB,C con probabilidad .58; sobre otra población esas listas valen .60 y .48. El 0.174 es la pérdida de adaptación dentro de la clase laminar. No es la brecha dinámica–estática general: el mejor plan fijo con cruces vale .627, dato comprobado y reservado a las notas.
3. La operación determinista del piloto se llama «quitar repetidos», para distinguirla del recorte aleatorio de Nick. Se muestra qué hace: ABC,ABD,ACD pasa a ABC,D. El árbol lleva dentro la condición ABC mixto; 1.50 y 1.75 siguen siendo valores de las políticas completas.
4. La descripción del cociente indica que compara planes locales de hasta tres pruebas, limitados por el presupuesto. Las notas precisan que los candidatos maximizan utilidad local antes de comparar sus cocientes. El piloto no demuestra eficiencia a escala ni una garantía general.
5. Sobre Nick se dice que su prueba no se aplica tal cual a conteos y soft clearing. No se afirma que un factor dos haya sido refutado en otro modelo. El recorte puede incluso crear cruces: de ABC,AB puede producir AC,AB con probabilidad 1/32.
6. La consulta sobre el intercambio de individuales explicita conteos, soft clearing y el estado de revisión humana pendiente según el plan. La garantía 1/G queda acotada a historial inicial vacío, independencia y soporte completo; no se aplica sin otro argumento a un historial condicionado.
7. La agenda indica cambiar antes de ejecutar el primer cruce y comparar también con el óptimo laminar desde el inicio, para separar la pérdida del prefijo. La referencia a λ=.30/.31 apunta al notebook 27.

## Comprobación independiente

El [script de auditoría](../../results/sesion_2026-10-05/revision_opus/verifica_revision.py) usa solo la biblioteca estándar, sin importar los solvers ni los evaluadores del proyecto. Sus [resultados](../../results/sesion_2026-10-05/revision_opus/verificacion_independiente.json) confirman:

- Los 114 óptimos generales del mapa, por enumeración recursiva de todas las acciones factibles y los 16 estados de salud. Diferencia exacta cero.
- El óptimo estático laminar .600 entre 44 familias admisibles de hasta dos pruebas; el estático general .627 entre 56 familias.
- El camino simulado .5304 y los valores 2, 1.50 y 1.75 de las tres políticas del piloto.

Esto complementa las 228 evaluaciones de políticas y los 114 cotejos del óptimo laminar ya guardados. Es verificación finita, no una garantía de aproximación.

Se cotejó además la consulta de Λ con el [Teorema 3.3(i) y su demostración](https://arxiv.org/html/2601.22419v2#S3.SS2): en su modelo binario, q_i≤1/2 implica que existe un óptimo individual. No se traslada esa conclusión a nuestro modelo de conteos.

## Matices del informe y pendientes

La preocupación de Opus por la optimalidad general se cerró con la enumeración independiente anterior. Su frase «casi un planificador exacto» no se incorporó: un plan óptimo dentro de un grupo no es una optimización global entre todos los grupos.

Hay ocho casos en `results/precio_laminar_2026-09-21/resultados.json`: seis desde historial vacío y dos condicionados. La tabla abreviada del plan omite dos; para el experimento se usará el JSON. Los controles donde no hay cruce estricto deben conservarse. La definición de cruce y la reducción de ancestros siguen siendo una tarea previa al experimento.

El memo de transferencia de §5, el experimento del primer cruce y las revisiones humanas de la proposición y de la Parte 2 siguen pendientes. Este notebook contiene antecedentes, explicación y agenda; la revisión no los marca como entregados. El plan maestro y el documento de Vladimir no se modificaron.
