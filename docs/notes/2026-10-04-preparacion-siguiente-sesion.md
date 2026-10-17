# Preparación tras la reunión del 29 de septiembre

Material principal: [28_pooling_y_precio_laminar.ipynb](../../augmented/notebooks/28_pooling_y_precio_laminar.ipynb). La versión visual del 5 de octubre tiene cuatro pantallas para unos ocho minutos y siete consultas, incluidas dos para estudiar el paper. El [guion actualizado](2026-10-05-guia-visual-notebook-28.md) explica el mapa ampliado de 114 casos y el orden de estudio. El notebook 27 queda como evidencia de la comparación de heurísticas.

Actualización tras la revisión con Opus: [dirección de prueba y discusión](2026-10-04-revision-y-direccion-prueba.md). Hay una garantía elemental de 1/G y una obstrucción para una salida siempre estática laminar cuando G crece. La existencia de una constante universal para políticas dinámicas laminares sigue abierta. El próximo experimento debe separar adaptación, laminaridad y calidad de la continuación.

La fuente de los encargos es el resumen automático de Fathom adjuntado por Héctor, «Dynamic Augmented Pooled Testing», 29 de septiembre, 48 minutos. No es una transcripción literal ni permite reconstruir todas las condiciones de las conjeturas. La lectura de abajo distingue lo recogido por Fathom, los cálculos verificados y nuestras propuestas nuevas. No se ha enviado ningún mensaje a Francisco.

## Qué pide la nueva dirección

Según el resumen, hay tres encargos de investigación: precisar y refutar la afirmación de que agrupar siempre es óptimo, terminar la lectura del nuevo paper con atención a §5 y estudiar una transformación de políticas generales con conteos en políticas laminares con poca pérdida. El resumen también pide enviar preguntas sobre el paper a Francisco; primero deben quedar preparadas para revisión de Héctor.

La pregunta principal pasa a ser cuánto cuesta imponer laminaridad. La comparación de π_L y π_ratio sigue siendo útil para construir continuaciones, pero ampliar otra vez el mapa de ganadores no es la prioridad de esta preparación.

## Correcciones necesarias al resumen automático

- Λ, mayúscula, es una cantidad de la instancia del paper; λ, minúscula, es el precio que elegimos en π_L. No son intercambiables. π_ratio, con α=1, no tiene ese parámetro λ.
- Usamos q para salud y p para infección. El ejemplo presentado con λ=0.30 y 0.31 tenía q=0.20 y p=0.80; ambos abrían el mismo trío y diferían después. Fathom parece mezclar esos datos.
- En la malla del notebook 27 la referencia era el óptimo laminar. El óptimo general con cruces es otra referencia y debe nombrarse explícitamente.
- Las reglas que dividían por pruebas reservadas no destacaban todas en salud rara. Los resultados favorables señalados correspondían principalmente a π_L y π_ratio.
- El argumento del paper termina en asignaciones estáticas que pueden cruzarse. Convertir a estático y convertir a laminar son restricciones diferentes. Además, la factorización laminar facilita la inferencia, pero por sí sola no demuestra que toda optimización sea eficiente.

## Qué ya quedó hecho

El contraejemplo solicitado tiene n=2, B=1, G=2, q_A=0.9, q_B=0.1 y utilidades 1. A individual vale 0.9, B vale 0.1 y AB vale 0.18. Las cuatro realizaciones de salud fueron enumeradas exactamente. Con R=1 ninguna identidad queda determinada con certeza. Esto refuta que siempre convenga una prueba de dos o más personas; no refuta la clase de políticas que permite tanto grupos como individuales.

Para B>=2 hay que precisar qué se afirma. Bajo independencia, n>B, G>=2, probabilidades estrictamente entre cero y uno y utilidades positivas, sustituir las dos primeras individuales del mejor plan individual por un par y su posible refinamiento mejora estrictamente ese plan. El par recupera la información de las dos individuales, y ahorra una prueba cuando ambos estados coinciden. Esta prueba ahorrada se usa en la persona B+1. La identidad y su alcance están en la consulta del notebook; se comprobó además en seis ejemplos heterogéneos. No es una afirmación sobre la optimalidad de cualquier grupo ni sobre todas las primeras acciones óptimas.

Se revalidó el caso n=4, B=3, G=3, q_i=0.5, u_i=1: las pruebas fijas ABC, ABD y ACD identifican los 16 perfiles. Alcanzan utilidad 2, la máxima posible. Dos solvers coinciden en que el óptimo laminar vale 1.75. El ejemplo ya existía; no se presenta como descubrimiento nuevo.

El piloto nuevo compara dos conversiones de ese plan:

1. Recortar de cada grupo las personas que ya aparecieron: queda ABC, D y un vacío. Se ejecutan los grupos no vacíos; cualquier presupuesto sobrante se usa en la acción laminar de mayor cobro inmediato. Valor: 1.50.
2. Conservar solo ABC y continuar con π_ratio. Valor: 1.75. Cuando ABC es mixto, resuelve primero dentro de ABC y decide después si prueba D.

Así, la primera regla pierde 0.50 frente al plan original: 0.25 corresponde a la restricción de clase y 0.25 a esa regla concreta. La segunda recupera la parte evitable en esta instancia. Todas las utilidades se integraron sobre los mismos 16 perfiles y se verificó B, G y laminaridad en cada trayectoria. Es un piloto, no una garantía general ni una validación del muestreo de caminos: aquí la política fuente es estática. La segunda regla ignora el resto del camino fuente después de su primera prueba.

## Qué estudiar ahora: 75 minutos

1. Durante 15 minutos, reconstruye el árbol con B=1 y explica exactamente qué refuta. Después lee el argumento del par con B>=2 y distingue «mejor que solo individuales» de «óptimo global».
2. Durante 15 minutos, trabaja el ejemplo de cuatro personas. Comprueba que ABC y ABD se cruzan, y explica por qué 2 es una cota superior alcanzada al identificar a todos. No hace falta memorizar los 16 perfiles.
3. Durante 30 minutos, lee la estrategia de prueba de [§5](https://arxiv.org/html/2601.22419v2#S5), el teorema 5.5 y §5.1 junto al ejemplo del [apéndice B](https://arxiv.org/html/2601.22419v2#A2). Dibuja el perfil simulado que selecciona el camino y, separado, el perfil real donde se evalúa. En esta sección Z=1 indica infección. Luego anota qué representan una consulta y una oportunidad de acreditar a la persona elegida. Deja la demostración completa de §5.2–5.4 para la siguiente sesión de estudio.
4. Durante 15 minutos, explica por qué las dos conversiones del piloto rinden distinto. Con ABC mixto, la primera prueba D antes de resolver ABC; la segunda puede aprovechar que ABC quede resuelto pronto. Ensaya las cuatro pantallas con una observación y una pregunta por pantalla.

El punto de partida matemático es precisar qué garantía buscamos: para G fijo ya podemos justificar 1/G mediante individuales. Una transformación T con E_ω[V(T(π,ω))] >= c V(π), donde c tampoco dependa de G, sería una meta más fuerte. El azar ω es independiente del perfil real; cada política resultante respeta B y G. V ya promedia sobre la salud. Antes de comprometernos con la constante universal, debemos investigar si puede ser falsa. La revisión contiene una obstrucción para salidas estáticas y una cota informacional para probabilidades y utilidades iguales.

## Siguiente trabajo de implementación, acotado

Usar primero los seis casos con historial inicial vacío ya guardados en `results/precio_laminar_2026-09-21/resultados.json`; mantener aparte los dos casos con historial condicionado. No hace falta una malla grande para esta primera comparación.

El control de reproducir caminos virtuales ya se evaluó exactamente en esos seis casos, con perfil virtual y real independientes. En el caso n=4, B=2, G=2, q=0.3, una política que ya es laminar baja de 0.774 a 0.5304 al fijar así su calendario. La mejor estática laminar vale 0.60: hay pérdida de adaptación y pérdida adicional del procedimiento. No atribuir ninguna de ellas a eliminar cruces.

La candidata prioritaria cambia: seguir la política original con resultados reales mientras siga siendo laminar y, ante el primer cruce, continuar primero con el óptimo laminar desde esa historia y después comparar con π_ratio. Añadir π_ratio desde el inicio como control. Esta comparación aún no está implementada. Mantener el muestreo independiente como un control separado, no como paso obligatorio de la conversión.

Guardar el óptimo general, el óptimo laminar, el valor de conservar el prefijo real con continuación óptima y el de usar π_ratio. Sus diferencias separan pérdida de clase, compromiso con el prefijo y heurística. Fijar también los desempates entre políticas fuente óptimas: sus caminos muestreados pueden valer distinto. Los controles deben comprobar legalidad en todas las ramas. Un resultado favorable en estos casos solo justifica seguir investigando.

## Qué discutir con Francisco

- «La afirmación literal tiene un contraejemplo con B=1. Con B>=2, un par sí mejora el mejor plan puramente individual bajo estas condiciones. ¿Cuál era el alcance preciso de la conjetura que queremos estudiar?»
- «El recorte aleatorio del paper no garantiza laminaridad: para un camino AB, AC, ambos grupos quedan intactos con probabilidad 1/16. Necesitamos otra transformación».
- «En soft clearing se puede acreditar a alguien por inferencia sin una prueba negativa que lo contenga. ¿Qué objeto debería sustituir a las oportunidades de acreditación del árbol de §5.1?»
- «En el piloto, replanear recupera toda la pérdida evitable de esa instancia. Propongo conservar el prefijo que ocurre con resultados reales y comparar la continuación exacta con el cociente al llegar al primer cruce».
- «Tenemos una garantía básica de 1/G. ¿Nos interesa mejorarla con G acotado o comprobar primero si puede existir una constante independiente también de G?».

No atribuir al paper una constante para nuestro caso ni dar por probado el precio de laminaridad. Las preguntas anteriores son material para preparar el intercambio; no se han enviado.

## Fuentes y reproducción

[Paper v2, 25 de septiembre](https://arxiv.org/html/2601.22419v2) · [Evidencia de esta preparación](../../results/sesion_2026-10-04/evidencia_28.json) · [Experimento](../../augmented/experimento_sesion_28.py) · [Generador](../../augmented/notebooks/build_28_sesion.py).

Desde la raíz del repositorio: `python -m augmented.experimento_sesion_28`, seguido de `python augmented/notebooks/build_28_sesion.py`. El notebook se puede presentar sin ejecutar código.
