# Dirección y alcance de los resultados que busca Francisco

Nota guardada por petición explícita de Héctor el 15 de octubre de 2026. Fuente: mensaje de Francisco copiado por Héctor en esta conversación. Se conserva como referencia para orientar el paper, los experimentos, los notebooks y las sesiones de estudio.

## Mensaje de Francisco

> Pero por cierto creo que puedo comprobar que para el caso de u_i = 1 y q_i = 1/2 para todos, que el gap entre augmented overlapping y laminar es de log(G) Basicamente creo que en ese caso laminar puede ser acotado como O(B) y con los detection matrices podemos tener algo com O(log (G) B)

Su expresión «creo que puedo comprobar» describe una propuesta de resultado. El mensaje por sí solo no constituye una demostración entregada.

## Qué dirección y qué magnitud señala

Francisco apunta a una caracterización estructural: cuánto cuesta restringir a laminar incluso cuando se toman las mejores decisiones posibles. La magnitud buscada es el orden de crecimiento de la separación en función de G, con hipótesis explícitas y constantes uniformes; una diferencia porcentual en una instancia pequeña es un paso hacia ese resultado.

La lectura matemática que trabajaremos, diferenciada de la cita literal, requiere:

1. Una cota $V^*_{\rm lam}\le CB$ que cubra todas las políticas laminares dinámicas, con C independiente de n, B y G, bajo $q_i=1/2$, $u_i=1$, independencia y soft clearing.
2. Una familia de instancias y políticas generales admisibles que alcance $V_{\rm general}\ge cB\log G$, con c uniforme y los mismos recursos. Una cota superior $O(B\log G)$, por sí sola, no proporciona esta segunda pieza.
3. Comparar ambas sobre esa familia: el cociente óptimo general/óptimo laminar sería al menos de orden $\log G$. Para caracterizar el orden por ambos lados se necesita además una cota superior correspondiente; el argumento de información del notebook apunta a esa parte y conserva su estado de revisión.

Se trata de familias de peores casos con n y B compatibles con la construcción. La utilidad total esperada no supera n/2; con B≥n ambas clases pueden identificar a todos y empatan. No se afirma una ventaja logarítmica en cada setup, ni que q=1/2 sea el peor régimen entre todas las probabilidades.

## Criterio para nuestro trabajo

Al preparar una propuesta para Francisco, preguntar: ¿qué afirmación general sugiere?, ¿qué mecanismo explica?, ¿qué familia de instancias o desigualdad permitiría demostrarla?, ¿qué hipótesis y límites tiene?

Los ejemplos pequeños sirven para establecer existencia y hacer visible el mecanismo; las mallas, para detectar patrones y falsificar hipótesis; las construcciones y cotas, para determinar el alcance general. La malla actual hasta G=5 no demuestra crecimiento logarítmico. Ampliarla debe responder a una pregunta que ayude a construir o refutar el argumento.

Si la separación propuesta se demuestra, ninguna mejora de una heurística laminar podrá eliminar esa pérdida de la clase en el peor caso. Q1 —aproximar eficientemente el óptimo laminar— sigue siendo relevante, pero se distingue de Q2 —el costo de exigir laminaridad frente al general—.

Para el estudio de Héctor, la prioridad es poder explicar esas dos piezas y su cociente, distinguir un límite para toda una clase de un fallo de una regla, y llegar con una pregunta concreta sobre el argumento que podría acotar la utilidad laminar por CB. Los encargos, responsables y fechas siguen en el [plan semanal](2026-10-14-plan-semana.md).

## Prioridad de estudio ratificada por Héctor — 17 de octubre

Héctor pide que el repaso construya progresivamente hacia la dirección propuesta de log G. Esa es la meta de preparación para la discusión; no implica dar por demostrada la propuesta.

Ordenar el estudio por lo que ayuda a entender las dos piezas: el límite para toda política laminar y la utilidad alcanzable mediante matrices, con los mismos recursos. Conectar cada ejemplo explícitamente con la pieza que ilustra, el paso que pone a prueba o la hipótesis que necesita. Después trabajar el cociente y la cota de información que limita su crecimiento.

Usar las intervenciones pasadas de Francisco como contexto, sin convertir su orden cronológico en un recorrido obligatorio. Retomar individuales, binario frente a laminar o heurísticas cuando resuelvan una duda necesaria, o para cerrar sus encargos separados; evitar que desplacen el objetivo principal. Dar un paso y una pregunta cada vez. La meta es que Héctor pueda explicar qué está verificado, qué falta demostrar y qué ejemplo, argumento o experimento puede aportar a esa discusión.
