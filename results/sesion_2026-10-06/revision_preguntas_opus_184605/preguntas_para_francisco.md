# Preguntas para Francisco · revisión con Claude Opus

Revisión real con claude-opus-5-5; selección y contraste posterior contra §5 del paper. Si solo hay tiempo para tres, priorizar 2, 3 y 6. Las preguntas de lectura se presentan como dudas de Héctor; las de investigación no afirman resultados ya demostrados.

1. Subcubos e independencia. «Z y Z′ se generan independientemente, pero después de fijar dónde coinciden y dónde difieren quedan complementarios dentro del subcubo. ¿Puedes mostrarme en un ejemplo cuándo usamos la independencia y cuándo esa complementariedad?» Aclaración de lectura. No suponer independencia de los dos perfiles después de ese condicionamiento.

2. Qué tendríamos que demostrar. «Para adaptar la simetrización, ¿necesitamos demostrar nuestra desigualdad en todos los subcubos, incluyendo personas fijadas sanas o infectadas, además de las que tienen probabilidad1/2? ¿Qué tendría que conservar la transformación para poder promediar después?» Aclaración con dirección de investigación. Probar solo el cubo completo con q_i=1/2 no basta para invocar sin más el argumento de §5.4.

3. Soft clearing. «Con AB=1 y A=1 acreditamos a B sin una prueba negativa que lo contenga. ¿Podemos formular el argumento alrededor de pequeños conjuntos de conteos que, juntos, demuestran que alguien está sano? ¿Qué propiedad necesitarían para conservar utilidad al transformar la política?» Investigación. En laminar ya disponemos de bloques residuales con conteo conocido; el objetivo nuevo sería una desigualdad de conservación, no renombrar ese mecanismo.

4. El factor dos. «¿Puedes mostrarme con el ejemplo del apéndice B cómo se equilibran perder a la persona que queríamos acreditar y eliminar a sus compañeros infectados? Quiero entender por qué esa cuenta conserva al menos la mitad en promedio.» Aclaración de lectura. Retener cada integrante con probabilidad1/2 no implica por sí solo conservar la mitad de utilidad.

5. El alcance de la garantía. «Ya tenemos una cota elemental1/G. ¿El objetivo es una fracción de utilidad que se mantenga incluso al aumentar G, o nos interesa mejorar la garantía para grupos pequeños? Eso cambia qué familias de ejemplos conviene estudiar.» Investigación. El dato3 frente a2.5 solo impone c≤5/6 a una garantía universal; no descarta una constante positiva independiente de G.

6. Una clase intermedia medible. «En nuestro caso el general obtiene3 y el laminar2.5. Propongo calcular qué ocurre si permitimos una sola excepción a la laminaridad por trayectoria. ¿Ese control nos ayudaría a saber si necesitamos una clase mucho más amplia?» Investigación. Definición concreta para un experimento: al eliminar a lo sumo una prueba de cada trayectoria, los grupos restantes deben formar una familia laminar. La prueba eliminada aquí solo define la clase: sí se ejecuta y consume presupuesto en la política evaluada. Ese experimento no está ejecutado.

Siguiente paso posible: primero comparar la clase intermedia con los dos óptimos en el caso n6,B4,G3,q=.5. En el experimento ya propuesto de conservar un prefijo real y seguir de modo laminar, medir la pérdida frente a2.5; ninguna política laminar puede superar ese óptimo.
