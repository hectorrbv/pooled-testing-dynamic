# Preparación de la sesión del 29 de septiembre

Abre [27_sesion_francisco.ipynb](../../augmented/notebooks/27_sesion_francisco.ipynb). Son cuatro pantallas, unos ocho minutos y medio y una celda de consulta con las definiciones. El objetivo es llevar observaciones, una explicación de lo que ocurre y una propuesta de experimento.

## Qué pidió Francisco y qué llevamos

En el [acta del 22 de septiembre](2026-09-22-sesion-francisco.md), Francisco pidió ejemplos donde las reglas toman decisiones diferentes (D31:18–32:22), propuso dividir utilidad esperada entre pruebas esperadas (D35:17) y preguntó cómo cambia la política al variar λ (D36:16).

Nuestra respuesta sigue esas tres preguntas. Ampliamos la malla de 36 a 114 setups: 19 probabilidades de salud, dos presupuestos y tres tamaños máximos. Cada punto tiene su propio óptimo laminar como referencia. También refinamos el barrido de λ y encontramos una decisión que el barrido anterior había saltado.

Abrir con: «Probamos el cociente y ampliamos los regímenes. Ayuda cuando hay pocos sanos, pero el presupuesto cambia qué regla conviene. Encontramos además una pérdida de π_L que podemos explicar por su continuación».

## Guion para las cuatro pantallas

1. Mapa global. «Con pocos sanos, mirar solo el cobro inmediato pierde utilidad. El costo esperado permite agrupar y aprovechar las pruebas que sobran. Con muchos sanos, las diferencias se reducen». Señala el contraste de colores, sin recitar porcentajes. π_3 funciona bien con G=2, pero cae con G=4: permitir grupos mayores no asegura que una heurística los use bien.

2. Cociente frente a penalización. «Con q=0.60 y G=3, π_L gana con B=2 y el cociente con B=3». En el primer caso abren grupos distintos; en el segundo ambas abren tres y difieren al continuar. No basta un mapa de la primera acción ni elegir política solo por q. Muchos puntos empatan. La ampliación también elimina la conclusión anterior de que el cociente tenía el mejor mínimo.

3. Árbol del par. «El par queda resuelto con una prueba el 82% de las veces. En esas ramas podemos usar la segunda en C». El cociente reconoce ese ahorro dentro de AB; cobrar dos pruebas reservadas lo oculta. La política completa usa dos pruebas y obtiene 0.282, frente a 0.200 de dos individuales. Este árbol explica un mecanismo, no demuestra que el cociente domine siempre.

4. Precio y continuación. «λ=0.30 alcanza el óptimo; 0.31 abre el mismo trío y luego pierde». Si ABC sale todo sano o todo infectado, quedan DEF y dos pruebas. Con 0.30 se prueba DE; con 0.31 ningún plan local tiene valor neto positivo y la implementación vuelve al cobro inmediato. Así acaba haciendo dos individuales. La pérdida tiene una causa concreta que podemos aislar.

Cerrar con una propuesta: «Compararía por separado ajustar λ al presupuesto restante y usar el cociente cuando ningún plan supera el precio. Fijaría las reglas antes de probarlas en casos nuevos».

## Repaso: 45 minutos

- 0–10: recorre ambos mapas. Explica qué representa una casilla, cuál es el óptimo de referencia y por qué el color cambia con B aunque q sea el mismo. No memorices 114 números.
- 10–20: dibuja AB y sus tres resultados. Comprueba 0.01×2.1 + 0.18×1 + 0.81×0.1 = 0.282. Para el plan de resolver solo AB, E[U]=0.2 y E[T]=1.18; C pertenece a la política completa.
- 20–30: reconstruye el fallo de λ. ABC queda resuelto con probabilidad 0.2³+0.8³=0.52. Resolver DE da utilidad local 0.4 y cuesta 1.32 pruebas en promedio. Su neto es 0.004 con λ=0.30 y −0.0092 con 0.31. La continuación completa del par vale 0.536 y la individual 0.4: 0.52×(0.536−0.4)=0.07072 explica toda la pérdida.
- 30–40: presenta las cuatro pantallas en voz alta. En cada una di qué observamos, por qué creemos que ocurre y qué comprobarías después.
- 40–45: elige una pregunta del paper para conversar, sin añadirla a las diapositivas.

## Ideas para llevar a la discusión

La prioridad es distinguir las causas de las pérdidas. Para π_L, comparar λ fijo, λ dependiente del presupuesto y reemplazar la vuelta al cobro inmediato por el cociente. Cambiar una cosa por vez y evaluar las reglas sin escoger λ mirando el óptimo de cada caso. Para π_3, comprobar si su pérdida con G=4 viene de sobrevalorar grupos que luego no alcanza a resolver.

La pregunta teórica del par es si puede construirse una separación entre costo esperado y pruebas reservadas que no dependa del desempate. Aquí π_C y π_R empatan entre el individual y el par en su score y eligen individual; π_M sí prefiere estrictamente el individual por cobro inmediato.

Los mapas todavía fijan n=6 y utilidad 1. Una siguiente evaluación útil usaría probabilidades y utilidades heterogéneas y otros tamaños, con criterios fijados previamente. Estos resultados orientan esa evaluación; no son garantías generales.

## Para discutir el paper con Nick


La [v1](https://arxiv.org/html/2601.22419v1) comparaba algoritmos y aprendizaje. La [v2](https://arxiv.org/html/2601.22419v2) añade garantías teóricas. Su modelo usa pruebas binarias y hard clearing: solo se acredita a quienes aparecen en una prueba negativa. Nosotros usamos conteos y soft clearing.

La pregunta del resultado principal es cuánto aporta adaptar las pruebas a los resultados. Una estrategia estática fija todas las pruebas al inicio; una dinámica elige la siguiente según lo que observa. Ambas permiten cruces. El [Teorema 5.5](https://arxiv.org/html/2601.22419v2#S5.SS4) demuestra que el óptimo dinámico no supera el doble del mejor estático con cruces. Es un límite al valor de adaptar en el modelo binario.

El [Teorema 4.2](https://arxiv.org/html/2601.22419v2#S4) garantiza que el greedy de cobro inmediato, usando probabilidades conjuntas exactas, alcanza al menos $1/(e+1)$ del óptimo estático con cruces. Al combinarlo con el factor 2, obtiene al menos $1/[2(e+1)]$ del óptimo dinámico. Esta garantía no corresponde a nuestras políticas con conteos.

Para discutir por qué la demostración no pasa directamente a conteos, basta recordar AB con un infectado: si ambos tenían $q=0.3$, ahora cada uno tiene probabilidad de salud $0.5$. El [lema 3.1](https://arxiv.org/html/2601.22419v2#S3.SS1) del modelo binario impide superar el prior para personas todavía no acreditadas. Nuestro resultado observado no satisface esa propiedad.

La segunda conexión es una referencia exacta para la separación entre modelos. En $n=48$, $B=7$, $G=16$, $q_i=0.05$, $p_i=0.95$, $u_i=1$, el [Teorema 3.3](https://arxiv.org/html/2601.22419v2#S3.SS2) da óptimo binario con individuales: $Bq=0.35$. Nuestra construcción laminar con conteos y soft clearing alcanza al menos $1-0.95^{48}=0.9147$. Compara dos modelos de observación y acreditación; medir el costo de laminaridad dentro de conteos es otra pregunta.

## Material para consultar

[27_mapa4_greedies_con_planeacion.ipynb](../../augmented/notebooks/27_mapa4_greedies_con_planeacion.ipynb) conserva los cálculos originales de las seis reglas y el barrido inicial. [27_precio_laminar_y_repaso.ipynb](../../augmented/notebooks/27_precio_laminar_y_repaso.ipynb) conserva los ejemplos con cruces para la otra pregunta de la sesión: cuánto cuesta restringirse a laminar. Esa comparación queda disponible si Francisco la pide; no se mezcla con el mapa de políticas.

El [CSV ampliado](../../results/sesion_2026-09-29/regimenes_politicas.csv) contiene los 114 setups y reproduce los 36 originales. El [barrido refinado](../../results/sesion_2026-09-29/lambda_refinado.csv) contiene 26 valores de λ. Los casos destacados se verificaron enumerando los 64 perfiles de salud, con discrepancia numérica menor que 10⁻¹⁵.

En los mapas λ=0.01 y el cociente es E[U]/E[T]. Ambas políticas planean hasta tres pruebas dentro de un grupo, limitadas por las pruebas restantes, ejecutan una y vuelven a decidir. π_ratio compara planes que maximizan utilidad para cada grupo y límite de pruebas: no busca el mejor cociente entre todos los árboles posibles. Es otra decisión de diseño abierta a discusión.

Los mínimos antiguos de 95.24% y 95.00% correspondían a los 36 setups. En la malla ampliada son 93.37% y 95.00%, respectivamente. No presentar los números antiguos como resumen actual ni estos mínimos como garantías. El 0.9307 de la sesión anterior corresponde a otra instancia; no es λ.
