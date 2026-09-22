# Repaso y presentación: los contraejemplos primero

Preparado el 21-sep-2026. Se ejecutaron los notebooks 26 esencial y 27 de principio a fin. Las comparaciones nuevas usan posterior-zero en ambas clases. El notebook 25 modificado por el usuario se preservó.

## Objetivo de comprensión

Poder explicar cada ejemplo con cuatro frases: **qué afirmación cuestiona, cómo funciona, qué cambió a partir de él y qué no demuestra**. El hilo de la presentación es la sucesión de dificultades que obligaron a cambiar la forma de decidir.

## Material y orden de apertura

| Prioridad | Abrir | Qué mostrar |
|---|---|---|
| 1 | [26 esencial](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/26_esencial.ipynb), §1 y §5 | Los dos scores y el contraejemplo de no-reentrada. |
| 2 | Mismo notebook, §7 | La instancia de seis personas, el menú resumido y la rama R(AEF)=1 del árbol. |
| 3 | Mismo notebook, §8–9 | π_L, sensibilidad a λ y diferencia entre los dos comparadores. |
| 4 | [27: precio de laminaridad](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/27_precio_laminar_y_repaso.ipynb), §2–4 | El cruce AC después de AB, el certificado de los tres tríos y el óptimo general de seis personas. |
| Apoyo | [24](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/24_caso_sesion.ipynb), Acto 4 | Una misma prueba pasa de ganancia 0.05 a 1/3 después de observar un conteo. |
| Apoyo | [25-II](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/25_resultados_y_peticiones_parte_ii.ipynb), §6; [26 completo](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/notebooks/26_costo_local_y_no_reentrada.ipynb), §2 y §4–6 | Derivación del colapso, definición del costo local y barrido de α, solo si surge una pregunta. |

El notebook 26 esencial conserva sus ejemplos históricos y los etiqueta. Su menú de seis personas ahora muestra una selección legible de 12 de las 56 primeras acciones. C3 y π_L empiezan por ADEF; las tres políticas π_M, π_C y π_R empiezan por F. El Q de ese menú presupone continuación óptima y no es el valor completo de esas heurísticas.

## Sesión de 100 minutos con papel o iPad

| Minutos | Actividad | Producto a mano |
|---|---|---|
| 0–20 | No-reentrada, 26 §5 | Tabla A/CD/AB: score, información nueva, cobro. Explicar por qué repetir no ayuda. |
| 20–40 | Instancia heterogénea, 26 §7 | Tabla de p y u; dibujar AEF y separar sus cuatro conteos. Reconstruir la rama R=1. |
| 40–55 | π_L y λ, 26 §8 | Explicar qué se penaliza, cuándo se abandona y por qué cambiar λ cambia la política. Leer sus dos ratios. |
| 55–75 | Cruces, 27 §2–3 | Calcular 1.30 frente a 1.45 desde AB=1 y después 0.42×0.15=0.063. Verificar al menos dos perfiles con ABC/ABD/ACD. |
| 75–85 | Límites de las conclusiones | Distinguir fallo de un score, fallo de una batería, falta de submodularidad y pérdida por laminaridad. |
| 85–100 | Ensayo | Presentar en diez minutos y usar cinco para reparar la explicación menos clara. |

Para 45 minutos: no-reentrada 10, seis personas 15, cruces 10, ensayo 10. La comparación histórica de convenciones queda como referencia; no hace falta repetir sus cuentas completas.

## Cuatro ejemplos y su mensaje

### A. No-reentrada: reconocer utilidad no basta para cobrarla

AB tiene un infectado; q_C=q_D=0.3 y u=1. El score de masa sana dentro del pool da V(A)=0.5, V(CD)=0.6 y V(AB)=1. El retest maximiza el score aunque aporta información cero. Bajo posterior-zero y con una prueba restante, refinar A cobra 1 con certeza; CD cobra 0.18 en esperanza y repetir AB cobra cero.

La tabla histórica de costo del 26 usa acreditación estricta: por eso muestra 0.5 para la recompensa de A y costos 1, 1.63 y 2.5. No convertir esos costos automáticamente a posterior-zero. El menú canónico actual excluye el retest; el ejemplo explica por qué fallaba el score y por qué surgió la idea de costo local.

**Frase oral:** «Este score confunde tener un sano dentro de un grupo con haber elegido una acción que permita identificarlo».

### B. Seis personas: una batería puede fallar aunque un score funcionara bien antes

p=(0.9,0.825,0.875,0.8,0.95,0.85), u=(2,1,1,1,4,2), B=3, G=4. El óptimo laminar abre AEF, con valor 1.0645140625. Que AEF salga limpio tiene probabilidad 0.00075; ese no es su principal atractivo.

Después de R(AEF)=1, probar E cobra 4 en las dos respuestas: E sana aporta 4; E infectada demuestra que A y F están sanas, aportando 2+2. Es cobro inmediato de ese refinamiento, no el valor total de toda la rama. Si R(AEF)=3, la política deja ese grupo y busca utilidad en otro.

| Política | Utilidad esperada | / óptimo laminar | / óptimo general |
|---|---:|---:|---:|
| π_M, π_C y π_R | 0.700000 | 0.657577 | 0.630367 |
| C3 | 0.69904125 | 0.656676 | 0.629504 |
| π_L, λ=0.001, horizonte 3, no-parálisis | 1.026265 | 0.964069 | 0.924177 |

Estas cifras se reevaluaron. La corrección del score inmediato no cambia este caso. El contraejemplo afecta a las cuatro políticas concretas, no a cualquier greedy imaginable. C3 fue diseñado y entrenado en un régimen más limitado; su buen resultado allí no garantiza buen rendimiento heterogéneo.

**Frase oral:** «La primera prueba puede pagar por las continuaciones que abre. Tenemos que explicar esas ramas, no solo mirar la probabilidad de que el pool esté limpio».

### C. Cruces: un buen óptimo laminar todavía puede perder

En n=4, q=0.3, B=3, G=2, el óptimo laminar vale 1.074 y el general 1.137. Tras R(AB)=1, el laminar tiene continuación 1.30; cruzar AC permite 1.45. Ese estado ocurre con probabilidad 0.42, así que la ganancia inicial es 0.063.

En n=4, q=0.5, B=3, G=3, laminar vale 1.75 y general 2. Las pruebas fijas ABC, ABD y ACD distinguen los 16 perfiles. El general alcanza toda la utilidad sana esperada, que es una cota superior. Este certificado es independiente de Bellman.

En las seis personas de B, el general vale 1.1104640625 y laminar/general=0.958621. El ratio de π_L cambia al cambiar de comparador: 0.964069 frente al laminar y 0.924177 frente al general.

Los árboles permiten localizar la ganancia: tras R(AEF)=1, el laminar prueba E y el general cruza DE; la mejora ponderada desde el inicio es 0.004195. Tras R(AEF)=2, el laminar prueba F y el general cruza ADE; aporta 0.041755 más. Las ramas 0 y 3 empatan. En total, 0.004195+0.041755=0.045950. El refinamiento que cobra 4 en ambas respuestas pertenece al óptimo **laminar** de esa rama, no es la decisión del óptimo general.

**Frase oral:** «Estamos midiendo dos pérdidas distintas: la de la política dentro de laminar y la de restringir las pruebas a esa clase».

### D. La información puede aumentar el valor de una misma prueba

En el Acto 4 del 24, q=0.05, u=1. Probar A da 0.05 antes del historial. Después de observar dos infectados en ABC, da 1/3. Este testigo también funciona bajo posterior-zero: si A sale infectada, B y C todavía no quedan identificadas.

Esto contradice rendimientos marginales siempre decrecientes bajo información adicional. Por tanto, no basta aplicar automáticamente una garantía basada en submodularidad adaptativa. El ejemplo no demuestra que no pueda existir ninguna otra garantía.

## Guion para presentar en 15–20 minutos

| Tiempo | Pantalla | Qué decir |
|---|---|---|
| 0–1 | 26, portada | «Quiero ordenar los avances por los ejemplos que hicieron cambiar nuestras reglas de decisión». |
| 1–5 | 26, §5 | Presentar estado AB=1, tres acciones y por qué V elige una repetición inútil. Explicar el papel del costo local. |
| 5–10 | 26, §7 | Mostrar parámetros, menú resumido y rama R(AEF)=1. Distinguir tres políticas que abren F de C3 que abre ADEF. |
| 10–12 | 26, §8 | «π_L mejora esta instancia. Con λ=.001 alcanza 0.964 del óptimo laminar; con λ=.5 vuelve a 0.658». El horizonte y λ forman parte del algoritmo. |
| 12–16 | 27, §2–3 | Mostrar un cruce útil con q=.3 y el certificado de los tres tríos con q=.5. |
| 16–18 | 27, §4–5 | Nombrar los dos comparadores y pedir a Francisco que priorice una pregunta teórica. |
| Reserva | 24, Acto 4 | Usar el testigo 0.05→1/3 si surge la pregunta sobre garantías greedy. |

Para diez minutos: 2 en no-reentrada, 4 en seis personas, 3 en cruces, 1 en la pregunta final. Dejar el barrido histórico de α y el portafolio de 7,000 instancias fuera del relato principal: requieren más contexto del que aportan aquí.

## Preguntas de ensayo

1. AB tiene conteo 1. ¿Qué información nueva aporta repetir AB? ¿Por qué el score antiguo lo premiaba?
2. En AEF con conteo 1, ¿qué se cobra inmediatamente en cada resultado de probar E?
3. ¿Por qué una primera acción con Q=0.872 no implica que la heurística que la elige cobre 0.872?
4. Si C3 y π_L abren ADEF, ¿cómo pueden terminar con valores distintos?
5. ¿Por qué multiplicamos 0.42 por 0.15 en el ejemplo de cuatro personas?
6. ¿A qué comparador se refiere el 0.964 de π_L? ¿Cuál es su ratio frente al general?
7. ¿Qué demuestra el diseño ABC/ABD/ACD que no requiere confiar en el solver?
8. ¿Qué falta para convertir cualquiera de estos ratios observados en una garantía universal?

<details><summary>Respuestas para revisar después del intento</summary>

1. Ninguna; valora la masa sana del pool, no el efecto de la acción sobre lo que puede identificarse.
2. E sana: 4. E infectada: A y F sanas por deducción, 2+2=4.
3. Q usa continuación óptima; la heurística puede tomar malas decisiones después.
4. La política incluye las decisiones posteriores al conteo, no solo la apertura.
5. 0.42 es la probabilidad de llegar a AB=1; 0.15 es la mejora de utilidad esperada condicionada a estar ahí.
6. Al laminar; frente al general es aproximadamente 0.924177.
7. Que tres pruebas cruzadas identifican todos los perfiles de cuatro personas y alcanzan la cota superior de utilidad.
8. Un enunciado con hipótesis, parámetros y algoritmo especificados, acompañado de una prueba para todas las instancias del alcance declarado.

</details>

## Evidencia y reproducción

[Tabla de ocho casos](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/results/precio_laminar_2026-09-21/comparacion.csv), [fracciones, árboles y huellas de fuentes](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/results/precio_laminar_2026-09-21/resultados.json), [enumerador por perfiles](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/bellman_perfiles.py), [script de experimentos](/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic/augmented/experimento_precio_laminar.py).

Desde la raíz del repositorio, usando el entorno del proyecto:

```bash
../venv/bin/python -m pytest augmented/tests_bellman_perfiles.py
../venv/bin/python -m augmented.experimento_precio_laminar
```

Validación realizada: 28 pruebas, con cuentas manuales, reconstrucción directa de 16 perfiles, cotejo contra el solver laminar de átomos, cotejo homogéneo contra la DP conjunta anterior, casos heterogéneos sembrados, evaluación de los árboles en cada perfil, casos de probabilidad cero y regresión del crédito del complemento. Son ejemplos y controles pequeños, no una cota de peor caso general. El máximo n de la tabla es 6 y el máximo presupuesto restante es 3.
