# Dos loops estilo autoresearch: diseños generales (log G) y scorer laminar (Q1) — 17 de octubre de 2026

Héctor pidió escribir y correr en Hermeneuta dos loops en los que un agente edita un algoritmo y un juez fijo lo mide con aritmética exacta. A diferencia del loop del 15 de octubre, que buscaba instancias, estos buscan diseños y políticas. Esta nota fija la especificación antes de la primera noche; el protocolo pide que A la apruebe y que cualquier récord se re-resuelva por vía independiente antes de llamarlo resultado.

## Loop 1: `augmented/autoresearch_logg/` — cuántos sanos acreditan B pruebas de tamaño ≤ G

Hipótesis de Francisco (13 oct): el óptimo general acredita del orden de $cB\log_2 G$ sanos y el laminar queda por debajo de $CB$. La cota superior $B\log_2(G+1)$ ya está derivada. Lo que nadie ha medido es cuánto alcanzan diseños explícitos a $G$ finito, fuera del techo de población de la malla.

- Juez fijo (`judge.py`): evalúa la política de `design.py` enumerando los $2^n$ perfiles con $q=1/2$, $u=1$, soft clearing. Objetivos fijos: $n=16$, $(B,G)\in\{(4,2),(4,4),(4,8),(4,16),(8,8),(8,16)\}$. Reporta por objetivo el valor esperado entre $B$ y entre $B\log_2(G+1)$. Score = suma de los seis valores por prueba.
- Puertas: pool inválido, más de $B$ pruebas, política no determinista (se evalúa dos veces), valor por encima de la cota de información. Anclas al arranque: individuales, par, detector ABC/ABD/ACD, y el árbol óptimo irrestricto del solver de referencia en $(4,3,3)$, $(6,3,3)$ y $(7,4,3)$ reproducido por el evaluador del juez.
- Línea base: bloques de cuatro con el detector de tres pruebas más individuales, score 3.625. Prueba de humo local (dos experimentos del agente): 3.625 → 3.766 (pares adaptativos en $G=2$) → 4.370 (DP exacto sobre bloques secuenciales de hasta cuatro personas; 0.742 por prueba en $B=4$, 0.751 en $B=8$). El valor por prueba queda plano para $G>4$ porque el DP no cruza bloques; ese es el frente que la noche debe atacar.
- Tests: `tests_judge_logg.py` (6). Tiempo por experimento: 16 s local, 31 s en Hermeneuta, dominado por las anclas.

## Loop 2: `augmented/autoresearch_q1/` — una regla laminar barata cerca del óptimo laminar

- Juez fijo (`judge.py`): evalúa el scorer de `scorer.py` sobre una batería de 125 celdas con óptimo laminar exacto conocido: las 32 de la malla del 14 oct ($q=1/2$), 81 de las mallas del 6 oct ($q\in\{0.3,0.5,0.7\}$, $n=4$–$6$) y 12 heterogéneas ($n=5$, $q$ y $u$ aleatorios de las semillas). Evaluación exacta en racionales con memoización; candidatos restringidos a pools laminares de tamaño ≤ G. Score = peor cociente valor/óptimo; desempatan la media y el número de celdas exactas.
- Puertas: pool no candidato (cruce, tamaño), valor por encima del óptimo, caída del scorer. Anclas: individuales, árbol laminar óptimo del solver reproducido en cuatro celdas (incluida una heterogénea), rechazo de un cruce.
- Línea base: greedy miope exacto; peor cociente 0.720 (celda $q=0.3$, $n=6$, $B=3$, $G=3$), media 0.873, 31 celdas exactas, 1.3 s por experimento. La batería es barata, así que caben reglas con mirada hacia adelante; el programa pide reportar el costo.
- Tests: `tests_judge_q1.py` (7).

## Ejecución en Hermeneuta

Directorio `~/agent-workspace/groupcount-loops-20261017/` con el solver de referencia, los dos loops y los CSV de la batería; intérprete `.venv311` (Python 3.11, numpy 2.4.6). CLI `claude` 2.1.302 instalado en `~/.local/bin`. Lanzamiento como unidad transitoria de usuario con `augmented/autoresearch_vps/launch.sh logg|q1|all [N]` (memoria 1.5 GiB, 1.5 CPU, 10 h, nice 10); seguimiento con `status.sh`. La autenticación va en `~/.config/groupcount/claude.env` (token de `claude setup-token`); no se copian credenciales locales.

Qué no se afirma: ningún valor de estos loops es un teorema ni una cota; el mejor diseño por objetivo y la mejor regla se re-resuelven por vía independiente antes de la sesión del 20 de octubre. Los TSV son de solo añadido; el agente edita un único archivo por loop.

## Resultados de la primera noche (18–19 de octubre)

La unidad del 18 de octubre murió por memoria a los 25 minutos (pico 1.5 GiB: el DP del agente con bloques de 6 más el propio CLI). Se relanzó el 19 de octubre con el tope del juez en 700 MB, `OOMPolicy=continue` y sesiones nuevas por ronda de 10 experimentos. La segunda unidad terminó sin fallos en 2 h 34 min de CPU, pico 978 MB. Registros en `results/sesion_2026-10-19/loops_hermeneuta/`; TSV y notas del agente en cada carpeta del loop.

Loop 1 (20 experimentos del agente, 8 descartados por límite de tiempo o memoria). Mejor diseño: DP exacto sobre bloques secuenciales de hasta 6 personas, pools de hasta 5 dentro del bloque, caché acotada. Sanos acreditados por prueba: $G=2$: 0.688; $B=4$: 0.836 en $G=4$ y 0.845 en $G=8$ y $16$; $B=8$: 0.826 en $G=8$ y $16$. La curva sube de $G=2$ a $G=4$ y queda plana después. Todos los diseños encontrados cruzan pools dentro del bloque y nunca entre bloques; el agente reporta que la clase de bloques secuenciales está en su pared de memoria y tiempo. El dato no muestra crecimiento logarítmico en $n=16$ con $B\le8$; tampoco lo descarta, porque faltan diseños con cruces entre bloques y presupuestos proporcionales a $G$. Siguiente estructura: cruces entre bloques o canonicalización por simetría del estado.

Loop 2 (31 experimentos del agente). Mejor regla: en la raíz se evalúan todos los pools laminares candidatos; cada uno se puntúa con su ganancia inmediata exacta más una mirada hacia adelante de profundidad 4 en la que cada nivel interno toma los cuatro pools de mayor ganancia y además el mejor pool de cada tamaño; la hoja es el greedy. Peor cociente sobre la batería: 0.720 (greedy) → 0.997; media 0.99998; 124 de 125 celdas exactas; 150 s por batería. La única celda no exacta es $q=0.7$, $n=6$, $B=3$, $G=4$. La lección del agente: ampliar por rango de ganancia no mueve la peor celda; diversificar por tamaño de pool sí. Ningún valor superó el óptimo en ninguna corrida.

Qué falta antes de decir algo: re-resolver por vía independiente el mejor diseño por objetivo del loop 1 y la mejor regla del loop 2; medir la regla en $n=50$ contra el rollout con Monte Carlo (`augmented/laminar_benchmarks.py`), porque la batería es pequeña y la regla se ajustó sobre ella. Los TSV son de solo añadido y nada está commiteado.
