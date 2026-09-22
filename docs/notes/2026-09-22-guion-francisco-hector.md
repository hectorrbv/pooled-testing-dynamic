# Héctor · cierre y guion para Francisco

Preparado el 21-sep-2026 para la sesión del **martes 22, 19:00 Praga**.
Sync previsto a las **16:00** para congelar el guion.

**Reparto actualizado tras integrar el guion conjunto del 22-sep:** Héctor
presenta este relato en **10 minutos**; después A presenta la proposición de
brecha de convención en **7 minutos**; Héctor retoma **3 minutos** para
validación, ancla y C1. Las cinco preguntas conjuntas están en
[guion de sesión](2026-09-22-guion-sesion.md).

Tu parte es explicar qué objeción motivó cada cambio, qué mide cada número y
qué pregunta queda abierta. El recorrido principal es **contraejemplo → π_L →
0.9307**. Los mapas son evidencia adicional para la discusión.

## Estado de los cinco encargos

| Encargo | Resultado y alcance |
|---|---|
| 1. BM17 contra enumerador | **296/296 pasan, diferencia exacta cero.** `tests_bm17.py` compara átomos con `pathwise_enum.py`, que enumera perfiles e impone compatibilidad con todo el historial. Matriz local reproducible: 8 anclas + 120 estados heterogéneos × 2 convenciones + 48 raíces homogéneas. No se afirma haber recuperado el script privado de 296 casos mencionado en el despacho. |
| 2. Reserva del ancla | Default G0: **k=3**, n=48, cota **0.914742409666**. `convencion='strict'` conserva k=2 y 0.806288515541. **33 tests del acid test pasan.** Los checks históricos de comportamiento de scorers siguen usando su evaluador estricto, identificado en el módulo. |
| 3. Mapa 1 | **665 celdas completas.** q=.05,.10,…,.95; G=2,3,4,5,8; B=2,…,8; u=1; n=B·G. En **665/665** empezar con un grupo supera empezar con un individuo. Se guardan todos los tamaños óptimos, incluidos empates entre grupos, y el valor de cada alternativa relevante. |
| 4. Mapa 3 | Ancla completa: límite de **2,500,000 estados** en 25.45 s, sin OPT certificado. Frontera resuelta: **OPT_lam=0.865393951284**, n=24, G=8, B=6. En la población original del ancla, restringir las acciones a G≤8 ya alcanza **1.123282733640** con B=7. |
| 5. Atlas y C1 | `run_showcase` separa gana / empata / pierde; CSV y sidecar regenerados; C1 de §9 corregido. Greedy global **10.9 / 56.1 / 33.0%**. |

Validación conjunta: **450 tests pasan** (BM17, acid test, ambos backends,
CBS por enumeración, perfiles, solvers y procedencia). Bellman por tipos usa
estados exactos y aritmética float64; el oráculo de perfiles y BM17 comparan
fracciones exactas. El backend compilado conserva la misma recursión y se cotejó
con el Python y con el solver etiquetado, incluidas las acciones de raíz.

Los artefactos registran convención, población, parámetros, commit base,
árbol de trabajo modificado y hashes de las fuentes. Se generaron antes del
commit de cierre; los hashes identifican las fuentes efectivamente ejecutadas.

## Guion de 10 minutos con el notebook 26

Abre `augmented/notebooks/26_esencial.ipynb`, ya ejecutado. Recorrido:
**§5 → §7 → §8 → §10**. §11 y notebook 27 son apoyo, no hay que recorrerlos enteros.

### Pantallas y foco para hoy

| Tiempo | Pantalla | Mensaje que debes dejar claro |
|---|---|---|
| 0:00–1:30 | 26 §5, tabla posterior-zero | Tener utilidad sana dentro del grupo no equivale a poder cobrarla. Incluso excluyendo repetir AB, V prefiere CD a A y se equivoca. |
| 1:30–4:00 | 26 §7, instancia y rama AEF → E; tabla de valores de §8 | Las cuatro políticas concretas quedan alrededor del 66% del óptimo laminar. Explica una rama para mostrar la continuación que compra la primera prueba. |
| 4:00–6:30 | 26 §8, fórmula y fila de π_L | El costo depende del resultado: una exploración puede terminar después de una prueba. π_L alcanza 96.41% del óptimo laminar en esa misma instancia. |
| 6:30–8:30 | 26 §10, curva de λ | En otra instancia, de siete personas, la rejilla encuentra λ=1.238469 y ratio 0.930703. La calibración forma parte de la política. |
| 8:30–10:00 | 26 §10, cierre | Qué respalda el 0.9307 y qué falta para una garantía; pregunta sobre la especificación de λ. Los resultados de validación y mapas van en tu bloque adicional de 3 min. |

**Frase de apertura:** «Voy a mostrar qué falla en las reglas que probamos,
cómo π_L recupera valor al considerar el gasto de pruebas según el resultado,
y por qué la elección de λ afecta al 0.9307 que reportamos».

Usa Bellman como referencia con una frase: «Calculamos el óptimo laminar por
programación dinámica para medir la pérdida de las heurísticas». El árbol y
la comparación de valores sostienen el argumento. Las derivaciones de Bellman,
el barrido histórico de α y las convenciones estrictas quedan para preguntas.

**Dos distinciones al hablar:** 0.70 es utilidad esperada por partida, mientras
que 0.6576 es su razón frente al óptimo; 0.9641 y 0.9307 pertenecen a instancias
distintas. Todos esos ratios del relato principal usan el óptimo laminar.

**Material de apoyo:** notebook 27 §2–3 si surge la pregunta de cuánto cuesta
imponer laminaridad; notebook 26 §11 para los mapas. El guion de 15–20 minutos
del 21-sep fue sustituido para esta sesión por este recorrido de diez minutos.

**0:00–1:30 · Qué estaba fallando (§5).**

«Queremos elegir pruebas para acreditar utilidad sana con un presupuesto
limitado. El score viejo contaba utilidad sana que existe dentro del grupo,
aunque la acción no la hiciera cobrable. Por eso podía recomendar repetir AB
cuando ya sabíamos que su conteo era uno. El contraejemplo obliga a valorar
qué cobra una acción y qué continuación deja al presupuesto restante».

En el ejemplo AB con R=1, una prueba restante y utilidades uno: repetir AB
cobra cero; probar A cobra uno bajo posterior-zero en cualquiera de sus
resultados. Si A está infectado, B queda sano por deducción. Ese es el hecho
que debes poder explicar a mano. No hace falta presentar toda la curva de α.

**Si Francisco objeta que repetir una prueba debe excluirse:** «Sí, el solver
actual ya lo excluye. Pero el defecto de V persiste: V(A)=0.5 y V(CD)=0.6,
con C y D nuevos y q=0.3; el score elige CD. Con una prueba restante, A cobra 1
y CD cobra solamente 2·0.3²=0.18 en esperanza». Esta comparación usa únicamente
acciones informativas, posterior-zero y el mismo presupuesto.

**1:30–4:00 · El contraejemplo que afectó a la batería (§7).**

«Pasamos de reparar un score a probar varias reglas completas. En seis
personas, B=3 y G=4, las tres políticas del companion valen 0.7 y C3 vale
0.699041. El óptimo laminar vale 1.064514: estamos alrededor del 65.7%».

Muestra la raíz AEF y una continuación. Tras R(AEF)=1, probar E cobra
inmediatamente 4 en ambas ramas: E sana aporta 4; E infectada deja A y F
sanos, que aportan 2+2. La prueba inicial compra una continuación útil.
El número 4 es el cobro inmediato en ese estado, no todo el valor del árbol.

Si pregunta por la batería: π_M elige el mejor cobro inmediato; π_C valora
planes por utilidad/pruebas reservadas y mantiene el compromiso con el bloque;
π_R usa densidad y reevalúa tras observar; C3 combina cobro inmediato,
promesa ajustada al presupuesto y una reserva para personas aún sin probar.
Son las cuatro implementaciones evaluadas, no todas las heurísticas posibles.

**4:00–6:30 · Qué cambia con π_L (§8).**

«π es una política: dice qué acción tomar tras cada historial. π_L compara
proyectos locales por la utilidad esperada que consiguen menos λ por cada
prueba que realmente consumen. Permite abandonar una exploración local cuando
el resultado ya la hace inútil. Así reconoce una exploración que falla barato».

\[
I_\lambda(C)=\sup_{\pi\text{ local}}\mathbb E[U_{\rm cobrada}-\lambda T].
\]

λ es un precio de planificación; **no se resta de la utilidad final reportada**
al comparar políticas. El índice penalizado elige; la evaluación mide utilidad
bruta acreditada. Esta implementación usa horizonte local min(b,3), replanifica
y, si ningún proyecto tiene índice positivo, aplica no-parálisis: elige el mejor
cobro inmediato. No confundir el abandono de un proyecto con terminar la partida.

En el ejemplo de seis personas, λ=.001 da **1.026265**, el **96.41% del óptimo
laminar**. Esta mejora motivó agregar π_L al portafolio.

**6:30–8:30 · De dónde sale 0.9307 (§10).**

«La candidata registrada combina tres personas casi seguras, tres premios
grandes con infección .975 y una persona intermedia. Con n=7, B=3, G=4, el
óptimo del harness vale 6.226133. π_L obtiene 5.794681: ratio 0.930703».

El valor se reprodujo el 21-sep con **λ=1.238469**, horizonte 3, no-parálisis
y rejilla geométrica de razón 1.4 construida desde el prior y las utilidades.
El gráfico muestra por qué la rejilla importa: puede saltarse la ventana buena.
La selección ex ante maximiza el valor esperado de cada política, sin usar OPT;
OPT es la referencia posterior para medir su calidad.

**Frase defendible:** «El 0.9307 está reproducido en la candidata registrada
con esta especificación. La búsqueda lo propone como conjetura empírica;
no tenemos una garantía universal del 93%». La verificación de hoy no vuelve
a evaluar las aproximadamente 7,000 instancias de la nota histórica.

**8:30–10:00 · Cierre del relato (§10).**

«El contraejemplo muestra una limitación de las cuatro reglas evaluadas;
π_L recupera gran parte del valor en esa instancia. El 0.9307 se reprodujo
en otra candidata con la rejilla declarada. La pregunta ahora es qué regla
para λ, horizonte y no-parálisis fijamos antes de buscar una garantía».

Da paso al bloque de A sobre la brecha de convención.

## Bloque adicional de Héctor: 3 minutos, después de A (26 §11)

**Primer minuto:** el solver se cotejó con dos enumeradores independientes.
Tras integrar las verificaciones de A, `tests_bm17.py` y
`tests_brecha_convencion.py` pasan **626 tests**, ejecutados el 22-sep.
La segunda suite incluye óptimos y primeras acciones forzadas para n≤5, B≤3;
la primera añade, entre otros casos, raíces homogéneas hasta n=6.

**Segundo minuto:** G0 permite k=3 en el ancla, con **cota 0.914742**.
La frontera del Mapa 3 tiene óptimo 0.865394; la variante CBS explícita que se
detiene al primer cobro vale 0.727127. En el ancla, una política factible con
G≤8 vale 1.123283 y supera a esa variante CBS, que vale 0.939440.
El óptimo completo con G=16 sigue sin certificarse.

**Tercer minuto:** C1 ya separa victorias, empates y derrotas: greedy global
10.9 / 56.1 / 33.0%. Remite a las cinco preguntas del guion conjunto.

«El enumerador independiente respalda el solver en los casos probados y el
Mapa 1 respalda a escala la preferencia inicial por grupos. El Mapa 3 muestra
que la cota del ancla tiene holgura. Quedan por fijar la política CBS completa
y la especificación uniforme del portafolio para discutir garantías».

Preguntas concretas:

1. ¿Qué hipótesis de población y homogeneidad llevará la proposición de primera acción?
2. ¿Qué continuación incluye cover-then-bisect cuando encuentra sanos antes de agotar B?
3. ¿Fijamos horizonte, no-parálisis y rejilla de λ antes de volver a medir toda la batería?

## Cómo leer el Mapa 3 sin confundir cota con política

| Caso | Cota CBS | CBS explícita que se detiene al primer cobro | Óptimo laminar calculado |
|---|---:|---:|---:|
| Ancla: n=48, G=16, B=7 | 0.914742 | 0.939440 | No certificado; **≥1.123283**, por la política restringida a G≤8 |
| Frontera: n=24, G=8, B=6 | 0.708011 | 0.727127 | **0.865394**; primera acción de tamaño 8 |

La CBS evaluada cubre hasta el primer grupo con algún sano; lo biseca,
prefiere el hijo izquierdo si ambos contienen sanos y termina al primer cobro.
Cuenta toda utilidad acreditada por el test y por su complemento. Su expectativa
se calculó con fracciones y se verificó por enumeración de perfiles pequeños.

**Conclusión precisa:** esa versión de CBS es subóptima en ambos casos. La
cota 0.914742 tampoco es ajustada. No se ha evaluado toda posible extensión
que continúe aprovechando las pruebas sobrantes; “CBS” necesita esa definición
antes de atribuirle un único valor. El ancla completa G=16 sigue abierta
computacionalmente. Los valores son utilidad esperada, no probabilidades,
y por eso pueden superar uno.

El Mapa 1 usa población suficiente n=B·G (también vale para n mayor). No
extender el claim estricto a toda población finita: n=4, q=.3, G=2, B=3
admite empate entre abrir un individuo y un par, ambos con valor 1.074.

El Mapa 3 mide **OPT dentro de la clase laminar por trayectoria**. El precio
de imponer esa clase frente a pools generales se estudia aparte en el notebook 27.

## Atlas: cifras corregidas y alcance

Las tres cuotas globales de greedy son 10.88 / 56.10 / 33.02%; las de rollout,
28.32 / 46.33 / 25.35%. Se usa tolerancia 1e-9 en la razón.

La región alta del código es `base_p >= 0.6`, 1,008 casos: greedy
4.46 / 91.57 / 3.97%; rollout 37.60 / 60.62 / 1.79%. Se conservó esa definición.
Los 1.4 / 97.5 / 1.1% anticipados en la nota de tareas corresponden a
**base_p >= 0.7**, 720 casos. Esa subregión se agregó como fila explícita del
CSV para conservar ambas definiciones. C1 identifica los umbrales.
Es un recuento del atlas histórico; no una migración de sus políticas a G0.

## Archivos y reproducción

- Notebook principal: `augmented/notebooks/26_esencial.ipynb`.
- Mapa 1: `results/mapa_homogeneo_1.csv`, `.csv.meta.json` y `.png`.
- Mapa 3: `results/mapa_homogeneo_3.csv`, `.csv.meta.json` y `.runtime.json`.
- Portafolio: `results/verificacion_constante_093.csv` y `.csv.meta.json`.
- Atlas: `augmented/data/laminar_week/showcase_regions.csv` y `.csv.meta.json`.

Desde la raíz del repositorio, usando el Python del entorno del proyecto:

El backend compilado usa la dependencia opcional `requirements-types.txt`;
sin ella, el solver Python y el enumerador funcionan y pytest salta con razón
visible únicamente las comprobaciones que necesitan Numba.

```bash
python -m pytest augmented/tests_bm17.py augmented/tests_acid_falsificador.py augmented/tests_bellman_tipos.py augmented/tests_mapa_ancla.py augmented/tests_bellman_perfiles.py augmented/tests_solvers.py augmented/tests_provenance.py -ra
python -m augmented.mapas_homogeneos
python -m augmented.mapa_ancla
python -m augmented.verificar_constante_093
python -m augmented.experiments_laminar_week showcase
python augmented/notebooks/build_cierre_22sep.py
```

El último comando actualiza las celdas del cierre y la figura del Mapa 1;
hay que ejecutar después el notebook para renovar sus salidas. No regenerar
el notebook con builders antiguos durante la presentación.
