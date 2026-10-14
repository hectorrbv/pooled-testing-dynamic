# Encargos del 6-oct: binario-gana-a-laminar y el precio de la laminaridad en q = ½ (diagnóstico IA, 2026-10-12)

Actualización del 14-oct: las secciones 1–3 conservan el corte original. La [sección 4](#4-actualización-de-héctor-14-oct) registra el contraste independiente, los notebooks locales y la familia posterior con cociente límite 10/9. La comprobación computacional no equivale a la firma personal de Héctor ni a una revisión matemática de Vladimir.

**Estatuto §25: diagnóstico en casos exactos chicos.** Solver nuevo `augmented/dp_perfiles_binario_conteo.py` (DP exacta sobre conjuntos de perfiles, canal binario o de conteo, pools libres ≤ G, posterior-zero, Fractions), escrito por el soporte IA y **no validado por B**; reproduce cinco valores conocidos: 387/500, 1137/1000 y 2 (aumentado irrestricto, `results/precio_laminar_2026-09-21/`) y 3/5, 9/10 (binario dinámico, Thm 3.3(i) en q = 0.3). El óptimo laminar aumentado sale de `augmented.bellman_perfiles.BellmanPerfiles(laminar=True)` (validado en septiembre). Scripts: `augmented/busqueda_binario_vs_laminar.py`, `augmented/malla_precio_laminar_q.py`, `augmented/arbol_testigo_binario_vs_laminar.py`. Datos: `results/encargos_2026-10-06/`.

## 1. Encargo: un ejemplo donde el binario dinámico óptimo gana a laminar [D 35:58, 57:22]

Por V4 del acta del 6-oct el ejemplo solo puede tener Λ > 1 (con Λ ≤ 1, Thm 3.3(i) da Binary = NonOverlap ≤ Laminar). Se muestrearon instancias con Λ > 1: n ∈ {3, 4, 5}, B ∈ {2, 3}, G ∈ {2, 3}, q_i y u_i heterogéneos (tres semillas, 240 s cada una).

| Instancias con Λ > 1 | Binario > laminar | Con q homogénea | Razón mínima laminar/binario |
|---|---|---|---|
| 6,800 | 63 (0.9%) | 0 | **0.978** (n = 4, B = 2, G = 2) |

Testigo más limpio: n = 3, B = 2, G = 3, u ≡ 1, q = (0.9, 0.7, 0.7), Λ = 1.75.

| Clase | Valor | Árbol óptimo |
|---|---|---|
| Binario dinámico | 2079/1000 = 2.079 | {a,b}; negativo (0.63): {c}; positivo (0.37): **cruce {a,c}**, paga 2 con prob. 0.511 |
| Laminar aumentado | 1031/500 = 2.062 | {a,b}; conteo 0: {c}; conteo 1 (0.34): refina {a}, paga exactamente 1 por deducción; conteo 2: {c} |
| Aumentado irrestricto | 2121/1000 = 2.121 | {a,b}; conteo 1: cruce {a,c} (vale 1.1735 con conteos); conteo 0 o 2: {c} |

Mecanismo: tras el positivo, a es el sano probable (0.9 contra 0.7) y c es fresco y probablemente sano; el cruce {a,c} apuesta por dos personas (2 × 0.511 = 1.02) y gana a la deducción segura de una (1.00). Con todo q_i ≤ ½ la deducción segura siempre gana: ese es el régimen de la Proposición. Son las "dos fuerzas" de Francisco [D 31:41–32:31], medidas. Lo que **no** se dice: nada sobre el peor caso ni sobre la constante; el ¼ no se acerca en n ≤ 5.

## 2. Encargo: el factor laminar vs aumentado irrestricto en q = ½ [D 42:21–45:13, 57:50]

Malla exacta, u ≡ 1, q ∈ {0.3, 0.5, 0.7}, n ∈ {4, 5, 6}, G ∈ {2, 3, 4}, B ∈ {2, 3, 4} (81 celdas; `precio_laminar_malla_n4_5.csv`, `precio_laminar_malla_n6.csv`). Peor razón laminar/irrestricto por (n, q):

| n | q = 0.3 | q = 0.5 | q = 0.7 |
|---|---|---|---|
| 4 | 0.895 (G3,B3) | **0.875** (G3,B3) | 0.937 (G3,B3) |
| 5 | 0.874 (G3,B3) | **0.857** (G3,B3) | 0.895 (G4,B3) |
| 6 | 0.860 (G3,B4) | **0.833** (G3,B4) | 0.879 (G4,B3) |

q = ½ es el peor de los tres en todo n, y la pérdida crece con n y con B (0.875 → 0.857 → 0.833). Con B = 2 la razón es 1 en toda celda (con dos pruebas no hay cruce rentable, cf. caso par_B2 del 21-sep). Consistente con la conjetura de Francisco; no dice nada sobre el límite ni sobre G grande (la cota de información V5 sugiere O(log G); sin evidencia aún).

## 3. Lo que sigue

- B valida el solver (dos vías: contra `BellmanPerfiles` sin laminar en la malla; contra `static_binary_value` en instancias con B pequeño) antes de que cualquier número de aquí entre al guion con etiqueta distinta de "diagnóstico".
- Extender la malla en n (7–8) y G = 5 para ver si la razón en q = ½ sigue bajando; buscar el peor testigo binario-vs-laminar con n ≤ 6 y u libre.
- Preguntas para el 13-oct: (31) el ejemplo debe tener Λ > 1 — confirmado en la búsqueda; (32) ¿O(log G) en q = ½?; (33) la referencia de las matrices decodificadoras.

## 4. Actualización de Héctor — 14-oct

### Contraste independiente del encargo del miércoles

Se ejecutó `python -m augmented.verifica_encargos_oct14` después del pull hasta `698d1d2`. El informe completo, con filas, fracciones, semilla y hashes de fuentes, está en [validacion_cruzada.json](../../results/sesion_2026-10-14/validacion_cruzada.json).

| Comprobación | Casos | Resultado |
|---|---:|---|
| Columna general de los dos CSV frente a `BellmanPerfiles(laminar=False)` | 81 | Coincidencia exacta en todos |
| Binario de los CSV frente a `BellmanBinario`, con hard clearing explícito y evaluación del árbol sobre cada perfil | 20 | Coincidencia exacta en todos |
| B=1: DP nuevo, Bellman binario y `static_binary_value` | 20 | Coincidencia racional entre los dos primeros; error ≤ 1e-10 contra el control flotante |

Los 20 casos son una selección reproducible, semilla 20261014, de los tres CSV binarios. No se afirma haber validado las 6,800 filas. En el régimen de priors interiores usado aquí, una persona puede identificarse sana a partir de resultados binarios si y solo si pertenece a algún pool negativo: por eso el crédito por inferencia del DP importado coincide con el hard clearing del comparador independiente. Esta equivalencia no se extiende sin revisión a priors con certezas iniciales.

Además, pasaron las 989 pruebas de `tests_bellman_binario_exacto.py`, `tests_busqueda_brecha_ampliada.py` y `tests_brecha_convencion.py`. Esto incluye el certificado simbólico de la familia siguiente. No se ejecutó en esta actualización el barrido aleatorio de 120 segundos de la proposición; su lectura y firma humana siguen pendientes.

### Separaciones ya disponibles antes del plan nuevo

El [notebook 29](../../augmented/notebooks/29_tres_clases_y_contraejemplo.ipynb) usa un testigo más separado y con grupos más pequeños: n=3, B=G=2, q=(0.95,0.8,0.8), u=1. Binario = 2.432; laminar = 2.366; general = 2.448. Sus árboles y las seis posibles primeras acciones laminares están contrastados por enumeración independiente. El testigo de Vlad también se reprodujo: 2.079, 2.062 y 2.121, respectivamente.

La búsqueda ampliada local y en Hermeneuta produjo la familia

\[
n=4,\quad B=G=2,\quad q=(s,1/2,1/2,s),\quad u=(1,3,3,2),\quad 19/20\le s<1.
\]

Para toda esa familia, los óptimos son

\[
V_{\rm bin}=\frac52s(1+s),\qquad V_{\rm lam}=3+\frac32s^2,
\qquad \lim_{s\uparrow1}\frac{V_{\rm bin}}{V_{\rm lam}}=\frac{10}{9}.
\]

El binario abre BD. Si sale negativo, probabilidad s/2, acredita 5 y prueba AC, con utilidad adicional esperada 2s. Si BD sale positivo, prueba AD; el evento BD positivo y AD negativo tiene probabilidad conjunta s²/2 y acredita 3. Así obtiene (s/2)(5+2s)+3s²/2.

El laminar abre BC. Con conteo 0, probabilidad 1/4, acredita 6 y prueba AD; con conteo 1, probabilidad 1/2, prueba B e identifica al sano de BC, de utilidad 3; con conteo 2, probabilidad 1/4, prueba AD. AD acredita 3 con probabilidad s². El valor es (6+3s²)/4+3/2+3s²/4.

La optimalidad se comprueba comparando todas las primeras acciones y sus continuaciones con 454 desigualdades polinómicas sobre el intervalo, no solo evaluando los dos árboles propuestos. Código: `augmented/certifica_brecha_ampliada.py`; [certificado](../../results/sesion_2026-10-12/brecha_ampliada/certificados.json); [resumen de la búsqueda](../../results/sesion_2026-10-12/brecha_ampliada/resumen.json). En s=0.99 se obtiene 4.92525 frente a 4.47015; el cociente laminar/binario es 0.9075986. La familia aproxima 0.9 en esa orientación, mejor que el 0.978 del corte original. No demuestra una cota superior universal de 10/9. En s=1 cambia el problema: A y D ya son sanos conocidos y el óptimo laminar vale 6; la fórmula describe un límite por la izquierda.

La búsqueda incluyó 31 optimizaciones, 6,199,986 evaluaciones flotantes y 2,856 comparaciones exactas adicionales; 122 casos limitados por recursos quedaron sin certificar. Los evaluadores exactos y el certificado se mantuvieron separados de la búsqueda de candidatos. Son configuraciones seleccionadas hasta n=8, no una malla exhaustiva ni evidencia de máximo global.

### Relación con el plan y la sesión de estudio

Los notebooks 27–29 y sus datos quedan disponibles para el equipo. El 29 conserva los cinco bloques y sus figuras; las notas de estudio incorporan la familia posterior y el orden para preparar el 20-oct: árbol pequeño, intercambio general del par, argumento de información con q=1/2, y comparación de políticas.

Q2 sigue pendiente: ampliar la malla en q=1/2, u=1, G=2…5 y n=6–7, y redactar la cota de información en palabras de Vladimir. No está probado que q=1/2 sea el peor caso, ni una garantía independiente de G.

Q1 sigue pendiente en la malla nueva: las columnas greedy y rollout del plan no son π_L y π_ratio del notebook. `greedy_laminar_value` y `rollout_laminar_value` trabajan sobre una jerarquía balanceada fija; el comparador correcto del plan es `BellmanPerfiles(laminar=True)`, que impone laminaridad por trayectoria. La pérdida observada puede incluir tanto la elección de jerarquía como la regla. `laminar_augmented_value` de `laminar_benchmarks.py` optimiza una biblioteca fija y no debe reemplazar silenciosamente ese comparador.

Corrección numérica del plan: n=4, B=2, G=2, q=0.3, u=1 da T₄=1.2 ≥ 0.774 ≥ 0.6=T₂. El 1.074 corresponde al óptimo laminar con B=3; el general en ese caso vale 1.137.

El siguiente loop de búsqueda debe tomar esta familia como referencia binaria. Las misiones de precio laminar y crecimiento en G continúan abiertas. El push de estos archivos no significa que se hayan enviado a Francisco ni que haya comenzado el loop nocturno.
