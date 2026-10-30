"""Keep the presentation short; retain earlier figures in the study notebook."""
from copy import deepcopy
import nbformat as nb
from figuras_plan_29 import plan_figures
from respuestas_discusion_29 import add_answers


def align(presentation,study,cell,images,root):
    new_images,summary=plan_figures(root)
    images.update(new_images)
    previous=deepcopy(presentation.cells)
    first=previous[0]
    first.source=first.source.replace('Comparamos utilidades esperadas óptimas con los mismos', 'Comparamos políticas y óptimos con los mismos')
    first.source+='\n\nTres preguntas: binario frente a laminar; laminar frente a general; reglas eficientes frente al óptimo laminar.'
    binary=previous[3];binary.source=binary.source.replace('## 3.','## 2.',1)
    q2=cell(r'''
## 3. La pérdida laminar aumenta al pasar de seis a siete personas

$n\in\{6,7\}$, $B=1,2,3,4$, $G\in\{2,3,4,5\}$, $q_i=1/2$, $u_i=1$. Son 32 casos.

Cada celda muestra $100(1-V_{\rm lam}/V_{\rm aug})$: el porcentaje de utilidad que se pierde al exigir laminaridad.

![Pérdida por laminaridad para cada presupuesto y tamaño máximo de pool, separada por n=6 y n=7.](attachment:precio_laminar_G.png)

Con $B=4,G=3$, el general obtiene 3.25 y el laminar 2.5625 cuando $n=7$: una pérdida del 21.15%, frente al 16.67% con $n=6$. Con una o dos pruebas empatan en toda esta malla.

La pérdida no crece de forma monótona con $G$. Estos tamaños pequeños todavía no establecen la dependencia asintótica que busca Francisco.
''','precio_laminar_G')
    information=cell(r'''
## 4. ¿Por qué aparece log G?

$q_i=1/2$, $u_i=1$, estados independientes y $1\le B\le n$.

Una prueba de tamaño máximo $G$ tiene como mucho $G+1$ resultados. Con $B$ pruebas, la información obtenida está limitada por $B\log_2(G+1)$. Identificar a una persona sana fija uno de los bits del perfil.

El argumento da $V_{\rm aug}\le B\log_2(G+1)$. Como las individuales son laminares y obtienen $B/2$:

$$\frac{V_{\rm aug}}{V_{\rm lam}}\le 2\log_2(G+1).$$

Es el argumento que Vlad debe revisar y escribir para la proposición. Las 32 celdas respetan la cota; la demostración requiere el paso sobre información, explicado en las notas.

Para discutir: ¿qué construcción consigue una separación del mismo orden? Esa es la otra mitad de la dirección propuesta por Francisco.
''')
    q1=cell(r'''
## 5. Rollout mejora las decisiones, pero la jerarquía también cuesta

Los mismos 32 casos. Greedy elige el mayor cobro inmediato; en los empates de esta malla empieza por individuales. Rollout compara cada prueba suponiendo que después seguirá greedy; ejecuta una y vuelve a comparar.

Ambos usan la misma jerarquía de grupos, fijada al inicio, y soft clearing. Cada celda muestra la mayor pérdida frente al óptimo laminar entre $n=6$ y $n=7$.

![Pérdida de greedy y rollout frente al óptimo laminar, con la misma escala. Cada celda toma el peor de n=6 y n=7.](attachment:greedy_rollout_G.png)

Rollout pierde como máximo 5.26% en esta malla. No toda la pérdida viene de sus decisiones: también restringimos los grupos a una jerarquía concreta.

En $n=7,B=4$, pasar de $G=4$ a $G=5$ deja el óptimo laminar en 2.671875, pero rollout baja de 2.671875 a 2.53125. La construcción automática cambia la jerarquía y empeora el resultado.

Para discutir: ¿conviene mejorar la elección de la jerarquía antes de complicar la regla de planificación?
''','greedy_rollout_G')
    presentation.cells=[first,previous[1],binary,q2,information,q1]
    presentation.metadata['weekly_plan']='../../docs/notes/2026-10-14-plan-semana.md'
    presentation.metadata['q1_q2_evidence']='../../results/precio_q05_G/malla_2026-10-14_completa.csv'

    study.cells[0].source=r'''# Estudio y discusión del notebook 29

Actualizado para el plan del 14 de octubre y la sesión del martes 20. La presentación conserva cinco bloques; las figuras anteriores de matrices y de $\pi_L/\pi_{ratio}$ siguen aquí como consultas.

## Orden de estudio

1. Bloque 2: reconstruir el árbol binario y el laminar. Distinguir el resultado observado, la inferencia y lo que se cobra. Una rama cada vez.
2. Bloque 1: explicar por qué resolver el par pronto libera una prueba para C; después leer el intercambio general escrito por Vlad.
3. Bloque 3: interpretar una sola celda del mapa. El denominador es el óptimo general; 21.15% es pérdida de utilidad esperada, no probabilidad de una rama.
4. Bloque 4: resultados posibles → información → sanos acreditados → cociente con individuales. No estudiar toda la sección 5 del paper como requisito previo.
5. Bloque 5: entender una decisión de greedy y una de rollout. Separar la pérdida de la jerarquía elegida de la pérdida de decidir dentro de ella.

## Guion breve

Par: 1 minuto; binario frente a laminar: 3 minutos; mapa de laminaridad: 2 minutos; argumento de log G: 2 minutos; decisiones y jerarquía: 2 minutos. La familia 10/9 queda como resultado adicional; no hace falta aprenderla antes de entender el ejemplo pequeño.
'''
    for c in study.cells[1:]:
        c.source=c.source.replace('En la matriz del bloque 4','En la matriz de consulta')
        c.source=c.source.replace('2. Consolidar las comprobaciones independientes del solver y los CSV del plan del 14 de octubre.',
                                  '2. El contraste inicial (81+20+20) está cerrado. La malla nueva tiene 32 casos, cinco contrastes laminares independientes y evaluaciones de las políticas sobre todos los perfiles de esas cinco celdas.')
        c.source=c.source.replace('4. En $q=1/2$, $u_i=1$, estudiar primero la cota de información y ampliar la malla en G.',
                                  '4. En $q=1/2$, $u_i=1$, estudiar la cota de información y discutir la malla ya ampliada en G.')
        c.source=c.source.replace('5. Comparar greedy y rollout sobre una jerarquía fija con el óptimo laminar por trayectoria.',
                                  '5. Interpretar la comparación ya calculada entre greedy, rollout, el óptimo de la jerarquía fija y el óptimo laminar por trayectoria.')
    study.cells.append(cell(r'''
## El paso de información, sin identificarlo con diagnóstico completo

Sea H el historial final de una política determinista con B pruebas. Cada resultado tiene a lo sumo G+1 valores, así que hay a lo sumo $(G+1)^B$ historiales. Su entropía satisface $\mathcal H(H)\le B\log_2(G+1)$.

Con $q_i=1/2$ independiente, los $2^n$ perfiles iniciales son equiprobables. Si una hoja h es compatible con $N_h$ perfiles, su probabilidad es $N_h/2^n$. Si acredita $k_h$ sanos, esas coordenadas son cero en todos sus perfiles: $N_h\le 2^{n-k_h}$. Por tanto $k_h\le n-\log_2 N_h$.

Promediando las hojas se obtiene $E[k_H]\le n-E[\log_2 N_H]=\mathcal H(H)\le B\log_2(G+1)$. No se exige resolver a todos; se acota lo acreditado. Para una política aleatoria se fija primero su semilla independiente y luego se promedia.

Si $u_i=1$ y $B\le n$, probar B individuos da B/2 y es laminar. Así $V_{\rm aug}/V_{\rm lam}\le 2\log_2(G+1)$. Si B supera n, todos pueden probarse y los óptimos coinciden. El argumento usa q=1/2: no se traslada automáticamente a probabilidades heterogéneas.

El chequeo numérico respeta esta desigualdad en los 32 casos; no reemplaza su revisión matemática por Vlad. Tampoco demuestra que q=1/2 sea el peor régimen o que la dependencia log G sea necesaria.
'''))
    study.cells.append(cell(r'''
## Qué se midió en Q1 y qué falta explicar

Las funciones antiguas de `laminar_benchmarks.py` cobraban solo cuando se observaba un cero. Se añadió una opción explícita `clearing="soft"`, manteniendo el comportamiento antiguo por defecto. En el control n=3, B=G=2, q=.3, rollout pasa de .6 con hard clearing a .774 con soft clearing. Esa diferencia es de convención, no evidencia de un score superior.

La jerarquía ordena personas por $q_i u_i$, agrupa bloques de tamaño hasta G y los divide en mitades. Todos sus grupos son laminares. Es solo una de las familias permitidas por el óptimo laminar por trayectoria.

Greedy maximiza la utilidad nueva inmediata; el desempate usa la máscara menor. En la malla q=.5, el individuo y el par empatan inicialmente en .5; el desempate elige individuo. Los resultados del greedy dependen de esa especificación. Un experimento siguiente útil sería cambiar solo el desempate.

Rollout evalúa una primera prueba y, en cada resultado, la continuación completa de greedy hasta agotar el presupuesto. Hace la prueba ganadora y recalcula. Su evaluación real sigue rollout después, no se compromete a usar greedy para siempre.

Para n=7, B=4, G=5: óptimo laminar 171/64; mejor política en la jerarquía fija 41/16; rollout 81/32. La pérdida total 9/64 se descompone en 7/64 por la jerarquía y 2/64 por las decisiones de rollout. Con G=4, la jerarquía y rollout sí alcanzan 171/64. Permitir G=5 no obliga a usar bloques de cinco; es nuestra construcción la que los cambia.

Comprobación: cinco celdas con óptimo laminar por átomos y por perfiles, más replay independiente de greedy y rollout. Los testigos n=7,B=4,G=3 y G=4/5 también tienen árboles o recorridos comprobados sobre sus 128 perfiles. Datos completos en `results/precio_q05_G/`.

Preguntas para Francisco: ¿conviene optimizar primero la jerarquía? ¿Cuánto cambia greedy al priorizar pares en los empates? ¿Qué decisión de rollout explica la pérdida restante aun dentro de una buena jerarquía?
'''))
    for old in [previous[2],previous[4],previous[5]]:
        old.source='## Consulta conservada\n\n'+old.source
        study.cells.append(old)
    for notebook in [presentation,study]:
        add_answers(notebook)
        nb.validate(notebook)
    return presentation,study
