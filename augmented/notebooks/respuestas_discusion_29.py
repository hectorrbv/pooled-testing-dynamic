"""Short discussion answers, reviewed with Opus and tied to saved evidence."""

ANSWERS = {
    '¿qué hipótesis permiten que la ganancia sea estrictamente positiva?': r'''Respuesta: con conteos y soft clearing, el par conserva lo que cobraríamos por A y B y libera una prueba para alguien fuera del plan de individuales. Bastan $G\ge2$, $2\le B<n$, independencia, probabilidades interiores y utilidades positivas: la ganancia es $(q_Aq_B+p_Ap_B)q_Cu_C>0$. No hace falta $q<1/2$.''',
    '¿qué limita cuánto puede crecer esta ventaja?': r'''Respuesta: todavía no sabemos el máximo. Nuestra familia se acerca a $10/9$; la cota general disponible permite hasta $4$. La libertad de volver a incluir a alguien probablemente sano explica el mecanismo, pero no fija su límite. Buscaría familias que amplifiquen esa ventaja variando probabilidades y utilidades.''',
    '¿qué construcción consigue una separación del mismo orden?': r'''Respuesta: estudiaría matrices detectoras dentro de bloques de G personas. Si resolvemos cada bloque con $O(G/\log G)$ pruebas, cobramos $G/2$ en promedio. Para obtener una brecha de orden $\log G$ falta además demostrar $V_{\rm lam}\le KB$ con K constante. Es una ruta pendiente: necesita diseños admisibles y suficientes personas y pruebas.''',
    '¿conviene mejorar la elección de la jerarquía antes de complicar la regla de planificación?': r'''Respuesta: en este caso empezaría por la jerarquía: explica $7/64$ de la pérdida total de $9/64$. Conservaría como opción la jerarquía de G=4, también válida con G=5. A esta escala podemos evaluar cada jerarquía completa y elegir una antes de empezar. Después mediría la pérdida restante de rollout; falta comprobar si esta prioridad se mantiene en otros casos.''',
    '¿cuánto puede crecer esa separación?': r'''Respuesta: conocemos una familia que se acerca a $V_{\rm bin}/V_{\rm lam}=10/9$ y una garantía que permite hasta 4. No sabemos cerrar esa distancia. Los dos factores dos provienen de comparaciones distintas: sus peores casos podrían no coincidir. Probaría ampliar la familia a más personas o pruebas.''',
    r'¿se puede demostrar $V_{\rm lam}\le K B$ con $K$ independiente de $n,B,G$?': r'''Respuesta: es la hipótesis que intentaría probar en $q_i=1/2$, $u_i=1$. Buscaría una cantidad asociada al estado laminar que permita acotar la ganancia acumulada por prueba. No basta acotar una rama: una inferencia puede acreditar muchos sanos de golpe. Sin ese argumento, una matriz detectora eficiente todavía no demuestra una separación frente al laminar.''',
    '¿qué decisión explica la pérdida de las heurísticas en este control?': r'''Respuesta: casi toda la pérdida aparece al abrir cuatro personas, mientras el óptimo laminar abre un par. Incluso continuando óptimamente tras esa apertura, obtendríamos 4.56914 frente a 4.64621. Las decisiones posteriores añaden una pérdida mucho menor. Compararía primero las aperturas con el mismo criterio de continuación; el cálculo por ramas está en las notas siguientes.''',
}

CONTROL_DETAIL = r'''### Dónde se pierde utilidad en el control de cinco personas

Estas son $\pi_L$ y $\pi_{\rm ratio}$, distintas del greedy y rollout de la malla nueva. Ambas empiezan por ABCD; el óptimo laminar empieza por AB.

Forzar ABCD y continuar después de forma óptima da $731063/160000=4.56914375$, frente al óptimo laminar $371697/80000=4.6462125$. La apertura ya pierde $0.07706875$.

Las continuaciones añaden pérdidas pequeñas y diferentes. Si ABCD tiene dos infectados, $\pi_{\rm ratio}$ prueba E; probar AB sería mejor. Si tiene tres infectados, $\pi_L$ prueba AB; probar A sería mejor. Ponderadas por la probabilidad de llegar a esas ramas, las pérdidas adicionales son $0.00045125$ y $0.000095$, respectivamente.

La enumeración directa de las dos pruebas restantes coincide con Bellman. [Diagnóstico y fracciones](../../results/sesion_2026-10-14/diagnostico_control_29.json); reproducción: `python -m augmented.diagnostico_control_29`. Esto localiza el fallo en este control; no establece una preferencia general por abrir pares.
'''


def add_answers(notebook):
    for cell in notebook.cells:
        if cell.cell_type != 'markdown':
            continue
        cell.source = cell.source.replace(
            '¿qué decisión de continuación explica la pérdida de las heurísticas en este control?',
            '¿qué decisión explica la pérdida de las heurísticas en este control?')
        cell.source = cell.source.replace(
            'aproximadamente $0.0945$ por laminaridad y $0.0772$ adicionales al usar cualquiera de las dos reglas.',
            'aproximadamente $0.0945$ por laminaridad y pérdidas adicionales de $0.07716$ con $\\pi_L$ y $0.07752$ con $\\pi_{\\rm ratio}$.')
        for question, answer in ANSWERS.items():
            if 'Para discutir:' in cell.source and question in cell.source and answer not in cell.source:
                cell.source += '\n\n'+answer
        if '¿qué decisión explica la pérdida de las heurísticas en este control?' in cell.source and CONTROL_DETAIL not in cell.source:
            cell.source += '\n\n'+CONTROL_DETAIL
    return notebook
