"""A short oral arc: one comparison, one case, the map, then the research question."""


def add_presentation(add):
    add(r'''
## 1. ¿Cuánto cuesta exigir laminaridad?

General: cualquier grupo de hasta $G$ personas.
Laminar: en cada rama, los grupos son disjuntos o uno contiene al otro.

AB seguido de A es laminar; AB seguido de AC no lo es.

Comparamos el mejor valor esperado de cada clase. Ambas adaptan las pruebas y tienen los mismos datos y presupuesto $B$.

$n=6$, $u_i=1$, estados independientes; $q$ es la probabilidad de estar sano y $p=1-q$. Observamos el conteo de infectados $R$ y acreditamos a quienes sabemos sanos con certeza: soft clearing.
''', 75, cell_id='sesion28-pregunta', notes='''
Apertura: Francisco, hoy traigo una sola comparación. Si exigimos que la política sea laminar, ¿cuánto valor perdemos frente a la mejor política sin esa restricción? Mantuvimos los mismos datos. Primero un caso; después el mapa.
Definir G como tamaño máximo de grupo y B como máximo de pruebas en cada rama. La clase general incluye a la laminar; no implica que toda política general sea no laminar. Un óptimo es el mejor valor esperado entre todas las políticas de su clase. AB/A y AB/AC ilustran solo la condición de laminaridad. El modelo y las convenciones compartidas se dan aquí una sola vez; el setup del caso siguiente fija los parámetros restantes. La línea de investigación busca construir políticas laminares con una garantía de valor.
Transición: antes del mapa, un caso que da significado a la pérdida.
''')

    add(r'''
## 2. Con los mismos recursos: 3 frente a 2.5

$n=6$, $B=4$, $G=3$, $q_i=p_i=0.5$, $u_i=1$.

- Óptimo general: 3 sanos acreditados en promedio.
- Óptimo laminar: 2.5.

La restricción cuesta una sexta parte del valor general: aproximadamente 16.7%.
''', 75, cell_id='sesion28-caso', notes='''
Frase oral: las personas y las pruebas son las mismas. Lo único que cambiamos es exigir laminaridad. Incluso eligiendo la mejor política laminar, queda media unidad de utilidad por debajo del óptimo general.
Ambos valores incluyen las mejores decisiones posteriores a cada resultado. General3 coincide con nq=3, toda la utilidad esperada disponible; laminar2.5 está cotejado entre dos solvers. La pérdida exacta es (3−2.5)/3=1/6;16.7% está redondeado. Los números son promedios sobre los64estados. El mecanismo detallado de la política n6 no lo presenta esta pantalla; si se pregunta, queda por explicar. El ejemplo n4 de las consultas no es esta instancia.
Transición: ¿es un caso aislado, o aparece un patrón al cambiar el presupuesto y las probabilidades?
''')

    add(r'''
## 3. La pérdida aparece con presupuestos intermedios

228 casos: $B=1,\ldots,6$, $G\in\{2,3\}$ y 19 valores de $q$ entre 0.05 y 0.95.

Color: pérdida porcentual respecto al general. Blanco: empate. El contorno delimita los casos con pérdida.

![Pérdida por laminaridad: se destaca el caso B=4, G=3, q=0.5 de la pantalla anterior.](attachment:general_vs_laminar.png)

En esta malla: empate con $B=1,2,6$; separación en 110 de los 114 casos con $B=3,4,5$.
''', 120, 'general_vs_laminar', cell_id='sesion28-mapa', notes='''
Frase oral: el caso anterior es esta celda señalada. Con una, dos o seis pruebas, el mapa queda en blanco. En los presupuestos intermedios, exigir laminaridad casi siempre cuesta.
Señalar primero la única etiqueta16.7%, después las filas B1/B2 yB6, y terminar en las filas B3aB5. El contorno marca separación estricta, incluso cuando el color es muy tenue; los bordes corresponden a celdas calculadas, no a umbrales continuos. Cada panel corresponde aG2oG3. Hay19probabilidades×6presupuestos×2cotas=228instancias;118empates110separaciones. La pérdida es100(1−laminar/general). B3separa34/38; B4yB5separan38/38cadauno. El máximo16.666...% coincide con el caso anterior. Los mayores valores por cada B,G con separación aparecen en q=.45o.5. B6=n permite resolver a todos; el empateB2 es observado en esta malla. La malla no proporciona una garantía universal de retener5/6.
Transición: esta pérdida también limita lo que una conversión podría prometer en todos los casos.
''')

    add(r'''
## 4. ¿Qué podemos garantizar al imponer laminaridad?

El caso de 3 frente a 2.5 descarta una garantía universal superior a $5/6$ del valor general. Todavía falta demostrar cuánto podemos garantizar.

Propuesta pendiente: seguir la política general y cambiar, antes de perder laminaridad, a la mejor continuación laminar. Compararla con el óptimo laminar desde el inicio.

¿Conviene conservar esas primeras decisiones o modificarlas antes?

[Guion oral](../../docs/notes/2026-10-06-guion-notebook-28.md) · [Consultas](28_notas_de_estudio.ipynb).
''', 90, cell_id='sesion28-direccion-prueba', notes='''
Cierre: en esta malla, el costo aparece con presupuestos intermedios. En el caso destacado, exigir laminaridad pierde una sexta parte incluso haciendo lo mejor posible. El siguiente paso es estudiar una conversión concreta: conservar las primeras decisiones de la general y cambiar antes de perder laminaridad.
Por qué el techo5/6 es válido: en esta instancia, cualquier salida laminar admisible vale como máximo2.5, también si aleatorizamos sobre políticas laminares. Aplicada a una política general óptima de valor3, una garantía E[V(T(pi))]>=c V(pi) exigiría3c<=2.5; luego c<=5/6. Esta es una cota superior sobre cualquier garantía uniforme para una familia de instancias que incluya este caso. No prueba que una garantía5/6 exista ni que no haya ejemplos peores. El alcance experimental continúa siendo la malla finita.
El experimento propuesto aún no se ejecutó. Cambiar antes de la primera prueba incompatible mantiene un prefijo laminar; la continuación debe ser compatible con todos sus grupos. Comparar con el óptimo laminar desde la raíz mide cuánto cuesta conservar ese prefijo. Antes de ejecutar, fijar definición tras reducir ancestros y desempate favorable a acciones laminares entre óptimas. Empezar con los ocho casos acordados. La comparación con heurísticas y la sección5 del paper quedan en consultas separadas.
Pregunta para Francisco: para intentar una garantía, ¿conservarías el prefijo de la política general o buscarías transformar sus decisiones desde el inicio?
''')
