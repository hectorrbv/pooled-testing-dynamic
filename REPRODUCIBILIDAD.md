# Reproducibilidad (B-M4, plan maestro §22)

Estado: **cerrado el 17 de agosto de 2026**. Este documento es la receta congelada
y el registro de lo que se encontró al cerrarlo.

## La receta

```bash
git clone <repo> && cd group-count-dynamic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # base + pytest
pytest
```

Resultado esperado: **suite verde, sin fallo de colección, con las clases
opcionales saltadas y la razón visible en el reporte**. Verificado en un venv
limpio: 143 pasan, 22 se saltan, cero fallos.

Con todas las clases instaladas (`pip install -r requirements-all.txt`) y las
licencias comerciales vigentes: 170 pasan, cero se saltan.

## Clases de dependencia

El requirements único se partió en cinco archivos. La regla es que la suite base
nunca necesita nada fuera de `requirements.txt`.

| Archivo | Contenido | Qué pasa sin él |
|---|---|---|
| `requirements.txt` | numpy, scipy, matplotlib, pandas | nada corre |
| `requirements-dev.txt` | pytest | no hay suite |
| `requirements-viz.txt` | graphviz, seaborn | `tests_visualizer.py` se salta |
| `requirements-rl.txt` | gymnasium | `tests_rl_fixes.py` se salta |
| `requirements-solvers.txt` | MOSEK, Gurobi (con licencia) | 20 tests de solver se saltan |
| `requirements-all.txt` | todo lo anterior | — |

## Los tres hallazgos del cierre

**El mecanismo de skip no era un skip.** `tests_solvers.py` definía una excepción
propia, `class SkipTest(Exception)`, y `_require_mosek()` la lanzaba. pytest no
reconoce esa clase, así que los tests no se saltaban: **fallaban**. Esa es la causa
de los cinco fallos MOSEK por licencia vencida que §22 registró el 1 de agosto. Se
sustituyó por `pytest.skip()` con razón.

**La caída silenciosa a la heurística era real y no estaba cubierta.** Los solvers
comerciales caen a `_heuristic_best_pool` cuando faltan, con un `RuntimeWarning`.
Los tests de Gurobi no tenían ninguna guarda, así que en una máquina sin Gurobi
validaban la heurística y pasaban por accidente. El caso extremo era
`test_compare_all_includes_solver_strategies`, que en el venv limpio **falla** con
8.966 contra 8.982 esperado: la heurística es medible peor. Se añadió
`_require_gurobi()` (con verificación de licencia, no solo de import) y se aplicaron
guardas a los 20 tests que ejercitan un solver comercial. Los tests que validan el
fallback a propósito viven en `tests_correctness_fixes.py` y se dejaron intactos.

**La colección se rompía por dos módulos, no uno.** §22 nombraba `gymnasium`; el
segundo era `graphviz` en `tests_visualizer.py`. Ambos usan ahora
`pytest.importorskip` con una razón que dice qué instalar.

## Semillas y artefactos

Toda aleatoriedad pasa por `augmented.provenance.seeded_rng(seed)`, que envuelve
`np.random.default_rng`. Nunca el estado global de `np.random`, porque no se puede
registrar.

Todo artefacto experimental se escribe con
`augmented.provenance.write_canonical_csv`, que produce dos archivos:

- `<nombre>.csv` — solo datos, con saltos de línea Unix explícitos, **byte-idéntico
  entre corridas con la misma semilla**. Un diff vacío es evidencia de
  reproducibilidad.
- `<nombre>.csv.meta.json` — generador, semilla, parámetros, versiones de python y
  de las bibliotecas base, commit y si el árbol estaba sucio, y la marca de tiempo.

La separación es deliberada: comparar resultados es un diff sobre el CSV; auditar
de dónde salieron es leer el sidecar. Si el timestamp viviera dentro del CSV,
ningún diff sería vacío nunca.

`read_stamp` falla explícitamente si el sidecar no existe, para que un artefacto
generado por fuera del mecanismo no se confunda con uno trazable.

## Ejemplar de referencia

`augmented/experiments_separacion_n10.py` es el patrón a copiar: enumeración
exacta, sin aleatoriedad (`seed=None` documentado como tal), CSV canónico en
`results/separacion_n10_q02.csv` con su sidecar, y anclas de regresión en
`augmented/tests_separacion_n10.py` que fijan los números citados en
`augmented/cuentas-n10-q02.md`.

## Lo que queda fuera

No hay CI configurado. La receta es reproducible a mano y está verificada, pero
nadie la corre automáticamente en cada push. Montar CI base determinista más suite
opcional es trabajo separado y no bloquea ningún gate actual.

El entorno conda `pooled-testing` no tiene pytest instalado, así que la suite no
corre ahí. Se dejó como está: la vía soportada es el venv de la receta.
