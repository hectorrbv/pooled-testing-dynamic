"""Extrae la Parte II del notebook 25 como un notebook autocontenido.

No editar el .ipynb resultante a mano: regenerarlo con este archivo desde
``group-count-dynamic``.
"""

from copy import deepcopy
from pathlib import Path

import nbformat as nbf


NOTEBOOKS = Path(__file__).parent
SOURCE = NOTEBOOKS / "25_resultados_y_peticiones.ipynb"
DESTINATION = NOTEBOOKS / "25_resultados_y_peticiones_parte_ii.ipynb"
PART_II_START = 33


def main() -> None:
    source = nbf.read(SOURCE, as_version=4)
    setup = deepcopy(source.cells[1])
    part_ii = [deepcopy(cell) for cell in source.cells[PART_II_START:]]

    portada = nbf.v4.new_markdown_cell(
        r"""
        # Resultados y peticiones de Francisco — Parte II

        Extracto autocontenido de `25_resultados_y_peticiones.ipynb`.
        Contiene únicamente el material de trabajo de la sesión del 6 de agosto:
        el colapso de \(V(T)\), el valor bajo presupuesto, el costo simulado con
        greedy y el régimen \(q=0.7\).

        **Convención.** `q` es la probabilidad de que una persona esté sana; el
        régimen de interés es `q < 0.5` (prevalencia alta). Los resultados de esta
        parte son exploratorios, no resultados cerrados de la Parte I.
        """
    )

    notebook = nbf.v4.new_notebook()
    notebook.cells = [portada, setup, *part_ii]
    notebook.metadata = deepcopy(source.metadata)
    nbf.write(notebook, DESTINATION)
    print(f"escrito: {DESTINATION.relative_to(NOTEBOOKS.parent)} ({len(notebook.cells)} celdas)")


if __name__ == "__main__":
    main()
