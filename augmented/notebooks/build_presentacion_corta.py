"""Build the five-screen talk from saved evidence, without rerunning solvers."""

import base64
import csv
import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/precio_laminar_2026-09-21/resultados.json"
GRID = ROOT / "results/verificacion_constante_093.csv"
META = GRID.with_suffix(".csv.meta.json")
data = json.loads(EVIDENCE.read_text())
params = json.loads(META.read_text())["params"]
case = next(c for c in data["cases"] if c["instance"]["id"] == "seis_heterogeneo")
policies = {p["policy"]: p for p in data["policy_six_person"]}
opt = float(Fraction(case["laminar"]["value"]))
with GRID.open() as stream:
    rows = [r for r in csv.DictReader(stream) if r["policy"] == "pi_L"]
best = max(rows, key=lambda r: float(r["value"]))
best_ratio = float(best["ratio"])
assert abs(opt - 1.0645140625) < 1e-12
assert abs(best_ratio - 0.9307030280342034) < 1e-12
assert case["laminar"]["first_pool"] == [0, 4, 5]
assert [float(r["lambda_value"]) for r in rows] == params["lambdas"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
xs = [float(r["lambda_value"]) for r in rows]
ys = [float(r["ratio"]) for r in rows]
# Each point is an evaluated lambda; do not imply verification between points.
ax.scatter(xs, ys, s=46, color="#296D88", zorder=3)
ax.scatter([float(best["lambda_value"])], [best_ratio], s=90, color="#14816F", zorder=4)
ax.axhline(1, color="#667085", lw=1, ls="--")
ax.annotate("0.9307", xy=(float(best["lambda_value"]), best_ratio),
            xytext=(0.8, 0.97), fontsize=15, fontweight="bold", color="#14816F",
            arrowprops={"arrowstyle": "-", "color": "#14816F"})
ax.text(3.55, 1.005, "Óptimo laminar", ha="right", va="bottom", fontsize=10, color="#475467")
ax.set(xlim=(-0.08, 3.65), ylim=(0.75, 1.035), xlabel="λ: penalización por prueba",
       ylabel="Utilidad / óptimo laminar", yticks=[0.8, 0.9, 1.0])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E4E7EC", lw=0.7)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=160, facecolor="white")
plt.close(fig)


def screen(text, seconds, notes, **extra):
    return nbf.v4.new_markdown_cell(
        text.strip(),
        metadata={"slideshow": {"slide_type": "slide"}, "talk_seconds": seconds,
                  "speaker_notes": notes, **extra},
    )


cells = [
    screen(r"""
# 1. La utilidad que existe no siempre se puede cobrar

**Queda una prueba.** AB ya dio un infectado; C y D siguen sin probar.
Priors independientes: $q=0.3$ sano; utilidad 1 por persona.
Con *posterior-zero*, cobramos al identificar a alguien sano con certeza.

| Próxima prueba | Score antiguo $V$ | Utilidad esperada que podemos cobrar |
|:---|---:|---:|
| A | 0.5 | **1**: identificamos al sano de AB |
| CD | **0.6** | 0.18: $2\times0.3^2$ |

**V prefiere CD, pero A permite cobrar más con el presupuesto disponible.**
""", 90, "V suma utilidad sana esperada dentro del grupo. En CD, R=1 deja un sano sin identificar. No hace falta usar una prueba redundante para mostrar el fallo."),
    screen(rf"""
## 2. Encontramos un caso donde las cuatro reglas pierden

**Seis personas · 3 pruebas · grupos de hasta 4 · posterior-zero.**

En orden A–F: $p_{{\rm infectado}}=(.9,.825,.875,.8,.95,.85)$; $u=(2,1,1,1,4,2)$.

| Política | Utilidad esperada de la partida completa |
|:---|---:|
| $\pi_M$, $\pi_C$, $\pi_R$ — cada una | {policies['π_M']['value']:.4f} |
| C3 | {policies['C3']['value']:.4f} |
| **Óptimo laminar, calculado con Bellman** | **{opt:.4f}** |

**Incluso la mejor de las cuatro alcanza solo el 65.8% del óptimo laminar.**
""", 110, "Este es el contraejemplo principal para la batería concreta. 0.70 son unidades de utilidad promedio, no 70%. Bellman calcula el mejor valor posible dentro de la clase laminar con ese presupuesto."),
    screen(r"""
## 3. El óptimo aprovecha la información para continuar

Abre **AEF**. Miremos solo la rama donde sale **un infectado**.
Las utilidades son $u_A=2$, $u_E=4$, $u_F=2$. Después prueba **E**:

| Resultado de E | Qué sabemos ahora | Cobro inmediato |
|:---|:---|---:|
| Sano | E está sano | **4** |
| Infectado | A y F están sanos | **2 + 2 = 4** |

Si E sale sano, la última prueba en A permite cobrar **2 adicionales**.
Si AEF hubiera dado tres infectados, abandonaríamos ese grupo tras una prueba.

**La continuación y las pruebas que consumimos dependen del resultado.**
""", 120, "Es una rama del árbol óptimo, no el valor promedio de toda la política. Si E sale infectado queda una prueba para otras personas. Abandonar AEF no significa terminar toda la partida."),
    screen(rf"""
## 4. $\pi_L$ valora planes con costo y posibilidad de abandonar

Compara planes locales usando

$$\mathbb{{E}}[U]-\lambda\,\mathbb{{E}}[T].$$

$U$: utilidad que se acredita. $T$: pruebas realmente consumidas.
$\lambda$: penalización por prueba; el máximo de **3 pruebas** sigue vigente.

En el mismo caso de seis personas, con $\lambda=0.001$:

| Política | Utilidad esperada | Fracción del óptimo laminar |
|:---|---:|---:|
| Mejor de las cuatro anteriores | 0.7000 | 65.8% |
| **$\pi_L$** | **{policies['π_L']['value']:.4f}** | **96.4%** |

**El 96.4% compara utilidad cobrada, sin restar la penalización.**
""", 140, "π significa política: una regla que decide según el historial. Configuración reproducida: horizonte 3 y no-parálisis activada. λ sirve para elegir acciones; el rendimiento se mide como utilidad bruta esperada.",
           experiment_config={"horizon": 3, "lambda": 0.001, "no_paralisis": True}),
    screen(rf"""
## 5. El 0.9307 depende de cómo elegimos $\lambda$

**Otra instancia: siete personas, 3 pruebas, grupos de hasta 4.**
Cada punto muestra un $\lambda$ evaluado para $\pi_L$.

![Utilidad relativa para los quince valores de lambda evaluados. El máximo observado es 0.9307.](attachment:lambda.png)

Mejor resultado de esta malla: $\lambda={float(best['lambda_value']):.6f}$,
$\mathbb{{E}}[U]={float(best['value']):.4f}$ y óptimo laminar ${float(best['opt_laminar']):.4f}$.

**0.9307 es el cociente en esta candidata; no demuestra una garantía del 93%.**

Para discutir: **¿qué regla para elegir $\lambda$ fijamos antes de buscar una garantía?**
""", 140, "Hay empate en el mejor valor para λ=1.238469 y 1.733857. Se elige λ antes de observar resultados, maximizando la utilidad esperada de la política en la malla declarada. Esta candidata usa el estado inicial histórico BM17 registrado en el archivo de evidencia; no es un mínimo certificado sobre 7000 casos.",
           experiment_config=params),
]
cells[-1].attachments = {"lambda.png": {"image/png": base64.b64encode(buf.getvalue()).decode()}}
for index, cell in enumerate(cells, 1):
    cell.id = f"presentacion-corta-{index}"
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "title": "Contraejemplo → π_L → 0.9307 · 10 minutos",
    "language_info": {"name": "python"},
    "presentation_minutes": 10,
    "evidence_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (EVIDENCE, GRID, META)
    },
})
nbf.validate(nb)
assert len(nb.cells) == 5 and all(c.cell_type == "markdown" for c in nb.cells)
assert sum(c.metadata.talk_seconds for c in nb.cells) == 600
dest = Path(__file__).with_name("26_presentacion_corta.ipynb")
nbf.write(nb, dest)
print(f"{dest}: 5 pantallas, 0 celdas de código, 10 minutos.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_bytes(buf.getvalue())
