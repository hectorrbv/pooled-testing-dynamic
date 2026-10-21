"""Build the short talk from saved evidence, without rerunning solvers."""

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
import graphviz


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

AZUL, AMBAR, VERDE, GRIS = "#2563eb", "#d97706", "#059669", "#64748b"
figures = {}
diagram_sources = {}


def tree(name):
    g = graphviz.Digraph(name)
    g.attr(rankdir="TB", bgcolor="white", pad="0.15", nodesep="0.24",
           ranksep="0.5", dpi="150", ordering="out")
    g.attr("node", shape="box", style="rounded", fontname="Helvetica",
           fontsize="16", fontcolor="#172033", color=AZUL,
           penwidth="1.5", margin="0.16,0.1")
    g.attr("edge", fontname="Helvetica", fontsize="13", color=GRIS,
           fontcolor=GRIS, arrowsize="0.65")
    return g


def keep_tree(name, g):
    diagram_sources[name] = g.source
    figures[name] = g.pipe(format="png")


# Two complete one-test trees. A and CD are alternative choices, not outcomes.
g = tree("una_prueba")
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("A", "Probar A\nV = 0.5", color=VERDE)
    s.node("CD", "Probar CD\nV = 0.6", color=AMBAR)
g.node("A0", "A sano\nCobro +1", color=VERDE)
g.node("A1", "B sano\nCobro +1", color=VERDE)
g.edge("A", "A0", "R = 0\nP = 0.5")
g.edge("A", "A1", "R = 1\nP = 0.5")
g.node("CD0", "C y D sanos\nCobro +2", color=VERDE)
g.node("CD1", "Un sano sin identificar\nCobro 0", color=GRIS)
g.node("CD2", "Ningún sano\nCobro 0", color=GRIS)
for r, prob in enumerate((0.09, 0.42, 0.49)):
    g.edge("CD", f"CD{r}", f"R = {r}\nP = {prob:.2f}")
keep_tree("una_prueba", g)

# First level of the recorded optimal laminar policy; deeper levels are omitted.
recorded_tree = case["laminar"]["tree"]
by_count = {b["infected_count"]: b["next"] for b in recorded_tree["branches"]}
letters = "ABCDEF"
g = tree("primera_prueba")
g.node("root", "Probar AEF\n3 pruebas disponibles")
for r, next_node in by_count.items():
    healthy = next_node["newly_healthy"]
    reward = sum(Fraction(case["instance"]["u"][i]) for i in healthy)
    action = "".join(letters[i] for i in next_node["pool"])
    labels = {
        0: f"AEF sanos: +{reward}\nProbar {action}",
        1: f"Sin cobro aún\nProbar {action}",
        2: f"Sin cobro aún\nProbar {action}",
        3: f"Ningún sano en AEF\nCambiar a {action}",
    }
    color = AZUL if r == 1 else GRIS
    g.node(f"r{r}", labels[r], color=color,
           style="rounded,filled" if r == 1 else "rounded", fillcolor="#eff6ff")
    g.edge("root", f"r{r}", f"R = {r}", color=color, fontcolor=color)
keep_tree("primera_prueba", g)

# Complete subtree conditional on R(AEF)=1, extracted from recorded decisions.
subtree = by_count[1]
assert subtree["pool"] == [4]
g = tree("rama_un_infectado")
g.attr(nodesep="0.45")
g.attr("edge", fontsize="12")
g.node("E", "AEF tiene un infectado\nProbar E")
for b in subtree["branches"]:
    r = b["infected_count"]
    child = b["next"]
    healthy = child["newly_healthy"]
    reward = sum(Fraction(case["instance"]["u"][i]) for i in healthy)
    assert reward == 4
    action = "".join(letters[i] for i in child["pool"])
    name = f"e{r}"
    healthy_label = " y ".join(letters[i] for i in healthy)
    g.node(name, f"{healthy_label} {'sanos' if len(healthy)>1 else 'sano'}: +{reward}\nProbar {action}", color=VERDE)
    g.edge("E", name, "E sano" if r == 0 else "E infectado")
    for branch in child["branches"]:
        result = branch["infected_count"]
        leaf = branch["next"]
        assert leaf["remaining_tests"] == 0
        new = leaf["newly_healthy"]
        more = sum(Fraction(case["instance"]["u"][i]) for i in new)
        label = f"{' y '.join(letters[i] for i in new)} sano\n+{more}" if new else "Sin nuevo cobro\n+0"
        leaf_id = f"{name}_{result}"
        g.node(leaf_id, label, color=VERDE if more else GRIS)
        g.edge(name, leaf_id, f"{action} {'sano' if result==0 else 'infectado'}")
keep_tree("rama_un_infectado", g)


def keep_plot(name, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white")
    figures[name] = buf.getvalue()
    plt.close(fig)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(8, 2.6), layout="constrained")
ratios = [policies["π_M"]["ratio_laminar"], policies["π_L"]["ratio_laminar"], 1]
ax.barh([2, 1, 0], ratios, color=[AMBAR, VERDE, AZUL], height=0.5)
ax.set(yticks=[2, 1, 0], yticklabels=["Mejor de las cuatro", r"$\pi_L$", "Óptimo laminar"],
       xlim=(0, 1.16), xticks=[0, 0.5, 1], xticklabels=["0%", "50%", "100%"],
       xlabel="Fracción del óptimo laminar")
for y, ratio in zip([2, 1, 0], ratios):
    ax.text(ratio + 0.025, y, f"{100*ratio:.1f}%", va="center", fontsize=14)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
keep_plot("comparacion", fig)

fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
xs = [float(r["lambda_value"]) for r in rows]
ys = [float(r["ratio"]) for r in rows]
# Each point is an evaluated lambda; do not imply verification between points.
ax.scatter(xs, ys, s=46, color=AZUL, zorder=3)
ax.scatter([float(best["lambda_value"])], [best_ratio], s=90, color=VERDE, zorder=4)
ax.axhline(1, color="#667085", lw=1, ls="--")
ax.annotate("0.9307", xy=(float(best["lambda_value"]), best_ratio),
            xytext=(0.8, 0.97), fontsize=15, color=VERDE,
            arrowprops={"arrowstyle": "-", "color": VERDE})
ax.text(3.55, 1.005, "Óptimo laminar", ha="right", va="bottom", fontsize=10, color="#475467")
ax.set(xlim=(-0.08, 3.65), ylim=(0.75, 1.035), xlabel="λ: penalización por prueba",
       ylabel="Utilidad / óptimo laminar", yticks=[0.8, 0.9, 1.0])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E4E7EC", lw=0.7)
keep_plot("lambda", fig)


def screen(text, seconds, notes, figure=None, **extra):
    cell = nbf.v4.new_markdown_cell(
        text.strip(),
        metadata={"slideshow": {"slide_type": "slide"}, "talk_seconds": seconds,
                  "speaker_notes": notes, **extra},
    )
    if figure is not None:
        cell.attachments = {f"{figure}.png": {"image/png": base64.b64encode(figures[figure]).decode()}}
    if figure in diagram_sources:
        cell.metadata["graphviz_source"] = diagram_sources[figure]
    return cell


cells = [
    screen(r"""
# 1. El contraejemplo con una prueba restante

$n=4$ (A, B, C y D)  
$B=2$ (total), $b=1$ (restante)  
$G=2$  
$p_i=0.7$, $q_i=0.3$, $u_i=1$ para cada persona  
Historial: $R(AB)=1$  
Soft clearing.

Las probabilidades son las iniciales, con estados independientes.
Después de probar AB, sabemos que uno está sano y el otro infectado, pero no quién es quién.

![Árboles de las dos alternativas: probar A acredita una unidad en ambas ramas; probar CD solo acredita dos si R es cero.](attachment:una_prueba.png)

Si probamos A, cualquiera de los dos resultados nos permite identificar al sano de AB. Cobramos 1 seguro.

Si probamos CD, cobramos 2 solo cuando ambos están sanos. Eso ocurre el 9% de las veces, así que cobramos 0.18 en promedio.

V cuenta los sanos esperados dentro del grupo: 0.5 en A y 0.6 en CD.
Por eso elige CD, aunque con la última prueba conviene A.
""", 90, "V suma utilidad sana esperada dentro del grupo. En CD, R=1 deja un sano sin identificar. No hace falta usar una prueba redundante para mostrar el fallo.", "una_prueba"),
    screen(r"""
## 2. Las cuatro reglas que comparamos

Probamos cuatro formas de elegir qué grupo analizar.

La regla $\pi_M$ elige la prueba que permite cobrar más en promedio en ese momento.

La regla $\pi_C$ compara planes por su utilidad esperada por prueba reservada. Se queda con el mejor y le asigna ese bloque de pruebas, aunque termine antes.

La regla $\pi_R$ usa el mismo criterio que $\pi_C$, pero hace una sola prueba y vuelve a decidir con el resultado que acaba de observar.

La regla C3 estima cuánto podría cobrar ahora y después, teniendo en cuenta las pruebas que quedan y las personas que deja sin probar.

En el siguiente ejemplo, las cuatro se quedan por debajo del óptimo.
""", 80, "Correspondencia con el código: cobrar ahora = π_M; reservar pruebas para un grupo = π_C; reconsiderar después de cada prueba = π_R; estimar el cobro futuro = C3. Las dos reglas intermedias maximizan H_c/c. La segunda descuenta todo c aunque el plan local termine antes. C3 es una fórmula aproximada experimental, no una continuación óptima. El primer contraejemplo se refiere a V; el siguiente evalúa estas cuatro políticas completas.",
           source_files=["augmented/densidad_companion.py", "docs/notes/2026-09-01-mision-vhat-C3.md"]),
    screen(rf"""
## 3. El caso de seis personas

$B=3$  
$G=4$  
Soft clearing.

En orden A–F: $p_{{\rm infectado}}=(.9,.825,.875,.8,.95,.85)$; $u=(2,1,1,1,4,2)$.

Con cualquiera de las cuatro reglas cobramos alrededor de 0.70 en promedio.
La mejor estrategia laminar alcanza {opt:.4f} y empieza probando AEF:

![Primera prueba del óptimo laminar: AEF. Según el conteo, continúa con BD, E, F o BD.](attachment:primera_prueba.png)

Aquí mostramos la primera prueba y qué haríamos después de cada resultado.
""", 100, "π_M, π_C y π_R valen 0.7000; C3 vale 0.69904125. 0.70 son unidades de utilidad promedio. Bellman calcula el mejor valor dentro de la clase laminar. El cobro de AEF completamente sano ocurre con probabilidad 0.00075.", "primera_prueba"),
    screen(r"""
## 4. La rama con un infectado

$n=6$ (A, B, C, D, E y F)  
$B=3$ (total), $b=2$ (restante)  
$G=4$  
$p=(0.9,\,0.825,\,0.875,\,0.8,\,0.95,\,0.85)$, $q_i=1-p_i$  
$u=(2,\,1,\,1,\,1,\,4,\,2)$  
Historial: $R(AEF)=1$  
Soft clearing.

Los vectores siguen el orden A–F. Las probabilidades son las iniciales, con estados independientes.
Ya sabemos que AEF contiene exactamente un infectado.

![Subárbol completo después de R(AEF)=1: se prueba E y luego A o D según el resultado. Los nodos indican el nuevo cobro.](attachment:rama_un_infectado.png)

Al probar E cobramos 4 en ambos casos: si está sano, cobramos su utilidad;
si está infectado, sabemos que A y F están sanos y cobramos 2 por cada uno.
Los signos + indican lo que cobramos en cada paso.
""", 110, "Este es el subárbol condicional, no el valor promedio de la política completa. Los cobros totales de las cuatro hojas son 6, 6, 5 y 4. Si AEF hubiera dado tres infectados se cambia de grupo.", "rama_un_infectado"),
    screen(rf"""
## 5. Cobro menos costo de las pruebas

Esta regla, que llamamos $\pi_L$, compara planes restando a la utilidad esperada un costo por las pruebas que consumen:

$$\mathbb{{E}}[U]-\lambda\,\mathbb{{E}}[T].$$

$U$ es la utilidad que cobramos y $T$ el número de pruebas que usamos.
$\lambda$ fija la penalización por prueba. Podemos dejar un grupo si no conviene seguir con él.

![Gráfica de barras: la mejor de las cuatro reglas alcanza 65.8% del óptimo laminar y pi L alcanza 96.4%.](attachment:comparacion.png)

Con $\lambda=0.001$, esta regla obtiene {policies['π_L']['value']:.4f}, un 96.4% del óptimo laminar.
En la gráfica comparamos lo que cobramos, sin restar la penalización.
""", 110, "π significa política: una regla que decide según el historial. Configuración reproducida: horizonte 3 y no-parálisis activada. λ sirve para elegir acciones; el rendimiento se mide como utilidad bruta esperada.",
           "comparacion",
           experiment_config={"horizon": 3, "lambda": 0.001, "no_paralisis": True}),
    screen(rf"""
## 6. La elección de $\lambda$ y el 0.9307

Otro caso:

$n=7$  
$B=3$  
$G=4$

Probamos varios valores de $\lambda$ para ver cómo cambia el resultado.

![Utilidad relativa para los quince valores de lambda evaluados. El máximo observado es 0.9307.](attachment:lambda.png)

Con $\lambda={float(best['lambda_value']):.6f}$ obtenemos uno de los mejores resultados de los valores probados:
${float(best['value']):.4f}$ de utilidad, frente a ${float(best['opt_laminar']):.4f}$ del óptimo laminar.

Al dividirlos sale 0.9307. Este resultado corresponde a este caso. Todavía queda por demostrar una garantía general.

La pregunta para discutir es cómo elegir $\lambda$ antes de buscar esa garantía.
""", 110, "Hay empate en el mejor valor para λ=1.238469 y 1.733857. Se elige λ antes de observar resultados, maximizando la utilidad esperada de la política en la malla declarada. Esta candidata usa el estado inicial histórico BM17 registrado en el archivo de evidencia; no es un mínimo certificado sobre 7000 casos.",
           "lambda",
           experiment_config=params),
]
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
assert len(nb.cells) == 6 and all(c.cell_type == "markdown" for c in nb.cells)
assert sum(c.metadata.talk_seconds for c in nb.cells) == 600
assert all("|" not in c.source for c in nb.cells)
dest = Path(__file__).with_name("26_presentacion_corta.ipynb")
nbf.write(nb, dest)
print(f"{dest}: 6 pantallas, 3 diagramas de árboles, 2 gráficas, sin tablas.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        preview = Path(sys.argv[1])
        preview.mkdir(parents=True, exist_ok=True)
        for name, png in figures.items():
            (preview / f"{name}.png").write_bytes(png)
        for name, source in diagram_sources.items():
            (preview / f"{name}.dot").write_text(source)
