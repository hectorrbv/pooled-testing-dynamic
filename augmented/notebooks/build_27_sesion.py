"""Four research discussion screens for Francisco, from recorded experiments.

Run from any directory. Optional argument writes figure previews.
The old presentation is retained in results/sesion_2026-09-29/.
"""
from pathlib import Path
from hashlib import sha256
import base64
import io
import json
import sys

import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/sesion_2026-09-29"
FILES = ["results/sesion_2026-09-29/regimenes_politicas.csv",
         "results/sesion_2026-09-29/lambda_refinado.csv"]
grid = pd.read_csv(ROOT / FILES[0])
lambda_grid = pd.read_csv(ROOT / FILES[1])
meta = json.loads((ROOT / (FILES[0]+".meta.json")).read_text())
for path, digest in meta["params"]["source_sha256"].items():
    assert sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
audit = json.loads((RESULTS/"regimenes_auditoria_perfiles.json").read_text())
assert audit["comparisons"] == 12 and audit["max_error"] < 1e-9
assert len(grid) == 114 and len(grid.drop_duplicates(["q","B","G"])) == 114
policies = ["pi_M","pi_C","pi_R","C3","pi_L","pi_ratio"]
labels = [r"$\pi_M$  inmediato", r"$\pi_C$  reserva bloque",
          r"$\pi_R$  reserva, recalcula", r"$\pi_3$  estima futuro",
          r"$\pi_L$  resta precio", r"$\pi_{ratio}$  utilidad / pruebas"]
qs = sorted(grid.q.unique())
assert len(qs) == 19 and qs[0] == .05 and qs[-1] == .95
assert grid[policies].max().max() <= 1+1e-9
INK, GRAY, TEAL, PURPLE, ORANGE = "#172033", "#64748b", "#00856a", "#7b46ad", "#b95716"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":13,
                     "axes.spines.top":False, "axes.spines.right":False})
images = {}
dots = {}
figure_data = {}


def keep(name, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=.12)
    images[name] = buf.getvalue()
    plt.close(fig)


# Each tile is one policy in one setup: no averaging over B or G.
loss_cmap = LinearSegmentedColormap.from_list(
    "loss", ["#f8fafc", "#f6d6ad", "#df9e56", "#aa581c", "#68310a"])
fig, axes = plt.subplots(2, 3, figsize=(13.4,7.2), sharex=True, sharey=True)
fig.subplots_adjust(left=.205, right=.98, top=.89, bottom=.20, wspace=.12, hspace=.27)
all_losses = {}
for row, B in enumerate([2,3]):
    for col, G in enumerate([2,3,4]):
        ax = axes[row,col]
        sub = grid[(grid.B==B)&(grid.G==G)].set_index("q").loc[qs]
        losses = np.clip(100*(1-sub[policies].to_numpy().T),0,100)
        all_losses[f"B{B}_G{G}"] = losses.tolist()
        im = ax.imshow(losses, cmap=loss_cmap, vmin=0, vmax=50, aspect="auto")
        ax.set_title(f"B={B}, G={G}", fontsize=16, pad=10)
        ax.set_yticks(range(6), labels=labels, fontsize=13)
        ax.set_xticks([0,5,9,13,18], labels=[".05",".30",".50",".70",".95"], fontsize=12)
        ax.tick_params(axis="both", length=0, pad=7, labelbottom=True)
        ax.set_xticks(np.arange(-.5,19,1), minor=True)
        ax.set_yticks(np.arange(-.5,6,1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.595,.13,"q: probabilidad de estar sano",ha="center",fontsize=14)
cax=fig.add_axes([.32,.065,.54,.023])
cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[0,10,20,30,40,50])
cb.set_label("Utilidad perdida frente al óptimo (%)",labelpad=8,fontsize=13)
cb.ax.set_xticklabels(["0: óptimo","10","20","30","40","50"])
cb.outline.set_visible(False)
fig.text(.205,.97,"Cada columna cambia q; cada panel fija B y G",fontsize=17,va="top",color=INK)
figure_data["mapa_global"]={"policies":policies,"q":qs,"loss_percent":all_losses}
keep("mapa_global",fig)

# Exact head-to-head comparison in all 114 setups, retaining ties.
pairs=[(B,G) for B in [2,3] for G in [2,3,4]]
diff=np.array([100*(grid[(grid.B==B)&(grid.G==G)].set_index("q").loc[qs,"pi_ratio"]
                   -grid[(grid.B==B)&(grid.G==G)].set_index("q").loc[qs,"pi_L"])
               for B,G in pairs])
winners=np.where(diff>1e-7,1,np.where(diff < -1e-7,-1,0))
assert (winners==1).sum()==13 and (winners==-1).sum()==10
fig,ax=plt.subplots(figsize=(12.7,5.0))
fig.subplots_adjust(left=.15,right=.985,top=.81,bottom=.16)
ax.imshow(winners,cmap=ListedColormap([PURPLE,"#e8ecf0",TEAL]),vmin=-1,vmax=1,aspect="auto")
ax.set_yticks(range(6),labels=[f"B={B}, G={G}" for B,G in pairs],fontsize=14)
ax.set_xticks(range(19),labels=[f"{q:.2f}".lstrip("0") for q in qs],fontsize=11)
ax.set_xlabel("q: probabilidad de estar sano",labelpad=12,fontsize=14)
ax.set_xticks(np.arange(-.5,19,1),minor=True)
ax.set_yticks(np.arange(-.5,6,1),minor=True)
ax.grid(which="minor",color="white",linewidth=2)
ax.tick_params(which="both",length=0,pad=7)
ax.axhline(2.5,color="white",linewidth=6)
for spine in ax.spines.values():spine.set_visible(False)
for i,j in zip(*np.where(winners!=0)):
    ax.text(j,i,f"{abs(diff[i,j]):.1f}",ha="center",va="center",color="white",fontsize=11)
fig.legend(handles=[Patch(facecolor=TEAL,label=r"Gana $\pi_{ratio}$"),
                    Patch(facecolor=PURPLE,label=r"Gana $\pi_L$"),
                    Patch(facecolor="#e8ecf0",label="Empate")],
           loc="upper center",bbox_to_anchor=(.59,1.01),ncol=3,frameon=False,fontsize=15)
fig.text(.15,.87,"Los números indican la ventaja en puntos porcentuales",fontsize=13,color=GRAY)
figure_data["mapa_candidatas"]={"q":qs,"rows":pairs,"winner":winners.tolist(),"difference_pp":diff.tolist()}
keep("mapa_candidatas",fig)

# One complete policy, not a single selected outcome.
g=graphviz.Digraph("par_completo")
g.attr(rankdir="TB",bgcolor="white",pad=".12",nodesep=".35",ranksep=".48",dpi="140",ordering="out")
g.attr("node",shape="box",style="rounded",fontname="Helvetica",fontsize="16",
       color=TEAL,fontcolor=INK,penwidth="1.4",margin=".16,.1")
g.attr("edge",fontname="Helvetica",fontsize="13",color=GRAY,fontcolor=GRAY,arrowsize=".6")
g.node("root","Primera prueba: AB")
for r,prob,label in [
    (0,"1%","A y B sanos: cobramos 2\nÚltima prueba: C\nValor esperado total: 2.1"),
    (1,"18%","Un sano y un infectado\nÚltima prueba: A\nCobramos 1 seguro"),
    (2,"81%","A y B infectados\nÚltima prueba: C\nValor esperado total: 0.1")]:
    g.node(str(r),label)
    g.edge("root",str(r),f"R = {r}\n{prob}")
images["par_completo"]=g.pipe(format="png")
dots["par_completo"]=g.source
assert abs(.01*2.1+.18*1+.81*.1-.282)<1e-12

# Show what changes in the policy when lambda changes, including continuations.
selected=lambda_grid.set_index("lambda").loc[[.01,.30,.31,.40]]
assert list(selected.first_pool_size)==[4,3,3,1]
fig,ax=plt.subplots(figsize=(10.8,3.9))
fig.subplots_adjust(left=.075,right=.985,top=.80,bottom=.23)
bars=ax.bar(range(4),selected.value,color=[GRAY,TEAL,ORANGE,GRAY],width=.55)
for i,(_,r) in enumerate(selected.iterrows()):
    ax.text(i,r.value+.035,f"{r.value:.3f}",ha="center",va="bottom",fontsize=15,color=INK,
            bbox={"facecolor":"white","edgecolor":"none","pad":.6})
ax.set(ylim=(0,1.06),ylabel="Utilidad esperada")
ax.set_xticks(range(4),labels=[f"λ={lam:.2f}\nAbre {int(r.first_pool_size)} {'persona' if r.first_pool_size == 1 else 'personas'}"
                            for lam,r in selected.iterrows()],fontsize=13)
ax.axhline(.91072,color=TEAL,linestyle="--",linewidth=1,zorder=0)
ax.text(.995,.91072,"Óptimo",transform=ax.get_yaxis_transform(),ha="right",va="bottom",color=TEAL,fontsize=12)
ax.set_yticks([0,.3,.6,.9])
ax.grid(axis="y",alpha=.12);ax.set_axisbelow(True)
fig.text(.075,.94,"El mismo primer grupo puede dar resultados distintos",fontsize=17,color=INK)
figure_data["lambda_decisiones"]=selected.reset_index().to_dict("records")
keep("lambda_decisiones",fig)

cells=[]
def add(source, seconds, figure=None, notes="", consultation=False, cell_id=None):
    c=nbf.v4.new_markdown_cell(source.strip(),metadata={
        "slideshow":{"slide_type":"skip" if consultation else "slide"},
        "talk_seconds":seconds,"speaker_notes":notes})
    c.id=cell_id or f"mapa-global-{len(cells)+1}"
    if figure:
        c.attachments={f"{figure}.png":{"image/png":base64.b64encode(images[figure]).decode()}}
        if figure in dots:c.metadata.graphviz_source=dots[figure]
    cells.append(c)

add(r"""
## 1. La planeación aporta más cuando hay pocos sanos

$n=6$, $u_i=1$, $p_i=1-q$. Soft clearing, estados iniciales independientes.
Probamos 19 valores de $q$ entre 0.05 y 0.95, $B\in\{2,3\}$ y $G\in\{2,3,4\}$: 114 setups.

![Mapa de las seis políticas en los 114 setups. Más oscuro significa mayor pérdida de utilidad frente al óptimo laminar.](attachment:mapa_global.png)

Con $q\leq0.50$, $\pi_L$ y $\pi_{ratio}$ superan claramente al cobro inmediato. Con $q\geq0.80$, el inmediato ya queda muy cerca del óptimo en los puntos probados.

óptimo laminar de cada setup. $\pi_L$ usa $\lambda=0.01$ y $\pi_{ratio}$ usa $E[U]/E[T]$.
""",150,"mapa_global",notes="""Abrir: «Probamos tu propuesta de dividir por las pruebas esperadas y ampliamos el barrido para ver en qué regímenes ayuda». Cada celda es un setup exacto, sin promediar B y G. Las columnas varían q; las filas son políticas; B cambia entre filas de paneles y G entre columnas. Blanco/pálido indica poca pérdida. La referencia es el óptimo laminar propio de cada caso. No decir que el color significa probabilidad de acertar.
Pi_M cobra ahora; pi_C y pi_R dividen por pruebas reservadas, con compromiso y recálculo respectivamente; pi_3 es C3, una estimación aproximada del cobro futuro; pi_L resta lambda por prueba esperada; pi_ratio divide entre pruebas esperadas. L y ratio consideran hasta min(3,b) pruebas dentro de un grupo y ejecutan una antes de recalcular. En q<=.5, pi_3 es óptima para G=2 y casi óptima para G=3, pero puede caer mucho con G=4. En q>=.8, pi_M pierde menos del 1% en los seis setups de cada q muestreado. Todo el mapa tiene n=6 y utilidad homogénea: no extrapolar la garantía a n mayor o utilidades distintas.""")
add(r"""
## 2. El presupuesto cambia cuál de las dos conviene

Mismos 114 setups. Comparamos $\pi_{ratio}$ con $\pi_L$, que puntúa $E[U]-\lambda E[T]$ con $\lambda=0.01$.

![Mapa de la ventaja entre cociente y penalización según q, B y G. Verde favorece al cociente, morado a la penalización y gris indica empate.](attachment:mapa_candidatas.png)

Con $q=0.60$ y $G=3$, gana $\pi_L$ si $B=2$, pero gana $\pi_{ratio}$ si $B=3$.
Elegir una regla solo por la probabilidad de salud no basta.
""",130,"mapa_candidatas",notes="""Responder a «¿dónde difieren?» con el cambio de presupuesto manteniendo q y G. En q=.60,G=3,B=2, ratio abre 2 y L abre 3: valores .9641 y 1 del óptimo. Con B=3 ambas abren 3, pero ratio obtiene el óptimo y L .9668: la diferencia está en las continuaciones. El mapa cuenta 13 victorias del cociente, 10 de L y 91 empates. El mínimo del cociente en esta malla ampliada es .9337068, menor que el .95 de L. Por eso la frase antigua «mejor mínimo 95.24%» solo correspondía a los 36 casos originales. Hay muchos empates, no una ganadora universal. La ventaja numérica usa (V_ratio - V_L)/V_opt * 100; es diferencia de rendimiento, no el score U/T. Los colores representan diferencias por encima de 1e-9 en rendimiento.""")
add(r"""
## 3. Una prueba grupal puede dejar libre la siguiente

$n=6$, $B=2$, $G=4$, $q_i=0.10$, $p_i=0.90$, $u_i=1$.

Dos individuales dan 0.200. $\pi_{ratio}$ y $\pi_L$ empiezan por AB y alcanzan 0.282, el óptimo.

![Política completa al abrir AB. Solo el resultado con un infectado requiere otra prueba dentro del par.](attachment:par_completo.png)

El par se resuelve con una sola prueba el 82% de las veces. La segunda queda disponible para C.
Ese ahorro dentro del grupo es lo que aprovecha el costo esperado.
""",130,"par_completo",notes="""La política completa siempre usa dos pruebas. El ahorro se refiere solo a resolver AB: su costo esperado es 1+.18=1.18, y su utilidad local es .2. El cociente local es .2/1.18=.16949, superior al .1 de un individual. Al dividir por dos pruebas reservadas, el par queda en .1 y empata al individual; pi_C y pi_R eligen el individual con su desempate actual. Pi_M prefiere estrictamente el individual: cobro inmediato .1 frente a .02 para AB. El .282 total incluye C: .01*2.1+.18*1+.81*.1. Pi_3 abre cuatro y vale .17852 en esta instancia, pero su árbol parcial anterior se retiró porque no explicaba el valor completo. La comparación inmediata frente al par muestra el mecanismo sin ese árbol. Pregunta útil: ¿podemos demostrar una ventaja del costo esperado en salud rara que no dependa del desempate de la densidad reservada?""")
add(r"""
## 4. Ajustar $\lambda$ puede arreglar una decisión y empeorar otra

$n=6$, $B=3$, $G=4$, $q_i=0.20$, $p_i=0.80$, $u_i=1$.

![Cuatro valores de lambda: 0.30 alcanza el óptimo; 0.31 abre el mismo grupo de tres pero pierde utilidad en las continuaciones.](attachment:lambda_decisiones.png)

Al refinar el barrido encontramos que $\lambda=0.30$ alcanza el óptimo.
Ambas abren un trío. Si queda resuelto, 0.30 agrupa otro par; 0.31 vuelve al cobro inmediato y prueba individuales.

""",100,"lambda_decisiones",notes="""Este es un experimento nuevo motivado por la pregunta D36:16 de Francisco. El barrido original saltaba de .2 a .4 y no detectaba que .30 es óptimo. Ahora se probaron 26 precios, con pasos .01 entre .2 y .4. .30 es el único óptimo entre esos puntos, no un intervalo certificado ni el único valor real posible. Valores exactos evaluados: lambda .01 -> .87232, .30 -> .91072, .31 -> .84, .40 -> .6. ABC queda resuelto si R=0 o R=3, con probabilidad .2^3+.8^3=.52. Quedan DEF y dos pruebas. Resolver DE tiene utilidad local .4 y costo esperado 1.32: con lambda .30 el neto es .004; con .31 no hay plan local de valor positivo y se activa el cobro inmediato. La continuación completa con el par vale .536; dos individuales valen .4. Esa diferencia explica toda la pérdida: .52*(.536-.4)=.07072. Las otras dos ramas tienen igual valor con ambos precios aunque pueden elegir distintos subconjuntos. No presentar .30 como recomendación universal: se encontró después de evaluar este caso. Propuesta para discutir: comparar por separado un precio dependiente de b y una vuelta al cociente cuando ningún plan supera el precio, fijando cada regla antes de evaluar nuevos casos. Mantener pi_ratio como referencia sin ajuste de lambda.""")
add(r"""


- $\pi_M$: maximiza lo que cobra inmediatamente.
- $\pi_C$: divide la utilidad del plan entre las pruebas reservadas y compromete el bloque.
- $\pi_R$: usa ese mismo cociente, hace una prueba y vuelve a decidir.
- $\pi_3$: estima cobro inmediato, cobro futuro y oportunidades fuera del grupo mediante una fórmula aproximada.
- $\pi_L$: busca el mejor plan local según $E[U]-\lambda E[T]$.
- $\pi_{ratio}$: compara planes locales por $E[U]/E[T]$.

""",0,consultation=True,cell_id="consulta-politicas")

nb=nbf.v4.new_notebook(cells=cells,metadata={
    "title":"Qué funciona, dónde falla y qué probar después",
    "language_info":{"name":"python"},
    "presentation_seconds":sum(c.metadata.talk_seconds for c in cells),
    "policy_display_names":{"C3":r"\pi_3"},
    "planning_limits":{"pi_L":3,"pi_ratio":3},
    "evidence_sha256":{p:sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
    "experiment_rows":114,
    "discussion_notes":"docs/notes/2026-09-29-estudio-y-guion-francisco.md",
    "prior_presentation":"results/sesion_2026-09-29/presentacion_antes_mapa_global.json",
    "pedagogical_focus":"Observación experimental, mecanismo y siguiente pregunta de investigación.",
})
nbf.validate(nb)
assert all("**" not in c.source and "posterior-zero" not in c.source for c in cells)
nbf.write(nb,Path(__file__).with_name("27_sesion_francisco.ipynb"))
(RESULTS/"datos_figuras_mapa_global.json").write_text(json.dumps(figure_data,indent=2)+"\n")
if len(sys.argv)>1:
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    for name,data in images.items():(out/f"{name}.png").write_bytes(data)
    for name,data in dots.items():(out/f"{name}.dot").write_text(data)
print(f"4 pantallas + 1 consulta; {nb.metadata.presentation_seconds/60:.1f} minutos.")
