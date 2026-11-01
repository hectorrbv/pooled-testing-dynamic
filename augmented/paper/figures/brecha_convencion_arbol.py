#!/usr/bin/env python3
"""Figura: el juego minimo de la brecha de convencion (n=4, q_sano=0.3, G=2, B=2).

Un solo arbol, la politica "par primero", con el cobro de cada rama bajo las dos
reglas de acreditacion (soft = posterior-zero, hard = estricta) y los tres numeros
verificados: 0.774 (soft), 0.564 (hard), 0.6 (dos singletons). Todo numero sale de
`tests_brecha_convencion.py` / `tests_bm17.py` (dos vias, fracciones exactas).

Uso:  python augmented/paper/figures/brecha_convencion_arbol.py
Salida: brecha_convencion_arbol.{png,pdf,svg} junto a este archivo.
Paleta: azul #2a78d6 (soft) / naranja #eb6834 (hard), par validado CVD.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, EDGE = "#1a1a19", "#5a5a55", "#9a9a95"
SOFT, HARD = "#2a78d6", "#eb6834"
SOFT_FILL, HARD_FILL = "#e6f0fb", "#fdeee6"

OUT = Path(__file__).resolve().parent
fig = plt.figure(figsize=(15, 8.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(cx, cy, w, h, text, *, fc="white", ec=EDGE, lw=1.2, size=11.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=color, weight=weight, zorder=3,
            linespacing=1.35)
    return (cx - w / 2, cx + w / 2, cy)


def edge(a, b, label=None, color=EDGE, lw=1.4, lab_color=INK2, dy=0.018):
    (_, xa, ya), (xb, _, yb) = a, b
    ax.plot([xa, xb], [ya, yb], color=color, lw=lw, zorder=1, solid_capstyle="round")
    if label:
        ax.text((xa + xb) / 2, (ya + yb) / 2 + dy, label, ha="center", va="bottom", fontsize=11,
                color=lab_color, zorder=3)


# ---------------------------------------------------------------- titulo
ax.text(0.5, 0.965, "Cuatro personas, probabilidad de sano q = 0.3, pools de hasta 2, presupuesto de 2 pruebas",
        ha="center", va="center", fontsize=15, color=INK, weight="bold")
ax.text(0.5, 0.928, "La misma política, \"par primero\", cobrada bajo las dos reglas de acreditación",
        ha="center", va="center", fontsize=12.5, color=INK2)

# ---------------------------------------------------------------- columnas
X0, X1, X2, X3, X4 = 0.075, 0.285, 0.485, 0.705, 0.915
Y0, Y1, Y2 = 0.815, 0.555, 0.285          # ramas R=0, R=1, R=2
Y1a, Y1b = 0.615, 0.487                   # hojas de la rama R=1

for x, t in ((X0, "Prueba 1"), (X1, "Conteo del par"), (X2, "Prueba 2"), (X3, "Resultado y cobro"),
             (X4, "Total de la rama\nsoft | hard")):
    ax.text(x, 0.885, t, ha="center", va="center", fontsize=10.5, color=INK2, style="italic")

root = box(X0, Y1, 0.12, 0.10, "Probar el par\n{1, 2}", size=12.5, weight="bold")

r0 = box(X1, Y0, 0.20, 0.105, "R = 0  (prob 0.09)\n1 y 2 sanos: cobra 2")
r1 = box(X1, Y1, 0.20, 0.105, "R = 1  (prob 0.42)\nuno sano, uno infectado:\ncobra 0 por ahora")
r2 = box(X1, Y2, 0.20, 0.105, "R = 2  (prob 0.49)\nambos infectados: cobra 0")
edge(root, r0); edge(root, r1); edge(root, r2)

t0 = box(X2, Y0, 0.16, 0.095, "Probar {3},\npersona nueva")
t1 = box(X2, Y1, 0.16, 0.095, "Probar {1},\nrefinar el par")
t2 = box(X2, Y2, 0.16, 0.095, "Probar {3},\npersona nueva")
edge(r0, t0); edge(r1, t1); edge(r2, t2)

l0 = box(X3, Y0, 0.225, 0.095, "3 sano con prob 0.3:\n+0.3 esperado, ambas reglas")
l1a = box(X3, Y1a, 0.225, 0.095, "{1} sano (½): cobra a 1\nsoft 1  |  hard 1")
l1b = box(X3, Y1b, 0.225, 0.095, "{1} infectado (½): 2 sano por resta\nsoft 1  |  hard 0",
          fc=SOFT_FILL, ec=SOFT, lw=2.0, weight="bold")
l2 = box(X3, Y2, 0.225, 0.095, "3 sano con prob 0.3:\n+0.3 esperado, ambas reglas")
edge(t0, l0); edge(t1, l1a, "½", dy=0.012); edge(t1, l1b, "½", dy=-0.034); edge(t2, l2)
ax.text(X3, Y1b - 0.068, "← aquí está toda la brecha: 0.42 × ½ = 0.21",
        ha="center", va="center", fontsize=11, color=SOFT, weight="bold")

# totales por rama, soft | hard, con color por regla y etiqueta de texto
def total(cx, cy, s, h):
    ax.add_patch(FancyBboxPatch((cx - 0.055, cy - 0.045), 0.11, 0.09, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="white", ec=EDGE, lw=1.2, zorder=2))
    ax.text(cx - 0.025, cy, s, ha="center", va="center", fontsize=14, color=SOFT, weight="bold", zorder=3)
    ax.text(cx, cy, "|", ha="center", va="center", fontsize=14, color=EDGE, zorder=3)
    ax.text(cx + 0.026, cy, h, ha="center", va="center", fontsize=14, color=HARD, weight="bold", zorder=3)
    return (cx - 0.055, cx + 0.055, cy)

s0 = total(X4, Y0, "2.3", "2.3")
s1 = total(X4, Y1, "1.0", "0.5")
s2 = total(X4, Y2, "0.3", "0.3")
edge(l0, s0); edge(l1a, s1); edge(l1b, s1); edge(l2, s2)

# ---------------------------------------------------------------- resumen
ax.plot([0.03, 0.97], [0.218, 0.218], color=EDGE, lw=0.8)
ax.text(0.04, 0.165, "Valor de la política  =  Σ  probabilidad de la rama  ×  total de la rama",
        ha="left", va="center", fontsize=11.5, color=INK2, style="italic")
ax.text(0.04, 0.118, "Soft clearing (posterior-zero):   0.09·2.3 + 0.42·1.0 + 0.49·0.3  =  0.774",
        ha="left", va="center", fontsize=13, color=SOFT, weight="bold")
ax.text(0.04, 0.074, "Hard clearing (estricta):             0.09·2.3 + 0.42·0.5 + 0.49·0.3  =  0.564",
        ha="left", va="center", fontsize=13, color=HARD, weight="bold")
ax.text(0.04, 0.030, "Dos pruebas individuales {1}, {2}:   0.3 + 0.3  =  0.6,  igual bajo las dos reglas",
        ha="left", va="center", fontsize=13, color=INK, weight="bold")

box(0.795, 0.128, 0.30, 0.066, "Soft:  0.774 > 0.6  →  abrir el par", fc=SOFT_FILL, ec=SOFT, lw=2, size=13,
    color=INK, weight="bold")
box(0.795, 0.050, 0.30, 0.066, "Hard:  0.564 < 0.6  →  nunca agrupar", fc=HARD_FILL, ec=HARD, lw=2, size=13,
    color=INK, weight="bold")
ax.text(0.795, 0.190, "La convención cambia la política óptima, no solo el valor",
        ha="center", va="center", fontsize=12, color=INK2, style="italic")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"brecha_convencion_arbol.{ext}", dpi=200, facecolor="white")
print("ok:", ", ".join(str(OUT / f"brecha_convencion_arbol.{e}") for e in ("png", "pdf", "svg")))
