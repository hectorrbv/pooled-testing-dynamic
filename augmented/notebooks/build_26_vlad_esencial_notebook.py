"""Construye una versión corta y pedagógica del notebook 26.

Integra los insights de las notas de Vlad con los resultados exactos del
notebook 26. Mantiene solo dos figuras y usa la convención original:
r cuenta personas infectadas y z_i = 1 significa infectada.
"""

from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(r"""
# Notebook 26 esencial: de información a certificados

Esta versión combina la intuición de las notas de Vlad con los cálculos
verificados del notebook 26. La pregunta que organiza todo es:

> **¿Cuánto valor sano parece haber en una acción y cuántas pruebas cuesta
> convertirlo en certificados?**

Al terminar deben quedar claras seis ideas:

1. S0 solo valora el cobro inmediato.
2. V valora la salud que podría extraerse si el seguimiento fuera gratis.
3. Un conteo puede fijar el total sin revelar quién porta cada estado.
4. Conocer que alguien es sano no equivale a acreditarlo.
5. El costo local aproxima el trabajo de acreditación.
6. V/C corrige el contraejemplo canónico, pero no produce una regla universal.
""")

md(r"""
## 0. Convenciones

- q es la probabilidad de estar **sana**.
- p = 1-q es la probabilidad de estar infectada.
- La prueba devuelve r, el **número de infectadas** del pool.
- r = 0 es una prueba limpia.
- Hard clearing: solo una prueba limpia acredita; deducir informa, pero no paga.
- La utilidad es 1 por persona acreditada.

La frase de Vlad que conviene conservar es:

> **Conocido no significa cobrado.**
""")

code(r"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Q = 0.30
P = 1 - Q

AZUL = "#2563eb"
AMBAR = "#d97706"
VERDE = "#059669"
GRIS = "#64748b"
TINTA = "#172033"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

print(f"q = {Q:.2f} sana | p = {P:.2f} infectada")
""")

md(r"""
## 1. Los dos extremos: S0 y V

Para un pool fresco homogéneo de tamaño k:

$$
S_0(k)=q^k k.
$$

S0 pregunta cuánto se cobra **con esta prueba**. Solo paga si las k personas
están sanas, es decir, si r = 0.

El score de presupuesto mágico es:

$$
V(T)=\sum_{i\in T}u_i\,P(i\text{ sana}\mid h).
$$

V pregunta cuánto podría extraerse si todas las pruebas de seguimiento fueran
gratis. Para un pool fresco con utilidad unitaria, V(k)=qk.

Así aparecen los dos extremos de Vlad:

- S0 es tímido: evita pools grandes.
- V es glotón: prefiere pools grandes, aunque después no sepa cosecharlos.
""")

code(r"""
ks = np.arange(1, 7)
tabla_scores = pd.DataFrame({
    "tamaño k": ks,
    "S0: cobro inmediato": (Q ** ks) * ks,
    "V: seguimiento gratis": Q * ks,
})

assert tabla_scores.loc[0, "S0: cobro inmediato"] == 0.3
assert abs(tabla_scores.loc[1, "S0: cobro inmediato"] - 0.18) < 1e-12
assert abs(tabla_scores.loc[1, "V: seguimiento gratis"] - 0.6) < 1e-12
tabla_scores
""")

code(r"""
# FIGURA 1 DE 2
fig, ax = plt.subplots(figsize=(7.2, 3.7))
ax.plot(ks, tabla_scores["S0: cobro inmediato"], marker="o", lw=2.4,
        color=AMBAR, label="S0: cobrar ahora")
ax.plot(ks, tabla_scores["V: seguimiento gratis"], marker="o", lw=2.4,
        color=AZUL, label="V: extraer con seguimiento gratis")
ax.set_xlabel("Tamaño del pool k")
ax.set_ylabel("Score")
ax.set_xticks(ks)
ax.set_title("S0 se encoge; V crece con el tamaño del pool")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
""")

md(r"""
**Lectura de la figura 1.** S0 nunca da el primer paso grande en el régimen
q < 1/2. V sí da ese paso, pero lo hace porque finge que el trabajo posterior
es gratis. El notebook 26 estudia precisamente ese trabajo omitido.
""")

md(r"""
## 2. El átomo mínimo: total fijo, identidad incierta

Se prueba AB y se observa r = 1. Hay exactamente una infectada y una sana.
Los únicos mundos compatibles son:

| Mundo | Peso previo | Posterior |
|---|---:|---:|
| A sana, B infectada | 0.3 × 0.7 = 0.21 | 0.21 / 0.42 = 1/2 |
| A infectada, B sana | 0.7 × 0.3 = 0.21 | 0.21 / 0.42 = 1/2 |

Chequeo: 1/2 + 1/2 = 1.

La intuición de Vlad:

> **El total del átomo está cerrado, pero sus piezas todavía se mueven.**

Sabemos qué contiene AB en total; no sabemos quién lleva cada estado.
""")

md(r"""
## 3. Conocido no significa cobrado

Reentramos probando A:

| Resultado de A | Qué sabemos | Qué cobramos con esta prueba |
|---|---|---:|
| r_A = 0 | A es sana y queda acreditada | 1 |
| r_A = 1 | A es infectada; B queda deducida sana | 0 |

Con una prueba restante, la reentrada vale 1/2. Con dos pruebas restantes,
la rama mala puede probar B, obtener r_B=0 y cobrar también. Entonces la misma
acción vale 1.

**Insight:** el valor de una acción depende del presupuesto restante. V omite
esa dimensión porque supone seguimiento gratuito.
""")

md(r"""
## 4. De dónde sale el costo local

Para un par fresco CD, primero se prueba el par. Esa primera prueba no forma
parte del costo extra; después contamos las subpruebas necesarias dentro de CD.

| Resultado inicial | Probabilidad | Subpruebas posteriores |
|---|---:|---:|
| r = 0: ambas sanas | q² = 0.09 | 0 |
| r = 1: una infectada | 2q(1-q) = 0.42 | 1.5 en promedio |
| r = 2: ambas infectadas | (1-q)² = 0.49 | 0 |

Dentro de r=1, el primer singleton sale sano la mitad de las veces y cuesta
una subprueba. La otra mitad sale infectado; la persona restante queda deducida
sana, pero necesita otra prueba para ser acreditada, así que cuesta dos.

$$
E[\text{subpruebas}\mid r=1]=\tfrac12(1)+\tfrac12(2)=1.5.
$$

Por tanto:

$$
\text{costo extra}=0.42\times1.5=0.63,
\qquad C(CD)=1+0.63=1.63.
$$

El 0.63 no es una probabilidad ni un resultado posible de una sola corrida:
es el promedio de trabajo posterior al abrir el par.
""")

code(r"""
p_r = np.array([Q**2, 2*Q*(1-Q), (1-Q)**2])
subtests = np.array([0.0, 1.5, 0.0])
costo_extra_par = float(p_r @ subtests)
costo_total_par = 1 + costo_extra_par

assert abs(p_r.sum() - 1) < 1e-12
assert abs(costo_extra_par - 0.63) < 1e-12
assert abs(costo_total_par - 1.63) < 1e-12

pd.DataFrame({
    "r": [0, 1, 2],
    "probabilidad": p_r,
    "subpruebas posteriores": subtests,
    "aporte al costo extra": p_r * subtests,
})
""")

md(r"""
## 5. El contraejemplo completo y la reparación

Estado: AB ya fue probado con r=1; C y D son vírgenes. Comparamos tres acciones:

- reentrar con A;
- abrir el par virgen CD;
- retestear AB.

V cuenta la utilidad sana localizada. C cuenta la prueba actual más el trabajo
posterior esperado. El valor realizable con una sola prueba cuenta lo que
realmente puede acreditarse ahora.
""")

code(r"""
acciones = pd.DataFrame({
    "acción": ["Reentrar A", "Par virgen CD", "Retest AB"],
    "V": [0.5, 0.6, 1.0],
    "C total": [1.0, 1.63, 2.5],
    "realizable con 1 prueba": [0.5, 0.18, 0.0],
})
acciones["V/C"] = acciones["V"] / acciones["C total"]

assert acciones.loc[acciones["V"].idxmax(), "acción"] == "Retest AB"
assert acciones.loc[acciones["V/C"].idxmax(), "acción"] == "Reentrar A"
assert acciones.loc[acciones["realizable con 1 prueba"].idxmax(), "acción"] == "Reentrar A"

acciones[["acción", "V", "C total", "V/C", "realizable con 1 prueba"]]
""")

code(r"""
# FIGURA 2 DE 2
criterios = [
    ("V", "V: localiza salud", AZUL),
    ("realizable con 1 prueba", "Cobro realizable ahora", VERDE),
    ("V/C", "V/C: valor por trabajo", AMBAR),
]

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
for ax, (col, titulo, color) in zip(axes, criterios):
    vals = acciones[col].to_numpy()
    bars = ax.bar(acciones["acción"], vals, color=color, alpha=0.88)
    ax.set_title(titulo, fontsize=10.5)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

axes[0].set_ylabel("Valor del criterio")
axes[0].set_ylim(0, 1.12)
fig.suptitle("El ranking se invierte cuando cobrar deja de ser gratis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
""")

md(r"""
**Lectura de la figura 2.**

- V elige el retest porque ve una unidad sana atrapada en AB.
- El cobro realizable elige A porque es la única acción que puede acreditar
  esa utilidad con una prueba.
- V/C también elige A: penaliza el trabajo que V había fingido gratuito.

Esta es la contribución conceptual central del notebook 26.
""")

md(r"""
## 6. Qué queda abierto

El notebook completo estudia la familia:

$$
\frac{V}{C^\alpha}.
$$

En el contraejemplo, el retest y la reentrada empatan cuando
$2.5^\alpha=2$, es decir, cuando
$\alpha^*=\ln 2/\ln 2.5\approx0.756$.

Sin embargo, el barrido exacto muestra que ningún alpha domina en todos los
regímenes. Un castigo grande ayuda cuando q es bajo, pero puede impedir agrupar
cuando q es alto. También importa si el costo se restringe a T o incluye otros
átomos abiertos.

Por eso V/C es una **reparación diagnóstica**, no una regla final congelada.
""")

md(r"""
## Cierre de pizarrón

La historia esencial en cinco frases:

1. S0 solo paga el presente y por eso no abre pools en prevalencia alta.
2. V valora la salud localizable y sí abre pools, pero supone seguimiento gratis.
3. Un conteo fija totales y deja incierta la identidad dentro de los átomos.
4. Hard clearing separa saber de cobrar; esa separación genera costo de cierre.
5. V/C pone precio a cosechar y corrige la no-reentrada canónica, aunque alpha
   y el alcance del costo siguen abiertos.

Fuentes conceptuales: notas de Vlad de las semanas 6 y 8.  
Fuente de cálculos y verificaciones: notebook 26, costo local y no-reentrada.
""")


nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3"}

output = Path("augmented/notebooks/26_vlad_esencial.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f"escrito: {output} ({len(cells)} celdas, 2 figuras)")
