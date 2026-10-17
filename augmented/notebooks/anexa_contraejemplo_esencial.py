"""Anexa la seccion del contraejemplo universal al 26_esencial.ipynb.

El 26_esencial es una version curada a mano (11 celdas) que ya no coincide
con su build-script; por eso este script ANEXA celdas al final sin tocar las
existentes, en vez de reconstruir. Idempotente: si la seccion ya esta, no
duplica.
"""

import nbformat as nbf

RUTA = 'augmented/notebooks/26_esencial.ipynb'
MARCA = 'El contraejemplo universal'

nb = nbf.read(RUTA, as_version=4)
if any(MARCA in c.source for c in nb.cells):
    print('la seccion ya existe; sin cambios')
    raise SystemExit

md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())

nuevas = [
md(r"""
## 7. El contraejemplo universal, visto como árbol

La sesión del 1-sep encargó buscar una instancia donde **todas** las
heurísticas greedy fallen a la vez. Existe: seis personas con probabilidades
de infección 0.8–0.95 y utilidades desiguales (E vale 4), B = 3, G = 4, bajo
posterior-zero (aquí la deducción **sí** acredita — es la convención nueva de
G0, distinta del hard clearing de las secciones anteriores). El inmediato, las
dos densidades del companion y C3 caen todas a **0.6576** del óptimo.

Dos vistas lo explican: el **menú** de la primera decisión (contra cuántas
alternativas ganó el óptimo y dónde quedaron las golosas) y el **árbol** de la
política óptima (qué hace el ganador, rama por rama).
"""),
code(r"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))
from fractions import Fraction
from augmented.bm17_toy_solver import SolverLaminar

p_ce = {0: Fraction('0.9'), 1: Fraction('0.825'), 2: Fraction('0.875'),
        3: Fraction('0.8'), 4: Fraction('0.95'), 5: Fraction('0.85')}
u_ce = {i: Fraction(v) for i, v in {0: 2, 1: 1, 2: 1, 3: 1, 4: 4, 5: 2}.items()}
sol = SolverLaminar(p_ce, u_ce, 4, 'posterior_zero')
U0 = frozenset(p_ce)
OPT = float(sol.V(U0, (), 3))
NOM = 'ABCDEF'

# el menu completo de la raiz: las 56 primeras jugadas con su Q exacto
menu = sorted(((float(sol._q_accion(U0, (), 3, a)),
                ''.join(NOM[i] for i in a[1]))
               for a in sol._acciones(U0, ())), reverse=True)

# Autoverificacion: optimo, la mejor jugada, y los singletons al fondo.
assert abs(OPT - 1.0645) < 5e-5 and menu[0][1] == 'AEF'
assert {s for _, s in menu[-6:]} == set('ABCDEF')
print(f'optimo exacto {OPT:.4f}; mejor jugada {menu[0][1]}; '
      f'las 6 peores jugadas del menu son los 6 singletons')
print(f'ratio de las cuatro heuristicas: 0.6576 (scoreboard de la mision)')

fig, ax = plt.subplots(figsize=(7.0, 10.5))
colores, etiquetas = [], []
for q_, s in menu:
    if s == 'AEF':
        colores.append(AZUL); etiquetas.append(f'{s}  <- optimo')
    elif s == 'F':
        colores.append(AMBAR); etiquetas.append(f'{s}  <- las 4 heuristicas')
    else:
        colores.append('#93c5fd' if ('E' in s and len(s) >= 3) else GRIS)
        etiquetas.append(s)
ax.barh(range(len(menu)), [q_ for q_, _ in menu], color=colores, height=0.7)
ax.set_yticks(range(len(menu)))
ax.set_yticklabels(etiquetas, fontsize=7, fontfamily='monospace')
ax.invert_yaxis(); ax.set_xlim(0.75, 1.09)
ax.set_xlabel('Q exacto de la primera jugada')
ax.set_title('El menú de la raíz: 56 jugadas ordenadas por valor exacto')
plt.tight_layout(); plt.show()
"""),
md(r"""
**Lectura del menú.** Todo el tercio superior son pools de 3+ que contienen a
E (la persona de u = 4); los seis singletons son las seis peores jugadas, y la
elección de las heurísticas (F sola) ocupa el lugar 51 de 56. El error de las
golosas no es de afinación sino de categoría: cobrar en vez de explorar.
"""),
code(r"""
import graphviz

g = graphviz.Digraph(format='svg')
g.attr(rankdir='TB', fontname='Helvetica')
g.attr('node', fontname='Helvetica', fontsize='11')
g.attr('edge', fontname='Helvetica', fontsize='9')
cont = [0]

def dibuja(U, atomos, b):
    U, atomos = sol._canoniza(U, atomos)
    v = float(sol.V(U, atomos, b))
    acc = sol.argmax.get((U, atomos, b))
    yo = f'n{cont[0]}'; cont[0] += 1
    if acc is None or b == 0:
        g.node(yo, '', shape='point', width='0.07', color='#9aa5b1')
        return yo
    verbo = 'ABRIR' if acc[0] == 'open' else 'REFINAR'
    pool = acc[1] if acc[0] == 'open' else acc[2]
    etq = f"{verbo} {''.join(NOM[i] for i in pool)}\nb={b}  EV {v:.3f}"
    g.node(yo, etq, shape='box', style='rounded',
           color=AZUL if verbo == 'ABRIR' else '#7048b6', penwidth='1.5')
    if acc[0] == 'open':
        S = acc[1]; zs = sol._z(S)
        ramas = []
        for s_, prob in enumerate(zs):
            if prob == 0: continue
            rew, nuevos = sol._pieza(S, s_, probado=True)
            ramas.append((s_, float(prob), float(rew),
                          ''.join(NOM[i] for i in S) if s_ == 0 else '',
                          (U - set(S), tuple(sorted(atomos + nuevos)))))
    else:
        _, (A, r), S = acc
        resto = tuple(sorted(set(A) - set(S)))
        zS, zR, zA = sol._z(S), sol._z(resto), sol._z(A)
        otros = tuple(a for a in atomos if a != (A, r))
        ramas = []
        for s_ in range(len(zS)):
            if not (0 <= r - s_ < len(zR)): continue
            prob = zS[s_] * zR[r - s_] / zA[r]
            if prob == 0: continue
            rS, nS = sol._pieza(S, s_, probado=True)
            rR, nR = sol._pieza(resto, r - s_, probado=False)
            acred = (''.join(NOM[i] for i in S) if s_ == 0 else '') + \
                    (''.join(NOM[i] for i in resto) if r - s_ == 0 else '')
            ramas.append((s_, float(prob), float(rS + rR), acred,
                          (U, tuple(sorted(otros + nS + nR)))))
    for s_, prob, rew, acred, (U2, at2) in ramas:
        hijo = dibuja(U2, at2, b - 1)
        etqr = f'r={s_}  p={prob:.3f}'
        if rew > 0:
            g.edge(yo, hijo, label=etqr + f'\n+{rew:.0f} ({acred})',
                   color='#0f7a5a', fontcolor='#0f7a5a', penwidth='1.5')
        else:
            g.edge(yo, hijo, label=etqr, color='#9aa5b1', fontcolor=GRIS)
    return yo

dibuja(U0, (), 3)
print(f'el arbol de la POLITICA optima: {cont[0]} nodos — extraido de un '
      f'problema con {len(sol.memo)} estados distintos '
      f'(el arbol completo sin compartir tendria ~116 mil nodos)')
g
"""),
md(r"""
**Lectura del árbol.** El óptimo abre el trío AEF aunque casi nunca sale
limpio (p = 0.001): lo abre porque cada conteo compra una continuación
distinta. La joya son los refinamientos de **doble filo**: tras conteo 1,
probar a E cobra 4 salga lo que salga — si E está sana cobra sus 4, y si está
infectada la deducción acredita a A y F (2+2). Eso — repartir presupuesto
según el conteo y cosechar por deducción — es lo que ningún score de un paso
ve, y por eso las cuatro heurísticas caen a 0.6576. La frase de Vlad aplica
invertida: aquí, **lo cobrable ni siquiera parece valioso** hasta que el
conteo llega.
"""),
]

nb.cells.extend(nuevas)
nbf.write(nb, RUTA)
print(f'anexadas {len(nuevas)} celdas a {RUTA} ({len(nb.cells)} total)')
