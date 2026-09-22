"""Append the verified meeting close to notebook 26; preserve notebooks 25/27."""
from pathlib import Path
import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results'
frame = pd.read_csv(OUT/'mapa_homogeneo_1.csv')
frame['smallest'] = frame.tamanos_optimos.astype(str).map(lambda s: min(map(int, s.split(';'))))
fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
cmap = plt.get_cmap('viridis', 7)
norm = BoundaryNorm(np.arange(1.5, 9.5), cmap.N)
for ax, G in zip(axes.flat, [2, 3, 4, 5, 8]):
    sub = frame[frame.G == G]
    values = sub.pivot(index='B', columns='q_sano', values='smallest')
    im = ax.imshow(values, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(0, 19, 2), [f'{q:.2f}' for q in values.columns[::2]])
    ax.set_yticks(range(7), values.index)
    ax.set(xlabel='q = probabilidad de estar sano', ylabel='Presupuesto B', title=f'Tamaño máximo G = {G}')
    for _, row in sub.iterrows():
        if ';' in str(row.tamanos_optimos):
            ax.text(round(row.q_sano*20)-1, row.B-2, '*', ha='center', va='center', color='white')
axes.flat[-1].axis('off')
axes.flat[-1].text(.05, .85,
    '665 de 665 casos:\nconviene empezar con un grupo.\n\nColor: menor tamaño óptimo.\n*: empate entre tamaños de grupo.\n\nn = B·G, posterior-zero, u = 1.\nEvidencia de malla, no demostración.',
    va='top', fontsize=13, linespacing=1.5)
fig.colorbar(im, ax=list(axes.flat[:5]), ticks=range(2, 9), label='Personas en el primer grupo óptimo', shrink=.7)
fig.suptitle('Mapa 1 · Primera acción de Bellman laminar', fontsize=17)
fig.savefig(OUT/'mapa_homogeneo_1.png', dpi=150)
plt.close(fig)

path = ROOT/'augmented/notebooks/26_esencial.ipynb'
nb = nbf.read(path, 4)
nb.cells = [c for c in nb.cells if not c.metadata.get('cierre_22sep')]
nb.cells[0].source = '''# Contraejemplos → π_L → 0.9307

**Sesión con Francisco: martes 22-sep, 19:00 Praga. Guion de 10 minutos.**
Recorrido principal: **§5 → §7 → §8 → §10**. §11 contiene el cierre de tareas
y los mapas para la discusión. §1 y el notebook 27 quedan como apoyo.

q = P(sano), p = P(infectado), R = número de infectados. Desde §7 usamos
posterior-zero; la tabla histórica de costos en §5 conserva la variante estricta.

Resultados revisados el 21-sep. Guion oral y pendientes precisos en
`docs/notes/2026-09-22-guion-francisco-hector.md`.
'''
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.startswith('## 8.'):
        c.source += r'''

$$I_\lambda(C)=\sup_{\pi\text{ local}}\mathbb E[U_{\mathrm{cobrada}}-\lambda T].$$

**π** es la regla que decide qué probar después de cada resultado. **λ** pone
precio a cada prueba efectivamente consumida; abandonar localmente vale cero.
En esta implementación, el horizonte local es min(b, 3) y se replanifica.
Si ningún proyecto supera el precio, no-parálisis elige el mejor cobro inmediato.
'''
        # Make rerunning this builder idempotent.
        marker = '\n\n$$I_\\lambda'
        chunks = c.source.split(marker)
        if len(chunks) > 2:
            c.source = chunks[0] + marker + chunks[1]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [md('''## 10. El 0.9307 y por qué λ es parte de la política

La candidata tiene n=7, B=3, G=4: tres personas casi seguras, tres premios
muy grandes con infección 0.975 y una persona intermedia. Contra el mismo
óptimo laminar del harness histórico, **6.226133**, π_L alcanza **0.930703**
con **λ=1.238469**, horizonte 3 y no-parálisis.

La rejilla se calcula a partir del prior y las utilidades. Se elige λ por el
valor esperado de la política, antes de jugar. OPT sirve después para medir
la razón; no interviene en esa selección.

**Alcance de la verificación de hoy:** esta candidata y esta rejilla. La nota
del 2-sep resume una búsqueda de unas 7,000 instancias; no se reevaluó aquí toda
esa batería con una especificación uniforme. 0.9307 es evidencia empírica,
no una garantía del 93% para cualquier instancia.
'''), code('''from hashlib import sha256
c093 = ROOT/'results/verificacion_constante_093.csv'
meta093 = json.loads(c093.with_suffix('.csv.meta.json').read_text())
for p, digest in meta093['params']['source_sha256'].items():
    assert sha256((ROOT/p).read_bytes()).hexdigest() == digest, f'Regenerar: {p}'
t093 = pd.read_csv(c093)
best093 = t093.loc[t093.groupby('policy')['value'].idxmax()]
display(best093[['policy','lambda_value','value','opt_laminar','ratio']].round(6))
curve = t093[t093.policy=='pi_L'].sort_values('lambda_value')
fig, ax = plt.subplots(figsize=(9.5, 4.1))
ax.semilogx(curve.lambda_value, curve.ratio, 'o-', color='#2563eb')
winner = curve.loc[curve.ratio.idxmax()]
ax.annotate(f"λ={winner.lambda_value:.6f}\\nratio={winner.ratio:.6f}",
            (winner.lambda_value,winner.ratio), xytext=(-125,-10),
            textcoords='offset points', arrowprops={'arrowstyle':'->'})
ax.set(xlabel='λ: precio por prueba', ylabel='Valor de π_L / óptimo laminar',
       title='La rejilla debe encontrar la ventana de λ que permite explorar', ylim=(.75,.97))
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/constante_093_lambda.png',dpi=150)
plt.show()
'''), md('''## 11. Cierre verificable y preguntas para Francisco

- **BM17:** 296 comparaciones exactas con enumeración de perfiles e historial;
  diferencia cero. Matriz local declarada: 8 anclas, 240 estados heterogéneos,
  48 raíces homogéneas; incluye ambas convenciones en los dos primeros grupos.
- **Ancla G0:** k=B−⌈log₂G⌉=3. 0.914742 es una cota inferior realizable.
- **Mapa 1:** 665/665 casos prefieren estrictamente empezar con un grupo;
  población n=B·G, q=.05,.10,…,.95, G=2,3,4,5,8, B=2,…,8.
- **Mapa 3:** la frontera G=8, B=6, n=24 sí se resolvió. El ancla G=16,
  B=7, n=48 alcanzó 2.5 millones de estados sin certificar OPT.

Bellman aquí optimiza dentro de la clase **laminar por trayectoria**; el
notebook 27 compara esa clase con acciones generales en poblaciones pequeñas.
'''), code('''m3 = pd.read_csv(ROOT/'results/mapa_homogeneo_3.csv')
display(m3[['case','n','G','B','cota_cbs','cbs_stop','optimo','status']].round(6))
from IPython.display import Image
display(Image(filename=str(ROOT/'results/mapa_homogeneo_1.png')))
'''), md('''### Qué concluimos del Mapa 3

En la frontera: **OPT_lam=0.865394 > 0.708011**. La versión CBS que se detiene
al primer cobro vale **0.727127**, de modo que esa política explícita es subóptima.

En el ancla: permitir solamente grupos de hasta 8 ya da **1.123283**, una
política factible con el límite original G=16. Supera tanto la cota **0.914742**
como esa versión CBS (**0.939440**). El óptimo completo G=16 sigue sin calcularse.

**Para discutir:** ¿qué continuación precisa debe incluir “cover-then-bisect”
si sobran pruebas? ¿Qué hipótesis hacen demostrable la preferencia inicial por
grupos? ¿Cómo fijamos la rejilla de λ antes de una evaluación uniforme?

No confundir la garantía de encontrar al menos un sano con el número esperado
de sanos acreditados: este último puede superar uno.
''')]
for c in cells:
    c.metadata['cierre_22sep'] = True
    c.metadata['slideshow'] = {'slide_type': 'slide' if c.cell_type=='markdown' else 'fragment'}
nb.cells.extend(cells)
nbf.validate(nb)
nbf.write(nb, path)
print(path)
