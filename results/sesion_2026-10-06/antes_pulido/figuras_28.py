"""Plain plots and decision trees for notebook 28; reuse the saved evidence."""
from fractions import Fraction as F
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, GRAY, BLUE = '#262626', '#777777', '#34536b'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'text.color': INK, 'axes.labelcolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': GRAY, 'axes.linewidth': .7,
    'xtick.color': INK, 'ytick.color': INK,
})


def canvas(width, height, x=12, y=4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, x), ylim=(0, y))
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, size=13, color=INK, **kwargs):
    ax.text(x, y, text, ha='center', va='center', fontsize=size,
            color=color, linespacing=1.5, **kwargs)


def edge(ax, start, end):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=GRAY, lw=.9)


def make_figures(grid, evidence, replay):
    images, figure_data = {}, {}

    def keep(name, fig, data):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                    pad_inches=.12, facecolor='white')
        images[name] = buf.getvalue()
        plt.close(fig)
        figure_data[name] = data

    # Only straight segments between the measured points; no fitted curves.
    rows = grid['rows']
    qs = sorted({r['q_decimal'] for r in rows})
    budgets = [2, 3, 4]
    settings = [(b, g) for b in budgets for g in (2, 3)]
    losses = np.array([
        [next(r['loss_percent'] for r in rows
              if (r['B'], r['G'], r['q_decimal']) == (b, g, q)) for q in qs]
        for b, g in settings
    ])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.88, wspace=.16)
    for panel, (ax, b) in enumerate(zip(axes, budgets)):
        for g, color, marker, style in [(2, GRAY, 's', '--'), (3, BLUE, 'o', '-')]:
            values = losses[settings.index((b, g))]
            ax.plot(qs, values, style, color=color, marker=marker, ms=4,
                    mfc='white' if g == 2 else color, mew=.9, lw=1.2,
                    label=f'G = {g}', zorder=3 if g == 2 else 2)
        ax.set_title('B = 4 = n (control)' if b == 4 else f'B = {b}', fontsize=13, pad=10)
        ax.set(xlim=(0, 1), ylim=(-.45, 14), xticks=[.1, .3, .5, .7, .9],
               yticks=[0, 5, 10, 12.5])
        ax.tick_params(length=3, width=.7)
        ax.yaxis.grid(True, color='#e8e8e8', lw=.6)
        ax.set_axisbelow(True)
        if panel == 0:
            ax.legend(frameon=False, loc='upper left', fontsize=11)
        if b in (2, 4):
            ax.text(.5, 5, 'G = 2 y G = 3: empate', ha='center', fontsize=10, color=GRAY)
    axes[0].set_ylabel('Pérdida relativa (%)')
    fig.supxlabel('q (probabilidad de estar sano)', fontsize=12, y=.025)
    axes[1].annotate('12.5%', (.5, 12.5), xytext=(0, 9),
                     textcoords='offset points', ha='center', fontsize=11)
    keep('mapa_precio_laminar', fig, {
        'q': qs, 'settings': settings, 'loss_percent': losses.tolist(),
        'normalization': '100*(1 - optimal_laminar/optimal_general)',
    })

    # Observed counts and the resulting inference, with no decorative nodes.
    fig, ax = canvas(9, 2.25, y=3)
    label(ax, 2.4, 1.5, 'R(ABC) = 1\nR(ABD) = 1\nR(ACD) = 2', size=16)
    ax.annotate('', xy=(7.3, 1.5), xytext=(4.6, 1.5),
                arrowprops={'arrowstyle': '->', 'color': GRAY, 'lw': 1})
    label(ax, 9.2, 1.5, 'A y B sanos\nC y D infectados', size=16)
    keep('conteos_identifican', fig, grid['count_witness'])

    # The comparison with fixed plans is stated in the cell, outside the tree.
    par = next(r for r in replay['path_replay'] if r['id'] == 'par_B2')
    assert par['general'] == '387/500' and par['virtual_path_replay'] == '663/1250'
    fig, ax = canvas(10, 3.05)
    label(ax, 6, 3.65, 'Probar AB', size=15)
    for x, result, prob, action, value in [
        (1.6, 0, '0.09', 'C', '2.30'),
        (6, 1, '0.42', 'A', '1'),
        (10.4, 2, '0.49', 'C', '0.30'),
    ]:
        edge(ax, (6, 3.3), (x, 1.65))
        label(ax, x, 1.35, f'R = {result}  ({prob})', size=12)
        label(ax, x, .85, f'Probar {action}', size=14)
        label(ax, x, .25, f'Total esperado: {value}', size=12, color=GRAY)
    keep('adaptacion_y_muestreo', fig, {
        'setup': {'n': 4, 'B': 2, 'G': 2, 'q': '.3', 'u': 1},
        'dynamic': '387/500', 'best_static_laminar': '3/5',
        'virtual_path_replay': '663/1250',
        'dynamic_branch_probabilities': ['9/100', '42/100', '49/100'],
        'dynamic_conditional_values': ['23/10', '1', '3/10'],
        'static_gap': '87/500', 'sampling_extra_loss': '87/1250',
    })

    pilot = evidence['crossing_pilot']
    assert pilot['trimmed_with_reuse']['value'] == '3/2'
    assert pilot['first_pool_then_ratio']['value'] == '7/4'
    mixed = [row for row in pilot['first_pool_then_ratio']['traces']
             if row['history'][0][1] in (1, 2)]
    mixed_probability = sum(F(row['probability']) for row in mixed)
    assert mixed_probability == F(3, 4)
    assert all(row['history'][1][0] == [0] for row in mixed)
    resolved = sum(F(row['probability']) for row in mixed
                   if row['history'][2][0] == [3])
    assert resolved / mixed_probability == F(1, 3)
    assert all(row['history'][2][0] in ([1], [3]) for row in mixed)
    trimmed_mixed = [row for row in pilot['trimmed_with_reuse']['traces']
                     if row['history'][0][1] in (1, 2)]
    assert all(row['history'][1][0] == [3] and row['history'][2][0] == [0]
               for row in trimmed_mixed)
    fig, ax = canvas(10.5, 3.9, x=13, y=5.15)
    label(ax, 6.5, 4.85, 'Tras ABC con R = 1 o 2 (75%)', size=12, color=GRAY)
    label(ax, 2.6, 4.1, 'Quitar repetidos', size=13)
    label(ax, 9, 4.1, 'Replanear con el cociente', size=13)
    label(ax, 2.6, 3.25, 'Probar D', size=15)
    label(ax, 9, 3.25, 'Probar A', size=15)
    edge(ax, (2.6, 2.9), (2.6, 1.35))
    label(ax, 2.6, 1.05, 'Probar A', size=15)
    for x, condition, action in [
        (6.7, 'ABC resuelto\n1/3', 'D'),
        (11.4, 'Queda un par mixto\n2/3', 'B'),
    ]:
        edge(ax, (9, 2.9), (x, 2.15))
        label(ax, x, 1.7, condition, size=12, color=GRAY)
        label(ax, x, .85, f'Probar {action}', size=15)
    keep('orden_continuacion', fig, {
        'condition': 'ABC has count 1 or 2', 'probability': '3/4',
        'resolved_after_A_given_mixed': '1/3',
        'unresolved_after_A_given_mixed': '2/3',
        'trimmed_full_value': '3/2', 'replanned_full_value': '7/4',
    })

    # The surrounding cell carries the interpretation and the theorem's scope.
    fig, ax = canvas(11, 1.55, x=14, y=2)
    for x, text in [(1.45, 'Población ficticia\nelige AB → A'),
                    (5.1, 'Fijar\nAB → A'),
                    (8.55, 'Recortar\ncon monedas de 1/2'),
                    (12.3, 'Evaluar en otra\npoblación independiente')]:
        label(ax, x, 1, text, size=12)
    for start, end in [(2.95, 4.05), (6.1, 7), (10.12, 10.7)]:
        ax.annotate('', xy=(end, 1), xytext=(start, 1),
                    arrowprops={'arrowstyle': '->', 'color': GRAY, 'lw': 1})
    keep('nick_construccion', fig, {
        'source': 'https://arxiv.org/html/2601.22419v2#S5',
        'scope': 'binary tests, hard clearing; construction used in proof',
        'thinning': 'independent coin for every person-pool occurrence',
    })

    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    probs = [.09, .21, .21, .49]
    ax.bar(range(4), probs, width=.42, color=[GRAY, BLUE, BLUE, GRAY])
    ax.set_xticks(range(4), labels=['Sano / sano', 'Sano / infectado',
                                   'Infectado / sano', 'Infectado / infectado'], fontsize=10)
    ax.set(ylim=(0, .56), yticks=[0, .1, .2, .3, .4, .5],
           ylabel='Probabilidad', xlabel='Primer perfil / segundo perfil')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.yaxis.grid(True, color='#e8e8e8', lw=.6)
    ax.set_axisbelow(True)
    for x, prob in enumerate(probs):
        ax.text(x, prob + .015, f'{prob:.2f}', ha='center', fontsize=11)
    ax.plot([1, 1, 2, 2], [.295, .315, .315, .295], color=GRAY, lw=.8)
    ax.text(1.5, .34, 'Si difieren: 1/2 cada uno', ha='center', fontsize=11)
    keep('nick_dos_perfiles', fig, grid['independent_pair_example'])
    return images, figure_data
