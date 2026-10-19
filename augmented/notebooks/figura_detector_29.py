"""Detector figure for notebook 29: people are dots, each test is a contour around its pool."""
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from figuras_separacion_28 import INK, GRAY, BLUE, WARM

# Four people on the corners of a square; every pool of three is a triangle.
POSITIONS = {'A': (-1, 1), 'B': (1, 1), 'C': (-1, -1), 'D': (1, -1)}
TESTS = [('ABC', 1), ('ABD', 1), ('ACD', 2)]
HEALTHY = 'AB'


def outline(points, r, k=16):
    """Closed curve at distance r around the convex hull of points (circle, capsule or rounded polygon)."""
    P = np.array(points, float)
    if len(P) == 1:
        t = np.linspace(0, 2 * np.pi, 4 * k + 1)
        return P[0, 0] + r * np.cos(t), P[0, 1] + r * np.sin(t)
    c = P.mean(0)
    P = P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]  # counter-clockwise
    m = len(P); xs = []; ys = []
    for i in range(m):
        prev, nxt = P[i] - P[i - 1], P[(i + 1) % m] - P[i]
        a0 = np.arctan2(prev[1], prev[0]) - np.pi / 2
        turn = (np.arctan2(nxt[1], nxt[0]) - np.arctan2(prev[1], prev[0])) % (2 * np.pi)
        t = np.linspace(a0, a0 + turn, k)
        xs += list(P[i, 0] + r * np.cos(t)); ys += list(P[i, 1] + r * np.sin(t))
    xs.append(xs[0]); ys.append(ys[0])
    return xs, ys


def contour(ax, pool, color, pad=.58, ring=1.6):
    """Rounded hull around the pool, drawn as a line so contours can cross or nest."""
    xs, ys = outline([POSITIONS[p] for p in pool], pad)
    ax.plot(xs, ys, color=color, lw=ring, zorder=2, solid_joinstyle='round')


def dots(ax, hollow='', infected=''):
    for name, (x, y) in POSITIONS.items():
        if name in hollow:
            ax.plot(x, y, 'o', ms=11, mfc='white', mec=BLUE, mew=1.6, zorder=5)
        else:
            ax.plot(x, y, 'o', ms=9, color=WARM if name in infected else INK, zorder=5)
        ax.text(1.75 * x, 1.75 * y, name, ha='center', va='center',
                fontsize=13, color=GRAY, zorder=6)


def panel(ax):
    ax.set(xlim=(-2.3, 2.3), ylim=(-2.9, 2.3), aspect='equal')
    ax.axis('off')


def detector_figure():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98, wspace=.08)
    for ax, (pool, result) in zip(axes[:3], TESTS):
        panel(ax)
        contour(ax, pool, BLUE)
        dots(ax)
        ax.text(0, -2.55, f'R = {result}', ha='center', va='center',
                fontsize=17, color=BLUE)
    ax = axes[3]
    panel(ax)
    dots(ax, hollow=HEALTHY, infected='CD')
    ax.text(0, -2.55, 'A, B sanos', ha='center', va='center', fontsize=17, color=BLUE)
    return fig


def detector_png():
    fig = detector_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='white',
                bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return buf.getvalue()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    Path(sys.argv[1]).write_bytes(detector_png())
