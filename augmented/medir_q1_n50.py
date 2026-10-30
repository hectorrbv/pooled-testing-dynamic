"""Medición en n=50 de la regla ganadora del loop Q1, contra greedy y rollout (Monte Carlo pareado).

Convención (la misma que la batería del loop Q1): conteos exactos, soft clearing (una
persona queda acreditada cuando está sana en todos los perfiles compatibles de su
bloque), laminaridad por trayectoria dentro de una partición fija en bloques de G
personas (ordenadas por q·u descendente y agrupadas de G en G). Dentro de un bloque
cualquier pool compatible con los anteriores de ese bloque es candidato.

Instancias como en experiments_rollout_n50: N=50, B=5, G en {3,5}, u ~ U{1,2,3},
p ~ U[0,1] (p = probabilidad de infección). Políticas:
  greedy    : mayor ganancia inmediata esperada.
  rollout   : un paso de mirada con continuación greedy exacta hasta el final (base greedy).
  lookahead : regla del loop Q1 (profundidad 4; en el primer nivel interno los 4 pools de
              mayor ganancia más el mejor pool de cada tamaño; en los demás niveles el
              mejor más el mejor de cada tamaño; hoja = ganancia inmediata).
Poda común a las tres: candidatos de los bloques ya tocados más los k bloques intactos con
mayor ganancia inmediata (k = --blocks). El greedy no cambia con esta poda.

Salida: CSV por instancia y resumen con diferencias pareadas y errores estándar.
Uso: PYTHONPATH=. python augmented/medir_q1_n50.py --G 3 --instances 48 --draws 24 --procs 8
"""
import argparse
import csv
from itertools import combinations
from math import prod
from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np

D = 4  # profundidad total de la regla lookahead (raíz + 3 niveles internos)


class Block:
    """Un bloque: enumeración exacta de sus 2^m perfiles, pools locales y resultados."""

    def __init__(self, people, q, u):
        self.people = tuple(people)
        self.m = len(people)
        self.q = [float(q[i]) for i in people]
        self.u = [float(u[i]) for i in people]
        self.weights = [prod((1 - self.q[j]) if (z >> j) & 1 else self.q[j] for j in range(self.m))
                        for z in range(1 << self.m)]
        self.full = (1 << (1 << self.m)) - 1
        self.pools = [sum(1 << j for j in S) for k in range(1, self.m + 1)
                      for S in combinations(range(self.m), k)]
        self.outcomes = {}
        for a in self.pools:
            masks = [0] * (bin(a).count('1') + 1)
            for z in range(1 << self.m):
                masks[bin(z & a).count('1')] |= 1 << z
            self.outcomes[a] = masks
        self.compat = {a: {b for b in self.pools if not (a & b) or (a & b) == a or (a & b) == b}
                       for a in self.pools}
        self._mass, self._healthy, self._cands = {}, {}, {}

    def mass(self, support):
        v = self._mass.get(support)
        if v is None:
            v, rest = 0.0, support
            while rest:
                low = rest & -rest
                v += self.weights[low.bit_length() - 1]
                rest ^= low
            self._mass[support] = v
        return v

    def healthy(self, support):
        v = self._healthy.get(support)
        if v is None:
            infected, rest = 0, support
            while rest:
                low = rest & -rest
                infected |= low.bit_length() - 1
                rest ^= low
            v = ((1 << self.m) - 1) & ~infected
            self._healthy[support] = v
        return v

    def utility(self, mask):
        return sum(self.u[j] for j in range(self.m) if (mask >> j) & 1)

    def candidates(self, past):
        c = self._cands.get(past)
        if c is None:
            c = [a for a in self.pools if all(a in self.compat[b] for b in past)]
            self._cands[past] = c
        return c

    def branches(self, support, cleared, pool):
        total = self.mass(support)
        out = []
        for omask in self.outcomes[pool]:
            child = support & omask
            if child:
                h = self.healthy(child)
                out.append((self.mass(child) / total, child, h, self.utility(h & ~cleared)))
        return out

    def gain(self, support, cleared, pool):
        return sum(p * r for p, _, _, r in self.branches(support, cleared, pool))


class Forest:
    """Partición fija en bloques; estado = (supports, cleared, pasts) por bloque."""

    def __init__(self, p, u, G, blocks_limit):
        n = len(p)
        q = 1.0 - np.asarray(p, float)
        order = sorted(range(n), key=lambda i: (-(q[i] * u[i]), i))
        self.blocks = [Block(order[s:s + G], q, u) for s in range(0, n, G)]
        self.limit = blocks_limit
        self.memo_greedy, self.memo_look = {}, {}

    def start(self):
        return (tuple(b.full for b in self.blocks), tuple(0 for _ in self.blocks),
                tuple(() for _ in self.blocks))

    def options(self, state):
        """(gain, block index, pool) para los candidatos admitidos por la poda."""
        supports, cleared, pasts = state
        touched = [i for i, past in enumerate(pasts) if past]
        best_untouched = []
        scored = []
        for i, b in enumerate(self.blocks):
            opts = [(b.gain(supports[i], cleared[i], a), i, a) for a in b.candidates(pasts[i])]
            if not opts:
                continue
            if pasts[i]:
                scored.extend(opts)
            else:
                best_untouched.append((max(opts)[0], i, opts))
        best_untouched.sort(reverse=True)
        for _, _, opts in best_untouched[:self.limit]:
            scored.extend(opts)
        return scored

    def children(self, state, i, pool):
        supports, cleared, pasts = state
        b = self.blocks[i]
        out = []
        for prob, child, h, reward in b.branches(supports[i], cleared[i], pool):
            s = list(supports); s[i] = child
            c = list(cleared); c[i] = cleared[i] | h
            p = list(pasts); p[i] = pasts[i] + (pool,)
            out.append((prob, (tuple(s), tuple(c), tuple(p)), reward))
        return out

    # ---- políticas: devuelven (bloque, pool)
    def greedy_action(self, state):
        g, i, a = max(self.options(state), key=lambda t: (t[0], -t[1], -t[2]))
        return i, a

    def greedy_value(self, state, steps):
        if steps == 0:
            return 0.0
        key = (state, steps)
        v = self.memo_greedy.get(key)
        if v is None:
            i, a = self.greedy_action(state)
            v = sum(prob * (reward + self.greedy_value(child, steps - 1))
                    for prob, child, reward in self.children(state, i, a))
            self.memo_greedy[key] = v
        return v

    def rollout_action(self, state, steps):
        best = None
        for g, i, a in self.options(state):
            v = sum(prob * (reward + self.greedy_value(child, steps - 1))
                    for prob, child, reward in self.children(state, i, a))
            if best is None or v > best[0] + 1e-12:
                best = (v, i, a)
        return best[1], best[2]

    def _look(self, state, steps, depth):
        if steps <= 0:
            return 0.0
        key = (state, steps, depth)
        v = self.memo_look.get(key)
        if v is not None:
            return v
        opts = sorted(self.options(state), key=lambda t: (-t[0], t[1], t[2]))
        if depth <= 1:
            v = opts[0][0] if opts else 0.0
        else:
            k = 4 if depth == D - 1 else 1
            sel = opts[:k]
            seen = {bin(a).count('1') for _, _, a in sel}
            for g, i, a in opts:
                z = bin(a).count('1')
                if z not in seen:
                    seen.add(z); sel.append((g, i, a))
            v = 0.0
            for g, i, a in sel:
                val = g
                if steps > 1:
                    val += sum(prob * self._look(child, steps - 1, depth - 1)
                               for prob, child, _ in self.children(state, i, a))
                v = max(v, val)
        self.memo_look[key] = v
        return v

    def lookahead_action(self, state, steps):
        best = None
        for g, i, a in self.options(state):
            v = g
            if steps > 1:
                v += sum(prob * self._look(child, steps - 1, D - 1)
                         for prob, child, _ in self.children(state, i, a))
            if best is None or v > best[0] + 1e-12:
                best = (v, i, a)
        return best[1], best[2]

    def episode(self, policy, z, B):
        """Bienestar realizado para el perfil latente z (array bool, True = infectado)."""
        state = self.start()
        welfare = 0.0
        for step in range(B):
            steps = B - step
            if policy == 'greedy':
                i, a = self.greedy_action(state)
            elif policy == 'rollout':
                i, a = self.rollout_action(state, steps)
            else:
                i, a = self.lookahead_action(state, steps)
            b = self.blocks[i]
            count = sum(1 for j in range(b.m) if (a >> j) & 1 and z[b.people[j]])
            for prob, child, reward in self.children(state, i, a):
                if child[0][i] == state[0][i] & b.outcomes[a][count]:
                    state = child; welfare += reward
                    break
        return welfare


def one_instance(task):
    G, B, N, draws, blocks_limit, seed = task
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, N)
    u = rng.integers(1, 4, N)
    forest = Forest(p, u, G, blocks_limit)
    rows = {k: [] for k in ('greedy', 'rollout', 'lookahead')}
    times = {k: 0.0 for k in rows}
    for _ in range(draws):
        z = rng.uniform(0, 1, N) < p
        for k in rows:
            t0 = time.perf_counter()
            rows[k].append(forest.episode(k, z, B))
            times[k] += time.perf_counter() - t0
    return dict(seed=seed, G=G, B=B, draws=draws,
                **{f'w_{k}': float(np.mean(v)) for k, v in rows.items()},
                **{f's_{k}': round(times[k], 1) for k in times})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--G', type=int, default=3)
    ap.add_argument('--B', type=int, default=5)
    ap.add_argument('--N', type=int, default=50)
    ap.add_argument('--instances', type=int, default=48)
    ap.add_argument('--draws', type=int, default=24)
    ap.add_argument('--blocks', type=int, default=3, help='bloques intactos admitidos por decisión')
    ap.add_argument('--procs', type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='results/sesion_2026-10-19/q1_n50')
    a = ap.parse_args()
    tasks = [(a.G, a.B, a.N, a.draws, a.blocks, a.seed * 1_000_003 + i) for i in range(a.instances)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        out = pool.map(one_instance, tasks, chunksize=1)
    root = Path(__file__).resolve().parents[1]
    folder = root / a.out
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'q1_n50_G{a.G}_B{a.B}.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    wg = np.array([r['w_greedy'] for r in out]); wr = np.array([r['w_rollout'] for r in out])
    wl = np.array([r['w_lookahead'] for r in out])
    def line(name, x, base):
        d = x - base; se = d.std(ddof=1) / np.sqrt(len(d))
        return f'{name:9s} {x.mean():7.3f}  vs greedy {d.mean():+.4f} (se {se:.4f}) [{100*d.mean()/base.mean():+.2f}%]'
    dl = wl - wr; sel = dl.std(ddof=1) / np.sqrt(len(dl))
    summary = [f'N={a.N} B={a.B} G={a.G} instancias={a.instances} sorteos={a.draws} bloques_intactos={a.blocks} '
               f'tiempo={time.time()-t0:.0f}s',
               line('greedy', wg, wg), line('rollout', wr, wg), line('lookahead', wl, wg),
               f'lookahead vs rollout {dl.mean():+.4f} (se {sel:.4f}) [{100*dl.mean()/wr.mean():+.2f}%]',
               'segundos por instancia (media): ' + ', '.join(
                   f"{k} {np.mean([r[f's_{k}'] for r in out]):.1f}" for k in ('greedy', 'rollout', 'lookahead'))]
    (folder / f'q1_n50_G{a.G}_B{a.B}.txt').write_text('\n'.join(summary) + '\n')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
