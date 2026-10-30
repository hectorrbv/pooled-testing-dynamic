"""Exact comparison for the 9-by-15 detecting matrix in notebook 29.

Run: python -m augmented.experimento_matriz15_laminar

At q=1/2 and u=1, the Bellman numerator is an integer sum over all
compatible profiles. No floating point is used for decisions. The laminar
normal-form reduction under soft clearing permits fresh pools and splits of
one known-count atom; symmetry makes identities within each atom irrelevant.
All split sizes are considered. This independently implements the recurrence
used by bellman_tipos, then checks the selected labeled policy directly by
partitioning the 32,768 profiles by their actual observed counts.
"""
from functools import lru_cache
from fractions import Fraction as F
from math import comb, prod
from pathlib import Path
from time import perf_counter
import csv
import gzip
import hashlib
import json

from augmented.bellman_tipos import crear_solver

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sesion_2026-10-20/matriz15_laminar'


class ExactHomogeneousLaminar:
    def __init__(self, G):
        self.G = G
        self.argmax = {}
        self.total = lru_cache(None)(self._total)

    @staticmethod
    @lru_cache(None)
    def mass(v, atoms):
        return 2**v * prod(comb(m, r) for m, r in atoms)

    @staticmethod
    def pieces(parts):
        reward = sum(m for m, r in parts if r == 0)
        return reward, tuple(sorted((m, r) for m, r in parts if 0 < r < m))

    def options(self, v, atoms, b):
        if not b:
            return
        for k in range(1, min(v, self.G) + 1):
            value = 0
            for r in range(k + 1):
                reward, new = self.pieces(((k, r),))
                nxt = tuple(sorted(atoms + new))
                value += reward * self.mass(v-k, nxt) + self.total(v-k, nxt, b-1)
            yield ('open', k), value
        for atom in sorted(set(atoms)):
            rest = list(atoms)
            rest.remove(atom)
            m, r = atom
            for k in range(1, m):
                value = 0
                for s in range(max(0, r-m+k), min(k, r) + 1):
                    reward, new = self.pieces(((k, s), (m-k, r-s)))
                    nxt = tuple(sorted(tuple(rest) + new))
                    value += reward * self.mass(v, nxt) + self.total(v, nxt, b-1)
                yield ('ref', atom, k), value

    def _total(self, v, atoms, b):
        best, action = 0, None
        for candidate, value in self.options(v, atoms, b):
            if value > best:
                best, action = value, candidate
        self.argmax[v, atoms, b] = action
        return best

    def value(self, n, B):
        return F(self.total(n, (), B), 2**n)


def matrix15():
    """Section 3: A_15 = [Q tensor A_4 | I_3; I_3; I_3]."""
    Q = ((1, 0, 1), (0, 1, 1), (1, 1, 0))
    A4 = ((1, 0, 1, 1), (0, 1, 1, 0), (1, 1, 0, 0))
    return [[Q[a][b] * A4[i][j] for b in range(3) for j in range(4)]
            + [int(i == j) for j in range(3)]
            for a in range(3) for i in range(3)]


def replay(solver, n, B):
    """Evaluate terminal credit from compatible profiles, not DP rewards."""
    nodes = []
    max_used = max_pool = 0

    def visit(profiles, virgins, atoms, b, past):
        nonlocal max_used, max_pool
        node_id = len(nodes)
        node = dict(remaining_tests=b, profiles=len(profiles))
        nodes.append(node)
        types = tuple(sorted((len(people), r) for people, r in atoms))
        action = solver.argmax[len(virgins), types, b]
        max_used = max(max_used, B-b)
        if action is None:
            infection_union = 0
            for z in profiles:
                infection_union |= z
            healthy = ((1 << n) - 1) ^ infection_union
            node['healthy_mask'] = healthy
            return node_id, len(profiles) * healthy.bit_count()

        if action[0] == 'open':
            pool = virgins[:action[1]]
            rest_virgins = virgins[action[1]:]
            rest_atoms = atoms
        else:
            index = next(i for i, (people, r) in enumerate(atoms)
                         if (len(people), r) == action[1])
            people, r = atoms[index]
            pool = people[:action[2]]
            rest_atoms = atoms[:index] + atoms[index+1:]
            rest_virgins = virgins
        mask = sum(1 << i for i in pool)
        assert 1 <= len(pool) <= solver.G
        assert all(mask & prev in (0, mask, prev) for prev in past)
        max_pool = max(max_pool, len(pool))
        node.update(pool=list(pool), branches=[])
        branches = {}
        for z in profiles:
            branches.setdefault((z & mask).bit_count(), []).append(z)
        total = 0
        for count, group in sorted(branches.items()):
            parts = [(pool, count)]
            if action[0] == 'ref':
                parts.append((people[action[2]:], r-count))
            new = tuple((members, cnt) for members, cnt in parts if 0 < cnt < len(members))
            child, value = visit(group, rest_virgins, tuple(sorted(rest_atoms + new)),
                                 b-1, past + (mask,))
            node['branches'].append(dict(count=count, child=child))
            total += value
        return node_id, total

    _, total = visit(list(range(2**n)), tuple(range(n)), (), B, ())
    return dict(value=str(F(total, 2**n)), total_credited=total, profiles=2**n,
                max_tests_used=max_used, max_pool_size=max_pool, nodes=len(nodes),
                laminar_checked_on_every_path=True), nodes


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = matrix15()
    pools = [sum(int(bit) << i for i, bit in enumerate(row)) for row in matrix]
    signatures = {tuple((z & pool).bit_count() for pool in pools) for z in range(2**15)}
    assert len(signatures) == 2**15 and max(map(int.bit_count, pools)) == 7

    start = perf_counter()
    solver = ExactHomogeneousLaminar(7)
    value = solver.value(15, 9)
    seconds = perf_counter() - start
    check, tree = replay(solver, 15, 9)
    assert F(check['value']) == value and check['max_tests_used'] <= 9

    reference = crear_solver(.5, 7)
    float_value = reference(15, (), 9)
    root_values = list(solver.options(15, (), 9))
    for (a, exact), (a_ref, approx) in zip(root_values, reference.action_values(15, (), 9)):
        assert a == a_ref and abs(float(F(exact, 2**15)) - approx) < 1e-12

    # Previous exact grid: independent labeled-profile Bellman certificates.
    grid_path = ROOT / 'results/precio_q05_G/malla_2026-10-14_completa.csv'
    controls = []
    for row in csv.DictReader(grid_path.open()):
        n, B, G = (int(row[key]) for key in ('n', 'B', 'G'))
        actual = ExactHomogeneousLaminar(G).value(n, B)
        assert actual == F(row['exact_laminar'])
        controls.append(dict(n=n, B=B, G=G, value=str(actual)))
    assert len(controls) == 32
    assert ExactHomogeneousLaminar(3).value(4, 3) == F(7, 4)
    assert ExactHomogeneousLaminar(1).value(15, 9) == F(9, 2)
    assert ExactHomogeneousLaminar(7).value(15, 1) == F(1, 2)

    general = F(15, 2)
    source_paths = [Path(__file__).resolve(), ROOT/'augmented/bellman_tipos.py', grid_path]
    result = dict(
        setup=dict(n=15, B=9, G=7, q='1/2', u=1, clearing='soft'),
        matrix=dict(source='Soderberg and Shapiro (1963), Section 3',
                    entries=matrix, row_sizes=list(map(int.bit_count, pools)),
                    profiles=2**15, distinct_count_vectors=len(signatures)),
        general=dict(value=str(general), normalized=str(general/9),
                     optimal=True, reason='Full identification attains total expected healthy utility n/2.'),
        laminar=dict(value=str(value), decimal=float(value), normalized=str(value/9),
                     states=solver.total.cache_info().currsize, seconds=seconds,
                     first_pool_size=solver.argmax[15, (), 9][1],
                     root_action_values={str(a[1]): str(F(v, 2**15)) for a, v in root_values},
                     method='Integer Bellman over homogeneous residual atoms; all normal-form actions.',
                     scope='Pathwise laminar dynamic counting, soft clearing, hard branchwise budget.',
                     normal_form_reference='Companion Theorem 4.1; docs/notes/2026-08-25-mapeo-companion-francisco.md',
                     reference_float_value=float_value, replay=check),
        individual='9/2', general_over_laminar=str(general/value),
        laminar_loss_fraction=str(1-value/general), grid_controls=controls,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (OUT/'resultados.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (OUT/'matriz.csv').open('w', newline='') as f:
        csv.writer(f).writerows(matrix)
    with gzip.open(OUT/'politica_laminar.json.gz', 'wt') as f:
        json.dump(dict(root=0, nodes=tree), f, separators=(',', ':'))
    print(json.dumps({k:result[k] for k in ['setup','laminar','general','general_over_laminar',
                                          'laminar_loss_fraction']}, indent=2))
    return result


if __name__ == '__main__':
    run()
