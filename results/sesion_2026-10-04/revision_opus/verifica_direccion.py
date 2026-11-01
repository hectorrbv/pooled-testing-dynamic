"""Exact checks for the research review; no optimization algorithms changed.

Run from repository root with the project Python. Checks saved optimal trees,
independent virtual-path replay, the elementary 1/G bound, and the residual-atom
description of every laminar history of <=3 tests on four individuals.
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from hashlib import sha256
import json

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'results/precio_laminar_2026-09-21/resultados.json'


def weight(z, p):
    result = F(1)
    for i, pi in enumerate(p):
        result *= pi if z >> i & 1 else 1-pi
    return result


def path_on_profile(tree, z):
    path = []
    while tree.get('pool') is not None:
        pool = sum(1 << i for i in tree['pool'])
        path.append(pool)
        r = (pool & z).bit_count()
        tree = next(b['next'] for b in tree['branches'] if b['infected_count'] == r)
    return tuple(path)


def fixed_value(path, weights, u):
    signatures = defaultdict(list)
    for z in weights:
        signatures[tuple((z & a).bit_count() for a in path)].append(z)
    value = F(0)
    for profiles in signatures.values():
        possibly_infected = 0
        for z in profiles:
            possibly_infected |= z
        credit = sum(ui for i, ui in enumerate(u) if not possibly_infected >> i & 1)
        value += sum(weights[z] for z in profiles) * credit
    return value


def laminar(a, b):
    return a & b in (0, a, b)


def main():
    rows = []
    for case in json.loads(SOURCE.read_text())['cases']:
        instance = case['instance']
        if instance['history']:
            continue  # The product-prior argument is at empty initial history.
        p = tuple(map(F, instance['p']))
        u = tuple(map(F, instance['u']))
        n, B, G = len(p), instance['B'], instance['G']
        weights = {z: weight(z, p) for z in range(1 << n)}
        assert all(w > 0 for w in weights.values())
        tree = case['general']['tree']
        paths = defaultdict(F)
        original = F(0)
        for z, w in weights.items():
            path = path_on_profile(tree, z)
            assert len(path) <= B and all(0 < a.bit_count() <= G for a in path)
            paths[path] += w
            signature = tuple((z & a).bit_count() for a in path)
            compatible = [y for y in weights
                          if tuple((y & a).bit_count() for a in path) == signature]
            infected_union = 0
            for y in compatible:
                infected_union |= y
            original += w * sum(ui for i, ui in enumerate(u)
                                if not infected_union >> i & 1)
        assert sum(paths.values()) == 1
        assert original == F(case['general']['value'])
        replay = sum(prob * fixed_value(path, weights, u) for path, prob in paths.items())
        assert replay <= original
        individual = sum(sorted((ui*(1-pi) for ui, pi in zip(u, p)), reverse=True)[:B])
        assert individual >= original / G
        rows.append(dict(id=instance['id'], n=n, B=B, G=G,
                         general=str(original), laminar=case['laminar']['value'],
                         individual=str(individual), virtual_path_replay=str(replay),
                         replay_over_general=str(replay/original),
                         distinct_virtual_paths=len(paths)))

    families = histories = refinements = 0
    pools = range(1, 16)
    for b in range(4):
        for family in combinations(pools, b):
            if any(not laminar(a, c) for a, c in combinations(family, 2)):
                continue
            families += 1
            atoms = []
            covered = 0
            for a in family:
                strict_children = [c for c in family if c != a and c & a == c]
                children = [c for c in strict_children
                            if not any(c != d and c & d == c for d in strict_children)]
                residual = a
                for c in children:
                    residual &= ~c
                if residual:
                    atoms.append((a, tuple(children), residual))
                    covered |= residual
            # Every legal new test cuts at most one current residual atom.
            partition = [atom[2] for atom in atoms]
            static_value = sum((F(a.bit_count(), 2**a.bit_count())
                                for a in partition), F(0))
            assert static_value <= F(b, 2)
            if 15 & ~covered:
                partition.append(15 & ~covered)
            for new in pools:
                if all(laminar(new, a) for a in family):
                    assert sum(0 < (a & new) < a for a in partition) <= 1
                    refinements += 1
            supports = defaultdict(list)
            for z in range(16):
                supports[tuple((a & z).bit_count() for a in family)].append(z)
            for signature, support in supports.items():
                histories += 1
                counts = dict(zip(family, signature))
                credited = 0
                for a, children, residual in atoms:
                    r = counts[a] - sum(counts[c] for c in children)
                    if r == 0:
                        credited |= residual
                infected_union = 0
                for z in support:
                    infected_union |= z
                assert credited == (15 & ~infected_union)
    result = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
                  scope='Finite independent checks, not proofs or worst-case guarantees.',
                  path_replay=rows, atom_checks=dict(n=4, max_tests=3, families=families,
                                                     histories=histories, legal_refinements=refinements,
                                                     static_laminar_bound_checks=families))
    Path(__file__).with_name('verificacion_direccion.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
