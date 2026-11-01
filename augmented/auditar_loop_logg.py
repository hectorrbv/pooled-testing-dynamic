"""Independent leaf enumeration of a fixed log-G design (not an optimality proof).

Uses Python tuples and integer sums, without judge.evaluate, numpy supports,
or the design's internal value function for welfare accounting.
"""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/sesion_2026-10-20/loops_revision'
TARGETS = [(16, 4, 2), (16, 4, 4), (16, 4, 8), (16, 4, 16), (16, 8, 8), (16, 8, 16)]


def enumerate_leaves(policy_factory, n, B, G):
    next_pool = policy_factory(n, B, G)
    leaves = []
    nodes = []
    maximum_pool = 0
    maximum_component = 0
    nonlaminar = None

    def walk(history, worlds, parent=None, outcome=None):
        nonlocal maximum_pool, maximum_component, nonlaminar
        node_id = len(nodes)
        node = dict(id=node_id, parent=parent, result=outcome, profiles=len(worlds))
        nodes.append(node)
        pool = None if len(history) == B else next_pool(history)
        if pool is not None:
            pool = tuple(pool)
            assert all(type(i) is int and 0 <= i < n for i in pool)
            assert 1 <= len(pool) <= G and len(set(pool)) == len(pool)
            maximum_pool = max(maximum_pool, len(pool))
            node['pool'] = pool
            parts = {}
            for z in worlds:
                r = sum((z >> i) & 1 for i in pool)
                parts.setdefault(r, []).append(z)
            for r, profiles in sorted(parts.items()):
                walk(history + ((pool, r),), tuple(profiles), node_id, r)
            return
        possible_infected = set()
        for z in worlds:
            possible_infected.update(i for i in range(n) if (z >> i) & 1)
        healthy = sorted(set(range(n)) - possible_infected)
        node.update(healthy=healthy, stop=True)
        masks = [set(p) for p, _ in history]
        if nonlaminar is None and any(a & b and not a <= b and not b <= a
                                      for a in masks for b in masks):
            nonlaminar = history
        components = []
        for pool_set in masks:
            touching = [c for c in components if c & pool_set]
            components = [c for c in components if not c & pool_set]
            components.append(set.union(pool_set, *touching))
        maximum_component = max(maximum_component, max(map(len, components), default=0))
        leaves.append((len(worlds), len(healthy), len(history)))

    walk((), tuple(range(1 << n)))
    assert sum(size for size, _, _ in leaves) == 1 << n
    value = Fraction(sum(size * healthy for size, healthy, _ in leaves), 1 << n)
    return dict(value=str(value), value_per_B=float(value / B), leaves=len(leaves),
                nodes=len(nodes), max_tests=max(t for _, _, t in leaves),
                max_pool=maximum_pool, max_connected_component=maximum_component,
                nonlaminar_history=nonlaminar), nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    design_path = ROOT / 'augmented/autoresearch_logg/design_best.py'
    sha = hashlib.sha256(design_path.read_bytes()).hexdigest()
    records = list(csv.DictReader((design_path.parent / 'results_logg.tsv').open(), delimiter='\t'))
    records = [r for r in records if r['status'] == 'ok' and r['design_sha'] == sha[:10]]
    assert records, 'No recorded run matches the current design SHA'
    expected = records[-1]
    spec = importlib.util.spec_from_file_location('fixed_design_for_audit', design_path)
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    # Controls for the independent evaluator; no reference-judge code used.
    def individuals(n, B, G):
        return lambda h: (len(h),) if len(h) < min(n, B) else None
    assert enumerate_leaves(individuals, 4, 3, 3)[0]['value'] == '3/2'
    def detector(n, B, G):
        pools = ((0, 1, 2), (0, 1, 3), (0, 2, 3))
        return lambda h: pools[len(h)] if len(h) < 3 else None
    assert enumerate_leaves(detector, 4, 3, 3)[0]['value'] == '2'
    audit = dict(design_sha256=sha, method='independent full-profile leaf enumeration',
                 scope='value of the fixed design; no unrestricted optimality claim', targets=[])
    for n, B, G in TARGETS:
        started = time.perf_counter()
        result, tree = enumerate_leaves(design.policy, n, B, G)
        result.update(n=n, B=B, G=G, seconds=round(time.perf_counter() - started, 3))
        result['expected'] = expected[f'value_n{n}_B{B}_G{G}']
        result['difference'] = str(Fraction(result['value']) - Fraction(result['expected']))
        assert result['difference'] == '0', result
        tree_path = args.out / f'tree_n{n}_B{B}_G{G}.json.gz'
        with gzip.open(tree_path, 'wt') as f:
            json.dump(dict(n=n, B=B, G=G, nodes=tree), f)
        result['tree'] = str(tree_path.relative_to(ROOT))
        audit['targets'].append(result)
        (args.out / 'loop1_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({k:result[k] for k in ('n','B','G','value','difference','max_pool','max_connected_component','seconds')}), flush=True)
    audit['complete'] = True
    (args.out / 'loop1_audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
