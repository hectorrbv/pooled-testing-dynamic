"""FIXED JUDGE (loop Q1) — the agent must never edit this file.

Scores one experiment: imports ``choose`` from the editable file ``scorer.py``, a
laminar policy, and evaluates it EXACTLY on a fixed battery of small instances
whose laminar optimum is known from versioned CSVs. Convention: exact counts,
soft clearing (posterior zero), pools pairwise disjoint or nested along every
path, at most G people per pool, utilities paid once per accredited person.

Metric per cell: policy value / laminar optimum (<= 1). The experiment's score
is the WORST ratio over the battery; the mean ratio breaks ties. Higher is better.

Hard gates (status=fail, result discarded): a pool that is not a candidate
(crosses an earlier pool on the path, too large, empty, unknown people), a
value above the optimum (that would be a solver or judge bug: report it),
anchors not reproduced at startup, a crash inside the scorer.

Usage: PYTHONPATH=. python augmented/autoresearch_q1/judge.py DESCRIPTION
"""
import ast
import csv
import hashlib
import importlib
import os
import sys
import time
from fractions import Fraction as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from q1_tools import Instance, State, pool_mask, members  # noqa: E402


class Invalid(Exception):
    """The policy broke a rule; the experiment is discarded."""


# ---------------------------------------------------------------- battery
def battery():
    """(label, n, B, G, q, u, optimum) from the versioned grids. u = 1 unless stated."""
    cells = []
    path = os.path.join(ROOT, 'results/precio_q05_G/malla_2026-10-14_completa.csv')
    for r in csv.DictReader(open(path)):
        n = int(r['n']); q = [F(r['q'])] * n
        cells.append((f"q05:{r['n']},{r['B']},{r['G']}", n, int(r['B']), int(r['G']), q, [F(1)] * n,
                      F(r['exact_laminar'])))
    for name in ('precio_laminar_malla_n4_5.csv', 'precio_laminar_malla_n6.csv'):
        path = os.path.join(ROOT, 'results/encargos_2026-10-06', name)
        for r in csv.DictReader(open(path)):
            n = int(r['n']); q = [F(r['q'])] * n
            cells.append((f"q{r['q']}:{r['n']},{r['B']},{r['G']}", n, int(r['B']), int(r['G']), q,
                          [F(1)] * n, F(r['laminar'])))
    for seed in (1, 2, 3):
        path = os.path.join(ROOT, f'results/encargos_2026-10-06/binario_vs_laminar_seed{seed}.csv')
        taken = 0
        for i, r in enumerate(csv.DictReader(open(path))):
            if r['n'] == '5' and taken < 4:
                q = [F(x) for x in ast.literal_eval(r['q'])]
                u = [F(x) for x in ast.literal_eval(r['u'])]
                cells.append((f"het:s{seed}r{i}", 5, int(r['B']), int(r['G']), q, u, F(r['laminar_aug'])))
                taken += 1
    return cells


# ---------------------------------------------------------------- exact policy value
def policy_value(choose, n, B, G, q, u, *, counter=None):
    inst = Instance(q, u, G)
    memo = {}

    def V(step, support, cleared, past):
        key = (step, support, cleared, past)
        if key in memo:
            return memo[key]
        if step == B:
            memo[key] = F(0)
            return F(0)
        state = State(inst, B - step, support, cleared, past)
        if counter is not None:
            counter[0] += 1
        choice = choose(state)
        if choice is None:
            memo[key] = F(0)
            return F(0)
        pool = choice if isinstance(choice, int) else pool_mask(choice)
        if pool not in state.candidates:
            raise Invalid(f'pool {members(pool)} is not a candidate after {[members(p) for p in past]} '
                          f'(size > {G}, crossing, empty or unknown people)')
        total, m = F(0), inst.mass(support)
        for omask in inst.outcomes[pool]:
            child = support & omask
            if not child:
                continue
            h = inst.healthy(child)
            total += F(inst.mass(child), m) * (inst.utility(h & ~cleared)
                                               + V(step + 1, child, cleared | h, past + (pool,)))
        memo[key] = total
        return total

    cleared0 = inst.healthy(inst.root)
    return inst.utility(cleared0) + V(0, inst.root, cleared0, ())


# ---------------------------------------------------------------- anchors
def individuals(state):
    t = len(state.past)
    return (1 << t) if t < state.n else None


def replay_tree_value(solver, B):
    """Value of the solver's own optimal tree through the judge's evaluator, matched by counts."""
    tree = solver.tree(B)
    inst_q = [1 - x for x in solver.p]

    class Walker:
        def __init__(self):
            self.by_support = {}

        def __call__(self, state):
            node = self.by_support.get((len(state.past), state.support), tree if not state.past else None)
            if node is None or node.get('stop'):
                return None
            pool = pool_mask(node['pool'])
            inst = state.inst
            for b in node['branches']:
                child = state.support & inst.outcomes[pool][b['infected_count']]
                self.by_support[(len(state.past) + 1, child)] = b['next']
            return pool

    return policy_value(Walker(), solver.n, B, solver.G, inst_q, solver.u)


def anchors():
    from augmented.bellman_perfiles import BellmanPerfiles
    cells = battery()
    assert len(cells) == 32 + 54 + 27 + 12, len(cells)
    assert policy_value(individuals, 4, 3, 3, [F(1, 2)] * 4, [F(1)] * 4) == F(3, 2)
    picks = [c for c in cells if c[0] in ('q05:6,4,3', 'q0.3:4,2,2', 'q0.7:5,3,3')]
    picks += [c for c in cells if c[0].startswith('het:s1')][:1]
    assert len(picks) == 4, [c[0] for c in picks]
    for label, n, B, G, q, u, opt in picks:
        solver = BellmanPerfiles([1 - x for x in q], u, G, laminar=True)
        assert solver.solve(B).value == opt, (label, solver.solve(B).value, opt)
        assert replay_tree_value(solver, B) == opt, label
    try:
        def crossing(state):
            return pool_mask((0, 1)) if not state.past else pool_mask((1, 2))
        policy_value(crossing, 4, 2, 2, [F(1, 2)] * 4, [F(1)] * 4)
    except Invalid:
        pass
    else:
        raise AssertionError('crossing pool accepted')


def scorer_sha():
    return hashlib.sha256(open(os.path.join(HERE, 'scorer.py'), 'rb').read()).hexdigest()[:10]


def main():
    desc = ' '.join(sys.argv[1:])
    t0 = time.time()
    anchors()
    scorer = importlib.import_module('scorer')
    cells = battery()
    row = {'status': 'ok', 'worst_ratio': '', 'mean_ratio': '', 'exact_cells': '', 'cells': len(cells),
           'worst_cell': '', 'scorer_calls': 0, 'scorer_sha': scorer_sha()}
    ratios = []
    counter = [0]
    try:
        for label, n, B, G, q, u, opt in cells:
            value = policy_value(scorer.choose, n, B, G, q, u, counter=counter)
            if value > opt:
                raise Invalid(f'value {value} above optimum {opt} at {label} — report this')
            ratios.append((value / opt, label))
    except Invalid as exc:
        row['status'] = 'fail'
        row['error'] = str(exc)
    except Exception as exc:
        row['status'] = 'fail'
        row['error'] = f'{type(exc).__name__}: {exc}'
    if row['status'] == 'ok':
        worst = min(ratios)
        row['worst_ratio'] = float(worst[0])
        row['mean_ratio'] = float(sum(r for r, _ in ratios) / len(ratios))
        row['exact_cells'] = sum(1 for r, _ in ratios if r == 1)
        row['worst_cell'] = worst[1]
    row['scorer_calls'] = counter[0]
    row['seconds'] = round(time.time() - t0, 1)
    row['description'] = desc
    row.setdefault('error', '')
    out = os.path.join(HERE, 'results_q1.tsv')
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter='\t')
        if new:
            w.writeheader()
        w.writerow(row)
    detail = os.path.join(HERE, 'last_run_cells.tsv')
    with open(detail, 'w') as f:
        f.write('cell\tratio\n')
        for r, label in sorted(ratios):
            f.write(f'{label}\t{float(r):.6f}\n')
    print('\t'.join(f'{k}={v}' for k, v in row.items() if k != 'description'))


if __name__ == '__main__':
    main()
