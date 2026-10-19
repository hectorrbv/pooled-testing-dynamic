"""FIXED JUDGE (loop log-G) — the agent must never edit this file.

Scores one experiment: imports ``policy`` from the editable file ``design.py`` and
evaluates it EXACTLY on fixed targets (n, B, G) with q = 1/2, u = 1, independent
states and soft clearing (a person is accredited healthy when every profile
compatible with the observed counts has that person healthy).

The metric is the expected number of accredited healthy people divided by the
budget B, one value per target; the score is their sum (higher = better). The
evaluation enumerates all 2^n profiles, so nothing is sampled or approximated.

Hard gates (any failure -> status=fail, result discarded): an invalid pool (size
above G, repeated or unknown person), more than B tests, a policy that is not
deterministic (second evaluation from scratch must agree), anchor values not
reproduced at startup, or a value above the information bound B*log2(G+1).

Policy interface (design.py):
    policy(n, B, G) -> next_pool(history)
    history: tuple of (pool, count); pool: tuple of people; count: infected in it
    next_pool returns the next pool (iterable of distinct people, 1..G of them)
    or None to stop testing.

Usage: PYTHONPATH=. python augmented/autoresearch_logg/judge.py DESCRIPTION
"""
import csv
import hashlib
import importlib
import os
import sys
import time
from fractions import Fraction as F
from math import log2

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

TARGETS = ((16, 4, 2), (16, 4, 4), (16, 4, 8), (16, 4, 16), (16, 8, 8), (16, 8, 16))
NMAX = 20


class Invalid(Exception):
    """The policy broke a rule; the experiment is discarded."""


def validate_pool(pool, n, G):
    if pool is None:
        return None
    try:
        pool = tuple(int(i) for i in pool)
    except Exception as exc:
        raise Invalid(f'pool is not a sequence of people: {pool!r}') from exc
    if not 1 <= len(pool) <= G:
        raise Invalid(f'pool size {len(pool)} not in 1..{G}: {pool}')
    if len(set(pool)) != len(pool) or any(not 0 <= i < n for i in pool):
        raise Invalid(f'pool with repeated or unknown people: {pool}')
    return pool


def evaluate(policy_factory, n, B, G):
    """Exact E[#accredited healthy] of a deterministic adaptive policy, q = 1/2, u = 1.

    Returns (value: Fraction, max_tests_used: int, distinct_histories: int).
    """
    if not (1 <= n <= NMAX and 1 <= B and 1 <= G <= n):
        raise Invalid(f'target out of range: n={n} B={B} G={G}')
    N = 1 << n
    profiles = np.arange(N, dtype=np.uint32)          # bit i set = person i infected
    group = np.zeros(N, dtype=np.int64)                # history id of each profile
    histories = {0: ()}
    stopped = set()
    next_pool = policy_factory(n, B, G)
    used = 0
    for _ in range(B):
        ids = sorted(histories)
        masks = np.zeros(len(ids), dtype=np.uint32)
        pools = {}
        for gid in ids:
            if gid in stopped:
                continue
            pool = validate_pool(next_pool(histories[gid]), n, G)
            if pool is None:
                stopped.add(gid)
                continue
            pools[gid] = pool
            masks[gid] = sum(1 << i for i in pool)
        if not pools:
            break
        used += 1
        counts = np.bitwise_count(profiles & masks[group]).astype(np.int64)
        keys = group * (G + 1) + counts
        new_ids, inverse = np.unique(keys, return_inverse=True)
        new_histories, new_stopped = {}, set()
        for new_gid, key in enumerate(new_ids.tolist()):
            gid, count = divmod(int(key), G + 1)
            if gid in stopped:
                new_histories[new_gid] = histories[gid]
                new_stopped.add(new_gid)
            else:
                new_histories[new_gid] = histories[gid] + ((pools[gid], count),)
        group = inverse.astype(np.int64).reshape(-1)
        histories, stopped = new_histories, new_stopped
    infected_union = np.zeros(len(histories), dtype=np.uint32)
    np.bitwise_or.at(infected_union, group, profiles)
    cleared = np.bitwise_count((~infected_union) & np.uint32(N - 1))
    total = int(cleared[group].astype(np.int64).sum())
    return F(total, N), used, len(histories)


# ---------------------------------------------------------------- anchors
def individuals(n, B, G):
    def next_pool(history):
        t = len(history)
        return (t,) if t < min(n, B) else None
    return next_pool


def static(pools):
    def factory(n, B, G):
        def next_pool(history):
            t = len(history)
            return pools[t] if t < len(pools) else None
        return next_pool
    return factory


def tree_policy(tree):
    """Replay a decision tree of augmented.bellman_perfiles.BellmanPerfiles.tree()."""
    def factory(n, B, G):
        def next_pool(history):
            node = tree
            for pool, count in history:
                assert tuple(node['pool']) == tuple(pool)
                node = next(b['next'] for b in node['branches'] if b['infected_count'] == count)
            return None if node.get('stop') else tuple(node['pool'])
        return next_pool
    return factory


def anchors():
    from augmented.bellman_perfiles import BellmanPerfiles
    assert evaluate(individuals, 4, 3, 3)[0] == F(3, 2)
    assert evaluate(individuals, 8, 4, 2)[0] == 2
    assert evaluate(static([(0, 1)]), 2, 1, 2)[0] == F(1, 2)
    assert evaluate(static([(0, 1, 2), (0, 1, 3), (0, 2, 3)]), 4, 3, 3)[0] == 2
    half = [F(1, 2)]
    for n, B, G, value in ((4, 3, 3, F(2)), (6, 3, 3, F(9, 4)), (7, 4, 3, F(13, 4))):
        solver = BellmanPerfiles(half * n, [F(1)] * n, G, laminar=False)
        assert solver.solve(B).value == value
        assert evaluate(tree_policy(solver.tree(B)), n, B, G)[0] == value
    try:
        evaluate(static([(0, 1, 2, 3)]), 4, 1, 3)
    except Invalid:
        pass
    else:
        raise AssertionError('oversized pool accepted')


def design_sha():
    return hashlib.sha256(open(os.path.join(HERE, 'design.py'), 'rb').read()).hexdigest()[:10]


def main():
    desc = ' '.join(sys.argv[1:])
    t0 = time.time()
    anchors()
    design = importlib.import_module('design')
    row = {'status': 'ok', 'score': '', 'design_sha': design_sha()}
    detail = {}
    try:
        for n, B, G in TARGETS:
            first = evaluate(design.policy, n, B, G)
            second = evaluate(design.policy, n, B, G)
            if first[0] != second[0]:
                raise Invalid(f'policy is not deterministic at n={n} B={B} G={G}')
            value, used, hist = first
            if float(value) > B * log2(G + 1) + 1e-9:
                raise Invalid(f'information bound violated at n={n} B={B} G={G}: {value} — report this')
            detail[(n, B, G)] = (value, used, hist)
    except Invalid as exc:
        row['status'] = 'fail'
        row['error'] = str(exc)
    except Exception as exc:  # a crash in the design counts as a failed experiment
        row['status'] = 'fail'
        row['error'] = f'{type(exc).__name__}: {exc}'
    if row['status'] == 'ok':
        row['score'] = float(sum(v / B for (n, B, G), (v, _, _) in detail.items()))
    for n, B, G in TARGETS:
        tag = f'n{n}_B{B}_G{G}'
        v, used, hist = detail.get((n, B, G), ('', '', ''))
        row[f'value_{tag}'] = str(v)
        row[f'per_test_{tag}'] = float(v / B) if v != '' else ''
        row[f'bits_{tag}'] = float(v / (B * log2(G + 1))) if v != '' else ''
        row[f'tests_used_{tag}'] = used
    row['seconds'] = round(time.time() - t0, 1)
    row['description'] = desc
    row.setdefault('error', '')
    out = os.path.join(HERE, 'results_logg.tsv')
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter='\t')
        if new:
            w.writeheader()
        w.writerow(row)
    print('\t'.join(f'{k}={v}' for k, v in row.items() if k != 'description'))


if __name__ == '__main__':
    main()
