"""FIXED JUDGE — the agent must never edit this file.

Scores one experiment: imports generator.sample(rng) from the editable file,
draws instances for a fixed wall-clock budget, solves each one EXACTLY with the
repository solvers, and reports the minimum ratio found (lower = better) plus
the full argmin instance so a human can re-solve it. Two missions share it:

  binario   : minimize  laminar_augmented / binary_dynamic   (Francisco, request 1:
              "an example, the worst possible, where optimal binary beats laminar";
              only instances with Lambda > 1 count — with Lambda <= 1 the ratio is >= 1 provably)
  precio    : minimize  laminar_augmented / unrestricted_augmented  (request 2: price of
              laminarity; Francisco conjectures the worst case near q = 1/2)

Hard gates (any failure -> status=fail, result discarded): instance limits
(n<=NMAX, B<=4, G<=4, 0<q<1, u>0), anchor values reproduced at startup,
argmin re-solved a second time from scratch with identical results.

Usage: PYTHONPATH=. python augmented/autoresearch_ejemplos/judge.py MISSION BUDGET_SECONDS NMAX DESCRIPTION
"""
import importlib, os, random, sys, time, csv
from fractions import Fraction as F
from itertools import combinations
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmented.dp_perfiles_binario_conteo import DP, Lambda
from augmented.bellman_perfiles import BellmanPerfiles

def laminar(q, u, G, B):
    return BellmanPerfiles([1 - x for x in q], list(u), G, laminar=True).solve(B).value

def anchors():
    q = [F(3, 10)] * 4; u = [F(1)] * 4
    assert DP(q, u, 2, 'count').value(2) == F(387, 500)
    assert DP(q, u, 2, 'count').value(3) == F(1137, 1000)
    assert DP([F(1, 2)] * 4, u, 3, 'count').value(3) == 2
    assert DP(q, u, 2, 'binary').value(2) == F(3, 5)
    assert DP(q, u, 2, 'binary').value(3) == F(9, 10)
    assert laminar(q, u, 2, 3) == F(537, 500)
    assert laminar([F(9,10),F(7,10),F(7,10)], [F(1)]*3, 3, 2) == F(1031, 500)
    assert DP([F(9,10),F(7,10),F(7,10)], [F(1)]*3, 3, 'binary').value(2) == F(2079, 1000)

def validate(inst, NMAX):
    """Returns the instance with Fractions, or None if it breaks a limit (counted as invalid)."""
    try:
        n, B, G, q, u = inst
        if not (2 <= n <= NMAX and 1 <= B <= 4 and 2 <= G <= min(4, n) and len(q) == n and len(u) == n):
            return None
        q = [F(x) for x in q]; u = [F(x) for x in u]
        if not (all(0 < x < 1 for x in q) and all(x > 0 for x in u)):
            return None
        return n, B, G, q, u
    except Exception:
        return None

def score(mission, inst):
    n, B, G, q, u = inst
    lam = laminar(q, u, G, B)
    if mission == 'binario':
        if Lambda(q, G) <= 1:
            return None, lam, None
        other = DP(q, u, G, 'binary').value(B)
    else:
        other = DP(q, u, G, 'count').value(B)
    return lam / other, lam, other

def main():
    mission, budget, NMAX, desc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), ' '.join(sys.argv[4:])
    assert mission in ('binario', 'precio')
    anchors()
    gen = importlib.import_module('generator')
    rng = random.Random(20261013)
    t0 = time.time(); best = None; tried = 0; skipped = 0; invalid = 0
    while time.time() - t0 < budget:
        inst = validate(gen.sample(rng), NMAX); tried += 1
        if inst is None:
            invalid += 1
            if invalid > 50 and invalid > tried // 2:
                break
            continue
        r, lam, other = score(mission, inst)
        if r is None:
            skipped += 1; continue
        if best is None or r < best[0]:
            best = (r, inst, lam, other)
    status = 'fail'
    if best is not None and not (invalid > 50 and invalid > tried // 2):
        r2, _, _ = score(mission, best[1])      # second solve, from scratch
        status = 'ok' if r2 == best[0] else 'fail'
    row = {
        'mission': mission, 'ratio_min': float(best[0]) if best else '', 'status': status,
        'n': best[1][0] if best else '', 'B': best[1][1] if best else '', 'G': best[1][2] if best else '',
        'q': [str(x) for x in best[1][3]] if best else '', 'u': [str(x) for x in best[1][4]] if best else '',
        'laminar': str(best[2]) if best else '', 'other': str(best[3]) if best else '',
        'tried': tried, 'skipped_lambda_le_1': skipped, 'invalid': invalid, 'seconds': round(time.time() - t0, 1),
        'generator_sha': __import__('hashlib').sha256(open(os.path.join(os.path.dirname(__file__), 'generator.py'), 'rb').read()).hexdigest()[:10],
        'description': desc,
    }
    out = os.path.join(os.path.dirname(__file__), f'results_{mission}.tsv')
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter='\t')
        if new: w.writeheader()
        w.writerow(row)
    print('\t'.join(f'{k}={v}' for k, v in row.items() if k != 'description'))

if __name__ == '__main__':
    main()
