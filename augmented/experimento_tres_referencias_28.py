"""Add the laminar optimum to the existing 228-case n=6 general map.

Run: python -m augmented.experimento_tres_referencias_28
Same cases, priors, utility, budget and cap for individual, laminar and general.
"""
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import signal

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.experimento_sesion_28 import evaluate
from augmented.experimento_mapa_laminar_28 import policy_from_tree

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/diseno_mapa_2026-10-05/agrupar_vs_individuales.json'
OUT = ROOT/'results/sesion_2026-10-06/tres_referencias_28.json'


def timeout(*_):
    raise TimeoutError('Laminar case exceeded 40 seconds; optimum not certified')


def run():
    base = json.loads(BASE.read_text())
    assert len(base['rows']) == base['audit']['exact_cases'] == 228
    for source, digest in base['source_sha256'].items():
        assert sha256((ROOT/source).read_bytes()).hexdigest() == digest
    reference = {(r['G'], r['B'], r['q']): r for r in base['rows']}
    signal.signal(signal.SIGALRM, timeout)
    start, rows, atom_checks = perf_counter(), [], 0
    for cap in (2, 3):
        for numerator in range(1, 20):
            q = F(numerator, 20)
            lam = BellmanPerfiles([1-q]*6, [1]*6, cap, laminar=True, state_limit=500_000)
            atom = SolverLaminar(dict(enumerate([1-q]*6)), dict(enumerate([F(1)]*6)),
                                 cap, 'posterior_zero')
            for budget in range(1, 7):
                signal.alarm(40)
                try:
                    prior = reference[cap, budget, str(q)]
                    individual, general = F(prior['individual']), F(prior['grouping'])
                    if budget == 6:
                        value = 6*q
                        action = lambda support, history, remaining: (len(history),) if len(history)<6 else None
                        certificate = 'individuals attain total expected healthy utility'
                    else:
                        value = lam.solve(budget).value
                        action = policy_from_tree(lam.tree(budget))
                        assert value == atom.V(frozenset(range(6)), (), budget)
                        atom_checks += 1
                        certificate = 'profile Bellman and independent atom solver agree'
                    audited = evaluate(action, [q]*6, [F(1)]*6, budget, cap, require_laminar=True)
                    assert F(audited['value']) == value
                    assert individual <= value <= general <= 6*q
                    if budget == 1:
                        assert value == general == max(k*q**k for k in range(1, cap+1))
                    if 2 <= budget < 6:
                        assert value > individual
                    if budget == 6:
                        assert value == individual == general
                    rows.append({
                        'n': 6, 'B': budget, 'G': cap, 'q': str(q), 'q_decimal': float(q),
                        'individual': str(individual), 'laminar': str(value), 'general': str(general),
                        'laminar_gain': str(value-individual),
                        'crossing_gain': str(general-value),
                        'laminar_gain_percent': float(100*(value/individual-1)),
                        'crossing_loss_percent': float(100*(1-value/general)),
                        'retained_ratio': str(value/general),
                        'independent_profile_check': True, 'profiles': 64,
                        'certificate': certificate,
                    })
                finally:
                    signal.alarm(0)
            print(f'G={cap}, q={float(q):.2f}: {len(rows)}/228', flush=True)
    bykey = {(r['G'], r['B'], r['q']): r for r in rows}
    for (cap, budget, q), row in bykey.items():
        for nxt in [(cap+1, budget, q), (cap, budget+1, q)]:
            if nxt in bykey:
                assert F(row['laminar']) <= F(bykey[nxt]['laminar'])
    paths = ['augmented/experimento_tres_referencias_28.py', 'augmented/bellman_perfiles.py',
             'augmented/bm17_toy_solver.py', 'augmented/experimento_sesion_28.py',
             'augmented/experimento_mapa_laminar_28.py']
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'How much does laminar grouping improve on individuals, and how much does it lose to the general optimum?',
        'model': {**base['model'], 'laminar': 'nested or disjoint along every realized trajectory'},
        'reference_file': str(BASE.relative_to(ROOT)),
        'reference_sha256': sha256(BASE.read_bytes()).hexdigest(),
        'source_sha256': {s: sha256((ROOT/s).read_bytes()).hexdigest() for s in paths},
        'metrics': {'laminar_gain_percent': '100*(laminar/individual - 1)',
                    'crossing_loss_percent': '100*(1 - laminar/general)'},
        'rows': rows,
        'summary': {
            'cases': len(rows),
            'laminar_beats_individuals': sum(F(r['laminar']) > F(r['individual']) for r in rows),
            'laminar_ties_individuals': sum(F(r['laminar']) == F(r['individual']) for r in rows),
            'general_beats_laminar': sum(F(r['general']) > F(r['laminar']) for r in rows),
            'general_ties_laminar': sum(F(r['general']) == F(r['laminar']) for r in rows),
            'worst_laminar_case': max(rows, key=lambda r:r['crossing_loss_percent']),
            'by_budget': {str(b): {
                'cases': sum(r['B']==b for r in rows),
                'general_beats_laminar': sum(r['B']==b and F(r['general'])>F(r['laminar']) for r in rows),
                'max_loss_percent': max(r['crossing_loss_percent'] for r in rows if r['B']==b),
            } for b in range(1,7)},
        },
        'audit': {'profile_and_atom_solver_agreements': atom_checks,
                  'full_budget_certificates': 38, 'independent_laminar_policy_evaluations': len(rows),
                  'profiles_per_policy': 64, 'monotonicity_checked': True},
        'seconds': round(perf_counter()-start, 3),
    }
    assert len(rows) == 228 and atom_checks == 190
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summary': result['summary'], 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    run()
