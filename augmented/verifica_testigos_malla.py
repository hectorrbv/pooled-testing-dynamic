"""Replay the saved Q1/Q2 witnesses independently of their solver caches.

Run from the repository root: python -m augmented.verifica_testigos_malla
This checks achieved utility and feasibility, not optimality by replay alone.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

from augmented.experimento_sesion_28 import evaluate


def tree_policy(tree):
    def action(support, history, b):
        node = tree
        for pool, result in history:
            assert tuple(node['pool']) == tuple(pool)
            node = next(branch['next'] for branch in node['branches']
                        if branch['infected_count'] == result)
        return node.get('pool')
    return action


def trace_policy(traces):
    actions = {}
    for trace in traces:
        history = ()
        for pool, result in trace['history']:
            pool = tuple(pool)
            assert history not in actions or actions[history] == pool
            actions[history] = pool
            history += ((pool, result),)
    return lambda support, history, b: actions.get(tuple(history))


def main():
    root = Path(__file__).resolve().parents[1]
    path = root/'results/precio_q05_G/testigos_verificados.json'
    data = json.loads(path.read_text())['checks']
    results = {}
    for kind in ['laminar', 'general']:
        replay = evaluate(tree_policy(data[kind]['tree']), [F(1,2)]*7, [1]*7,
                          4, 3, require_laminar=kind == 'laminar')
        assert F(replay['value']) == F(data[kind]['value'])
        results[kind] = {k:v for k,v in replay.items() if k != 'traces'}
    for G in [4,5]:
        for kind in ['greedy','rollout']:
            saved = data[f'q1_G{G}'][kind+'_replay']
            replay = evaluate(trace_policy(saved['traces']), [F(1,2)]*7, [1]*7,
                              4, G, require_laminar=True)
            assert F(replay['value']) == F(saved['value'])
            results[f'{kind}_G{G}'] = {k:v for k,v in replay.items() if k != 'traces'}
    results['artifact_sha256'] = sha256(path.read_bytes()).hexdigest()
    results['replay_source_sha256'] = sha256((root/'augmented/experimento_sesion_28.py').read_bytes()).hexdigest()
    out = path.with_name('replay_testigos.json')
    out.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
