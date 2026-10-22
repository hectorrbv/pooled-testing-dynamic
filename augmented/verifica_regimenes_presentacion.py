"""Independent profile audit of selected new regime and lambda comparisons."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import pandas as pd
from augmented.indice_lagrangiano import PoliticaLagrangiana
from augmented.politica_ratio import PoliticaRatio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sesion_2026-09-29'


def enumerate_value(policy, q, B, G):
    profiles = list(product((0, 1), repeat=6))
    total = F(0)
    for actual in profiles:
        weight = q**actual.count(0)*(1-q)**actual.count(1)
        compatible = profiles[:]
        untested, atoms = frozenset(range(6)), ()
        for _ in range(B):
            action = policy.decide(untested, atoms, B-_)
            if action is None:
                break
            if action[0] == 'open':
                group = tuple(action[1])
                assert set(group) <= untested
                untested -= frozenset(group)
                pieces = [group]
                others = list(atoms)
            else:
                old, group = action[1], tuple(action[2])
                assert old in atoms and set(group) < set(old[0])
                pieces = [group, tuple(i for i in old[0] if i not in group)]
                others = [a for a in atoms if a != old]
            assert 0 < len(group) <= G
            count = sum(actual[i] for i in group)
            compatible = [x for x in compatible if sum(x[i] for i in group)==count]
            for piece in pieces:
                r = sum(actual[i] for i in piece)
                if 0 < r < len(piece):
                    others.append((tuple(sorted(piece)), r))
            atoms = tuple(sorted(others))
        credited = sum(all(x[i]==0 for x in compatible) for i in range(6))
        total += weight*credited
    return float(total)


def main():
    grid = pd.read_csv(OUT/'regimenes_politicas.csv')
    lambdas = pd.read_csv(OUT/'lambda_refinado.csv')
    checks = []
    for q, B, G in [(.6,2,3),(.6,3,3),(.65,2,3),(.95,3,4)]:
        row = grid[(grid.q==q)&(grid.B==B)&(grid.G==G)].iloc[0]
        for name, cls in [('pi_L',PoliticaLagrangiana),('pi_ratio',PoliticaRatio)]:
            qf = F(str(q)); p = {i:1-qf for i in range(6)}; u={i:1 for i in p}
            policy = cls(p,u,G)
            value = enumerate_value(policy,qf,B,G)
            expected = row.opt*row[name]
            assert abs(value-expected)<1e-9
            checks.append({'q':q,'B':B,'G':G,'policy':name,'value':value,
                           'reference':expected,'error':abs(value-expected)})
    for lam in [.01,.30,.31,.40]:
        row = lambdas[lambdas['lambda']==lam].iloc[0]
        policy = PoliticaLagrangiana({i:F(4,5) for i in range(6)},
                                    {i:1 for i in range(6)},4,lam=lam)
        value = enumerate_value(policy,F(1,5),3,4)
        assert abs(value-row.value)<1e-9
        checks.append({'q':.2,'B':3,'G':4,'policy':'pi_L','lambda':lam,
                       'value':value,'reference':row.value,'error':abs(value-row.value)})
    continuations = []
    for lam, size, value in [(.30,2,.536),(.31,1,.4)]:
        p = {i:F(4,5) for i in range(6)}
        policy = PoliticaLagrangiana(p,{i:1 for i in p},4,lam=lam)
        U = frozenset((3,4,5))
        action = policy.decide(U,(),2)
        assert action[0]=='open' and len(action[1])==size
        assert abs(policy.valor(U,(),2)-value)<1e-12
        without_fallback = PoliticaLagrangiana(p,{i:1 for i in p},4,
                                              lam=lam,no_paralisis=False)
        fallback_used = without_fallback.decide(U,(),2) is None
        assert fallback_used == (lam==.31)
        continuations.append({'lambda':lam,'action':action,
                              'value':value,'immediate_fallback':fallback_used})
    branch_probability = .2**3+.8**3
    explained_loss = branch_probability*(.536-.4)
    observed_loss = (lambdas.set_index('lambda').loc[.30,'value']
                     -lambdas.set_index('lambda').loc[.31,'value'])
    assert abs(explained_loss-observed_loss)<1e-12
    result = {'profiles_per_check':64, 'comparisons':len(checks),
              'max_error':max(c['error'] for c in checks), 'checks':checks,
              'lambda_continuation':{'first_pool':'ABC','outcomes':[0,3],
                  'probability':branch_probability,'untested':'DEF',
                  'remaining_budget':2,'continuations':continuations,
                  'explained_loss':explained_loss,'observed_loss':observed_loss}}
    (OUT/'regimenes_auditoria_perfiles.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':
    main()
