from fractions import Fraction as F
import random
import pytest
from augmented.busqueda_brecha_ampliada import TwoTests
from augmented.bellman_binario_exacto import BellmanBinario
from augmented.bellman_perfiles import BellmanPerfiles


@pytest.mark.parametrize('seed',range(32))
def test_two_test_closed_forms_against_two_independent_exact_bellmans(seed):
    rng=random.Random(seed);n=rng.choice([2,3,4,5,6]);G=rng.randint(1,n)
    q=[F(rng.randint(1,99),100) for _ in range(n)]
    u=[F(rng.randint(1,20),rng.randint(1,5)) for _ in range(n)]
    b,l=TwoTests(n,G).values(list(map(float,q)),list(map(float,u)))
    assert b==pytest.approx(float(BellmanBinario([1-x for x in q],u,G).solve(2).value),abs=1e-10)
    assert l==pytest.approx(float(BellmanPerfiles([1-x for x in q],u,G,laminar=True).solve(2).value),abs=1e-10)


@pytest.mark.parametrize('n,B,G',[(3,2,2),(4,3,2),(4,3,3),(5,3,3)])
def test_compiled_full_dag_against_exact_values(n,B,G):
    from augmented.busqueda_brecha_compilada import FixedDAG
    bd,ld=FixedDAG(n,B,G,'binary'),FixedDAG(n,B,G,'laminar')
    for seed in range(6):
        rng=random.Random(seed)
        q=[F(rng.randint(1,9999),10000) for _ in range(n)]
        u=[F(rng.randint(1,100),7) for _ in range(n)]
        qf,uf=list(map(float,q)),list(map(float,u))
        assert bd.value(qf,uf)==pytest.approx(float(BellmanBinario([1-x for x in q],u,G).solve(B).value),abs=1e-9)
        assert ld.value(qf,uf)==pytest.approx(float(BellmanPerfiles([1-x for x in q],u,G,laminar=True).solve(B).value),abs=1e-9)


def test_family_has_an_exact_interval_certificate_and_an_endpoint_jump():
    from augmented.certifica_brecha_ampliada import symbolic_family
    certificate=symbolic_family()
    assert certificate['limit_ratio']=='10/9'
    assert certificate['inequalities_checked']==454
    for s in [F(19,20),F(99,100),F(999999,1000000)]:
        p=[1-s,F(1,2),F(1,2),1-s];u=[1,3,3,2]
        assert BellmanBinario(p,u,2).solve(2).value==F(5,2)*s*(1+s)
        assert BellmanPerfiles(p,u,2,laminar=True).solve(2).value==3+F(3,2)*s*s
    assert BellmanPerfiles([0,F(1,2),F(1,2),0],[1,3,3,2],2,laminar=True).solve(2).value==6
