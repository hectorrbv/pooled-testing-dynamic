"""Check the optimized and compiled homogeneous recursions against labeled DP."""
from fractions import Fraction as F
import pytest
from augmented.bellman_tipos import crear_solver
from augmented.bm17_toy_solver import SolverLaminar


@pytest.mark.parametrize('q',[F(1,20),F(3,10),F(1,2),F(19,20)])
@pytest.mark.parametrize('n,G,B',[(3,2,2),(4,2,3),(4,3,3),(5,3,3),(6,4,3)])
@pytest.mark.parametrize('backend',['python','compiled'])
def test_backends_match_labeled_solver(q,n,G,B,backend):
    if backend == 'compiled':
        pytest.importorskip('numba', reason='optional compiled backend: requirements-types.txt')
    a=SolverLaminar(dict.fromkeys(range(n),1-q),dict.fromkeys(range(n),1),G,'posterior_zero')
    expected=float(a.V(frozenset(range(n)),(),B))
    v=crear_solver(float(1-q),G,backend=backend)
    assert v(n,(),B)==pytest.approx(expected,abs=1e-10)
    choices=v.action_values(n,(),B)
    for (_,k),actual in choices:
        target=float(a.valor_forzando_primera(frozenset(range(n)),(),B,('open',tuple(range(k)))))
        assert actual==pytest.approx(target,abs=1e-10)


@pytest.mark.parametrize('G',[2,3,4,8,16])
def test_compiled_finite_and_saturated_states(G):
    pytest.importorskip('numba', reason='optional compiled backend: requirements-types.txt')
    py=crear_solver(.95,G)
    jit=crear_solver(.95,G,backend='compiled')
    for b in [1,2,3]:
        for n,atoms in [(G,()),(b*G,()),(0,((G,G-1),)),(2,((G,1),))]:
            assert jit(n,atoms,b)==pytest.approx(py(n,atoms,b),abs=1e-10)


@pytest.mark.parametrize('backend',['python','compiled'])
def test_root_tie_and_saturation(backend):
    if backend == 'compiled':
        pytest.importorskip('numba', reason='optional compiled backend: requirements-types.txt')
    v=crear_solver(.7,2,backend=backend)
    assert set(v.optimal_actions(4,(),3))=={('open',1),('open',2)}
    assert v(6,(),3)==v(500,(),3)
    assert v.optimal_actions(4,(),2)==(('open',2),)
