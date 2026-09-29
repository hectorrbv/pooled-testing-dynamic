"""296 declared BM17 comparison cases: 8 anchors + 240 states + 48 roots.

This is a reproducible local matrix, not a claim to have recovered the private
reviewer's unpublished 296-case script. The oracle uses complete latent profiles
and full-history pathwise compatibility; BM17 uses residual atoms/convolutions.
"""

from fractions import Fraction as F
import random

import pytest

from augmented.bm17_toy_solver import SolverLaminar
from augmented.pathwise_enum import PathwiseEnumerator


def atom_state(n, history, conv):
    """Translate the explicitly generated one-root history, without solver calls."""
    if not history:
        return frozenset(range(n)), (), F(0)
    root, r = history[0]
    parts = [(tuple(root), r, True)]
    if len(history) == 2:
        sub, s = history[1]
        assert set(sub) < set(root)
        parts = [(tuple(sub), s, True), (tuple(i for i in root if i not in sub), r-s, False)]
    atoms, paid = [], F(0)
    for group, count, tested in parts:
        if count == len(group):
            continue
        if count == 0 and (conv == 'posterior_zero' or tested):
            continue  # already credited, so excluded from additional value
        atoms.append((group, count))
    return frozenset(set(range(n))-set(root)), tuple(sorted(atoms)), paid


ANCHORS = [(3,1,'strict','3/10'), (3,1,'posterior_zero','3/10'),
           (4,2,'strict','3/5'), (4,2,'posterior_zero','387/500'),
           (4,3,'strict','1011/1000'), (4,3,'posterior_zero','537/500'),
           (4,1,'strict','1/2'), (4,1,'posterior_zero','1')]


@pytest.mark.parametrize('idx,data', list(enumerate(ANCHORS)))
def test_anchor(idx, data):
    n,B,conv,expected=data
    p,u=dict.fromkeys(range(n),F(7,10)),dict.fromkeys(range(n),F(1))
    history=[((0,1),1)] if idx>=6 else []
    U,atoms,_=atom_state(n,history,conv)
    a=SolverLaminar(p,u,2,conv).V(U,atoms,B)
    e=PathwiseEnumerator(p,u,2,conv).value(B,history)
    assert a == e == F(expected)


@pytest.mark.parametrize('seed',range(120))
@pytest.mark.parametrize('conv',['strict','posterior_zero'])
def test_seeded_heterogeneous_conditional_state(seed,conv):
    rng=random.Random(22092026+seed)
    n=rng.randint(2,5); G=rng.randint(2,min(3,n)); B=rng.randint(1,3)
    p={i:F(rng.randint(1,9),10) for i in range(n)}
    u={i:F(rng.randint(1,5)) for i in range(n)}
    history=[]
    if seed % 3:
        root=tuple(sorted(rng.sample(range(n),rng.randint(2,G))))
        r=rng.randint(1,len(root)-1)
        history.append((root,r))
        if seed % 3 == 2:
            sub=tuple(sorted(rng.sample(root,rng.randint(1,len(root)-1))))
            s=rng.randint(max(0,r-len(root)+len(sub)),min(r,len(sub)))
            history.append((sub,s))
    U,atoms,_=atom_state(n,history,conv)
    expected=PathwiseEnumerator(p,u,G,conv).value(B,history)
    actual=SolverLaminar(p,u,G,conv).V(U,atoms,B)
    assert actual == expected, (seed,conv,p,u,history,B,G,actual,expected)


@pytest.mark.parametrize('n',[3,4,5,6])
@pytest.mark.parametrize('B',[1,2,3])
@pytest.mark.parametrize('q',[F(1,5),F(3,10),F(1,2),F(7,10)])
def test_homogeneous_root_against_two_independent_representations(n,B,q):
    from augmented.experiments_separacion_n10 import laminar_value
    p,u=dict.fromkeys(range(n),1-q),dict.fromkeys(range(n),F(1))
    a=SolverLaminar(p,u,n,'posterior_zero').V(frozenset(range(n)),(),B)
    e=PathwiseEnumerator(p,u,n).value(B)
    assert a == e
    assert float(a) == pytest.approx(laminar_value(n,B,float(q)), abs=1e-10)
