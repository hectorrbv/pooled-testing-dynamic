"""Exact unrestricted reference with root symmetry for homogeneous instances.

At the empty history, equal independent priors and utilities make all pools of
the same size equivalent under relabeling. Only the FIRST action is reduced;
every continuation retains the full action menu of BellmanPerfiles.
"""
from fractions import Fraction as F
from augmented.bellman_perfiles import BellmanPerfiles


def solve_general(q, u, B, G, *, state_limit=1_000_000):
    q,u=tuple(map(F,q)),tuple(map(F,u))
    solver=BellmanPerfiles([1-x for x in q],u,G,state_limit=state_limit)
    if len(set(q))!=1 or len(set(u))!=1 or B==0:
        result=solver.solve(B)
        return result.value,result.states,"all_root_pools"
    support=solver.root_support
    mass,credit,_=solver._info(support)
    best=mass*credit
    for k in range(1,min(G,len(q))+1):
        j=solver.pool_index[(1<<k)-1]
        children=[support & mask for mask in solver.outcomes[j] if support & mask]
        candidate=sum(solver._dp(B-1,child,solver.all_actions) for child in children)
        best=max(best,candidate)
    return F(best,mass*solver.utility_scale),len(solver.memo),"root_cardinality_symmetry"
