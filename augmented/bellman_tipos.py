"""Bellman con compresion por tipos (companion Prop 6.2), caso homogeneo.

La pregunta que responde: hasta que n se puede calcular el OPTIMO exacto en
una laptop. Respuesta medida: el n practicamente no cuesta — lo que cuesta es
el presupuesto B (y el tope de pool G).

Con poblacion homogenea el estado no necesita saber QUIENES, solo CUANTOS:

    (virgenes, atomos, b)   con atomos = tupla de (tamano, r_infectados)

Dos personas del mismo tipo son intercambiables, asi que n=120 y n=500 dan el
mismo arbol de estados. Las transiciones son binomial para pools virgenes e
hipergeometrica para refinamientos (el split de un atomo de conteo conocido).

Medido en esta maquina (ver __main__): n=120, B=8, G=5, q=0.9 en ~4 s con 277k
estados; n=500 identico; B=10 en ~33 s; B=12 y G=10 exceden 120 s. El muro es
exponencial en B, como anuncia la cota (6.1) del companion.

Siguiente paso natural (etapa 2 de B-M17): generalizar de homogeneo a M tipos
—- el estado pasa a vectores de multiplicidades por tipo y n sigue siendo
gratis mientras M sea chico.

Convencion posterior-zero; p_inf = probabilidad de infeccion.
"""

import time
from functools import lru_cache
from math import comb


def crear_solver(p_inf, G, u=1.0, *, max_states=None, backend='python'):
    """Exact homogeneous DP evaluated in floating point, with root argmax/ties.

    V.action_values(n, atoms, b) exposes legal action values. V.optimal_actions
    saves all numerical maximizers in V.argmax, so a singleton/group tie is not
    silently classified as a strict preference. G0/posterior-zero only.

    Exact reductions: cap virgin count at b*G; evaluate b=1 directly; merge
    identical successor multisets; test only the smaller side of a split since
    its complementary count is inferred for free. No beam search or pruning.
    """
    if backend == 'compiled':
        from augmented.bellman_tipos_compilado import CompiledSolver
        return CompiledSolver(p_inf,G,u)
    if backend != 'python':
        raise ValueError('Unknown backend')
    if not 0 <= p_inf <= 1 or G < 1 or u < 0:
        raise ValueError('Invalid homogeneous instance')
    p_inf, u = float(p_inf), float(u)
    q = 1 - p_inf
    openings = {}
    fresh_best = [0.0]
    refinements, immediate = {}, {}
    argmax = {}
    for k in range(1, G+1):
        probs = [comb(k,r)*p_inf**r*q**(k-r) for r in range(k+1)]
        branches = [(probs[0]+probs[-1], ())]
        branches += [(probs[r], ((k,r),)) for r in range(1,k) if probs[r]]
        openings[k] = k*u*probs[0], tuple(branches)
        fresh_best.append(max(fresh_best[-1], k*u*probs[0]))
    for m in range(2,G+1):
        for r in range(1,m):
            candidates=[]
            for j in range(1,m//2+1):
                aggregate={}; reward=0.0
                for s in range(max(0,r-m+j),min(j,r)+1):
                    pr=comb(j,s)*comb(m-j,r-s)/comb(m,r)
                    reward += pr*u*((j if s==0 else 0)+(m-j if r-s==0 else 0))
                    new=tuple(sorted((a,c) for a,c in ((j,s),(m-j,r-s)) if 0<c<a))
                    aggregate[new]=aggregate.get(new,0.0)+pr
                candidates.append((j,reward,tuple((pr,new) for new,pr in aggregate.items())))
            refinements[m,r]=tuple(candidates)
            immediate[m,r]=max(x[1] for x in candidates)

    def state(v, atoms, b):
        return min(v,b*G), tuple(sorted(atoms)), b

    def value(v, atoms, b):
        if b<=0:
            return 0.0
        v=min(v,b*G)
        if b==1:
            return max(fresh_best[min(G,v)], max((immediate[a] for a in atoms),default=0.0))
        return cached(v,atoms,b)

    def action_values(v, atoms, b):
        if b<=0:
            return ()
        values=[]
        for k in range(1,min(G,v)+1):
            reward,branches=openings[k]
            val=reward+sum(pr*value(v-k,tuple(sorted(atoms+new)),b-1) for pr,new in branches)
            values.append((('open',k),val))
        for atom in dict.fromkeys(atoms):
            rest=list(atoms);rest.remove(atom);rest=tuple(rest)
            for j,reward,branches in refinements[atom]:
                val=reward+sum(pr*value(v,tuple(sorted(rest+new)),b-1) for pr,new in branches)
                values.append((('ref',atom,j),val))
        return tuple(values)

    @lru_cache(maxsize=None)
    def cached(v,atoms,b):
        if max_states is not None and cached.cache_info().currsize>=max_states:
            raise RuntimeError('Exact-state limit reached; no optimum certified')
        return max((val for _,val in action_values(v,atoms,b)),default=0.0)

    def V(virgenes,atomos,b):
        if virgenes<0 or b<0 or any(not (0<r<m<=G) for m,r in atomos):
            raise ValueError('Expected nonnegative counts and unresolved atoms of size <= G')
        v,atoms,b=state(virgenes,atomos,b)
        return value(v,atoms,b)

    def optimal_actions(v,atoms,b,tol=1e-10):
        v,atoms,b=state(v,atoms,b)
        options=action_values(v,atoms,b)
        best=max((val for _,val in options),default=0.0)
        answer=tuple(a for a,val in options if abs(val-best)<=tol*max(1.,abs(best)))
        argmax[v,atoms,b]=answer
        return answer

    V.action_values=lambda v,a,b: action_values(*state(v,a,b))
    V.optimal_actions=optimal_actions
    V.argmax=argmax
    V.cache_info=cached.cache_info
    V.cache_clear=cached.cache_clear
    return V


if __name__ == '__main__':
    print('Bellman por tipos (homogeneo): el n no cuesta, el presupuesto si\n')
    print(f'{"n":>5} {"B":>3} {"G":>3} {"q_sano":>7} | {"optimo":>9} '
          f'{"estados":>9} {"tiempo":>8}')
    casos = [(12, 3, 3, 0.30), (120, 3, 3, 0.30), (120, 5, 5, 0.90),
             (120, 8, 5, 0.90), (500, 8, 5, 0.90), (120, 10, 5, 0.90)]
    vistos = {}
    for (n, B, G, q) in casos:
        t0 = time.time()
        V = crear_solver(1 - q, G)
        v = V(n, (), B)
        est, dt = V.cache_info().currsize, time.time() - t0
        vistos[(n, B, G, q)] = (v, est)
        print(f'{n:5d} {B:3d} {G:3d} {q:7.2f} | {v:9.4f} {est:9d} {dt:7.2f}s')

    # n = 12 y n = 120 coinciden: con B=3 solo se tocan <=9 personas.
    assert vistos[(12, 3, 3, 0.30)] == vistos[(120, 3, 3, 0.30)]
    # n = 120 y n = 500 coinciden: el n es gratis bajo compresion por tipos.
    assert vistos[(120, 8, 5, 0.90)] == vistos[(500, 8, 5, 0.90)]
    print('\nOK: n=12 == n=120 (B=3) y n=120 == n=500 (B=8): el tamano de la '
          'poblacion no entra al costo; el muro es exponencial en B.')
