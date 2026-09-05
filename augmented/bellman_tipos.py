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


def crear_solver(p_inf, G, u=1.0):
    """V(virgenes, atomos, b) exacto para poblacion homogenea."""
    q = 1 - p_inf

    def Z(k, r):
        """Binomial: prob de r infectados en k personas frescas."""
        return comb(k, r) * p_inf ** r * q ** (k - r)

    def hiper(m, r, j, s):
        """Hipergeometrica: prob de s infectados entre los j probados de un
        atomo de tamano m con r infectados (ec. 3.5 en el caso homogeneo)."""
        if not (0 <= s <= j and 0 <= r - s <= m - j):
            return 0.0
        return comb(j, s) * comb(m - j, r - s) / comb(m, r)

    def norm(atomos):
        """Descarta los conteos extremos: cobrados o todos infectados."""
        return tuple(sorted(a for a in atomos if 0 < a[1] < a[0]))

    @lru_cache(maxsize=None)
    def V(virgenes, atomos, b):
        if b == 0:
            return 0.0
        mejor = 0.0
        for k in range(1, min(G, virgenes) + 1):            # abrir pool virgen
            val = 0.0
            for r in range(k + 1):
                pr = Z(k, r)
                if pr == 0:
                    continue
                rew = k * u if r == 0 else 0.0
                nuevos = ((k, r),) if 0 < r < k else ()
                val += pr * (rew + V(virgenes - k, norm(atomos + nuevos), b - 1))
            mejor = max(mejor, val)
        for (m, r) in set(atomos):                          # refinar un atomo
            resto = list(atomos)
            resto.remove((m, r))
            for j in range(1, min(G, m - 1) + 1):
                val = 0.0
                for s in range(j + 1):
                    pr = hiper(m, r, j, s)
                    if pr == 0:
                        continue
                    rew = (j * u if s == 0 else 0.0) + \
                          ((m - j) * u if r - s == 0 else 0.0)
                    nuevos = tuple(x for x in ((j, s), (m - j, r - s))
                                   if 0 < x[1] < x[0])
                    val += pr * (rew + V(virgenes,
                                         norm(tuple(resto) + nuevos), b - 1))
                mejor = max(mejor, val)
        return mejor

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
