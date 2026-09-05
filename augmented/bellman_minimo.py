"""Recursion 5.5 del companion, version minima legible (posterior-zero).

Version de trabajo con flag de convencion y fracciones: bm17_toy_solver.py.
"""

from functools import lru_cache
from itertools import combinations


def probabilidades_conteo(personas, p):
    # Z(A, r) por convolucion (ec. 3.4): coeficientes de prod(1-p_i + p_i*z)
    coeficientes = [1.0]
    for i in personas:
        nuevos = [0.0] * (len(coeficientes) + 1)
        for r, c in enumerate(coeficientes):
            nuevos[r] += c * (1 - p[i])
            nuevos[r + 1] += c * p[i]
        coeficientes = nuevos
    return coeficientes


def crear_solver(p, u, G):

    def resolver_pedazo(miembros, r):
        # nu (ec. 5.2): r=0 cobra y sale; r=|A| sale sin cobrar; interior = atomo
        if r == 0:
            return sum(u[i] for i in miembros), None
        if r == len(miembros):
            return 0.0, None
        return 0.0, (tuple(sorted(miembros)), r)

    def esperanza(virgenes, atomos, b, resultados):
        # E[recompensa inmediata + V del sucesor] sobre los conteos posibles
        total = 0.0
        for prob, pedazos in resultados:
            if prob == 0:
                continue
            recompensa, nuevos_atomos = 0.0, []
            for miembros, r in pedazos:
                pago, atomo = resolver_pedazo(miembros, r)
                recompensa += pago
                if atomo is not None:
                    nuevos_atomos.append(atomo)
            siguiente = tuple(sorted(atomos + tuple(nuevos_atomos)))
            total += prob * (recompensa + V(virgenes, siguiente, b - 1))
        return total

    @lru_cache(maxsize=None)
    def V(virgenes, atomos, b):
        # ec. 5.5: max{0, mejor pool virgen, mejor refinamiento}
        if b == 0:
            return 0.0
        mejor = 0.0

        for k in range(1, min(G, len(virgenes)) + 1):
            for S in combinations(sorted(virgenes), k):
                z = probabilidades_conteo(S, p)
                resultados = [(z[r], [(S, r)]) for r in range(len(S) + 1)]
                mejor = max(mejor, esperanza(virgenes - set(S),
                                             atomos, b, resultados))

        for (A, r) in atomos:
            resto_atomos = tuple(a for a in atomos if a != (A, r))
            for k in range(1, len(A)):
                for S in combinations(A, k):
                    # conteo del subpool via ec. 3.5; el complemento sale gratis
                    S_c = tuple(i for i in A if i not in S)
                    zS = probabilidades_conteo(S, p)
                    zC = probabilidades_conteo(S_c, p)
                    zA = probabilidades_conteo(A, p)
                    resultados = []
                    for s in range(len(S) + 1):
                        if 0 <= r - s < len(zC):
                            prob = zS[s] * zC[r - s] / zA[r]
                            resultados.append((prob, [(S, s), (S_c, r - s)]))
                    mejor = max(mejor, esperanza(virgenes, resto_atomos,
                                                 b, resultados))
        return mejor

    return V


if __name__ == '__main__':
    # instancia del contraejemplo: 4 personas, 70% infectada, u = 1, G = 2
    p = {i: 0.7 for i in range(4)}
    u = {i: 1.0 for i in range(4)}
    V = crear_solver(p, u, G=2)

    todos = frozenset(range(4))
    print('optimo con B = 2:', round(V(todos, (), 2), 4))
    print('optimo con B = 3:', round(V(todos, (), 3), 4))
    print('atomo con conteo 1 y una prueba:',
          round(V(frozenset((2, 3)), (((0, 1), 1),), 1), 4))

    assert abs(V(todos, (), 2) - 387 / 500) < 1e-9
    assert abs(V(todos, (), 3) - 537 / 500) < 1e-9
    print('OK: coincide con el solver de trabajo (387/500 y 537/500)')
