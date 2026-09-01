"""La ecuacion de Bellman del companion, en su version minima legible.

Este archivo existe para ENTENDER, no para producir: es la recursion 5.5 del
companion (Thm 5.1) escrita con el minimo de codigo posible. La version de
trabajo, con flag de convencion, fracciones exactas y validaciones, es
`bm17_toy_solver.py`; los numeros de ambas coinciden.

El mapa codigo <-> companion:

    probabilidades_conteo   ec. 3.4   (el polinomio Z, por convolucion)
    prob. del subpool       ec. 3.5   (dentro de V, rama 'refinar')
    resolver_pedazo         ec. 5.2   (el normalizador: cobrar/descartar/atomo)
    la recompensa g         ec. 5.1   (implicita en resolver_pedazo)
    V                       ec. 5.5   (max{0, abrir, refinar}, memoizada)

Convencion (la del companion): cada persona i esta infectada con probabilidad
p[i]; la prueba de un pool devuelve cuantos infectados hay; una persona cobra
su utilidad u[i] cuando la historia demuestra que esta sana (posterior-zero:
la deduccion tambien acredita). Presupuesto B pruebas, pools de tamano <= G,
solo historias laminares (abrir pool virgen o partir un atomo).
"""

from functools import lru_cache
from itertools import combinations


def probabilidades_conteo(personas, p):
    """P(exactamente r infectados en `personas`), para r = 0 .. len(personas).

    Es el producto de polinomios (1-p_i + p_i*z), persona por persona: al
    multiplicar, el exponente de z va contando infectados (ec. 3.4).
    """
    coeficientes = [1.0]
    for i in personas:
        nuevos = [0.0] * (len(coeficientes) + 1)
        for r, c in enumerate(coeficientes):
            nuevos[r] += c * (1 - p[i])      # i salio sana: el conteo no sube
            nuevos[r + 1] += c * p[i]        # i salio infectada: sube en 1
        coeficientes = nuevos
    return coeficientes


def crear_solver(p, u, G):
    """Devuelve V(virgenes, atomos, b): el optimo laminar exacto.

    Estado (el suficiente del companion, Thm 5.1):
      virgenes : frozenset de personas nunca probadas
      atomos   : tuplas (miembros, r) — grupos probados con r infectados
                 adentro, aun sin resolver (0 < r < |miembros|)
      b        : pruebas restantes
    """

    def resolver_pedazo(miembros, r):
        """Que pasa con un pedazo cuyo conteo r acaba de quedar determinado.

        Los tres casos del normalizador (ec. 5.2): r = 0 -> todos sanos, se
        cobra su utilidad y salen del estado; r = |miembros| -> todos
        infectados, salen sin cobrar; conteo intermedio -> queda como atomo.
        """
        if r == 0:
            return sum(u[i] for i in miembros), None
        if r == len(miembros):
            return 0.0, None
        return 0.0, (tuple(sorted(miembros)), r)

    def esperanza(virgenes, atomos, b, resultados):
        """E[recompensa + V del estado siguiente] sobre los conteos posibles.

        `resultados` es una lista de (probabilidad, pedazos), donde cada
        pedazo (miembros, r) se pasa por resolver_pedazo.
        """
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
        """La ecuacion 5.5: max{0, mejor pool virgen, mejor refinamiento}."""
        if b == 0:
            return 0.0
        mejor = 0.0                                  # el 0 = derecho a parar

        # Abrir un pool virgen S: el conteo sale de Z(S, .) (ec. 5.3).
        for k in range(1, min(G, len(virgenes)) + 1):
            for S in combinations(sorted(virgenes), k):
                z = probabilidades_conteo(S, p)
                resultados = [(z[r], [(S, r)]) for r in range(len(S) + 1)]
                mejor = max(mejor, esperanza(virgenes - set(S),
                                             atomos, b, resultados))

        # Refinar un atomo (A, r) probando S ⊊ A: el conteo s de S se
        # distribuye como Z(S,s)·Z(A\S, r-s)/Z(A,r) (ec. 3.5), y el
        # complemento A\S queda determinado gratis (ec. 5.4).
        for (A, r) in atomos:
            resto_atomos = tuple(a for a in atomos if a != (A, r))
            for k in range(1, len(A)):
                for S in combinations(A, k):
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
    # La instancia del contraejemplo: 4 personas, 30% de estar sana cada una
    # (70% infectada), utilidad 1, pools de a lo mas 2.
    p = {i: 0.7 for i in range(4)}
    u = {i: 1.0 for i in range(4)}
    V = crear_solver(p, u, G=2)

    todos = frozenset(range(4))
    print('optimo con B = 2:', round(V(todos, (), 2), 4))    # 0.774
    print('optimo con B = 3:', round(V(todos, (), 3), 4))    # 1.074

    # El estado del contraejemplo: el par {0,1} probado con 1 infectado,
    # una prueba restante — la reentrada cobra 1.0 seguro (ratificacion G0).
    atomo = ((( 0, 1), 1),)
    print('atomo con conteo 1 y una prueba:', round(V(frozenset((2, 3)), atomo, 1), 4))

    assert abs(V(todos, (), 2) - 387 / 500) < 1e-9
    assert abs(V(todos, (), 3) - 537 / 500) < 1e-9
    print('OK: coincide con el solver de trabajo (387/500 y 537/500)')
