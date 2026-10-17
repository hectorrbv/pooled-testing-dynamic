"""Politica pi_ratio: la regla de cociente U/T que pidio Francisco (22-sep).

Encargo literal [D 35:17]: "algo interesante seria la prueba que maximiza la
division entre estos dos factores: el promedio de U dividido por el promedio
de T, o con un factor de lambda en uno versus el otro".

La regla. Para cada componente C (pool virgen candidato o atomo vivo) y cada
horizonte c <= b, se toma el MEJOR PLAN LOCAL con c pruebas y derecho a parar,
y se le miden dos numeros por separado:

    E[U](C, c) = utilidad esperada que ese plan acredita
    E[T](C, c) = numero ESPERADO de pruebas que ese plan gasta

El indice es E[U] / E[T]^alpha. Se elige el par (C, c) de indice maximo, se
ejecuta la PRIMERA accion de su plan y se recalcula todo (horizonte rodante).

Que la distingue de las otras dos reglas con planeacion:
  - densidad H_c/c (pi_C, pi_R): divide entre el horizonte RESERVADO c. Cobra
    pruebas que el plan nunca gasta cuando la exploracion muere temprano.
  - indice Lagrangiano I_lambda (pi_L): resta un precio, E[U] - lambda*E[T].
  - pi_ratio: divide entre el costo ESPERADO. Es la forma de cociente, con el
    mismo arreglo que el contraejemplo pedia (costo esperado, no reservado).

alpha = 1 es el cociente puro que pidio Francisco. alpha != 1 es la familia
V/C^alpha (candidata F, §14.8) con C = E[T]; se barre ex ante, eligiendo por
valor de politica y nunca por razon contra el optimo.

Estatuto (§25): DIAGNOSTICO. Sin garantia probada. Convencion posterior-zero;
r cuenta infectados.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from augmented.bm17_toy_solver import SolverLaminar, z_tabla


class PoliticaRatio:
    def __init__(self, p, u, G, alpha=1.0, horizonte=3, no_paralisis=True):
        self._p, self._u, self._G = dict(p), dict(u), G
        self._alpha, self._horizonte = alpha, horizonte
        self._no_paralisis = no_paralisis
        self._locales = {}

    p = property(lambda self: self._p)
    u = property(lambda self: self._u)
    G = property(lambda self: self._G)
    alpha = property(lambda self: self._alpha)
    horizonte = property(lambda self: self._horizonte)
    no_paralisis = property(lambda self: self._no_paralisis)

    # ---------------------------------------------------------- mecanica
    def _z(self, S):
        return z_tabla(tuple(sorted(S)), self.p)

    def _local(self, miembros):
        """Solver restringido al componente (mismas reglas, sub-poblacion)."""
        clave = tuple(sorted(miembros))
        if clave not in self._locales:
            self._locales[clave] = SolverLaminar(
                {i: self.p[i] for i in clave}, {i: self.u[i] for i in clave},
                self.G, 'posterior_zero')
        return self._locales[clave]

    def _pieza(self, X, r):
        X = tuple(sorted(X))
        if r == 0:
            return sum(float(self.u[i]) for i in X), None
        if r == len(X):
            return 0.0, None
        return 0.0, (X, r)

    def _ramas(self, U, atomos, accion):
        """[(prob, recompensa, U', atomos')] de una accion."""
        out = []
        if accion[0] == 'open':
            S = accion[1]
            for s, prob in enumerate(self._z(S)):
                if prob == 0:
                    continue
                rew, nuevo = self._pieza(S, s)
                out.append((float(prob), rew, U - frozenset(S),
                            tuple(sorted(atomos + ((nuevo,) if nuevo else ())))))
        else:
            _, (A, r), S = accion
            resto = tuple(i for i in A if i not in S)
            zS, zR, zA = self._z(S), self._z(resto), self._z(A)
            otros = tuple(a for a in atomos if a != (A, r))
            for s in range(len(zS)):
                if not (0 <= r - s < len(zR)):
                    continue
                prob = zS[s] * zR[r - s] / zA[r]
                if prob == 0:
                    continue
                rS, nS = self._pieza(S, s)
                rR, nR = self._pieza(resto, r - s)
                nuevos = tuple(x for x in (nS, nR) if x)
                out.append((float(prob), rS + rR, U,
                            tuple(sorted(otros + nuevos))))
        return out

    # ------------------------------------------- el plan local: E[U] y E[T]
    def _plan_local(self, sol, U, atomos, c):
        """(E[U], E[T]) del mejor plan local con c pruebas y derecho a parar.

        El plan es el argmax del solver restringido al componente. E[T] cuenta
        solo las pruebas que el plan gasta de verdad: si una rama para, esa
        rama no suma.
        """
        if c == 0:
            return 0.0, 0.0
        U, atomos = sol._canoniza(U, atomos)
        sol.V(U, atomos, c)
        accion = sol.argmax.get((U, atomos, c))
        if accion is None:
            return 0.0, 0.0                       # parar: ni utilidad ni costo
        valor, tests = 0.0, 1.0
        for prob, rew, U2, at2 in self._ramas(U, atomos, accion):
            v2, t2 = self._plan_local(sol, U2, at2, c - 1)
            valor += prob * (rew + v2)
            tests += prob * t2
        return valor, tests

    def _proyecto_virgen(self, S, c):
        """(E[U], E[T]) de abrir el root S y seguir adentro (ec. 8.10)."""
        if c < 1:
            return 0.0, 0.0
        sol = self._local(S)
        valor, tests = 0.0, 1.0
        for s, prob in enumerate(self._z(S)):
            if prob == 0:
                continue
            rew, nuevo = self._pieza(S, s)
            atomos = (nuevo,) if nuevo else ()
            v2, t2 = self._plan_local(sol, frozenset(), atomos, c - 1)
            valor += float(prob) * (rew + v2)
            tests += float(prob) * t2
        return valor, tests

    def _proyecto_atomo(self, A, r, c):
        return self._plan_local(self._local(A), frozenset(),
                                ((tuple(sorted(A)), r),), c)

    def _indice(self, valor, tests):
        if tests <= 0 or valor <= 0:
            return 0.0
        return valor / (tests ** self.alpha)

    # ---------------------------------------------------------- decision
    def _cobro_inmediato(self, U, atomos, accion):
        return sum(prob * rew for prob, rew, _, _ in
                   self._ramas(U, atomos, accion))

    def decide(self, U, atomos, b):
        tope = min(b, self.horizonte)
        mejor, mejor_idx = None, 1e-12
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(sorted(U), k):
                for c in range(1, tope + 1):
                    idx = self._indice(*self._proyecto_virgen(S, c))
                    if idx > mejor_idx:
                        mejor, mejor_idx = ('open', S), idx
        for (A, r) in atomos:
            sol = self._local(A)
            for c in range(1, tope + 1):
                idx = self._indice(*self._proyecto_atomo(A, r, c))
                if idx > mejor_idx:
                    accion = sol.politica(frozenset(),
                                          ((tuple(sorted(A)), r),), c)
                    if accion is not None:
                        mejor, mejor_idx = ('ref', (A, r), accion[-1]), idx
        if mejor is not None:
            return mejor
        if not self.no_paralisis:
            return None
        alt, alt_v = None, -1.0
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(sorted(U), k):
                m = self._cobro_inmediato(U, atomos, ('open', S))
                if m > alt_v:
                    alt, alt_v = ('open', S), m
        for (A, r) in atomos:
            for k in range(1, min(self.G, len(A) - 1) + 1):
                for S in combinations(A, k):
                    m = self._cobro_inmediato(U, atomos, ('ref', (A, r), S))
                    if m > alt_v:
                        alt, alt_v = ('ref', (A, r), S), m
        return alt

    @lru_cache(maxsize=None)
    def valor(self, U, atomos, b):
        """Valor esperado exacto de la politica (sin Monte Carlo)."""
        if b == 0:
            return 0.0
        accion = self.decide(U, atomos, b)
        if accion is None:
            return 0.0
        return sum(prob * (rew + self.valor(U2, at2, b - 1))
                   for prob, rew, U2, at2 in self._ramas(U, atomos, accion))


def optimo(p, u, B, G):
    pf = {i: Fraction(str(v)) for i, v in p.items()}
    uf = {i: Fraction(str(v)) for i, v in u.items()}
    return float(SolverLaminar(pf, uf, G, 'posterior_zero')
                 .V(frozenset(pf), (), B))


if __name__ == '__main__':
    import time
    t0 = time.time()

    # Instancia del contraejemplo universal (n=6, B=3, G=4).
    p_ce = {0: 0.9, 1: 0.825, 2: 0.875, 3: 0.8, 4: 0.95, 5: 0.85}
    u_ce = {0: 2, 1: 1, 2: 1, 3: 1, 4: 4, 5: 2}
    opt = optimo(p_ce, u_ce, 3, 4)
    pol = PoliticaRatio(p_ce, u_ce, 4, alpha=1.0)
    v = pol.valor(frozenset(p_ce), (), 3)
    print(f'contraejemplo universal: opt {opt:.4f} | pi_ratio {v:.4f} '
          f'| razon {v/opt:.4f}   (las cuatro golosas: 0.6576)')

    # B-M16: aqui el cociente debe abrir el par, no un singleton.
    p16 = {i: 0.7 for i in range(4)}
    u16 = {i: 1 for i in range(4)}
    opt16 = optimo(p16, u16, 2, 2)
    pol16 = PoliticaRatio(p16, u16, 2, alpha=1.0)
    v16 = pol16.valor(frozenset(p16), (), 2)
    primera = pol16.decide(frozenset(p16), (), 2)
    print(f'B-M16: opt {opt16:.4f} | pi_ratio {v16:.4f} | razon {v16/opt16:.4f}'
          f' | primera accion {primera}')

    # El costo es ESPERADO, no reservado: un par fresco con q=0.3 de sano
    # gasta menos de las 2 pruebas que la densidad reservaria.
    eu, et = pol16._proyecto_virgen((0, 1), 2)
    assert 1.0 < et < 2.0, et
    print(f'\npar fresco con horizonte 2: E[U] = {eu:.4f}, E[T] = {et:.4f}')
    print('OK: E[T] < 2 — el plan para temprano en las ramas resueltas, '
          'y el cociente no cobra las pruebas que no gasta')
    assert abs(v16 - opt16) < 1e-9, 'pi_ratio deberia ser optima en B-M16'
    print(f'[{time.time() - t0:.1f}s]')
