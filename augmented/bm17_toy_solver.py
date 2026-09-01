"""Prototipo B-M17: recursion exacta 5.5 del companion laminar (toy, n<=4, B<=3).

Implementa la ecuacion de Bellman del companion (Thm 5.1) sobre estados
(U, A, b) con atomos residuales, aritmetica exacta en fracciones y el flag de
convencion `posterior_zero | strict` (G0 / pregunta 21 de §34).

Convencion del companion y del paper: Z_i = 1 es infectado, q_i es la
probabilidad de estar sano, y la prueba devuelve R(T) = numero de INFECTADOS.

Validacion (spec B-M16 de A, 2026-08-20, tests de aceptacion bajo `strict`):
predictivas del par virgen, posterior tras R=1, optimo 0.6 (nunca agrupa) y
"par primero con continuacion perfecta" 0.564. Bajo `posterior_zero` el mismo
solver produce el primer numero dual de B-M18.

Uso: python -m augmented.bm17_toy_solver  (corre las validaciones y la demo).
"""

from fractions import Fraction
from itertools import combinations


def z_tabla(miembros, p):
    """Coeficientes Z(A, r) del companion (ec. 3.3-3.4): prob. exacta de r
    infectados en `miembros`, por convolucion de los factores (q_i + p_i z)."""
    coef = [Fraction(1)]
    for i in miembros:
        qi, pi = 1 - p[i], p[i]
        nuevo = [Fraction(0)] * (len(coef) + 1)
        for r, c in enumerate(coef):
            nuevo[r] += c * qi
            nuevo[r + 1] += c * pi
        coef = nuevo
    return coef


class SolverLaminar:
    """Recursion 5.5 memoizada con argmax guardado (Steps 1-6 de Prop 6.1)."""

    def __init__(self, p, u, G, convencion):
        assert convencion in ('posterior_zero', 'strict')
        self.p = {i: Fraction(pi) for i, pi in p.items()}
        self.u = {i: Fraction(ui) for i, ui in u.items()}
        self.G = G
        self.conv = convencion
        self.memo, self.argmax = {}, {}

    def _u(self, X):
        return sum(self.u[i] for i in X)

    def _z(self, X):
        return z_tabla(sorted(X), self.p)

    # --- normalizador de sucesores (nu, ec. 5.2, con el flag) ------------
    def _pieza(self, X, r, probado):
        """(recompensa, atomos_nuevos) para el pedazo X con conteo r.

        `probado` distingue el pool fisicamente probado (acredita bajo ambas
        convenciones si r=0) del complemento deducido (solo acredita bajo
        posterior_zero; bajo strict queda como atomo vivo de conteo 0)."""
        X = tuple(sorted(X))
        if not X:
            return Fraction(0), ()
        if r == len(X):                     # todos infectados: fuera, sin pago
            return Fraction(0), ()
        if r == 0:
            if self.conv == 'posterior_zero' or probado:
                return self._u(X), ()       # acredita y sale del estado
            return Fraction(0), ((X, 0),)   # strict: deducido limpio, vivo
        return Fraction(0), ((X, r),)       # conteo interior: atomo vivo

    # --- acciones --------------------------------------------------------
    def _acciones(self, U, atomos):
        """Menu legal: pools virgenes y refinamientos, todos con |S| <= G."""
        acc = []
        U = sorted(U)
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(U, k):
                acc.append(('open', S))
        for (A, r) in atomos:
            if r == 0:                      # deducido limpio: solo existe bajo strict
                tope = min(self.G, len(A))       # incluye el atomo completo si cabe
            else:
                tope = min(self.G, len(A) - 1)   # subconjunto propio, tope G
            for k in range(1, tope + 1):
                for S in combinations(A, k):
                    acc.append(('ref', (A, r), S))
        return acc

    @staticmethod
    def _canoniza(U, atomos):
        U = frozenset(U)
        atomos = tuple(sorted((tuple(sorted(A)), r) for A, r in atomos))
        ocupados = set()
        for A, r in atomos:
            if not (0 <= r <= len(A)):
                raise ValueError(f'conteo invalido {r} para el atomo {A}')
            if set(A) & (ocupados | U):
                raise ValueError(f'el atomo {A} pisa a U o a otro atomo')
            ocupados |= set(A)
        return U, atomos

    # --- recursion 5.5 ----------------------------------------------------
    def V(self, U, atomos, b):
        U, atomos = self._canoniza(U, atomos)
        clave = (U, atomos, b)
        if clave in self.memo:
            return self.memo[clave]
        mejor, mejor_acc = Fraction(0), None      # el 0 explicito: parar
        if b > 0:
            for accion in self._acciones(U, atomos):
                q = self._q_accion(U, atomos, b, accion)
                if q > mejor:
                    mejor, mejor_acc = q, accion
        self.memo[clave] = mejor
        self.argmax[clave] = mejor_acc
        return mejor

    def _q_accion(self, U, atomos, b, accion):
        """Q^open (5.3) o Q^ref (5.4) de una accion."""
        if accion[0] == 'open':
            S = accion[1]
            zs = self._z(S)
            total = Fraction(0)
            for s, prob in enumerate(zs):
                if prob == 0:
                    continue
                rew, nuevos = self._pieza(S, s, probado=True)
                total += prob * (rew + self.V(U - set(S),
                                              atomos + nuevos, b - 1))
            return total
        _, (A, r), S = accion
        resto = tuple(sorted(set(A) - set(S)))
        otros = tuple(a for a in atomos if a != (A, r))
        if r == 0:                          # strict: conteo 0 determinista
            rew, nuevos = self._pieza(S, 0, probado=True)
            rew2, nuevos2 = self._pieza(resto, 0, probado=False)
            return rew + rew2 + self.V(U, otros + nuevos + nuevos2, b - 1)
        zS, zR, zA = self._z(S), self._z(resto), self._z(A)
        total = Fraction(0)
        for s in range(len(zS)):
            if not (0 <= r - s < len(zR)):
                continue
            prob = zS[s] * zR[r - s] / zA[r]     # ec. 3.5
            if prob == 0:
                continue
            rewS, nS = self._pieza(S, s, probado=True)
            rewR, nR = self._pieza(resto, r - s, probado=False)
            total += prob * (rewS + rewR + self.V(U, otros + nS + nR, b - 1))
        return total

    def valor_forzando_primera(self, U, atomos, b, accion):
        """Evalua una primera accion fija (y legal) con continuacion optima."""
        U, atomos = self._canoniza(U, atomos)
        if b < 1 or accion not in self._acciones(U, atomos):
            raise ValueError(f'accion ilegal en este estado: {accion}')
        return self._q_accion(U, atomos, b, accion)

    def politica(self, U, atomos, b):
        """Primera accion optima (tras llamar V)."""
        U, atomos = self._canoniza(U, atomos)
        self.V(U, atomos, b)
        return self.argmax[(U, atomos, b)]


def _valida_y_demo():
    # Instancia de la spec B-M16: n=4, q_sano=0.3 (p_infectado=0.7), u=1.
    n = 4
    p = {i: Fraction(7, 10) for i in range(n)}
    u = {i: Fraction(1) for i in range(n)}
    G, B = 2, 2
    par, par_virgen = (0, 1), (2, 3)

    # Dos vias para Z: convolucion contra enumeracion directa.
    from itertools import product
    for A in [par, (0, 1, 2)]:
        zs = z_tabla(A, p)
        for r in range(len(A) + 1):
            fuerza = sum(
                prob for zvec in product((0, 1), repeat=len(A))
                if sum(zvec) == r
                for prob in [Fraction(1) and _peso(zvec, A, p)]
            )
            assert zs[r] == fuerza, (A, r)
    print('OK: Z(A,r) coincide con la enumeracion directa (dos vias)')

    # Predictivas del par virgen (spec: 0.09 / 0.42 / 0.49, R = infectados).
    zs = z_tabla(par_virgen, p)
    assert zs[0] == Fraction(9, 100) and zs[1] == Fraction(42, 100) \
        and zs[2] == Fraction(49, 100)
    print('OK: predictivas del par virgen 0.09 / 0.42 / 0.49')

    # Posterior tras R=1 en {a,b}: P(a sano) = 1/2, P(ambos sanos) = 0.
    zS, zR = z_tabla((0,), p), z_tabla((1,), p)
    zA = z_tabla(par, p)
    p_a_sano = zS[0] * zR[1] / zA[1]
    assert p_a_sano == Fraction(1, 2)
    print('OK: posterior tras R=1: P(a sano) = 1/2; P(ambos sanos | R=1) = 0')

    U0 = frozenset(range(n))
    resultados = {}
    for conv in ('strict', 'posterior_zero'):
        sol = SolverLaminar(p, u, G, conv)
        v = sol.V(U0, (), B)
        resultados[conv] = (sol, v)
        print(f'optimo laminar {conv:14s} (n=4, B=2, G=2): {v} = {float(v):.4f}')

    sol_s, v_s = resultados['strict']
    sol_z, v_z = resultados['posterior_zero']

    # Tests de aceptacion de la spec (strict): optimo 0.6, nunca agrupa;
    # par primero con continuacion perfecta 0.564.
    assert v_s == Fraction(3, 5), v_s
    assert sol_s.politica(U0, (), B)[0] == 'open' \
        and len(sol_s.politica(U0, (), B)[1]) == 1
    v_par = sol_s.valor_forzando_primera(U0, (), B, ('open', par))
    assert v_par == Fraction(564, 1000), v_par
    print('OK spec B-M16 (strict): optimo 3/5 con singleton primero; '
          'par primero = 0.564')

    # El numero dual (B-M18 minimo): la misma instancia bajo posterior-zero.
    v_par_z = sol_z.valor_forzando_primera(U0, (), B, ('open', par))
    print(f'dual posterior-zero: optimo {v_z} = {float(v_z):.4f} '
          f'(par primero {v_par_z} = {float(v_par_z):.4f}; '
          f'primera accion optima: {sol_z.politica(U0, (), B)})')
    assert v_z == Fraction(387, 500) and v_par_z == v_z
    assert v_z >= v_s and v_par_z > v_par

    # Reentrada del contraejemplo (atomo {a,b} con 1 infectado, 1 prueba):
    # strict 0.5 -> posterior_zero 1.0 (cuenta 2 de la ratificacion G0).
    atomo = ((par, 1),)
    r_s = sol_s.valor_forzando_primera(frozenset((2, 3)), atomo, 1,
                                       ('ref', (par, 1), (0,)))
    r_z = sol_z.valor_forzando_primera(frozenset((2, 3)), atomo, 1,
                                       ('ref', (par, 1), (0,)))
    assert r_s == Fraction(1, 2) and r_z == Fraction(1)
    print('OK ratificacion G0: reentrada 0.5 (strict) -> 1.0 (posterior_zero)')

    # Extra: B=3, mismos parametros, ambos flags (para la demo del martes).
    v3 = {}
    for conv in ('strict', 'posterior_zero'):
        sol = SolverLaminar(p, u, G, conv)
        v3[conv] = sol.V(U0, (), 3)
        print(f'B=3 {conv:14s}: {v3[conv]} = {float(v3[conv]):.4f} '
              f'(primera accion: {sol.politica(U0, (), 3)})')
    # Consistencia con el notebook 24: q(3q^2-3q+4) en q=0.3.
    assert v3['strict'] == Fraction(1011, 1000)

    # Regresiones del review adversarial (2026-08-31):
    # (a) el tope G aplica tambien a refinamientos — atomo de 4 con G=1.
    p2 = {i: Fraction(1, 2) for i in range(4)}
    at4 = (((0, 1, 2, 3), 1),)
    assert SolverLaminar(p2, u, 1, 'strict').V((), at4, 1) == Fraction(3, 4)
    assert SolverLaminar(p2, u, 1, 'posterior_zero').V((), at4, 1) == Fraction(3, 2)
    # (b) bajo strict, el atomo deducido de conteo 0 ofrece subconjuntos <= G.
    assert SolverLaminar(p2, u, 1, 'strict').V((), (((0, 1), 0),), 1) == 1
    # (c) las acciones ilegales se rechazan en la API publica.
    try:
        sol_s.valor_forzando_primera(U0, (), B, ('open', (0, 1, 2)))
        raise RuntimeError('debio rechazar la accion ilegal')
    except ValueError:
        pass
    print('OK regresiones: tope G en refinamientos, menu strict r=0 por '
          'subconjuntos, y rechazo de acciones ilegales')


def _peso(zvec, A, p):
    w = Fraction(1)
    for zi, i in zip(zvec, A):
        w *= p[i] if zi else 1 - p[i]
    return w


if __name__ == '__main__':
    _valida_y_demo()
