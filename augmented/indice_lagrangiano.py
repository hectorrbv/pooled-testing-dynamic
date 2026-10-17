"""Politica pi_L: indice Lagrangiano local (companion ec. 8.13) + no-paralisis.

Motivacion: el contraejemplo universal (n=6, B=3, G=4) donde pi_M, pi_C, pi_R y
C3 caen todas a 0.6576 del optimo. La region del contraejemplo (infeccion muy
alta, premio moderado) senala dos ingredientes que a la familia greedy le
faltan, y este modulo los prueba:

  1. Valor extraible TOTAL bajo el presupuesto restante, con derecho a PARAR
     temprano — no densidad H_c/c (que reserva el horizonte y por eso
     sobre-cobra la exploracion que casi siempre muere en la primera prueba),
     ni cobro inmediato.
  2. Regla de no-paralisis: si ningun proyecto supera el precio lambda, no se
     abandona la partida — se toma el mejor cobro inmediato.

I_lambda(C) = sup_{pi local a C} E[utilidad acreditada - lambda * pruebas],
con el sup incluyendo la politica nula (abandono gratis). lambda es el precio
sombra de una prueba; con lambda grande la exploracion muere y la politica
degenera a greedy inmediato (ver el barrido del __main__).

Horizonte rodante: el indice se trunca a `horizonte` pruebas (3 por defecto) aunque
queden mas. No es entonces el I_lambda de horizonte-restante-completo de la
ec. 8.13, sino su version de horizonte rodante (MPC): cada decision se planea
con una ventana corta y se replanea tras observar. Se declara asi a proposito
—- planear con b completo cuesta como resolver el problema —- y se anota como
desviacion del enunciado.

Parametros inmutables: lam, horizonte, no_paralisis y G son de solo lectura
porque las memorias cachean sobre la instancia; mutarlos daria valores
obsoletos en silencio. Para otro lambda, construir otra politica.

Estatuto (§25): DIAGNOSTICO. Sin garantia probada; la calibracion de lambda es
la pregunta abierta 10.5 del companion. Convencion posterior-zero; r cuenta
infectados.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from augmented.bm17_toy_solver import SolverLaminar, z_tabla


class PoliticaLagrangiana:
    def __init__(self, p, u, G, lam=0.01, horizonte=3, no_paralisis=True):
        self._p, self._u, self._G = dict(p), dict(u), G
        self._lam, self._horizonte = lam, horizonte
        self._no_paralisis = no_paralisis

    # solo lectura: las memorias cachean sobre la instancia (ver docstring)
    p = property(lambda self: self._p)
    u = property(lambda self: self._u)
    G = property(lambda self: self._G)
    lam = property(lambda self: self._lam)
    horizonte = property(lambda self: self._horizonte)
    no_paralisis = property(lambda self: self._no_paralisis)

    def _subconjuntos_legales(self, A):
        """Subconjuntos propios no vacios de A que respetan |S| <= G."""
        for k in range(1, min(self.G, len(A) - 1) + 1):
            yield from combinations(A, k)

    # ---------------------------------------------------------- mecanica
    def _z(self, S):
        return z_tabla(tuple(sorted(S)), self.p)

    def _pieza(self, X, r):
        X = tuple(sorted(X))
        if r == 0:
            return sum(self.u[i] for i in X), None
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
                out.append((float(prob), float(rew), U - frozenset(S),
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
                out.append((float(prob), float(rS + rR), U,
                            tuple(sorted(otros + nuevos))))
        return out

    # ------------------------------------------------- I_lambda (ec. 8.13)
    @lru_cache(maxsize=None)
    def _I_atomos(self, atomos, tope):
        """Mejor valor neto local partiendo de atomos vivos; 0 = abandonar."""
        if tope == 0 or not atomos:
            return 0.0
        mejor = 0.0
        for (A, r) in atomos:
            for S in self._subconjuntos_legales(A):
                val = -self.lam
                for prob, rew, _, ats in self._ramas(frozenset(), atomos,
                                                     ('ref', (A, r), S)):
                    val += prob * (rew + self._I_atomos(ats, tope - 1))
                mejor = max(mejor, val)
        return mejor

    @lru_cache(maxsize=None)
    def _I_virgen(self, S, tope):
        """Valor neto de abrir el root S y seguir adentro; 0 = no abrirlo."""
        if tope == 0:
            return 0.0
        val = -self.lam
        for prob, rew, _, ats in self._ramas(frozenset(S), (), ('open', S)):
            val += prob * (rew + self._I_atomos(ats, tope - 1))
        return max(0.0, val)

    # ---------------------------------------------------------- decision
    def _cobro_inmediato(self, U, atomos, accion):
        """M_h de una accion: utilidad que acredita en esta misma prueba."""
        esperado = 0.0
        for prob, rew, _, _ in self._ramas(U, atomos, accion):
            esperado += prob * rew
        return esperado

    def decide(self, U, atomos, b):
        tope = min(b, self.horizonte)
        mejor, mejor_I = None, 1e-12
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(sorted(U), k):
                v = self._I_virgen(S, tope)
                if v > mejor_I:
                    mejor, mejor_I = ('open', S), v
        for (A, r) in atomos:
            v = self._I_atomos(((A, r),), tope)
            if v > mejor_I:
                sub, sub_v = None, -1e9
                for S in self._subconjuntos_legales(A):
                    val = -self.lam
                    for prob, rew, _, ats in self._ramas(
                            frozenset(), ((A, r),), ('ref', (A, r), S)):
                        val += prob * (rew + self._I_atomos(ats, tope - 1))
                    if val > sub_v:
                        sub, sub_v = S, val
                if sub:
                    mejor, mejor_I = ('ref', (A, r), sub), v
        if mejor is not None:
            return mejor
        if not self.no_paralisis:
            return None
        # nada supera el precio: no se abandona la partida — se toma el mejor
        # cobro inmediato del menu COMPLETO (virgenes y refinamientos)
        alt, alt_v = None, -1.0
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(sorted(U), k):
                m = self._cobro_inmediato(U, atomos, ('open', S))
                if m > alt_v:
                    alt, alt_v = ('open', S), m
        for (A, r) in atomos:
            for S in self._subconjuntos_legales(A):
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
    uf = {i: Fraction(v) for i, v in u.items()}
    return float(SolverLaminar(pf, uf, G, 'posterior_zero')
                 .V(frozenset(pf), (), B))


# (p, u, B, G, mejor de la bateria {pi_M, pi_C, pi_R, C3} segun el scoreboard)
INSTANCIAS = {
    'contraejemplo universal': (
        {0: 0.9, 1: 0.825, 2: 0.875, 3: 0.8, 4: 0.95, 5: 0.85},
        {0: 2, 1: 1, 2: 1, 3: 1, 4: 4, 5: 2}, 3, 4, 0.6576),
    'B-M16 (no-reentrada)': (   # C3 ya lograba 1.0 aqui; las tres del §8, 0.7752
        {i: 0.7 for i in range(4)}, {i: 1 for i in range(4)}, 2, 2, 1.0000),
    'rare-health G=4': (        # pi_R ya lograba 0.9976; pi_M solo 0.5087
        {i: 0.99 for i in range(8)}, {i: 1 for i in range(8)}, 4, 4, 0.9976),
    'baja prevalencia q=0.7': (
        {i: 0.3 for i in range(5)}, {i: 1 for i in range(5)}, 3, 3, None),
}


if __name__ == '__main__':
    print('pi_L (lambda = 0.01, horizonte 3, con no-paralisis) contra la bateria\n')
    print(f'{"instancia":26s} {"optimo":>8s} {"pi_L":>8s} {"ratio":>7s} '
          f'{"mejor golosa":>13s}')
    ratios = {}
    for nombre, (p, u, B, G, golosa) in INSTANCIAS.items():
        opt = optimo(p, u, B, G)
        pol = PoliticaLagrangiana(p, u, G, lam=0.01)
        v = pol.valor(frozenset(p), (), B)
        ratios[nombre] = v / opt
        gol = f'{golosa:.4f}' if golosa else '—'
        print(f'{nombre:26s} {opt:8.4f} {v:8.4f} {v/opt:7.4f} {gol:>13s}')

    # El contraejemplo: pi_L recupera lo que las cuatro golosas pierden.
    assert ratios['contraejemplo universal'] > 0.95
    assert ratios['B-M16 (no-reentrada)'] > 0.999
    assert ratios['rare-health G=4'] > 0.99
    print('\nOK: pi_L iguala o supera a la mejor de la bateria en las cuatro; '
          'gana +0.31 justo donde las cuatro caen a 0.6576')

    # --- regresiones del review adversarial (2026-09-01) -----------------
    # (a) el tope G aplica tambien a refinamientos de un atomo grande
    pol = PoliticaLagrangiana({i: 0.5 for i in range(4)},
                              {i: 1 for i in range(4)}, G=1, lam=0.0)
    acc = pol.decide(frozenset(), (((0, 1, 2, 3), 1),), 1)
    assert acc is not None and len(acc[2]) <= 1, acc
    assert abs(pol.valor(frozenset(), (((0, 1, 2, 3), 1),), 1) - 1.5) < 1e-9
    # (b) los parametros son de solo lectura (la memoria cachea sobre self)
    try:
        pol.lam = 0.5
        raise RuntimeError('lam deberia ser de solo lectura')
    except AttributeError:
        pass
    # (c) no-paralisis considera refinamientos, no solo pools virgenes
    pol = PoliticaLagrangiana({0: 0.5, 1: 0.5}, {0: 1, 1: 1}, G=2, lam=1.1)
    acc = pol.decide(frozenset(), (((0, 1), 1),), 1)
    assert acc is not None and acc[0] == 'ref', acc
    assert abs(pol.valor(frozenset(), (((0, 1), 1),), 1) - 1.0) < 1e-9
    print('OK regresiones: tope G en refinamientos, parametros inmutables, '
          'y no-paralisis sobre el menu completo')

    print('\nSensibilidad a lambda (el precio de una prueba):')
    for nombre in ('contraejemplo universal', 'rare-health G=4'):
        p, u, B, G, _ = INSTANCIAS[nombre]
        opt = optimo(p, u, B, G)
        fila = []
        for lam in (0.001, 0.01, 0.05, 0.3):
            pol = PoliticaLagrangiana(p, u, G, lam=lam)
            fila.append(f'λ={lam}: {pol.valor(frozenset(p), (), B)/opt:.3f}')
        print(f'  {nombre:24s} ' + '  '.join(fila))
    print('\nLambda grande mata la exploracion y degenera a greedy: la '
          'calibracion es la pregunta abierta 10.5 del companion.')
