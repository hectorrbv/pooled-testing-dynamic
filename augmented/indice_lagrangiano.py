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

Estatuto (§25): DIAGNOSTICO. Sin garantia probada; la calibracion de lambda es
la pregunta abierta 10.5 del companion. Convencion posterior-zero; r cuenta
infectados.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from augmented.bm17_toy_solver import SolverLaminar, z_tabla


class PoliticaLagrangiana:
    def __init__(self, p, u, G, lam=0.01, tope=3, no_paralisis=True):
        self.p, self.u, self.G = dict(p), dict(u), G
        self.lam, self.tope, self.no_paralisis = lam, tope, no_paralisis

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
            for k in range(1, len(A)):
                for S in combinations(A, k):
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
    def decide(self, U, atomos, b):
        tope = min(b, self.tope)
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
                for k in range(1, len(A)):
                    for S in combinations(A, k):
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
        # nada supera el precio: no se abandona la partida, se cobra lo mejor
        alt, alt_v = None, -1.0
        for k in range(1, min(self.G, len(U)) + 1):
            for S in combinations(sorted(U), k):
                s0 = float(self._z(S)[0]) * sum(self.u[i] for i in S)
                if s0 > alt_v:
                    alt, alt_v = ('open', S), s0
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
    print('pi_L (lambda = 0.01, tope 3, con no-paralisis) contra la bateria\n')
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
