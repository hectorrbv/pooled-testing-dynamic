#!/usr/bin/env python3
"""Enumerador sobre historias -- via independiente del solver B-M17 (y de pathwise_enum.py de B).

Calcula el valor optimo laminar exacto (escala de juguete: n <= 5, B <= 3) sin
importar NADA del repo.  No comparte codigo, ni formula, ni estructura de datos
con `augmented/bm17_toy_solver.py`: ahi el estado son atomos residuales con su
conteo y la posterior sale de la convolucion Poisson-binomial; aqui todo se
calcula desde cero sobre perfiles explicitos e historias explicitas.  Es la
regla de dos vias independientes de §25 del plan maestro, aplicada al juez que
esta detras de todos los cocientes de los scoreboards.

Modelo (companion §2-§5; nota de diseno `docs/notes/2026-08-31-diseno-BM17.md`):

  * Personas 0..n-1.  La persona i esta infectada (Z_i = 1) con probabilidad
    p_i y sana con q_i = 1 - p_i, de forma independiente.  Un perfil z en
    {0,1}^n se guarda como bitmask (bit i encendido <=> i infectada) con peso
    exacto prod_i p_i^{z_i} q_i^{1-z_i}, en enteros sobre un denominador comun:
    todas las sumas son exactas.
  * Una prueba sobre un pool T (1 <= |T| <= G) devuelve el numero EXACTO de
    infectados en T (conteo aumentado).
  * Una historia es una tupla de pares (pool, conteo).  Los perfiles
    CONSISTENTES con una historia son los que reproducen el conteo observado en
    cada pool probado.  Toda probabilidad de aqui abajo es un cociente de sumas
    de pesos de perfiles consistentes: sin formula de posterior, sin convolucion,
    sin contabilidad de atomos.
  * Acreditacion (u_i se cobra una sola vez, al acreditarse i por primera vez):
      strict         : i estuvo fisicamente dentro de un pool probado con
                       conteo 0.
      posterior_zero : P(Z_i = 1 | historia) = 0, o sea ningun perfil
                       consistente tiene a i infectada -- incluye la deduccion
                       por resta de conteos.
    Ambas nociones son monotonas en la historia, asi que "recien acreditado en
    este paso" esta bien definido y nadie cobra dos veces.
  * El menu legal (laminar pathwise) se lee directamente de la historia; las
    reglas exactas y su procedencia estan en `legal_actions()`.
  * Valor (presupuesto B duro en toda rama):
        V(h, 0) = 0
        V(h, b) = max{ 0, max_{S legal} sum_c P(c|h) [ u(recien acreditados)
                                                       + V(h+(S,c), b-1) ] }
    memoizado sobre el CONJUNTO de pares (pool, conteo): todo lo de arriba
    depende de la historia solo a traves de ese conjunto, nunca del orden.

API publica:
    PathwiseEnumerator(p, u, G, convention)      # p, u secuencias; convention en CONVENTIONS
        .value(B)                                # optimo, Fraction exacta
        .value_forcing_first(B, pool)            # mejor politica cuya PRIMERA prueba es `pool`
        .best_first_action(B)                    # argmax de la primera accion (desempate lexicografico)
    homogeneous(n, q, u, G, convention)          # constructor comodo, q = P(sana)

Correr el archivo ejecuta los seis valores de referencia de la nota de diseno
(n=4, q=0.3 sana, u=1, G=2) y algunas comprobaciones internas.  El cotejo
completo contra el solver vive en `augmented/tests_brecha_convencion.py`.
"""

from fractions import Fraction
from itertools import combinations

CONVENTIONS = ('strict', 'posterior_zero')


# ----------------------------------------------------------------------------
# ayudas de bitmask (personas y pools son bitmasks sobre 0..n-1)
# ----------------------------------------------------------------------------
def popcount(x):
    return bin(x).count('1')


def bits(mask):
    """Lista ordenada de las personas en `mask`."""
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


def mask_of(people):
    m = 0
    for i in people:
        m |= 1 << i
    return m


def fmt_pool(mask):
    return '{' + ','.join(str(i) for i in bits(mask)) + '}'


# ----------------------------------------------------------------------------
class PathwiseEnumerator:
    def __init__(self, p, u, G, convention):
        if convention not in CONVENTIONS:
            raise ValueError(f'convencion desconocida {convention!r}; usa una de {CONVENTIONS}')
        self.p = [Fraction(x) for x in p]
        self.u = [Fraction(x) for x in u]
        if len(self.p) != len(self.u):
            raise ValueError('p y u deben tener la misma longitud')
        for pi in self.p:
            # 0 < p_i < 1 mantiene todo perfil con peso positivo, asi que
            # "consistente" y "con peso posterior positivo" coinciden (lo que
            # necesita posterior_zero para estar bien definido).
            if not (0 < pi < 1):
                raise ValueError(f'se necesita 0 < p_i < 1, llego {pi}')
        self.n = len(self.p)
        self.G = int(G)
        if self.G < 1:
            raise ValueError('G debe ser >= 1')
        self.convention = convention
        self.ALL = (1 << self.n) - 1

        # pesos exactos de los perfiles, como enteros sobre el denominador comun
        # prod_i denominator(p_i)
        self.den = 1
        for pi in self.p:
            self.den *= pi.denominator
        self.wnum = [self._weight_numerator(z) for z in range(1 << self.n)]
        assert sum(self.wnum) == self.den                    # los pesos suman 1

        self.memo = {}          # (historia canonica, b) -> Fraction
        self.best_action = {}   # (historia canonica, b) -> pool mask o None (= parar)

    # ---- perfiles ----------------------------------------------------------
    def _weight_numerator(self, z):
        w = 1
        for i in range(self.n):
            pi = self.p[i]
            w *= pi.numerator if (z >> i) & 1 else pi.denominator - pi.numerator
        return w

    def consistent(self, history):
        """Perfiles cuyos conteos coinciden con cada (pool, conteo) de la historia."""
        return [z for z in range(1 << self.n)
                if all(popcount(z & T) == c for T, c in history)]

    def prob_mass(self, profiles):
        return sum(self.wnum[z] for z in profiles)

    # ---- acreditacion ------------------------------------------------------
    def accredited(self, history, cons):
        """Bitmask de acreditados tras `history` (cons = sus perfiles consistentes)."""
        if self.convention == 'strict':
            acc = 0
            for T, c in history:
                if c == 0:
                    acc |= T                       # fisicamente dentro de un pool con conteo 0
            return acc
        possibly_infected = 0
        for z in cons:
            possibly_infected |= z                 # algun perfil consistente la tiene infectada
        return self.ALL & ~possibly_infected       # posterior P(infectada) == 0

    # ---- estructura laminar leida de la historia ---------------------------
    @staticmethod
    def tested_mask(history):
        t = 0
        for T, _ in history:
            t |= T
        return t

    def cells(self, history):
        """Atomos residuales: las personas probadas agrupadas por su firma de
        pertenencia a los pools probados hasta ahora (la particion generada por
        la familia laminar).  Se devuelven como lista de bitmasks."""
        groups = {}
        for i in bits(self.tested_mask(history)):
            sig = tuple((T >> i) & 1 for T, _ in history)
            groups[sig] = groups.get(sig, 0) | (1 << i)
        return sorted(groups.values())

    def legal_actions(self, history, cons):
        """Pools que se pueden probar a continuacion.  Reglas (menu laminar
        pathwise; son las de `augmented/bm17_toy_solver.py::_acciones` y de la
        nota de diseno del 2026-08-31, pero calculadas directamente sobre la
        historia):

          (1) Pool VIRGEN: S disjunto de todo pool probado, 1 <= |S| <= G.
          (2) REFINAMIENTO de una celda residual C (ver cells()):
              - conteo(C) == |C|  (toda infectada)   : C esta muerta.
              - conteo(C) == 0 y C acreditada        : C esta muerta
                (posterior_zero: siempre; strict: justo cuando C fue ella misma
                un pool probado con conteo 0).
              - conteo(C) == 0 y C NO acreditada     : solo pasa bajo strict (un
                complemento limpio por deduccion): cualquier S subconjunto de C,
                1 <= |S| <= G, incluida C entera si |C| <= G (prueba no
                informativa que acredita).
              - 0 < conteo(C) < |C|                  : cualquier subconjunto
                ESTRICTO S de C, 1 <= |S| <= min(G, |C| - 1).
          Nada cruza celdas, nada mezcla virgenes con probados, no hay
          superconjuntos y (se comprueba con assert) nunca se reofrece un pool
          identico a uno ya probado.
        """
        tested = self.tested_mask(history)
        virgin = self.ALL & ~tested
        acc = self.accredited(history, cons)
        pools = []
        for k in range(1, min(self.G, popcount(virgin)) + 1):
            for S in combinations(bits(virgin), k):
                pools.append(mask_of(S))
        for C in self.cells(history):
            counts = {popcount(z & C) for z in cons}
            # laminaridad => el conteo de cada celda queda determinado por la historia
            assert len(counts) == 1, (history, fmt_pool(C), counts)
            r = counts.pop()
            size = popcount(C)
            if r == size:
                continue                                   # toda infectada: muerta
            if r == 0:
                if acc & C == C:
                    continue                               # acreditada: muerta
                assert acc & C == 0, 'una celda nunca queda acreditada a medias'
                kmax = min(self.G, size)                   # strict, limpia por deduccion
            else:
                assert acc & C == 0
                kmax = min(self.G, size - 1)               # solo subconjunto estricto
            for k in range(1, kmax + 1):
                for S in combinations(bits(C), k):
                    pools.append(mask_of(S))
        already = {T for T, _ in history}
        assert not (set(pools) & already), 'se ofrecio dos veces un pool identico'
        assert len(set(pools)) == len(pools)
        return pools

    # ---- recursion de Bellman sobre historias ------------------------------
    @staticmethod
    def canonical(history):
        return tuple(sorted(history))

    def Q(self, history, b, S, cons):
        """Esperanza de (recompensa inmediata + continuacion optima) al probar S
        ahora, con b >= 1 pruebas restantes, en `history`, cuyos perfiles
        consistentes son `cons`."""
        assert b >= 1
        total_mass = self.prob_mass(cons)
        acc_before = self.accredited(history, cons)
        buckets = {}
        for z in cons:
            buckets.setdefault(popcount(z & S), []).append(z)
        total, prob_check = Fraction(0), Fraction(0)
        for c in sorted(buckets):
            cons_c = buckets[c]
            prob = Fraction(self.prob_mass(cons_c), total_mass)
            prob_check += prob
            h2 = history + ((S, c),)
            acc_after = self.accredited(h2, cons_c)
            assert acc_after & acc_before == acc_before, 'la acreditacion debe ser monotona'
            reward = sum((self.u[i] for i in bits(acc_after & ~acc_before)), Fraction(0))
            total += prob * (reward + self.V(h2, b - 1, cons_c))
        assert prob_check == 1
        return total

    def V(self, history=(), b=0, cons=None):
        """Utilidad esperada optima que queda por cobrar desde `history` con b pruebas."""
        if b == 0:
            return Fraction(0)
        if cons is None:
            cons = self.consistent(history)
        key = (self.canonical(history), b)
        if key in self.memo:
            return self.memo[key]
        best, arg = Fraction(0), None              # el 0 explicito: dejar de probar
        for S in self.legal_actions(history, cons):
            q = self.Q(history, b, S, cons)
            if q > best:
                best, arg = q, S
        self.memo[key] = best
        self.best_action[key] = arg
        return best

    # ---- conveniencias publicas --------------------------------------------
    def value(self, B):
        return self.V((), B)

    def value_forcing_first(self, B, pool):
        """Valor de la mejor politica cuya primera prueba es `pool` (iterable de
        personas o bitmask), seguida de una continuacion optima."""
        S = pool if isinstance(pool, int) else mask_of(pool)
        cons = self.consistent(())
        if B < 1:
            raise ValueError('se necesita B >= 1 para forzar una primera accion')
        if S not in self.legal_actions((), cons):
            raise ValueError(f'primer pool ilegal {fmt_pool(S)}')
        return self.Q((), B, S, cons)

    def best_first_action(self, B):
        self.V((), B)
        return self.best_action[((), B)]


def homogeneous(n, q, u, G, convention):
    """n personas, cada una sana con probabilidad q (infectada con 1 - q), utilidad u."""
    q = Fraction(q)
    return PathwiseEnumerator([1 - q] * n, [u] * n, G, convention)


# ----------------------------------------------------------------------------
def _selfcheck():
    # Valores de referencia de la nota de diseno / docstring del solver:
    # n = 4, q = 0.3 sana (p = 0.7 infectada), u = 1, G = 2.
    ref = {
        'strict':         {1: Fraction(3, 10), 2: Fraction(3, 5),    3: Fraction(1011, 1000)},
        'posterior_zero': {1: Fraction(3, 10), 2: Fraction(387, 500), 3: Fraction(537, 500)},
    }
    for conv in CONVENTIONS:
        for B, expected in ref[conv].items():
            got = homogeneous(4, Fraction(3, 10), 1, 2, conv).value(B)
            status = 'OK ' if got == expected else 'FALLA'
            print(f'{status} n=4 q=3/10 u=1 G=2 B={B} {conv:14s}: {got} = {float(got):.6f} '
                  f'(referencia {expected})')
            assert got == expected, (conv, B, got, expected)

    # Valores de "par primero" con B = 2 citados en la adenda del guion del 1-sep.
    e_s = homogeneous(4, Fraction(3, 10), 1, 2, 'strict')
    e_z = homogeneous(4, Fraction(3, 10), 1, 2, 'posterior_zero')
    v_pair_s = e_s.value_forcing_first(2, (0, 1))
    v_pair_z = e_z.value_forcing_first(2, (0, 1))
    print(f'par-primero B=2: strict {v_pair_s} = {float(v_pair_s)}, '
          f'posterior_zero {v_pair_z} = {float(v_pair_z)}')
    assert v_pair_s == Fraction(564, 1000) and v_pair_z == Fraction(387, 500)

    # Reentrada del contraejemplo: tras ({0,1}, 1) con una prueba restante,
    # refinar con {0} vale 1/2 bajo strict y 1 bajo posterior_zero.
    for conv, expected in (('strict', Fraction(1, 2)), ('posterior_zero', Fraction(1))):
        e = homogeneous(4, Fraction(3, 10), 1, 2, conv)
        h = ((mask_of((0, 1)), 1),)
        got = e.Q(h, 1, mask_of((0,)), e.consistent(h))
        print(f'OK  reentrada {conv:14s}: {got}')
        assert got == expected
    print('selfcheck OK')


if __name__ == '__main__':
    _selfcheck()
