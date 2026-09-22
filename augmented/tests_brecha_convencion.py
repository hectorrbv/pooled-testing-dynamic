"""Tests de la Proposicion de brecha de convencion + tercera via del solver B-M17.

Dos programas que no comparten codigo calculan el optimo laminar exacto:

  * `augmented.bm17_toy_solver.SolverLaminar`  -- recursion 5.5 del companion
    sobre estados (U, atomos, b), posterior por convolucion Poisson-binomial.
  * `augmented.enumerador_historias.PathwiseEnumerator` -- fuerza bruta sobre perfiles
    explicitos e historias explicitas, sin formula de posterior.

Regla de dos vias de §25 del plan maestro, aplicada al juez que esta detras de
todos los cocientes de los scoreboards (contraejemplo universal, pi_L, constante
del portafolio).  Todo se compara en `Fraction`: la igualdad es exacta, no hay
tolerancia.  Ademas se comprueban, con ambos programas, las propiedades de la
Proposicion de brecha de convencion en el juego minimo n=4, G=2, B=2
(acta 2026-09-01, fila §32 2026-09-01).

Escala: n <= 5, B <= 3 (la suite completa corre en decenas de segundos).
"""

import random
from fractions import Fraction
from itertools import combinations

import pytest

from augmented.bm17_toy_solver import SolverLaminar
from augmented.enumerador_historias import CONVENTIONS, PathwiseEnumerator, homogeneous


# ----------------------------------------------------------------------------
# ayudas
# ----------------------------------------------------------------------------
def _solver(p_list, u_list, G, conv):
    return SolverLaminar(
        {i: p for i, p in enumerate(p_list)},
        {i: u for i, u in enumerate(u_list)},
        G,
        conv,
    )


def _cotejo_completo(p_list, u_list, G, B, conv):
    """Optimo y cada primera accion legal forzada: enumerador == solver, exacto."""
    n = len(p_list)
    enum = PathwiseEnumerator(p_list, u_list, G, conv)
    sol = _solver(p_list, u_list, G, conv)
    U = frozenset(range(n))
    assert enum.value(B) == sol.V(U, (), B)
    for k in range(1, min(G, n) + 1):
        for S in combinations(range(n), k):
            assert enum.value_forcing_first(B, S) == sol.valor_forzando_primera(
                U, (), B, ("open", S)
            ), (n, G, B, conv, S)
    return enum, sol


# ----------------------------------------------------------------------------
# 1. Valores de referencia de la nota de diseno (n=4, q_sano=0.3, u=1, G=2)
# ----------------------------------------------------------------------------
REFERENCIA = {
    "strict": {1: Fraction(3, 10), 2: Fraction(3, 5), 3: Fraction(1011, 1000)},
    "posterior_zero": {1: Fraction(3, 10), 2: Fraction(387, 500), 3: Fraction(537, 500)},
}


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("B", [1, 2, 3])
def test_referencia_n4_q03_dos_vias(conv, B):
    q = Fraction(3, 10)
    p_list, u_list = [1 - q] * 4, [1] * 4
    enum, sol = _cotejo_completo(p_list, u_list, 2, B, conv)
    assert enum.value(B) == REFERENCIA[conv][B]
    assert sol.V(frozenset(range(4)), (), B) == REFERENCIA[conv][B]


def test_par_primero_y_reentrada_n4_q03():
    """Adenda del guion 1-sep: par-primero B=2 vale 0.564 (strict) / 0.774 (pz);
    tras ({0,1}, 1) con una prueba, refinar {0} vale 1/2 (strict) / 1 (pz)."""
    q = Fraction(3, 10)
    esperado_par = {"strict": Fraction(141, 250), "posterior_zero": Fraction(387, 500)}
    esperado_reentrada = {"strict": Fraction(1, 2), "posterior_zero": Fraction(1)}
    for conv in CONVENTIONS:
        enum = homogeneous(4, q, 1, 2, conv)
        sol = _solver([1 - q] * 4, [1] * 4, 2, conv)
        U = frozenset(range(4))
        assert enum.value_forcing_first(2, (0, 1)) == esperado_par[conv]
        assert sol.valor_forzando_primera(U, (), 2, ("open", (0, 1))) == esperado_par[conv]
        # reentrada: estado tras el par con conteo 1, una prueba restante
        h = ((0b11, 1),)                       # pool {0,1} con conteo 1, como bitmask
        assert enum.Q(h, 1, 0b01, enum.consistent(h)) == esperado_reentrada[conv]
        assert sol.valor_forzando_primera(
            frozenset({2, 3}), (((0, 1), 1),), 1, ("ref", ((0, 1), 1), (0,))
        ) == esperado_reentrada[conv]


# ----------------------------------------------------------------------------
# 2. Malla homogenea: n<=5, B<=3, G en {2, 3, n}, cuatro q, ambas convenciones
# ----------------------------------------------------------------------------
def _malla():
    casos = []
    for n in (2, 3, 4, 5):
        for G in sorted({2, 3, n} - {g for g in (3,) if g > n}):
            for B in (1, 2, 3):
                for q in (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2), Fraction(7, 10)):
                    for conv in CONVENTIONS:
                        casos.append((n, G, B, q, conv))
    return casos


@pytest.mark.parametrize("n,G,B,q,conv", _malla())
def test_malla_homogenea_dos_vias(n, G, B, q, conv):
    _cotejo_completo([1 - q] * n, [1] * n, G, B, conv)


# ----------------------------------------------------------------------------
# 3. Instancias heterogeneas aleatorias (semilla fija): raiz y primeras acciones
# ----------------------------------------------------------------------------
def _aleatorias(cuantas=40, semilla=20260913):
    rng = random.Random(semilla)
    casos = []
    for _ in range(cuantas):
        n = rng.choice([2, 3, 4, 5])
        p_list = [Fraction(rng.randint(1, 9), 10) for _ in range(n)]
        u_list = [rng.randint(1, 5) for _ in range(n)]
        G = rng.randint(2, n)
        B = rng.randint(1, 3)
        casos.append((tuple(p_list), tuple(u_list), G, B))
    return casos


@pytest.mark.parametrize("conv", CONVENTIONS)
@pytest.mark.parametrize("p_list,u_list,G,B", _aleatorias())
def test_heterogeneas_aleatorias_dos_vias(p_list, u_list, G, B, conv):
    _cotejo_completo(list(p_list), list(u_list), G, B, conv)


# ----------------------------------------------------------------------------
# 4. Proposicion de brecha de convencion, juego minimo n=4, G=2, B=2
#    (parte 1: dominacion; parte 3(b): formulas exactas para q <= 1/2 y
#    umbral 1/2 bajo strict; verificado con los dos programas)
# ----------------------------------------------------------------------------
Q_GRID = [Fraction(k, 10) for k in range(1, 10)]


@pytest.mark.parametrize("q", Q_GRID)
def test_dominacion_posterior_zero_sobre_strict(q):
    """Parte 1: OPT_str(B) <= OPT_pz(B), y tambien accion por accion."""
    for B in (1, 2, 3):
        e_s, e_z = homogeneous(4, q, 1, 2, "strict"), homogeneous(4, q, 1, 2, "posterior_zero")
        assert e_s.value(B) <= e_z.value(B)
        for k in (1, 2):
            for S in combinations(range(4), k):
                assert e_s.value_forcing_first(B, S) <= e_z.value_forcing_first(B, S)


@pytest.mark.parametrize("q", Q_GRID)
def test_formulas_juego_minimo_B2(q):
    """Parte 3(b): con pi_par = par, refinar si R=1, virgen si no (la mejor
    continuacion par-primero para q <= 1/2):
        V_pz(pi_par)  = 2q + q (q^2 + p^2)
        V_str(pi_par) = 2q + q^2 (2q - 1)
        V(pi_sing)    = 2q
    Para q > 1/2 la mejor continuacion cambia ({3,4} vale 2q^2 > q) y las
    formulas quedan como cota inferior estricta."""
    p = 1 - q
    v_pair_pz = homogeneous(4, q, 1, 2, "posterior_zero").value_forcing_first(2, (0, 1))
    v_pair_st = homogeneous(4, q, 1, 2, "strict").value_forcing_first(2, (0, 1))
    f_pz = 2 * q + q * (q * q + p * p)
    f_st = 2 * q + q * q * (2 * q - 1)
    if q <= Fraction(1, 2):
        assert v_pair_pz == f_pz
        assert v_pair_st == f_st
        assert v_pair_pz - v_pair_st == p * q          # la brecha sale de la rama R=1
    else:
        assert v_pair_pz > f_pz and v_pair_st > f_st
    assert f_pz - 2 * q == q * (q * q + p * p) > 0     # par > singletons bajo pz, todo q


@pytest.mark.parametrize("q", Q_GRID)
def test_primera_accion_optima_juego_minimo_B2(q):
    """Corolario en B=2: bajo posterior_zero la primera accion optima es un par
    para todo q; bajo strict es un singleton si q < 1/2, empate exacto en
    q = 1/2 y un par si q > 1/2 (umbral de la Prop 1 de [v4] reproducido)."""
    for conv in CONVENTIONS:
        enum = homogeneous(4, q, 1, 2, conv)
        sol = _solver([1 - q] * 4, [1] * 4, 2, conv)
        v_pair = enum.value_forcing_first(2, (0, 1))
        v_single = enum.value_forcing_first(2, (0,))
        # mejor continuacion tras un singleton: otro singleton (q) si q <= 1/2,
        # el par virgen (2q^2) si q > 1/2 -- "dos singletons" es la politica
        # fija pi_sing = 2q, optima entre las singleton-primero solo si q <= 1/2
        esperado_single = 2 * q if q <= Fraction(1, 2) else q + 2 * q * q
        assert v_single == esperado_single
        assert sol.valor_forzando_primera(frozenset(range(4)), (), 2, ("open", (0,))) == esperado_single
        if conv == "posterior_zero":
            assert v_pair > v_single
        elif q < Fraction(1, 2):
            assert v_pair < v_single
        elif q == Fraction(1, 2):
            assert v_pair == v_single
        else:
            assert v_pair > v_single
