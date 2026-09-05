"""Tests del tensor condicional de subpools.

Corre con ``pytest augmented/tests_laminar_tables.py`` (o ``pytest`` a secas
desde la raíz, que colecciona todo el repo).

La estrategia es de identidad: cada afirmación se comprueba contra un cálculo
independiente. La forma cerrada contra la enumeración de mundos, la caché
contra convoluciones directas, y los átomos hijos contra tensores construidos
desde cero.
"""

import numpy as np
import pytest

from augmented.laminar_tables import (
    split_after_test,
    subpool_tensor,
    subpool_tensor_brute,
    subset_pmf_cache,
)


def _instancia(seed, m_lo=2, m_hi=8):
    """Un pool aleatorio: cuántas personas y con qué priors."""

    rng = np.random.default_rng(seed)
    m = int(rng.integers(m_lo, m_hi + 1))
    return m, rng.uniform(0.05, 0.95, size=m).tolist()


# --------------------------------------------------------------------------
# La identidad central: dos vías independientes que deben coincidir
# --------------------------------------------------------------------------

def test_forma_cerrada_igual_a_enumeracion():
    """La fórmula y la fuerza bruta dan lo mismo, para todo r y todo subconjunto."""

    for seed in range(25):
        m, p = _instancia(seed)
        for r in range(m + 1):
            esperado = subpool_tensor_brute(p, r)
            obtenido = subpool_tensor(p, r)
            assert set(obtenido) == set(esperado)
            for s in esperado:
                assert np.allclose(obtenido[s], esperado[s], atol=1e-12), (
                    f"seed={seed} r={r} s={s:b}"
                )


def test_cache_es_poisson_binomial():
    """Φ[s] coincide con convolucionar a mano las Bernoulli del bloque."""

    m, p = _instancia(3, m_lo=6, m_hi=6)
    cache = subset_pmf_cache(p)
    for s in range(1 << m):
        a_mano = np.array([1.0])
        for person in range(m):
            if s & (1 << person):
                a_mano = np.convolve(a_mano, [1.0 - p[person], p[person]])
        assert np.allclose(cache[s], a_mano, atol=1e-13), f"bloque {s:b}"


# --------------------------------------------------------------------------
# Las tres propiedades que dictó Francisco en sesión
# --------------------------------------------------------------------------

def test_cada_columna_suma_uno():
    """'Por cada columna la suma de elementos es igual a 1.'"""

    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            for s, columna in subpool_tensor(p, r).items():
                assert abs(columna.sum() - 1.0) < 1e-12, f"s={s:b} r={r}"


def test_columna_del_pool_entero_es_indicadora():
    """'El posterior de T' cuando T'=T es nada más r': unos y ceros.

    Ya observaste el conteo del pool, así que ahí no queda incertidumbre.
    """

    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            columna = subpool_tensor(p, r)[(1 << m) - 1]
            esperado = np.zeros(m + 1)
            esperado[r] = 1.0
            assert np.allclose(columna, esperado, atol=1e-12)


def test_ley_de_soporte():
    """'Algunas entradas son vacías', y se sabe exactamente cuáles.

    El conteo de un subconjunto no puede contradecir al del pool. Con r
    positivos en total y ``m − |s|`` personas fuera del subconjunto, dentro
    tiene que haber al menos ``r − (m − |s|)``, y como mucho ``min(r, |s|)``.
    Fuera de ese rango la probabilidad es cero exacto; dentro es positiva.
    """

    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            for s, columna in subpool_tensor(p, r).items():
                tamano = s.bit_count()
                minimo = max(0, r - (m - tamano))
                maximo = min(r, tamano)
                for k in range(len(columna)):
                    if minimo <= k <= maximo:
                        assert columna[k] > 0.0, f"s={s:b} r={r} k={k}"
                    else:
                        assert columna[k] == 0.0, f"s={s:b} r={r} k={k}"


def test_la_celda_imposible_del_ejemplo_de_bolsillo():
    """Con 2 positivos entre 4, es imposible que tres personas estén limpias.

    Los dos positivos no caben en la única persona que queda fuera. Un
    producto de marginales independientes le daría masa positiva a esta celda.
    """

    tensor = subpool_tensor([0.2, 0.4, 0.6, 0.8], r=2)
    assert tensor[0b0111][0] == 0.0
    assert np.allclose(tensor[0b0011], [0.5353, 0.4498, 0.0149], atol=1e-4)


# --------------------------------------------------------------------------
# La división: los hijos salen de la caché del padre
# --------------------------------------------------------------------------

def test_los_hijos_reusan_los_arreglos_del_padre():
    """No es que coincidan: son el MISMO objeto en memoria.

    Es la forma más directa de comprobar que la división no recalcula nada.
    """

    from augmented.laminar_tables import _restricted_cache

    m, p = _instancia(7, m_lo=6, m_hi=6)
    padre = subset_pmf_cache(p)
    miembros = [0, 2, 5]
    hijo = _restricted_cache(padre, miembros)

    for hijo_mask in range(1 << len(miembros)):
        padre_mask = 0
        for j, persona in enumerate(miembros):
            if hijo_mask & (1 << j):
                padre_mask |= 1 << persona
        assert hijo[hijo_mask] is padre[padre_mask]


def test_division_igual_a_construir_desde_cero():
    """Los átomos de la división coinciden con tensores hechos de cero."""

    for seed in range(10):
        m, p = _instancia(seed, m_lo=4, m_hi=7)
        priors = np.asarray(p)
        rng = np.random.default_rng(1000 + seed)
        probado = int(rng.integers(1, (1 << m) - 1))
        residuo = ((1 << m) - 1) ^ probado

        for r in range(m + 1):
            for conteo in range(probado.bit_count() + 1):
                if not 0 <= r - conteo <= residuo.bit_count():
                    continue
                atomo_probado, atomo_residual = split_after_test(
                    p, r, probado, conteo
                )
                for atomo, mask, propio in (
                    (atomo_probado, probado, conteo),
                    (atomo_residual, residuo, r - conteo),
                ):
                    miembros = [i for i in range(m) if mask & (1 << i)]
                    desde_cero = subpool_tensor(priors[miembros], propio)
                    assert set(atomo.tensor) == set(desde_cero)
                    for s in desde_cero:
                        assert np.allclose(
                            atomo.tensor[s], desde_cero[s], atol=1e-12
                        )


def test_division_coincide_con_enumerar_la_historia_completa():
    """El conteo del hermano es irrelevante dentro de un átomo.

    Es la factorización entre átomos, comprobada contra la enumeración de
    mundos que respetan AMBOS conteos a la vez.
    """

    m = 6
    p = np.random.default_rng(99).uniform(0.15, 0.85, size=m)
    probado = 0b000111

    for r in range(1, m):
        for conteo in range(probado.bit_count() + 1):
            if not 0 <= r - conteo <= m - probado.bit_count():
                continue
            atomo, _ = split_after_test(p.tolist(), r, probado, conteo)

            for s_local in range(1 << len(atomo.members)):
                s_global = 0
                for j, persona in enumerate(atomo.members):
                    if s_local & (1 << j):
                        s_global |= 1 << persona

                pesos = np.zeros(s_local.bit_count() + 1)
                for world in range(1 << m):
                    if world.bit_count() != r:
                        continue
                    if (world & probado).bit_count() != conteo:
                        continue
                    prob = 1.0
                    for persona in range(m):
                        prob *= (p[persona] if world & (1 << persona)
                                 else 1.0 - p[persona])
                    pesos[(world & s_global).bit_count()] += prob
                if pesos.sum() <= 0.0:
                    continue
                assert np.allclose(
                    atomo.tensor[s_local], pesos / pesos.sum(), atol=1e-12
                )


# --------------------------------------------------------------------------
# Entradas degeneradas: fallar fuerte y explicando
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p, r, fragmento", [
    ([0.5, 0.5], 3, "fuera del rango"),
    ([0.5, 0.5], -1, "fuera del rango"),
    ([], 0, "vacío"),
    ([0.5, 1.5], 1, "debe estar en"),
    ([0.5, float("nan")], 1, "finito"),
])
def test_rechaza_entradas_invalidas(p, r, fragmento):
    for funcion in (subpool_tensor, subpool_tensor_brute):
        with pytest.raises(ValueError, match=fragmento):
            funcion(p, r)


def test_rechaza_conteo_de_probabilidad_nula():
    """Pedir 2 positivos cuando dos personas son imposibles no tiene sentido."""

    for funcion in (subpool_tensor, subpool_tensor_brute):
        with pytest.raises(ValueError, match="probabilidad nula"):
            funcion([0.0, 0.0, 0.5, 0.5], 4)


@pytest.mark.parametrize("probado, conteo, r, fragmento", [
    (0b1111, 2, 2, "no lo parte"),
    (0, 0, 2, "vacío"),
    (0b10000, 1, 2, "se sale del pool"),
    (0b0011, 0, 3, "incompatible"),
])
def test_division_rechaza_lo_imposible(probado, conteo, r, fragmento):
    with pytest.raises(ValueError, match=fragmento):
        split_after_test([0.4] * 4, r, probado, conteo)
