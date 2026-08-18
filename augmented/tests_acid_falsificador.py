"""Tests del harness del acid test (B-M7) y del falsificador (B-M8)."""

import math

import pytest

from augmented.acid_test import (
    AcidInstance,
    anchor_grid,
    anchor_instance,
    cbs_lower_bound,
    check8_robustez,
    check9_desempate,
    k_from_budget,
    run_acid_test,
    run_local_checks,
    singleton_baseline,
    verify_anchor,
)
from augmented.falsificador import atomos, clasificar, cruza, default_sweep, run
from augmented.scorers import get_scorer

S0 = get_scorer("S0")


# ------------------------------------------------------------------ B-M7 ancla

def test_el_ancla_de_la_seccion_16_cuadra():
    ok, detalle = verify_anchor()
    assert ok, detalle


def test_k_desde_el_presupuesto_reserva_el_test_acreditador():
    """k = B - ceil(log2 G) - 1; el -1 es la prueba que convierte deduccion en pago."""
    assert k_from_budget(7, 16) == 2      # 7 - 4 - 1
    assert k_from_budget(4, 4) == 1       # 4 - 2 - 1
    assert k_from_budget(2, 4) == 0       # no alcanza ni para una ruta
    assert k_from_budget(1, 2) == 0


def test_cota_cbs_es_la_probabilidad_de_hallar_al_menos_un_sano():
    inst = AcidInstance(q=0.05, G=16, k=2, B=7)
    assert cbs_lower_bound(inst) == pytest.approx(1 - 0.95 ** 32, abs=1e-12)
    assert inst.coverage == 32


def test_baseline_singleton_es_B_por_q():
    assert singleton_baseline(AcidInstance(q=0.05, G=16, k=2, B=7)) == pytest.approx(0.35)


def test_sin_presupuesto_para_una_ruta_la_cota_es_cero():
    assert cbs_lower_bound(AcidInstance(q=0.1, G=8, k=0, B=2)) == 0.0


def test_la_malla_analitica_cubre_G_hasta_16():
    filas = anchor_grid()
    assert {f.instance.G for f in filas} == {2, 4, 8, 16}
    assert any(f.ratio > 2.0 for f in filas), "la familia debe tener celdas con ventaja grande"
    assert any(f.ratio < 1.0 for f in filas), "y celdas donde el baseline gana; no es universal"


def test_el_ancla_da_una_razon_de_2_3():
    inst = anchor_instance()
    assert cbs_lower_bound(inst) / singleton_baseline(inst) == pytest.approx(2.30, abs=0.01)


# ------------------------------------------------------------------ B-M7 checks

def test_s0_falla_exactamente_los_checks_que_exigen_presupuesto():
    """Resultado esperado y documentado: S0 es el baseline, no un candidato.

    Falla el check 1 (no abre territorio virgen: con q chica el pool grande puntua
    q^G) y el 6 (es ciego al presupuesto). Ambos son justo lo que S3 debe corregir.
    """
    rep = run_acid_test("S0")
    fallidos = {c.number for c in rep.failed}
    assert fallidos == {1, 6}, f"cambio el diagnostico de S0: falla {sorted(fallidos)}"


def test_s0_no_duplica_utilidad_acreditada():
    rep = run_local_checks(S0, AcidInstance(q=0.15, G=4, k=2, B=5))
    check7 = next(c for c in rep.checks if c.number == 7)
    assert check7.passed is True


def test_s0_distingue_cero_observado_de_deduccion():
    rep = run_local_checks(S0, AcidInstance(q=0.15, G=4, k=2, B=5))
    check4 = next(c for c in rep.checks if c.number == 4)
    assert check4.passed is True


def test_los_veredictos_son_bool_de_python_no_de_numpy():
    """Sin normalizar, `np.bool_ is True` es falso y los conteos mienten."""
    for c in run_acid_test("S0").checks:
        assert c.passed is None or c.passed is True or c.passed is False


def test_check8_es_estable_para_s0():
    assert check8_robustez(S0).passed is True


def test_check9_sobrevive_a_romper_empates():
    assert check9_desempate(S0, AcidInstance(q=0.15, G=4, k=2, B=5)).passed is True


def test_instancia_demasiado_grande_falla_con_mensaje_util():
    with pytest.raises(ValueError, match="excede el limite"):
        run_local_checks(S0, AcidInstance(q=0.05, G=16, k=2, B=7))


# ------------------------------------------------------------------ B-M8 cruce

def test_definicion_formal_de_cruce():
    """t cruza T si se tocan y ninguno contiene al otro (§17)."""
    assert cruza(0b0110, 0b0011)               # comparten el bit 1, ninguno contiene al otro
    assert not cruza(0b0011, 0b1100)           # disjuntos
    assert not cruza(0b0011, 0b0111)           # descendiente
    assert not cruza(0b0111, 0b0011)           # ancestro
    assert not cruza(0b0011, 0b0011)           # repetida
    assert not cruza(0b0000, 0b0011)           # vacio


def test_cruce_es_simetrico():
    for a, b in [(0b0110, 0b0011), (0b1010, 0b0110), (0b0011, 0b1100)]:
        assert cruza(a, b) == cruza(b, a)


def test_atomos_de_dos_pools_que_se_cruzan():
    celdas = sorted(atomos([(0b0011, 1), (0b0110, 1)]))
    assert celdas == sorted([0b0001, 0b0010, 0b0100])


def test_atomos_de_pools_disjuntos():
    assert sorted(atomos([(0b0011, 0), (0b1100, 0)])) == sorted([0b0011, 0b1100])


# ------------------------------------------------------------------ B-M8 clases

def test_clase_virgen_cuando_no_toca_nada_probado():
    clase, _ = clasificar(0b1100, [(0b0011, 1)], 0, 4)
    assert clase == "virgen"


def test_clase_descendiente_dentro_de_un_pool_probado():
    clase, _ = clasificar(0b0001, [(0b0011, 1)], 0, 4)
    assert clase == "descendiente"


def test_clase_ancestro_contiene_un_pool_probado():
    clase, _ = clasificar(0b0111, [(0b0011, 1)], 0, 4)
    assert clase == "ancestro"


def test_clase_repetida():
    clase, _ = clasificar(0b0011, [(0b0011, 1)], 0, 4)
    assert clase == "repetida"


def test_clase_cruzada_y_su_flag():
    clase, flags = clasificar(0b0110, [(0b0011, 1)], 0, 4)
    assert clase == "cruzada" and flags["cruzada"] is True


def test_flag_atomo_virgen_mezcla_territorio():
    _, flags = clasificar(0b0101, [(0b0011, 1)], 0, 4)
    assert flags["atomo_virgen"] is True


def test_flag_dominada_cuando_todo_esta_acreditado():
    _, flags = clasificar(0b0011, [(0b0011, 0)], 0b0011, 4)
    assert flags["dominada"] is True


def test_los_flags_de_separabilidad_de_s3_se_declaran_no_disponibles():
    """Sin S3 no tienen contenido, y se marcan None en vez de inventar un False."""
    _, flags = clasificar(0b0001, [(0b0011, 1)], 0, 4)
    assert flags["valor_perdido_separabilidad"] is None
    assert flags["local_realizable_no_conjunta"] is None


# ------------------------------------------------------------------ B-M8 barrido

@pytest.fixture(scope="module")
def barrido():
    return run(sweep=[(4, 2, 2, 0.45), (5, 3, 3, 0.90), (4, 2, 3, 0.05)])


def test_el_barrido_produce_decisiones_para_las_tres_politicas(barrido):
    filas, resumen = barrido
    assert {r["policy"] for r in resumen} == {"S0", "rollout", "optimo"}
    assert len(filas) > 0


def test_las_masas_por_clase_suman_uno(barrido):
    _, resumen = barrido
    for r in resumen:
        total = sum(v for k, v in r.items() if k.startswith("w_"))
        assert total == pytest.approx(1.0, abs=1e-6), r["instance_id"]


def test_el_rollout_tiene_regret_local_cero_por_construccion(barrido):
    _, resumen = barrido
    for r in resumen:
        if r["policy"] == "rollout":
            assert r["mean_local_regret"] == pytest.approx(0.0, abs=1e-12)


def test_el_optimo_no_maximiza_Q_del_rollout(barrido):
    """Resultado, no bug: Q^g evalua con continuacion greedy, no con la optima."""
    _, resumen = barrido
    regrets = [r["mean_local_regret"] for r in resumen if r["policy"] == "optimo"]
    assert all(x >= -1e-12 for x in regrets)


def test_cada_decision_registra_probabilidad_positiva(barrido):
    filas, _ = barrido
    assert all(d.probability > 0.0 for d in filas)


def test_el_barrido_por_defecto_cubre_lo_declarado_en_la_seccion_17():
    casos = default_sweep()
    assert {c[0] for c in casos} == {4, 5, 6}
    assert {c[1] for c in casos} == {1, 2, 3}
    assert {c[2] for c in casos} == {2, 3}
