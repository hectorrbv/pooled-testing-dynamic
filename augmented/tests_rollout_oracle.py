"""Tests de la interfaz de scorers (B-M5) y del oraculo de rollout (B-M6, gate G5)."""

import pytest

from augmented.laminar_benchmarks import (
    ExactPolicyEvaluator,
    balanced_laminar_library,
    greedy_laminar_value,
    rollout_laminar_value,
)
from augmented.rollout_oracle import (
    G5_TOL,
    GreedyPolicy,
    RolloutPolicy,
    cross_check,
    oracle,
    value_by_belief_dp,
    value_by_latent_enumeration,
    verify_g5,
    verify_proposition_b,
)
from augmented.scorers import S0, REGISTRY, get_scorer, select_action, tie_break_key

# Bateria de instancias: homogenea, heterogenea, prevalencia alta, utilidades
# desiguales.  n pequeno a proposito: el evaluador latente recorre 2^n perfiles.
BATERIA = [
    ("homogenea q=0.3", [0.7] * 5, [1.0] * 5, 3, 2),
    ("heterogenea", [0.5, 0.8, 0.2, 0.9, 0.6], [1.0, 3.0, 2.0, 1.5, 0.5], 3, 3),
    ("prevalencia alta", [0.85] * 6, [1.0] * 6, 3, 3),
    ("utilidades desiguales", [0.4] * 4, [10.0, 1.0, 1.0, 1.0], 2, 2),
    ("presupuesto 1", [0.6, 0.3, 0.9], [2.0, 1.0, 3.0], 1, 2),
]


# ------------------------------------------------------------------ B-M5 scorers

def test_s0_declara_su_ficha_completa():
    spec = S0().spec
    assert spec.name == "S0"
    assert spec.uses_full_history and spec.hard_clearing and spec.discounts_credited
    assert not spec.budget_aware        # S0 es ciego al presupuesto, por definicion
    assert spec.realizable


def test_s0_es_probabilidad_de_cero_por_utilidad():
    """Instancia a mano: pool {0,1}, q=0.5 cada uno, u=1 cada uno."""
    ev = ExactPolicyEvaluator([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], 2, 2)
    pool = 0b011
    assert S0().score(ev, ev.all_worlds, 0, None, pool) == pytest.approx(0.25 * 2.0)


def test_s0_de_un_singleton_es_q_por_u():
    ev = ExactPolicyEvaluator([0.7, 0.5], [3.0, 1.0], 1, 1)
    assert S0().score(ev, ev.all_worlds, 0, None, 0b01) == pytest.approx(0.3 * 3.0)


def test_s0_descuenta_lo_ya_acreditado():
    """Un individuo en C(H) no vuelve a pagar (§5.7, sin doble conteo)."""
    ev = ExactPolicyEvaluator([0.5, 0.5], [1.0, 1.0], 2, 2)
    sin_acreditar = S0().score(ev, ev.all_worlds, 0, None, 0b11)
    con_uno_acreditado = S0().score(ev, ev.all_worlds, 0b01, None, 0b11)
    assert con_uno_acreditado == pytest.approx(sin_acreditar / 2)


def test_s0_nunca_agrupa_con_q_pequena():
    """Patologia conocida y esperada de S0 (§14.2): motiva S3."""
    ev = ExactPolicyEvaluator([0.99] * 4, [1.0] * 4, 2, 4)
    s = S0()
    single = s.score(ev, ev.all_worlds, 0, None, 0b0001)
    grupo = s.score(ev, ev.all_worlds, 0, None, 0b1111)
    assert single > grupo


def test_registro_solo_tiene_s0():
    assert set(REGISTRY) == {"S0"}


def test_s1_y_s2_no_se_cablean_y_el_error_dice_por_que():
    for muerto in ("S1", "S2"):
        with pytest.raises(KeyError) as exc:
            get_scorer(muerto)
        assert "S1 colapsa" in str(exc.value) and "S2 esta muerto" in str(exc.value)


# ------------------------------------------------------------------ desempate §5.11

def test_desempate_prefiere_mayor_score():
    n = 4
    ganador = select_action([(0.5, 0b0011), (0.9, 0b1100)], n)
    assert ganador[1] == 0b1100


def test_desempate_ante_score_igual_prefiere_pool_mas_chico():
    n = 4
    ganador = select_action([(0.5, 0b0111), (0.5, 0b0001)], n)
    assert ganador[1] == 0b0001


def test_desempate_ante_score_y_tamano_iguales_prefiere_menor_mascara():
    n = 4
    ganador = select_action([(0.5, 0b1000), (0.5, 0b0001)], n)
    assert ganador[1] == 0b0001


def test_desempate_es_orden_total_y_no_depende_del_orden_de_entrada():
    n = 4
    filas = [(0.5, 0b0011), (0.5, 0b0101), (0.5, 0b0001), (0.7, 0b1000)]
    assert select_action(filas, n) == select_action(list(reversed(filas)), n)


def test_desempate_sin_candidatas_falla_explicitamente():
    with pytest.raises(ValueError):
        select_action([], 4)


def test_clave_de_desempate_es_estable():
    assert tie_break_key(1.0, 0b0001, 4) > tie_break_key(1.0, 0b0011, 4)


# ------------------------------------------------------------------ B-M6 gate G5

@pytest.mark.parametrize("nombre,p,u,B,G", BATERIA, ids=[c[0] for c in BATERIA])
def test_g5_los_dos_evaluadores_coinciden(nombre, p, u, B, G):
    """DP sobre creencias contra enumeracion de perfiles latentes, a 1e-10."""
    for chk in verify_g5(p, u, B, G):
        assert chk.passes, f"{nombre}/{chk.label}: gap {chk.gap:.3e} > {G5_TOL:g}"


@pytest.mark.parametrize("nombre,p,u,B,G", BATERIA, ids=[c[0] for c in BATERIA])
def test_rollout_domina_a_greedy(nombre, p, u, B, G):
    _, greedy, roll = oracle(p, u, B, G)
    assert value_by_belief_dp(roll) >= value_by_belief_dp(greedy) - 1e-12


@pytest.mark.parametrize("nombre,p,u,B,G", BATERIA, ids=[c[0] for c in BATERIA])
def test_proposicion_b_se_verifica_estado_por_estado(nombre, p, u, B, G):
    rep = verify_proposition_b(p, u, B, G)
    assert rep.greedy_always_candidate, "hipotesis 3 rota: greedy fuera de las candidatas"
    assert rep.dominance_holds, f"dominancia rota, margen minimo {rep.worst_margin}"
    assert rep.states_checked > 0
    assert rep.root_margin >= -1e-12


def test_el_ancla_de_la_sesion_del_3_de_agosto():
    """n=5, q=0.3, B=3, G=2: greedy 0.900 y rollout 1.011, por camino nuevo."""
    p, u = [0.7] * 5, [1.0] * 5
    _, greedy, roll = oracle(p, u, 3, 2)
    assert value_by_belief_dp(greedy) == pytest.approx(0.900, abs=1e-9)
    assert value_by_belief_dp(roll) == pytest.approx(1.011, abs=1e-9)
    assert value_by_latent_enumeration(greedy) == pytest.approx(0.900, abs=1e-9)
    assert value_by_latent_enumeration(roll) == pytest.approx(1.011, abs=1e-9)


def test_presupuesto_cero_vale_cero():
    _, greedy, roll = oracle([0.5, 0.5], [1.0, 1.0], 0, 2)
    assert value_by_belief_dp(greedy) == 0.0
    assert value_by_belief_dp(roll) == 0.0


def test_biblioteca_vacia_falla_explicitamente():
    ev = ExactPolicyEvaluator([0.5, 0.5], [1.0, 1.0], 1, 2)
    with pytest.raises(ValueError):
        GreedyPolicy(ev, [], S0())


def test_el_rollout_replanifica_y_no_continua_con_greedy():
    """El rollout se separa del greedy en la raiz, y nunca queda por debajo de su Q.

    Q^g(raiz, t) evalua la accion t con continuacion GREEDY. Como el rollout
    replanifica, su valor real domina a ese Q; la desigualdad es debil porque hay
    instancias --- esta entre ellas --- donde despues del primer paso el rollout y
    el greedy vuelven a coincidir y el valor se alcanza con igualdad.
    """
    p, u = [0.7] * 5, [1.0] * 5
    ev, greedy, roll = oracle(p, u, 3, 2)
    mejor_q = max(valor for valor, _ in roll.q_values(ev.all_worlds, 0, 0))
    assert value_by_belief_dp(roll) >= mejor_q - 1e-12
    # La evidencia de que no es greedy: elige otra accion en la raiz, y con eso
    # gana 0.111 de welfare.
    assert roll.action(ev.all_worlds, 0, 0) != greedy.action(ev.all_worlds, 0, 0)
    assert value_by_belief_dp(roll) - value_by_belief_dp(greedy) == pytest.approx(0.111, abs=1e-9)


# ------------------------------------------------------------------ contraste cruzado

@pytest.mark.parametrize("nombre,p,u,B,G", BATERIA, ids=[c[0] for c in BATERIA])
def test_coincide_con_la_implementacion_previa(nombre, p, u, B, G):
    """El oraculo nuevo contra `greedy_and_rollout_values`, que ya existia.

    Son dos implementaciones independientes de la misma pareja de politicas sobre
    la misma biblioteca. Una divergencia aqui significa que alguna de las dos
    tiene una regla de desempate distinta, y §5.11 exige una sola congelada.
    """
    biblioteca = balanced_laminar_library(p, u, G)
    previo_g, previo_r = ExactPolicyEvaluator(p, u, B, G).greedy_and_rollout_values(biblioteca)
    _, greedy, roll = oracle(p, u, B, G, actions=biblioteca)
    assert value_by_belief_dp(greedy) == pytest.approx(previo_g, abs=1e-12)
    assert value_by_belief_dp(roll) == pytest.approx(previo_r, abs=1e-12)


def test_coincide_con_las_funciones_publicas_del_atlas():
    p, u, B, G = [0.7] * 5, [1.0] * 5, 3, 2
    _, greedy, roll = oracle(p, u, B, G)
    assert value_by_belief_dp(greedy) == pytest.approx(greedy_laminar_value(p, u, B, G), abs=1e-12)
    assert value_by_belief_dp(roll) == pytest.approx(rollout_laminar_value(p, u, B, G), abs=1e-12)
