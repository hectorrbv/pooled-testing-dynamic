"""Anclas de la separacion exacta en n=10, q=0.2 (B-M11) y de su artefacto (B-M4).

Los valores anclados aqui son los que se citan en `cuentas-n10-q02.md`. Si alguno
se mueve, o cambio el modelo o se rompio el solver; en ninguno de los dos casos se
actualiza el numero sin explicar por que.
"""

import csv
from pathlib import Path

import pytest

from augmented.experiments_separacion_n10 import (
    dynamic_value,
    laminar_value,
    static_value,
)
from augmented.provenance import read_stamp

N, Q = 10, 0.2
CSV = Path(__file__).resolve().parent.parent / "results" / "separacion_n10_q02.csv"
TOL = 1e-9


# ------------------------------------------------------------------ anclas rapidas

def test_presupuesto_1_todas_las_variantes_valen_q():
    for f in (lambda: static_value(N, 1, False, Q), lambda: static_value(N, 1, True, Q),
              lambda: dynamic_value(N, 1, False, Q), lambda: dynamic_value(N, 1, True, Q),
              lambda: laminar_value(N, 1, Q)):
        assert f() == pytest.approx(0.2, abs=TOL)


def test_presupuesto_2_escalera_conocida():
    assert static_value(N, 2, False, Q) == pytest.approx(0.400, abs=TOL)
    assert static_value(N, 2, True, Q) == pytest.approx(0.408, abs=TOL)
    assert dynamic_value(N, 2, False, Q) == pytest.approx(0.488, abs=TOL)
    assert dynamic_value(N, 2, True, Q) == pytest.approx(0.536, abs=TOL)


def test_laminar_no_pierde_en_presupuesto_2():
    assert laminar_value(N, 2, Q) == pytest.approx(dynamic_value(N, 2, True, Q), abs=TOL)


def test_estatico_binario_es_el_diseno_individual():
    """Con pruebas binarias lo individual SI es optimo estatico; con conteos no."""
    for B in (1, 2):
        assert static_value(N, B, False, Q) == pytest.approx(B * Q, abs=TOL)
    assert static_value(N, 2, True, Q) > 2 * Q + TOL


def test_orden_de_las_variantes():
    """Contar y adaptarse solo pueden ayudar; laminar solo puede estorbar."""
    B = 2
    assert static_value(N, B, False, Q) <= static_value(N, B, True, Q) + TOL
    assert static_value(N, B, True, Q) <= dynamic_value(N, B, True, Q) + TOL
    assert dynamic_value(N, B, False, Q) <= dynamic_value(N, B, True, Q) + TOL
    assert laminar_value(N, B, Q) <= dynamic_value(N, B, True, Q) + TOL


def test_laminar_crece_con_el_presupuesto():
    vals = [laminar_value(N, b, Q) for b in range(1, 6)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------------ el artefacto

def test_el_artefacto_existe_y_trae_procedencia():
    assert CSV.exists(), "corre: python3 -m augmented.experiments_separacion_n10"
    stamp = read_stamp(CSV)
    assert stamp["generator"] == "augmented.experiments_separacion_n10.main"
    assert stamp["params"]["n"] == N and stamp["params"]["q"] == Q
    assert stamp["params"]["certification_rule"] == "permissive"


def test_el_artefacto_ancla_la_escalera_del_presupuesto_3():
    fila = next(r for r in csv.DictReader(CSV.open()) if r["B"] == "3")
    esperado = {
        "static_binary": 0.600, "dynamic_binary": 0.7902848,
        "static_augmented": 0.800, "dynamic_augmented_laminar": 0.9282432,
        "dynamic_augmented": 1.00032,
    }
    for col, val in esperado.items():
        assert float(fila[col]) == pytest.approx(val, abs=TOL), col


def test_la_laminaridad_cuesta_siete_por_ciento_en_presupuesto_3():
    fila = next(r for r in csv.DictReader(CSV.open()) if r["B"] == "3")
    perdida = 1 - float(fila["dynamic_augmented_laminar"]) / float(fila["dynamic_augmented"])
    assert 0.070 < perdida < 0.074
