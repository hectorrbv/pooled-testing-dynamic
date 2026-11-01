"""Tests de procedencia y determinismo de artefactos (B-M4, plan maestro §22)."""

import json

import numpy as np
import pytest

from augmented.provenance import (
    STAMP_VERSION,
    env_versions,
    git_commit,
    read_stamp,
    run_stamp,
    seeded_rng,
    sidecar_path,
    write_canonical_csv,
)


def _rows(rng, k=5):
    return [{"i": i, "x": float(rng.random()), "etiqueta": f"fila-{i}"} for i in range(k)]


# ---------------------------------------------------------------- semillas

def test_seeded_rng_es_determinista():
    a = seeded_rng(270726).random(10)
    b = seeded_rng(270726).random(10)
    np.testing.assert_array_equal(a, b)


def test_seeded_rng_distingue_semillas():
    a = seeded_rng(1).random(10)
    b = seeded_rng(2).random(10)
    assert not np.array_equal(a, b)


def test_seeded_rng_rechaza_no_enteros():
    for malo in (0.5, "270726", None, True):
        with pytest.raises(TypeError):
            seeded_rng(malo)


# ---------------------------------------------------------------- el sello

def test_run_stamp_tiene_los_cinco_campos_que_exige_el_plan():
    stamp = run_stamp("modulo.funcion", 42, {"n": 10})
    assert stamp["generator"] == "modulo.funcion"
    assert stamp["seed"] == 42
    assert stamp["params"] == {"n": 10}
    assert stamp["versions"]["numpy"] is not None
    assert "commit" in stamp["git"]
    assert stamp["stamp_version"] == STAMP_VERSION


def test_run_stamp_exige_generador():
    with pytest.raises(ValueError):
        run_stamp("", 42)


def test_run_stamp_acepta_seed_nulo_para_experimentos_deterministas():
    assert run_stamp("modulo.exacto", None)["seed"] is None


def test_env_versions_reporta_python_y_base():
    v = env_versions()
    assert v["python"].count(".") >= 1
    assert set(v) >= {"python", "platform", "numpy", "scipy", "pandas"}


def test_git_commit_no_revienta_fuera_de_repo(tmp_path):
    g = git_commit(tmp_path)
    assert set(g) == {"commit", "dirty"}


def test_git_commit_encuentra_el_repo():
    g = git_commit()
    assert g["commit"] is None or len(g["commit"]) == 40


# ---------------------------------------------------------------- artefactos

def test_csv_canonico_es_byte_identico_con_la_misma_semilla(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_canonical_csv(a, _rows(seeded_rng(7)), generator="t.gen", seed=7)
    write_canonical_csv(b, _rows(seeded_rng(7)), generator="t.gen", seed=7)
    assert a.read_bytes() == b.read_bytes()


def test_csv_canonico_usa_saltos_unix(tmp_path):
    p = tmp_path / "x.csv"
    write_canonical_csv(p, _rows(seeded_rng(1)), generator="t.gen", seed=1)
    assert b"\r\n" not in p.read_bytes()


def test_sidecar_se_escribe_junto_al_csv(tmp_path):
    p = tmp_path / "x.csv"
    write_canonical_csv(p, _rows(seeded_rng(1)), generator="t.gen", seed=1,
                        params={"q": 0.2})
    side = sidecar_path(p)
    assert side.name == "x.csv.meta.json"
    stamp = json.loads(side.read_text())
    assert stamp["rows"] == 5
    assert stamp["columns"] == ["i", "x", "etiqueta"]
    assert stamp["params"] == {"q": 0.2}
    assert stamp["artifact"] == "x.csv"


def test_read_stamp_recupera_lo_escrito(tmp_path):
    p = tmp_path / "x.csv"
    write_canonical_csv(p, _rows(seeded_rng(3)), generator="t.gen", seed=3)
    assert read_stamp(p)["seed"] == 3


def test_read_stamp_falla_si_el_artefacto_no_tiene_procedencia(tmp_path):
    p = tmp_path / "suelto.csv"
    p.write_text("i,x\n1,2\n")
    with pytest.raises(FileNotFoundError):
        read_stamp(p)


def test_no_se_escriben_artefactos_vacios(tmp_path):
    with pytest.raises(ValueError):
        write_canonical_csv(tmp_path / "x.csv", [], generator="t.gen", seed=1)


def test_filas_incompletas_se_rechazan(tmp_path):
    filas = [{"a": 1, "b": 2}, {"a": 3}]
    with pytest.raises(ValueError):
        write_canonical_csv(tmp_path / "x.csv", filas, generator="t.gen", seed=1)


def test_orden_de_columnas_explicito_se_respeta(tmp_path):
    p = tmp_path / "x.csv"
    write_canonical_csv(p, [{"a": 1, "b": 2}], generator="t.gen", seed=1,
                        fieldnames=["b", "a"])
    assert p.read_text().splitlines()[0] == "b,a"


def test_crea_directorios_intermedios(tmp_path):
    p = tmp_path / "hondo" / "mas" / "x.csv"
    write_canonical_csv(p, [{"a": 1}], generator="t.gen", seed=1)
    assert p.exists() and sidecar_path(p).exists()
