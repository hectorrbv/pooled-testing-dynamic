"""Procedencia y reproducibilidad de artefactos experimentales (B-M4, plan maestro §22).

El plan exige que cada artefacto experimental registre generador, semilla,
versiones, parametros y commit, y que exista un CSV canonico por experimento y no
solo notebooks. Este modulo es la unica via soportada para escribirlos.

La separacion de responsabilidades es deliberada: el CSV contiene solo datos y es
byte-identico entre corridas con la misma semilla, de modo que un diff vacio es
evidencia de reproducibilidad. Todo lo que varia entre corridas --- la marca de
tiempo, el commit, las versiones de biblioteca --- vive en un archivo hermano
`<nombre>.meta.json`. Comparar resultados es diff sobre el CSV; auditar de donde
salieron es leer el sidecar.

Uso tipico:

    from augmented.provenance import seeded_rng, write_canonical_csv

    rng = seeded_rng(270726)
    rows = [...]
    write_canonical_csv(
        "results/mi_experimento.csv", rows,
        generator="augmented.experiments_mio.barrido", seed=270726,
        params={"n": 10, "B": 3, "q": 0.2},
    )
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Version del formato del sidecar. Subir SOLO si cambia la forma del dict, para
# que un lector viejo pueda detectar que no entiende un archivo nuevo.
STAMP_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit(repo_root: Path | str | None = None) -> dict:
    """Commit actual y si el arbol tiene cambios sin commitear.

    Devuelve ``{"commit": None, "dirty": None}`` fuera de un repo git en vez de
    fallar: un artefacto generado desde un tarball sigue siendo valido, solo que
    su procedencia es menos precisa y el sidecar lo dice.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    def _git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=10,
        )
    try:
        rev = _git("rev-parse", "HEAD")
        if rev.returncode != 0:
            return {"commit": None, "dirty": None}
        status = _git("status", "--porcelain")
        return {
            "commit": rev.stdout.strip(),
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def env_versions() -> dict:
    """Versiones de las dependencias base, que son las que mueven los numeros."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def seeded_rng(seed: int) -> np.random.Generator:
    """Generador sembrado. Unica forma soportada de obtener aleatoriedad.

    Existe para que la semilla sea siempre un argumento explicito y nunca el
    estado global de ``np.random``, que no se puede registrar en un sidecar.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"la semilla debe ser un entero, no {type(seed).__name__}")
    return np.random.default_rng(int(seed))


def run_stamp(generator: str, seed: int | None, params: dict | None = None) -> dict:
    """Sello de procedencia de una corrida."""
    if not generator:
        raise ValueError("generator es obligatorio: identifica quien produjo el artefacto")
    return {
        "stamp_version": STAMP_VERSION,
        "generator": generator,
        "seed": None if seed is None else int(seed),
        "params": dict(params or {}),
        "versions": env_versions(),
        "git": git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def sidecar_path(csv_path: Path | str) -> Path:
    """Ruta del sidecar de procedencia que acompania a un CSV."""
    p = Path(csv_path)
    return p.with_suffix(p.suffix + ".meta.json")


def write_canonical_csv(
    path: Path | str,
    rows,
    *,
    generator: str,
    seed: int | None,
    params: dict | None = None,
    fieldnames=None,
) -> Path:
    """Escribe un CSV canonico mas su sidecar de procedencia.

    El CSV se escribe con ``newline=""`` y salto de linea Unix explicito para que
    dos corridas con la misma semilla produzcan bytes identicos en cualquier
    sistema operativo. Las filas deben ser dicts con las mismas claves; el orden
    de columnas se toma de ``fieldnames`` o de la primera fila.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no se escriben artefactos vacios: revisa el generador")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    missing = {k for r in rows for k in fieldnames if k not in r}
    if missing:
        raise ValueError(f"filas sin las columnas declaradas: {sorted(missing)}")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    stamp = run_stamp(generator, seed, params)
    stamp["rows"] = len(rows)
    stamp["columns"] = list(fieldnames)
    stamp["artifact"] = csv_path.name
    sidecar_path(csv_path).write_text(
        json.dumps(stamp, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path


def read_stamp(csv_path: Path | str) -> dict:
    """Lee el sidecar de procedencia de un CSV. Falla si no existe."""
    side = sidecar_path(csv_path)
    if not side.exists():
        raise FileNotFoundError(
            f"{side.name} no existe: el artefacto se genero sin procedencia "
            f"(§22 exige write_canonical_csv)"
        )
    return json.loads(side.read_text(encoding="utf-8"))
