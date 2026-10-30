"""Independent checks requested in the October 14 weekly plan.

Run from the repository root: python -m augmented.verifica_encargos_oct14
Checks saved exact results against a different solver; records every comparison.
This is a computational audit, not a human sign-off on the research claims.
"""
import ast
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

from augmented.bellman_binario_exacto import BellmanBinario, replay_binary_tree
from augmented.bellman_perfiles import BellmanPerfiles
from augmented.dp_perfiles_binario_conteo import DP
from augmented.laminar_benchmarks import static_binary_value

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/encargos_2026-10-06"
OUTPUT = ROOT / "results/sesion_2026-10-14/validacion_cruzada.json"


def rows(pattern):
    result = []
    for path in sorted(INPUT.glob(pattern)):
        with path.open() as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                result.append((str(path.relative_to(ROOT)), line, row))
    return result


def main():
    start = perf_counter()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": 20261014, "grid": [], "binary": [], "one_test": []}
    for source, line, row in rows("precio_laminar_malla_*.csv"):
        n, B, G = (int(row[k]) for k in ("n", "B", "G"))
        q = F(row["q"])
        actual = BellmanPerfiles([1-q]*n, [1]*n, G, laminar=False).solve(B).value
        report["grid"].append(dict(source=source, line=line, n=n, B=B, G=G,
            q=str(q), saved=row["unrestricted"], independent=str(actual),
            match=actual == F(row["unrestricted"])))
    print("grid", len(report["grid"]), "checked", flush=True)

    sample = random.Random(report["seed"]).sample(rows("binario_vs_laminar_seed*.csv"), 20)
    for source, line, row in sample:
        q, u = ([F(x) for x in ast.literal_eval(row[k])] for k in ("q", "u"))
        B, G = int(row["B"]), int(row["G"])
        solver = BellmanBinario([1-x for x in q], u, G)
        value = solver.solve(B).value
        replay = replay_binary_tree(solver.tree(B), q, u, B, G)
        report["binary"].append(dict(source=source, line=line, q=list(map(str,q)),
            u=list(map(str,u)), B=B, G=G, saved=row["binary_dyn"],
            independent=str(value), profile_replay=replay,
            match=value == F(row["binary_dyn"]) == F(replay["value"])))

        # With B=1 the dynamic and static optima must coincide.
        fresh = DP(q, u, G, "binary").value(1)
        direct = BellmanBinario([1-x for x in q], u, G).solve(1).value
        static = static_binary_value([float(1-x) for x in q], list(map(float,u)), 1, G)
        error = abs(float(fresh)-static)
        report["one_test"].append(dict(source=source, line=line, dynamic=str(fresh),
            independent=str(direct), static=static, absolute_error=error,
            match=fresh == direct and error <= 1e-10))
        # The imported reference keeps instance-bound caches at class scope.
        DP.V.cache_clear()
        DP.P.cache_clear()
    print("binary and one-test controls", len(sample), "checked each", flush=True)
    assert len(report["grid"]) == 81
    report["counts"] = {k: len(report[k]) for k in ("grid", "binary", "one_test")}
    report["mismatches"] = sum(not x["match"] for k in report["counts"] for x in report[k])
    report["seconds"] = perf_counter()-start
    sources = ["augmented/verifica_encargos_oct14.py", "augmented/bellman_binario_exacto.py",
        "augmented/bellman_perfiles.py", "augmented/dp_perfiles_binario_conteo.py",
        "augmented/laminar_benchmarks.py"]
    sources += sorted({x["source"] for k in report["counts"] for x in report[k]})
    report["sha256"] = {p: sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("counts", "mismatches", "seconds")}), flush=True)
    if report["mismatches"]:
        raise SystemExit("Cross-check failed; inspect the saved mismatches")


if __name__ == "__main__":
    main()
