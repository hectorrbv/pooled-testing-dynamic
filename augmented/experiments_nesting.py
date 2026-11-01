"""¿El greedy exacto alguna vez vuelve a probar DENTRO de un pool observado?

Es la pregunta que decide si el laminar dinámico se distingue del estático.
Si el greedy nunca anida --- si siempre elige territorio virgen --- entonces
la jerarquía no se usa nunca y el laminar dinámico degenera en un diseño
estático, como advirtió Francisco en la sesión del 27 de julio.

Método exacto, sin Monte Carlo: se camina el árbol de decisiones del greedy
(una acción por nodo, ramificación solo por resultados) ponderando cada
decisión por la probabilidad de su rama.

El greedy usa el conjunto de acciones COMPLETO (todos los pools de tamaño
<= G), no una biblioteca laminar. La pregunta es si, siendo libre, *elige*
anidar.

Clasificación exhaustiva de cada decisión respecto a lo ya probado:

- ``virgen``  : disjunta de todo pool probado.
- ``anidada`` : contenida en algún pool probado. Ése necesariamente salió con
  conteo positivo, porque los pools con conteo 0 acreditan a sus miembros y
  volver a probarlos da ganancia 0.
- ``mixta``   : toca territorio probado sin caber dentro de un solo pool.

Corre con::

    python -m augmented.experiments_nesting
"""

import csv
import json
from pathlib import Path

import numpy as np

from augmented.laminar_benchmarks import ExactPolicyEvaluator


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "laminar_week"

CATEGORIES = ("virgen", "anidada", "mixta")


def _classify(pool, tested):
    """Clasifica un pool respecto al historial de pools ya probados."""

    if not tested:
        return "virgen"
    if any(pool & probado and pool | probado == probado for probado in tested):
        return "anidada"
    if any(pool & probado for probado in tested):
        return "mixta"
    return "virgen"


def greedy_nesting_stats(p, u, B, G):
    """Reparto de las decisiones del greedy exacto entre las tres categorías.

    Devuelve un dict con la masa de probabilidad que cae en cada categoría,
    normalizada sobre el total de decisiones tomadas, más el welfare del
    greedy y la masa de decisiones anidadas que ocurren tras un conteo bajo.
    """

    evaluator = ExactPolicyEvaluator(p, u, B, G)
    reparto = dict.fromkeys(CATEGORIES, 0.0)
    total = 0.0
    anidadas_tras_conteo_bajo = 0.0
    valor = 0.0
    decisiones_libres = [0.0]   # había virgen disponible y algo que ganar
    anidadas_libres = [0.0]     # y aun así se prefirió estrictamente anidar

    def walk(step, worlds, cleared, tested, probability):
        nonlocal total, anidadas_tras_conteo_bajo, valor
        if step == B or probability <= 0.0:
            return

        mejor_pool, mejor_puntaje, mejores_ramas = None, -1.0, ()
        mejor_virgen = 0.0          # el mejor puntaje alcanzable sin tocar
        habia_virgen = False        # territorio ya probado
        for pool in evaluator.pools:
            ramas = evaluator.branches(worlds, cleared, pool)
            inmediato = sum(masa * premio for masa, _, _, premio in ramas)
            if _classify(pool, tested) == "virgen":
                habia_virgen = True
                mejor_virgen = max(mejor_virgen, inmediato)
            if inmediato > mejor_puntaje + 1e-15:
                mejor_pool, mejor_puntaje, mejores_ramas = pool, inmediato, ramas

        if mejor_pool is None:
            return

        categoria = _classify(mejor_pool, tested)
        reparto[categoria] += probability
        total += probability
        valor += probability * mejor_puntaje

        # La pregunta limpia: ¿eligió anidar TENIENDO alternativa virgen, y
        # ganando algo con ello?  Si no había virgen, anidar es forzado; si
        # todos los puntajes son cero, la elección es un empate arbitrario.
        if habia_virgen and mejor_puntaje > 1e-12:
            decisiones_libres[0] += probability
            if categoria != "virgen" and mejor_puntaje > mejor_virgen + 1e-12:
                anidadas_libres[0] += probability

        if categoria == "anidada":
            # ¿El padre había salido con un conteo bajo respecto a lo esperado?
            # Es el mecanismo que predice la nota de literatura: un conteo
            # inusualmente bajo deja un bloque posterior muy atractivo.
            padre = next(
                probado for probado in tested
                if mejor_pool | probado == probado
            )
            miembros = [i for i in range(evaluator.n) if padre & (1 << i)]
            esperado = float(np.asarray(p)[miembros].sum())
            observado = _observed_count(evaluator, worlds, padre)
            if observado is not None and observado < esperado:
                anidadas_tras_conteo_bajo += probability

        nuevo_historial = tested + (mejor_pool,)
        for masa, hijo, nuevo_cleared, _ in mejores_ramas:
            walk(step + 1, hijo, nuevo_cleared, nuevo_historial,
                 probability * masa)

    walk(0, evaluator.all_worlds, 0, (), 1.0)

    if total <= 0.0:
        return {c: 0.0 for c in CATEGORIES} | {
            "decisiones": 0.0, "anidada_tras_conteo_bajo": 0.0,
            "decisiones_libres": 0.0, "anida_pudiendo_no": 0.0,
        }
    libres = decisiones_libres[0]
    return {c: reparto[c] / total for c in CATEGORIES} | {
        "decisiones": total,
        "anidada_tras_conteo_bajo": anidadas_tras_conteo_bajo / total,
        "decisiones_libres": libres,
        # La métrica que responde la pregunta sin confusiones: de las
        # decisiones donde SÍ había virgen y SÍ había algo que ganar, ¿en qué
        # fracción se prefirió estrictamente salirse a territorio ya probado?
        "anida_pudiendo_no": (
            anidadas_libres[0] / libres if libres > 0 else 0.0
        ),
    }


def _observed_count(evaluator, worlds, pool):
    """El conteo de ``pool`` si es el mismo en todos los mundos vivos."""

    miembros = [i for i in range(evaluator.n) if pool & (1 << i)]
    conteos = set()
    rest = worlds
    while rest:
        bit = rest & -rest
        mundo = bit.bit_length() - 1
        conteos.add(int(evaluator.scenarios[mundo, miembros].sum()))
        if len(conteos) > 1:
            return None
        rest &= rest - 1
    return conteos.pop() if conteos else None


def _calibrate_mean(raw, target):
    raw = np.clip(np.asarray(raw, dtype=float), 1e-7, 1.0 - 1e-7)
    logits = np.log(raw / (1.0 - raw))
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if (1.0 / (1.0 + np.exp(-(logits + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 1.0 / (1.0 + np.exp(-(logits + (lo + hi) / 2.0)))


def run(reps=4):
    """Barrido sobre prevalencia y régimen de tasas."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filas = []
    for base_p in np.round(np.arange(0.1, 0.91, 0.1), 2):
        for rate_mode in ("homogeneous", "beta_bimodal"):
            for n, B, G in ((5, 3, 3), (6, 3, 3)):
                for rep in range(reps):
                    rng = np.random.default_rng(
                        770000 + int(base_p * 100) * 100 + n * 10 + rep
                        + (5 if rate_mode == "beta_bimodal" else 0)
                    )
                    if rate_mode == "homogeneous":
                        p = np.full(n, float(base_p))
                    else:
                        p = _calibrate_mean(rng.beta(0.4, 0.4, n), float(base_p))
                    u = np.ones(n)
                    stats = greedy_nesting_stats(p, u, B, G)
                    filas.append({
                        "base_p": float(base_p), "n": n, "B": B, "G": G,
                        "rate_mode": rate_mode, "replicate": rep,
                        **{k: float(v) for k, v in stats.items()},
                    })

    ruta = DATA_DIR / "nesting.csv"
    with ruta.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filas[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)
    return filas


def main():
    filas = run()
    import pandas as pd

    frame = pd.DataFrame(filas)
    print("=" * 70)
    print("¿EL GREEDY EXACTO ELIGE ANIDAR?")
    print("=" * 70)
    print("\nMasa de decisiones por categoría (global):\n")
    for categoria in CATEGORIES:
        print(f"  {categoria:>10}: {100 * frame[categoria].mean():6.2f}%")

    print("\n\nPor prevalencia (masa anidada):\n")
    print(f"  {'p':>6}{'homogénea':>13}{'dispersa':>12}")
    for base_p, grupo in frame.groupby("base_p"):
        hom = grupo[grupo.rate_mode == "homogeneous"].anidada.mean()
        dis = grupo[grupo.rate_mode == "beta_bimodal"].anidada.mean()
        print(f"  {base_p:>6.1f}{100 * hom:>12.1f}%{100 * dis:>11.1f}%")

    anida = (frame.anidada > 1e-9).mean()
    print(f"\n\nInstancias donde el greedy anida al menos una vez: "
          f"{100 * anida:.1f}%")
    print(f"Masa anidada que sigue a un conteo BAJO respecto al esperado: "
          f"{100 * frame.anidada_tras_conteo_bajo.mean():.2f}%")
    print(f"\nArtefacto: {DATA_DIR / 'nesting.csv'}")


if __name__ == "__main__":
    main()
