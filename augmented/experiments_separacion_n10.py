"""Separacion exacta estatico/dinamico x binario/aumentado en n=10 (B-M11).

Enumera el espacio de creencias completo --- sin simulacion, sin muestreo --- para
las cuatro variantes del problema mas la clase laminar, y emite un CSV canonico
con procedencia (plan maestro §22).

Modelo. n personas iid, cada una sana con probabilidad q. Utilidad de una politica
= numero esperado de personas certificadas sanas con certeza, bajo la regla
permisiva de certificacion: una persona cuenta si el sistema de restricciones la
determina sana, no solo si cae en un pool observado enteramente sano. Esa eleccion
importa y esta documentada en `cuentas-n10-q02.md`.

Prueba binaria: reporta si el pool tiene al menos un sano. Prueba aumentada:
reporta cuantos. Estatico: los pools se fijan de antemano. Dinamico: cada pool
puede depender de lo observado. Laminar: dos pools cualesquiera estan anidados o
son disjuntos, nunca se cruzan.

Corre con:  python3 -m augmented.experiments_separacion_n10
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product
from math import comb
from pathlib import Path

from augmented.provenance import write_canonical_csv

# El presupuesto 4 sin restringir no es tratable por enumeracion: el arbol de
# acciones crece como el producto de (tamano de atomo + 1) sobre atomos, y con
# cuatro pruebas no cierra en tiempo razonable. La clase laminar si llega lejos.
BMAX_IRRESTRICTO = 3
BMAX_LAMINAR = 8


# --------------------------------------------------------------- estado por atomos
# atoms: tupla de tamanos de celdas vivas del diagrama de Venn de los pools ya
# probados.  dist: tupla ordenada de (countvec, prob), con countvec[i] = numero de
# sanos en el atomo i.  Los atomos totalmente determinados se retiran del estado y
# su utilidad se banca, que es lo que mantiene el espacio de estados manejable.

def bank_prune(atoms, dist):
    """Banca atomos determinados; devuelve (ganancia, atoms', dist')."""
    gain = 0
    keep = []
    for i in range(len(atoms)):
        vals = {cv[i] for cv, _ in dist}
        if vals == {atoms[i]}:
            gain += atoms[i]
        elif vals != {0}:
            keep.append(i)
    agg = {}
    for cv, p in dist:
        k = tuple(cv[i] for i in keep)
        agg[k] = agg.get(k, 0.0) + p
    return gain, tuple(atoms[i] for i in keep), tuple(sorted(agg.items()))


def canon(atoms, dist):
    """Forma canonica bajo permutacion de atomos, para memoizar."""
    m = len(atoms)
    if m == 0:
        return ((), ())
    if m <= 6:
        best = None
        for perm in permutations(range(m)):
            cand = (
                tuple(atoms[i] for i in perm),
                tuple(sorted((tuple(cv[i] for i in perm), round(p, 12)) for cv, p in dist)),
            )
            if best is None or cand < best:
                best = cand
        return best
    order = sorted(range(m), key=lambda i: (atoms[i], sum(cv[i] * p for cv, p in dist)))
    return (
        tuple(atoms[i] for i in order),
        tuple(sorted((tuple(cv[i] for i in order), round(p, 12)) for cv, p in dist)),
    )


def outcomes(atoms, dist, t, augmented, q):
    """Aplica el pool que toma t[i] del atomo i. Devuelve (prob, gain, atoms', dist')."""
    buckets = {}
    m = len(atoms)
    for cv, p in dist:
        ranges = [range(max(0, cv[i] - (atoms[i] - t[i])), min(t[i], cv[i]) + 1) for i in range(m)]
        for ys in product(*ranges):
            w = p
            for i in range(m):
                w *= comb(t[i], ys[i]) * comb(atoms[i] - t[i], cv[i] - ys[i]) / comb(atoms[i], cv[i])
            if w == 0.0:
                continue
            R = sum(ys)
            key = R if augmented else (1 if R >= 1 else 0)
            nv = []
            for i in range(m):
                if t[i] > 0:
                    nv.append(ys[i])
                if atoms[i] - t[i] > 0:
                    nv.append(cv[i] - ys[i])
            buckets.setdefault(key, {})
            nv = tuple(nv)
            buckets[key][nv] = buckets[key].get(nv, 0.0) + w
    natoms = []
    for i in range(m):
        if t[i] > 0:
            natoms.append(t[i])
        if atoms[i] - t[i] > 0:
            natoms.append(atoms[i] - t[i])
    natoms = tuple(natoms)
    out = []
    for d in buckets.values():
        tot = sum(d.values())
        g, a2, d2 = bank_prune(natoms, tuple(sorted((k, v / tot) for k, v in d.items())))
        out.append((tot, g, a2, d2))
    return out


_MEMO: dict = {}


def _V(atoms, dist, b, augmented, q):
    if b == 0 or not atoms:
        return 0.0
    key = (canon(atoms, dist), b, augmented, q)
    if key in _MEMO:
        return _MEMO[key]
    best = 0.0
    for t in product(*[range(a + 1) for a in atoms]):
        if sum(t) == 0:
            continue
        val = sum(p * (g + _V(a2, d2, b - 1, augmented, q))
                  for p, g, a2, d2 in outcomes(atoms, dist, t, augmented, q))
        best = max(best, val)
    _MEMO[key] = best
    return best


def dynamic_value(n, b, augmented, q):
    """Optimo dinamico irrestricto (puede cruzar pools)."""
    dist = tuple(((k,), comb(n, k) * q ** k * (1 - q) ** (n - k)) for k in range(n + 1))
    g, atoms, d = bank_prune((n,), dist)
    return g + _V(atoms, d, b, augmented, q)


# --------------------------------------------------------------- laminar aumentado
# Bajo laminaridad con pruebas aumentadas, todo grupo probado termina con su conteo
# conocido EXACTO.  Por eso agregar un grupo entero a un pool nuevo solo suma una
# constante conocida y no informa: basta considerar pools frescos y subconjuntos de
# un grupo vivo.  Eso hace que el estado se factorice en urnas independientes.

@lru_cache(maxsize=None)
def _L(groups, m, b, q):
    if b == 0:
        return 0.0
    best = 0.0
    for k in range(1, m + 1):                      # pool fresco de tamano k
        val = 0.0
        for R in range(k + 1):
            p = comb(k, R) * q ** R * (1 - q) ** (k - R)
            if p == 0:
                continue
            if R == k:
                val += p * (k + _L(groups, m - k, b - 1, q))
            elif R == 0:
                val += p * _L(groups, m - k, b - 1, q)
            else:
                val += p * _L(tuple(sorted(groups + ((k, R),))), m - k, b - 1, q)
        best = max(best, val)
    for (s, R) in set(groups):                     # subconjunto de un grupo vivo
        rest = list(groups)
        rest.remove((s, R))
        for k in range(1, s):
            val = 0.0
            for r in range(max(0, R - (s - k)), min(k, R) + 1):
                p = comb(k, r) * comb(s - k, R - r) / comb(s, R)
                if p == 0:
                    continue
                gain, ng = 0, list(rest)
                for (ss, rr) in ((k, r), (s - k, R - r)):
                    if rr == ss:
                        gain += ss
                    elif rr != 0:
                        ng.append((ss, rr))
                val += p * (gain + _L(tuple(sorted(ng)), m, b - 1, q))
            best = max(best, val)
    return best


def laminar_value(n, b, q):
    """Optimo dinamico aumentado restringido a la clase laminar."""
    return _L((), n, b, q)


# --------------------------------------------------------------- estatico
def _static_value(counts, B, augmented, q):
    sigs = [(s, c) for s, c in enumerate(counts) if c > 0]
    groups = {}
    for xs in product(*[range(c + 1) for _, c in sigs]):
        p = 1.0
        for (_, c), x in zip(sigs, xs):
            p *= comb(c, x) * q ** x * (1 - q) ** (c - x)
        obs = []
        for j in range(B):
            R = sum(x for (s, _), x in zip(sigs, xs) if (s >> j) & 1)
            obs.append(R if augmented else (1 if R >= 1 else 0))
        groups.setdefault(tuple(obs), []).append((p, xs))
    total = 0.0
    for lst in groups.values():
        pg = sum(p for p, _ in lst)
        u = sum(c for idx, (s, c) in enumerate(sigs)
                if s != 0 and all(xs[idx] == c for _, xs in lst))
        total += pg * u
    return total


def _compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def static_value(n, B, augmented, q):
    """Optimo estatico por enumeracion de TODOS los disenios de B pools sobre n."""
    return max(_static_value(c, B, augmented, q) for c in _compositions(n, 2 ** B))


# --------------------------------------------------------------- artefacto
def build_rows(n=10, q=0.2):
    rows = []
    for B in range(1, BMAX_IRRESTRICTO + 1):
        rows.append({
            "n": n, "q": q, "B": B,
            "individual_baseline": round(B * q, 10),
            "static_binary": round(static_value(n, B, False, q), 10),
            "static_augmented": round(static_value(n, B, True, q), 10),
            "dynamic_binary": round(dynamic_value(n, B, False, q), 10),
            "dynamic_augmented_laminar": round(laminar_value(n, B, q), 10),
            "dynamic_augmented": round(dynamic_value(n, B, True, q), 10),
            "tractable": "exact",
        })
    for B in range(BMAX_IRRESTRICTO + 1, BMAX_LAMINAR + 1):
        rows.append({
            "n": n, "q": q, "B": B,
            "individual_baseline": round(B * q, 10),
            "static_binary": "", "static_augmented": "", "dynamic_binary": "",
            "dynamic_augmented_laminar": round(laminar_value(n, B, q), 10),
            "dynamic_augmented": "",
            "tractable": "laminar-only",
        })
    return rows


def main():
    n, q = 10, 0.2
    rows = build_rows(n, q)
    out = Path(__file__).resolve().parent.parent / "results" / "separacion_n10_q02.csv"
    write_canonical_csv(
        out, rows,
        generator="augmented.experiments_separacion_n10.main",
        seed=None,  # enumeracion exacta: no hay aleatoriedad que sembrar
        params={"n": n, "q": q, "certification_rule": "permissive",
                "bmax_unrestricted": BMAX_IRRESTRICTO, "bmax_laminar": BMAX_LAMINAR},
    )
    for r in rows:
        print(f"B={r['B']}  est.bin={r['static_binary']}  din.bin={r['dynamic_binary']}  "
              f"est.aum={r['static_augmented']}  laminar={r['dynamic_augmented_laminar']}  "
              f"din.aum={r['dynamic_augmented']}")
    print(f"\nartefacto: {out}")


if __name__ == "__main__":
    main()
