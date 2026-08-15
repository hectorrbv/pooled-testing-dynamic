"""Laminar por refinamiento vs irrestricto, y politica optima. n=10, q=0.2."""
from math import comb
from functools import lru_cache
from solver_n10 import dynamic, V, bank_prune, outcomes, MEMO, Q, N
from itertools import product

# ---- laminar: estado = (multiset de grupos vivos (s,R) con 0<R<s, m sin tocar) ----
@lru_cache(maxsize=None)
def L(groups, m, b):
    """valor esperado adicional. groups: tupla ordenada de (s,R) vivos."""
    if b == 0:
        return 0.0
    best = 0.0
    # (a) pool fresco de tamano k
    for k in range(1, m + 1):
        val = 0.0
        for R in range(k + 1):
            p = comb(k, R) * Q**R * (1 - Q)**(k - R)
            if p == 0:
                continue
            if R == k:
                val += p * (k + L(groups, m - k, b - 1))
            elif R == 0:
                val += p * L(groups, m - k, b - 1)
            else:
                ng = tuple(sorted(groups + ((k, R),)))
                val += p * L(ng, m - k, b - 1)
        best = max(best, val)
    # (b) subconjunto de tamano k dentro de un grupo vivo
    for i, (s, R) in enumerate(set(groups)):
        rest = list(groups)
        rest.remove((s, R))
        rest = tuple(rest)
        for k in range(1, s):
            val = 0.0
            for r in range(max(0, R - (s - k)), min(k, R) + 1):
                p = comb(k, r) * comb(s - k, R - r) / comb(s, R)
                if p == 0:
                    continue
                gain = 0
                ng = list(rest)
                for (ss, rr) in ((k, r), (s - k, R - r)):
                    if rr == ss:
                        gain += ss
                    elif rr == 0:
                        pass
                    else:
                        ng.append((ss, rr))
                val += p * (gain + L(tuple(sorted(ng)), m, b - 1))
            best = max(best, val)
    return best

# ---- politica optima irrestricta: primera accion y su continuacion ----
def best_action(atoms, dist, b, augmented):
    best, arg = -1.0, None
    for t in product(*[range(a + 1) for a in atoms]):
        if sum(t) == 0:
            continue
        val = 0.0
        for p, g, a2, d2 in outcomes(atoms, dist, t, augmented):
            val += p * (g + V(a2, d2, b - 1, augmented))
        if val > best:
            best, arg = val, t
    return best, arg

def trace(b, augmented, depth=2):
    dist = tuple(((k,), comb(N, k) * Q**k * (1 - Q)**(N - k)) for k in range(N + 1))
    g, atoms, d = bank_prune((N,), dist)
    lines = []
    val, t = best_action(atoms, d, b, augmented)
    lines.append(f"  paso 1: atomos={atoms} -> toma {t}  (valor {val:.4f})")
    for p, gg, a2, d2 in sorted(outcomes(atoms, d, t, augmented), key=lambda z: -z[0])[:3]:
        if b > 1 and a2:
            v2, t2 = best_action(a2, d2, b - 1, augmented)
            lines.append(f"    outcome p={p:.3f} banca {gg} -> atomos={a2} toma {t2}")
        else:
            lines.append(f"    outcome p={p:.3f} banca {gg} -> atomos={a2} (fin)")
    return "\n".join(lines)

if __name__ == "__main__":
    print(f"n={N} q={Q}")
    print(f"{'B':>2} {'din.aum irrestricto':>20} {'din.aum laminar':>17} {'din.bin irrestricto':>20}")
    for B in range(1, 4):
        di = dynamic(N, B, True)
        dl = L((), N, B)
        db = dynamic(N, B, False)
        print(f"{B:>2} {di:>20.5f} {dl:>17.5f} {db:>20.5f}")
    print("\nlaminar aumentado con presupuestos mayores (irrestricto no es tratable):")
    for B in range(4, 9):
        print(f"{B:>2} {'':>20} {L((), N, B):>17.5f}    (B*q = {B*Q:.2f})")
    print("\npolitica optima aumentada B=3:")
    print(trace(3, True))
    print("\npolitica optima binaria B=3:")
    print(trace(3, False))
