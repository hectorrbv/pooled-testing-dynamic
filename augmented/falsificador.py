"""Falsificador de comportamiento de politicas (B-M8, gate G6, plan maestro §17).

El falsificador no pregunta cuanto vale una politica sino **que hace**: clasifica
cada decision que toma, en cada estado alcanzable, ponderada por la probabilidad
de llegar ahi. Sirve para dos cosas. Una, describir: donde anida el greedy, cuando
cruza el optimo, que fraccion del welfare vive en acciones que la clase laminar
prohibe. Dos, falsificar: si una candidata de scorer pierde valor justo en las
acciones multiatomo, se habra identificado exactamente que complementariedad le
falta --- resultado, no fracaso.

**Cruce, formal (§17).** ``t`` cruza ``T_j`` si y solo si

    t ∩ T_j ∉ {∅, t, T_j}

es decir se tocan, ninguno contiene al otro, y ninguno es vacio. Quedan fuera del
cruce las disjuntas, las descendientes, las ancestras y las repeticiones. Una
accion es cruzada respecto de la historia H si cruza algun pool ya ejecutado. Esta
es la definicion que decide si una trayectoria sale o no de la clase laminar, y por
eso se implementa una sola vez y se testea sola.

**La curva se reporta tal como salga.** El plan lo dice explicitamente y aqui se
respeta: este modulo mide y escribe, no ajusta.

Corre con:  python3 -m augmented.falsificador
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from augmented.laminar_benchmarks import ExactPolicyEvaluator
from augmented.provenance import write_canonical_csv
from augmented.rollout_oracle import GreedyPolicy, RolloutPolicy
from augmented.scorers import get_scorer, select_action

# Precedencia de clases, en el orden declarado por §17. La primera que aplica es
# la clase primaria; el resto de la informacion viaja en los flags.
CLASES = ("repetida", "descendiente", "ancestro", "union_de_atomos",
          "compatible_mixta", "virgen", "cruzada", "dominada")


# ---------------------------------------------------------------- cruce y clases

def cruza(t: int, other: int) -> bool:
    """t cruza `other`: se tocan, y ninguno contiene al otro (§17)."""
    inter = t & other
    return inter != 0 and inter != t and inter != other


def atomos(executed) -> list:
    """Celdas del diagrama de Venn de los pools ejecutados (territorio tocado)."""
    celdas = []
    for pool, _ in executed:
        nuevas = []
        resto = pool
        for celda in celdas:
            comun = celda & pool
            if comun:
                nuevas.append(comun)
                if celda ^ comun:
                    nuevas.append(celda ^ comun)
                resto &= ~comun
            else:
                nuevas.append(celda)
        if resto:
            nuevas.append(resto)
        celdas = [c for c in nuevas if c]
    return celdas


def clasificar(t: int, executed, cleared: int, n: int) -> tuple[str, dict]:
    """Clase primaria y flags de una accion respecto de la historia ejecutada."""
    pools = [pool for pool, _ in executed]
    tocado = 0
    for pool in pools:
        tocado |= pool
    virgen_mask = ((1 << n) - 1) & ~tocado
    celdas = atomos(executed)

    flags = {
        "dominada": bool(t and not (t & ~cleared)),          # todo ya acreditado: paga 0
        "cruzada": any(cruza(t, pool) for pool in pools),
        "intraatomo": any(t and not (t & ~celda) for celda in celdas),
        "multiatomo": sum(1 for celda in celdas if t & celda) >= 2,
        "atomo_virgen": bool(t & tocado) and bool(t & virgen_mask),
        # Estos dos exigen S3 para tener contenido: miden valor perdido por la
        # forma knapsack del potencial. Sin candidata, no se inventan.
        "valor_perdido_separabilidad": None,
        "local_realizable_no_conjunta": None,
    }

    if t in pools:
        clase = "repetida"
    elif any(t != pool and not (t & ~pool) for pool in pools):
        clase = "descendiente"
    elif any(t != pool and not (pool & ~t) for pool in pools):
        clase = "ancestro"
    elif flags["cruzada"]:
        clase = "cruzada"
    elif t and not (t & tocado):
        clase = "virgen"
    elif celdas and all((t & celda) in (0, celda) for celda in celdas) and not (t & virgen_mask):
        clase = "union_de_atomos"
    elif flags["dominada"]:
        clase = "dominada"
    else:
        clase = "compatible_mixta"
    return clase, flags


# ---------------------------------------------------------------- politicas

class OptimalPolicy:
    """Politica optima irrestricta, con el mismo desempate congelado (§5.11)."""

    def __init__(self, evaluator, actions):
        self.ev = evaluator
        self.actions = tuple(actions)

    @lru_cache(maxsize=None)
    def value(self, worlds, cleared, step):
        if step >= self.ev.B:
            return 0.0
        mejor = 0.0
        for pool in self.actions:
            v = sum(prob * (reward + self.value(child, nc, step + 1))
                    for prob, child, nc, reward in self.ev.branches(worlds, cleared, pool))
            mejor = max(mejor, v)
        return mejor

    @lru_cache(maxsize=None)
    def action(self, worlds, cleared, step):
        puntuadas = []
        for pool in self.actions:
            v = sum(prob * (reward + self.value(child, nc, step + 1))
                    for prob, child, nc, reward in self.ev.branches(worlds, cleared, pool))
            puntuadas.append((v, pool))
        return select_action(puntuadas, self.ev.n)[1]


# ---------------------------------------------------------------- recorrido

@dataclass
class Decision:
    instance_id: str
    history_id: str
    probability: float
    policy: str
    action: int
    action_size: int
    depth: int
    klass: str
    flags: dict
    score: float
    immediate: float
    rollout_q: float
    local_regret: float
    final_value: float


def _history_id(executed) -> str:
    return "|".join(f"{pool:x}:{count}" for pool, count in executed) or "raiz"


def walk(ev, policy, nombre: str, instance_id: str, rollout: RolloutPolicy, value_fn):
    """Recorre los estados alcanzables por `policy`, registrando cada decision.

    La probabilidad de cada decision es la del historial que la precede, asi que
    las fracciones por clase quedan ponderadas por P^pi(H) como pide §17.
    ``value_fn(worlds, cleared, step)`` es el valor de la propia politica desde ese
    estado; se registra para poder cruzar comportamiento con welfare.
    """
    decisiones = []
    pila = [(ev.all_worlds, 0, 0, (), 1.0)]
    while pila:
        worlds, cleared, step, executed, prob = pila.pop()
        if step >= ev.B or prob <= 0.0:
            continue
        elegida = policy.action(worlds, cleared, step)
        clase, flags = clasificar(elegida, executed, cleared, ev.n)

        q_por_pool = {pool: q for q, pool in rollout.q_values(worlds, cleared, step)}
        mejor_q = max(q_por_pool.values())
        inmediato = sum(p * r for p, _, _, r in ev.branches(worlds, cleared, elegida))

        decisiones.append(Decision(
            instance_id=instance_id,
            history_id=_history_id(executed),
            probability=prob,
            policy=nombre,
            action=elegida,
            action_size=bin(elegida).count("1"),
            depth=step,
            klass=clase,
            flags=flags,
            score=inmediato,
            immediate=inmediato,
            rollout_q=q_por_pool.get(elegida, float("nan")),
            local_regret=mejor_q - q_por_pool.get(elegida, mejor_q),
            final_value=value_fn(worlds, cleared, step),
        ))
        # Se ramifica por CONTEO observado, no por rama anonima: el conteo es
        # parte del historial y sin el no se puede clasificar la accion siguiente.
        masa_total = ev.mass(worlds)
        for count, compatible in enumerate(ev.outcome_worlds[elegida]):
            child = worlds & compatible
            masa_hijo = ev.mass(child)
            if masa_hijo <= 0.0:
                continue
            nc = cleared | elegida if count == 0 else cleared
            pila.append((child, nc, step + 1,
                         executed + ((elegida, count),),
                         prob * masa_hijo / masa_total))
    return decisiones


# ---------------------------------------------------------------- barrido

def default_sweep():
    """Barrido de §17 recortado a lo que corre en segundos, no en horas."""
    casos = []
    for n in (4, 5, 6):
        for B in (1, 2, 3):
            for G in (2, 3):
                for prev in (0.05, 0.45, 0.90):
                    casos.append((n, B, G, prev))
    return casos


def run(sweep=None, out_path=None):
    sweep = sweep or default_sweep()
    filas = []
    resumen = []
    for n, B, G, prev in sweep:
        instance_id = f"n{n}_B{B}_G{G}_p{prev}"
        p, u = [prev] * n, [1.0] * n
        ev = ExactPolicyEvaluator(p, u, B, G)
        acciones = ev.pools
        greedy = GreedyPolicy(ev, acciones, get_scorer("S0"))
        roll = RolloutPolicy(ev, acciones, get_scorer("S0"), base=greedy)
        opt = OptimalPolicy(ev, acciones)

        from augmented.rollout_oracle import _rollout_value
        valores = {
            "S0": greedy.value,
            "rollout": lambda w, c, s_, _r=roll: _rollout_value(_r, w, c, s_),
            "optimo": opt.value,
        }
        for nombre, pol in (("S0", greedy), ("rollout", roll), ("optimo", opt)):
            decisiones = walk(ev, pol, nombre, instance_id, roll, valores[nombre])
            filas.extend(decisiones)
            masa = {}
            for d in decisiones:
                masa[d.klass] = masa.get(d.klass, 0.0) + d.probability
            total = sum(masa.values()) or 1.0
            resumen.append({
                "instance_id": instance_id, "n": n, "B": B, "G": G, "prevalence": prev,
                "policy": nombre,
                "decisions": len(decisiones),
                "w_cruzada": round(masa.get("cruzada", 0.0) / total, 10),
                "w_virgen": round(masa.get("virgen", 0.0) / total, 10),
                "w_descendiente": round(masa.get("descendiente", 0.0) / total, 10),
                "w_union_de_atomos": round(masa.get("union_de_atomos", 0.0) / total, 10),
                "w_compatible_mixta": round(masa.get("compatible_mixta", 0.0) / total, 10),
                "w_dominada": round(masa.get("dominada", 0.0) / total, 10),
                "mean_local_regret": round(
                    sum(d.local_regret * d.probability for d in decisiones) / total, 10),
                "mean_action_size": round(
                    sum(d.action_size * d.probability for d in decisiones) / total, 10),
            })

    if out_path is not None:
        detalle = [{
            "instance_id": d.instance_id, "history_id": d.history_id,
            "probability": round(d.probability, 10), "policy": d.policy,
            "action": d.action, "action_size": d.action_size, "depth": d.depth,
            "class": d.klass,
            "flags": ";".join(k for k, v in d.flags.items() if v is True),
            "score": round(d.score, 10), "immediate": round(d.immediate, 10),
            "rollout_q": round(d.rollout_q, 10),
            "local_regret": round(d.local_regret, 10),
            "final_value": round(d.final_value, 10),
        } for d in filas]
        base = Path(out_path)
        write_canonical_csv(
            base, detalle,
            generator="augmented.falsificador.run", seed=None,
            params={"sweep_size": len(sweep), "policies": ["S0", "rollout", "optimo"],
                    "classes": list(CLASES)})
        write_canonical_csv(
            base.with_name(base.stem + "_resumen.csv"), resumen,
            generator="augmented.falsificador.run", seed=None,
            params={"sweep_size": len(sweep), "level": "instancia"})
    return filas, resumen


# ---------------------------------------------------------------- demo

def main():
    root = Path(__file__).resolve().parent.parent / "results"
    filas, resumen = run(out_path=root / "falsificador_decisiones.csv")
    print(f"Falsificador (B-M8): {len(filas)} decisiones sobre {len(resumen) // 3} instancias\n")

    print("Masa de decisiones CRUZADAS por politica (fuera de la clase laminar):")
    for pol in ("S0", "rollout", "optimo"):
        filas_pol = [r for r in resumen if r["policy"] == pol]
        media = sum(r["w_cruzada"] for r in filas_pol) / len(filas_pol)
        con_cruce = sum(1 for r in filas_pol if r["w_cruzada"] > 1e-12)
        print(f"  {pol:<8s} media {media:.4f}   instancias con algun cruce: {con_cruce}/{len(filas_pol)}")

    print("\nRegret local medio contra Q del rollout:")
    for pol in ("S0", "rollout", "optimo"):
        filas_pol = [r for r in resumen if r["policy"] == pol]
        print(f"  {pol:<8s} {sum(r['mean_local_regret'] for r in filas_pol) / len(filas_pol):.6f}")

    print(f"\nartefactos en {root}/falsificador_decisiones.csv (+ _resumen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
