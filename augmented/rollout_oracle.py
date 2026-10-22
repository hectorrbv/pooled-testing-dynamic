"""Oraculo de rollout y su verificacion cruzada (B-M6, gate G5, plan maestro §15).

El rollout replanificado usa al greedy como politica base: puntua cada accion con
``Q_b^g(H,t) = E[r + V_{b-1}^g(H') | H,t]``, ejecuta el argmax, y en el estado
siguiente vuelve a aplicar la misma regla --- no continua con greedy. Eso es lo
que demuestra la Proposicion B (`augmented/paper/proposicion_b_policy_improvement.md`)
y lo que este modulo implementa; `verify_proposition_b` comprueba que el codigo
sea esa politica y no una variante parecida.

**Gate G5: dos evaluadores independientes a 1e-10.** El plan no acepta un solo
numero. Aqui hay dos caminos de codigo que no comparten nada mas que el modelo:

- ``value_by_belief_dp`` recorre hacia atras el arbol de creencias, ramificando
  por conteo observado y ponderando por probabilidad condicional.
- ``value_by_latent_enumeration`` recorre hacia adelante los 2^n perfiles latentes
  z, corre la politica de forma determinista sobre cada uno --- sin probabilidades
  en ningun paso --- y promedia con el prior.

Uno razona sobre creencias y esperanzas; el otro sobre realizaciones y frecuencias.
Que coincidan a 1e-10 es evidencia real de que la politica esta bien evaluada; que
un DP coincida consigo mismo no lo seria.

Corre con:  python3 -m augmented.rollout_oracle
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from augmented.laminar_benchmarks import ExactPolicyEvaluator, balanced_laminar_library
from augmented.scorers import Scorer, get_scorer, select_action

G5_TOL = 1e-10


# ---------------------------------------------------------------- politicas

class _PolicyBase:
    """Politica determinista sobre el estado de creencia (worlds, cleared, step)."""

    def __init__(self, evaluator: ExactPolicyEvaluator, actions, scorer: Scorer):
        self.ev = evaluator
        self.actions = tuple(actions)
        self.scorer = scorer
        if not self.actions:
            raise ValueError("la biblioteca de acciones esta vacia")

    def action(self, worlds: int, cleared: int, step: int) -> int:
        raise NotImplementedError


class GreedyPolicy(_PolicyBase):
    """argmax del scorer miope, con el desempate congelado de §5.11."""

    def action(self, worlds, cleared, step):
        return self._action(worlds, cleared, step)

    @lru_cache(maxsize=None)
    def _action(self, worlds, cleared, step):
        del step  # S0 no depende del presupuesto restante
        scored = [
            (self.scorer(self.ev, worlds, cleared, None, pool), pool)
            for pool in self.actions
        ]
        return select_action(scored, self.ev.n)[1]

    @lru_cache(maxsize=None)
    def value(self, worlds, cleared, step):
        """V_b^g exacto, incremental (misma convencion que §14.5)."""
        if step >= self.ev.B:
            return 0.0
        pool = self.action(worlds, cleared, step)
        return sum(
            prob * (reward + self.value(child, new_cleared, step + 1))
            for prob, child, new_cleared, reward
            in self.ev.branches(worlds, cleared, pool)
        )


class RolloutPolicy(_PolicyBase):
    """argmax de Q_b^g, replanificado en cada estado (§15, Proposicion B)."""

    def __init__(self, evaluator, actions, scorer, base: GreedyPolicy | None = None):
        super().__init__(evaluator, actions, scorer)
        self.base = base or GreedyPolicy(evaluator, actions, scorer)

    def q_values(self, worlds, cleared, step):
        """``[(Q_b^g(H,t), t), ...]`` para toda accion admisible."""
        out = []
        for pool in self.actions:
            q = sum(
                prob * (reward + self.base.value(child, new_cleared, step + 1))
                for prob, child, new_cleared, reward
                in self.ev.branches(worlds, cleared, pool)
            )
            out.append((q, pool))
        return out

    def action(self, worlds, cleared, step):
        return self._action(worlds, cleared, step)

    @lru_cache(maxsize=None)
    def _action(self, worlds, cleared, step):
        if step >= self.ev.B:
            raise ValueError("sin presupuesto restante: no hay accion que elegir")
        return select_action(self.q_values(worlds, cleared, step), self.ev.n)[1]


# ---------------------------------------------------------------- evaluador 1

def value_by_belief_dp(policy: _PolicyBase) -> float:
    """Valor por recursion hacia atras sobre el arbol de creencias."""
    ev = policy.ev

    @lru_cache(maxsize=None)
    def go(worlds, cleared, step):
        if step >= ev.B:
            return 0.0
        pool = policy.action(worlds, cleared, step)
        return sum(
            prob * (reward + go(child, new_cleared, step + 1))
            for prob, child, new_cleared, reward in ev.branches(worlds, cleared, pool)
        )

    return float(go(ev.all_worlds, 0, 0))


# ---------------------------------------------------------------- evaluador 2

def value_by_latent_enumeration(policy: _PolicyBase) -> float:
    """Valor corriendo la politica sobre cada perfil latente z, y promediando.

    No aparece ninguna probabilidad dentro del bucle: para un z fijo el conteo de
    cada pool es un entero determinista y la recompensa es aritmetica. El prior
    entra una sola vez, al promediar. Ese aislamiento es lo que hace que este
    camino sea independiente del DP sobre creencias.
    """
    ev = policy.ev
    total = 0.0
    for z in range(ev.world_count):
        weight = float(ev.weights[z])
        if weight <= 0.0:
            continue
        worlds, cleared, payoff = ev.all_worlds, 0, 0.0
        for step in range(ev.B):
            pool = policy.action(worlds, cleared, step)
            # Conteo observado bajo ESTE perfil: cuantos activos hay en el pool.
            count = int(ev.scenarios[z][list(_bits(pool))].sum()) if pool else 0
            if count == 0:
                payoff += float(ev.utility[pool & ~cleared])
                cleared |= pool
            worlds &= ev.outcome_worlds[pool][count]
        total += weight * payoff
    return total


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask &= mask - 1


# ---------------------------------------------------------------- gate G5

@dataclass(frozen=True)
class CrossCheck:
    """Resultado de contrastar los dos evaluadores sobre una politica."""

    label: str
    belief_dp: float
    latent_enumeration: float

    @property
    def gap(self) -> float:
        return abs(self.belief_dp - self.latent_enumeration)

    @property
    def passes(self) -> bool:
        return self.gap <= G5_TOL

    def __str__(self):
        estado = "OK" if self.passes else "FALLA"
        return (f"{self.label:<10s} dp={self.belief_dp:.12f}  "
                f"latente={self.latent_enumeration:.12f}  gap={self.gap:.2e}  {estado}")


def cross_check(policy: _PolicyBase, label: str) -> CrossCheck:
    return CrossCheck(label, value_by_belief_dp(policy), value_by_latent_enumeration(policy))


def oracle(p, u, B, G, actions=None, scorer="S0"):
    """Construye greedy y rollout sobre una instancia, listos para evaluar."""
    ev = ExactPolicyEvaluator(p, u, B, G)
    if actions is None:
        actions = balanced_laminar_library(p, u, G)
    s = get_scorer(scorer) if isinstance(scorer, str) else scorer
    greedy = GreedyPolicy(ev, actions, s)
    return ev, greedy, RolloutPolicy(ev, actions, s, base=greedy)


def verify_g5(p, u, B, G, actions=None, scorer="S0"):
    """Gate G5 sobre una instancia: ambos evaluadores, ambas politicas."""
    _, greedy, roll = oracle(p, u, B, G, actions, scorer)
    return [cross_check(greedy, "greedy"), cross_check(roll, "rollout")]


# ---------------------------------------------------------------- Proposicion B

@dataclass(frozen=True)
class PropositionBReport:
    """Verificacion de que el codigo ES la politica demostrada (P21-B3)."""

    states_checked: int
    greedy_always_candidate: bool      # hipotesis 3 del enunciado
    dominance_holds: bool              # V^r >= V^g en todo estado alcanzable
    worst_margin: float                # min(V^r - V^g); negativo seria refutacion
    root_margin: float                 # V^r - V^g desde el estado inicial
    replans: bool                      # el rollout replanifica, no sigue greedy

    @property
    def passes(self):
        return self.greedy_always_candidate and self.dominance_holds and self.replans


def verify_proposition_b(p, u, B, G, actions=None, scorer="S0") -> PropositionBReport:
    """Comprueba las hipotesis y la conclusion de la Proposicion B, estado por estado.

    La hipotesis 3 --- que la accion del greedy esta entre las candidatas del
    rollout --- se verifica en vez de asumirse, porque es la unica que un cambio
    en la biblioteca de acciones podria romper sin aviso.
    """
    ev, greedy, roll = oracle(p, u, B, G, actions, scorer)

    vistos = set()
    margen_raiz = 0.0
    hipotesis3 = True
    dominancia = True
    margen = float("inf")
    replanifica = False

    frontera = [(ev.all_worlds, 0, 0)]
    while frontera:
        worlds, cleared, step = frontera.pop()
        if (worlds, cleared, step) in vistos or step >= ev.B:
            continue
        vistos.add((worlds, cleared, step))

        if greedy.action(worlds, cleared, step) not in {pool for _, pool in roll.q_values(worlds, cleared, step)}:
            hipotesis3 = False

        vr = _rollout_value(roll, worlds, cleared, step)
        vg = greedy.value(worlds, cleared, step)
        margen = min(margen, vr - vg)
        if step == 0:
            margen_raiz = vr - vg
        if vr < vg - 1e-12:
            dominancia = False

        elegido = roll.action(worlds, cleared, step)
        if elegido != greedy.action(worlds, cleared, step):
            replanifica = True
        for _, child, new_cleared, _ in ev.branches(worlds, cleared, elegido):
            frontera.append((child, new_cleared, step + 1))

    return PropositionBReport(
        states_checked=len(vistos),
        greedy_always_candidate=hipotesis3,
        dominance_holds=dominancia,
        worst_margin=(0.0 if margen == float("inf") else margen),
        root_margin=margen_raiz,
        # `replans` documenta si en ESTA instancia el rollout se separo del greedy.
        # No es una hipotesis del teorema; es la senal de que la instancia ejercita
        # de verdad al rollout y no es un caso donde ambos coinciden trivialmente.
        replans=replanifica or B <= 1,
    )


def _rollout_value(roll: RolloutPolicy, worlds, cleared, step, _memo=None):
    """V_b^r exacto. El memo es por llamada raiz, no global: la politica depende
    de la instancia y cachearla por (worlds, cleared, step) entre instancias
    distintas mezclaria arboles."""
    if _memo is None:
        _memo = {}
    key = (worlds, cleared, step)
    if key in _memo:
        return _memo[key]
    ev = roll.ev
    if step >= ev.B:
        return 0.0
    pool = roll.action(worlds, cleared, step)
    val = sum(
        prob * (reward + _rollout_value(roll, child, new_cleared, step + 1, _memo))
        for prob, child, new_cleared, reward in ev.branches(worlds, cleared, pool)
    )
    _memo[key] = val
    return val


# ---------------------------------------------------------------- demo

def main():
    casos = [
        ("n=5 homogenea q=0.3", [0.7] * 5, [1.0] * 5, 3, 2),
        ("n=5 heterogenea", [0.5, 0.8, 0.2, 0.9, 0.6], [1.0, 3.0, 2.0, 1.5, 0.5], 3, 3),
        ("n=6 prevalencia alta", [0.85] * 6, [1.0] * 6, 3, 3),
    ]
    print(f"Gate G5: dos evaluadores independientes, tolerancia {G5_TOL:g}\n")
    todo_ok = True
    for nombre, p, u, B, G in casos:
        print(f"--- {nombre} (B={B}, G={G})")
        for chk in verify_g5(p, u, B, G):
            print(f"    {chk}")
            todo_ok &= chk.passes
        rep = verify_proposition_b(p, u, B, G)
        print(f"    Prop B: {rep.states_checked} estados, hipotesis3={rep.greedy_always_candidate}, "
              f"dominancia={rep.dominance_holds}, margen raiz={rep.root_margin:+.6f}, "
              f"margen minimo={rep.worst_margin:+.6f}")
        todo_ok &= rep.passes
    print("\nG5:", "APROBADO" if todo_ok else "NO APROBADO")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
