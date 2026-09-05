"""Interfaz de scorers miopes (B-M5, plan maestro §14).

Un scorer es una funcion ``S(estado, pool) -> float`` que puntua una accion sin
resolver el problema de control. La politica greedy es ``argmax`` del scorer con
la regla de desempate congelada de §5.11; el rollout usa al greedy como politica
base (§15). Esta es la unica via soportada para definir un criterio miope: si un
scorer nuevo no implementa este protocolo, no entra al oraculo ni al atlas.

**Que esta cableado y que no.** Solo ``S0``. El plan es explicito en que no se
deje un placeholder de S2, y la razon no es de estilo:

- ``S1`` colapsa (§14.3). Bajo strict hard clearing una deduccion no acredita, asi
  que ``S1_hard`` es identicamente ``S0``. No es una aproximacion parecida: es la
  misma funcion.
- ``S2`` esta muerto en sus dos variantes (§14.4). La global muere por la tower
  property: el incremento esperado es cero ante toda accion, asi que no ordena
  nada. La de cubierta recupera el primer movimiento pero es martingala bajo
  subdivision, asi que encuentra y nunca cosecha.
- ``S3`` (§14.5) requiere ``varphi_virgin`` y ``varphi``, que son diseño de A
  (A-M12) y estan detras del gate G4a. Cuando exista, entra por este mismo
  protocolo.

**Sin doble conteo.** El scorer puntua la ganancia incremental de la accion, no el
valor acumulado del estado. La utilidad ya acreditada ``U(C(H))`` no aparece
nunca: sumarla convertiria ``r`` en un termino contado dos veces (§14.5).
"""

from __future__ import annotations

from dataclasses import dataclass

from augmented.core import indices_from_mask

# ---------------------------------------------------------------- desempate

# Regla de desempate de §5.11, CONGELADA antes de comparar arboles.
#
# Orden: (1) mayor score; (2) criterio de tamano declarado; (3) menor mascara.
#
# El criterio de tamano que este proyecto declara es POOL MAS CHICO PRIMERO. Se
# elige asi porque ante score identico el pool menor deja mas individuos sin
# tocar, y por tanto mas acciones distinguibles disponibles despues; el pool
# mayor consume territorio virgen sin pagar por el. La tercera clave, la menor
# mascara, no tiene contenido semantico: existe solo para que la regla sea total
# y el arbol resultante no dependa del orden de iteracion.
TIE_BREAK = "mayor score, luego pool mas chico, luego menor mascara"


def tie_break_key(score: float, pool: int, n: int) -> tuple:
    """Clave de orden total para elegir accion. Mayor es mejor."""
    return (score, -_pool_size(pool, n), -pool)


def _pool_size(pool: int, n: int) -> int:
    return bin(pool & ((1 << n) - 1)).count("1")


def select_action(scored, n: int):
    """Elige entre ``[(score, pool, payload), ...]`` con la regla congelada.

    Devuelve la tupla ganadora completa. Falla con una lista vacia en vez de
    devolver un centinela: no hay accion valida y quien llama debe decidir.
    """
    scored = list(scored)
    if not scored:
        raise ValueError("no hay acciones candidatas que puntuar")
    return max(scored, key=lambda row: tie_break_key(row[0], row[1], n))


# ---------------------------------------------------------------- protocolo

@dataclass(frozen=True)
class ScorerSpec:
    """Ficha declarativa de un scorer, exigida por §14.1 y §14.8.

    Cada campo responde a un requisito del plan. Un scorer que no pueda llenar
    la ficha honestamente no esta listo para el atlas.
    """

    name: str
    uses_full_history: bool     # ve H completo, no solo un resumen
    hard_clearing: bool         # respeta strict hard clearing (§5.7)
    discounts_credited: bool    # descuenta C(H); no recobra lo ya acreditado
    budget_aware: bool          # depende de b (S0 no; S3 si)
    realizable: bool            # es valor de una politica ejecutable, no una cota suelta
    description: str


class Scorer:
    """Protocolo. Implementar ``score`` y declarar ``spec``."""

    spec: ScorerSpec

    def score(self, evaluator, worlds: int, cleared: int, budget: int, pool: int) -> float:
        raise NotImplementedError

    def __call__(self, evaluator, worlds, cleared, budget, pool):
        return self.score(evaluator, worlds, cleared, budget, pool)

    def __repr__(self):
        return f"<{type(self).__name__} {self.spec.name}>"


# ---------------------------------------------------------------- S0

class S0(Scorer):
    """S0(H,t) = P(R(t)=0 | H) * sum_{i in t \\ C(H)} u_i   (§14.2).

    Es el pago inmediato esperado de la accion bajo hard clearing, y nada mas: no
    mira el presupuesto restante ni el valor de la informacion. Su patologia
    conocida y esperada es que con q pequena nunca agrupa, porque el factor
    P(R=0) cae exponencialmente en el tamano del pool mientras la utilidad solo
    crece linealmente. Esa es precisamente la falla que S3 debe corregir.
    """

    spec = ScorerSpec(
        name="S0",
        uses_full_history=True,
        hard_clearing=True,
        discounts_credited=True,
        budget_aware=False,
        realizable=True,
        description="pago inmediato esperado; §14.2",
    )

    def score(self, evaluator, worlds, cleared, budget, pool):
        del budget  # S0 es ciego al presupuesto, por definicion
        total_mass = evaluator.mass(worlds)
        if total_mass <= 0.0:
            return 0.0
        clean_worlds = worlds & evaluator.outcome_worlds[pool][0]
        clean_mass = evaluator.mass(clean_worlds)
        if clean_mass <= 0.0:
            return 0.0
        reward = float(evaluator.utility[pool & ~cleared])
        return (clean_mass / total_mass) * reward


#: Registro de scorers disponibles. S3 se agrega aqui cuando pase G4a.
REGISTRY = {"S0": S0()}


def get_scorer(name: str) -> Scorer:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"scorer '{name}' no registrado; disponibles: {sorted(REGISTRY)}. "
            f"S1 colapsa a S0 (§14.3) y S2 esta muerto (§14.4); no se cablean."
        ) from None


def pool_members(pool: int, n: int):
    """Miembros de un pool, para scorers que necesiten el detalle por individuo."""
    return indices_from_mask(pool, n)
