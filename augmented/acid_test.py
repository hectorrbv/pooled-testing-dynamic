"""Harness del acid test para candidatas de scorer (B-M7, gate G4b, plan maestro §16).

El acid test pregunta una sola cosa: **¿el scorer recupera el plan que sabemos que
gana?** La familia es homogenea con q pequena, y ahi el plan bueno es cubrir con
pools raiz de tamano G y despues bajar por busqueda binaria hasta acreditar a un
sano. Un scorer que no valore ese plan no sirve, por elegante que sea su forma.

**El ancla ejecutable** es (q, G, k, B) = (0.05, 16, 2, 7): el plan cover-binary-search
cobra al menos 0.806u mientras el baseline de singletons cobra 0.35u. La forma
general bajo strict hard clearing es

    k = max(0, B - ceil(log2 G) - 1)      pools raiz
    valor >= u * (1 - (1-q)^(kG))

y el ``-1`` no es decoracion: bajo strict hard clearing una deduccion no acredita
(§5.8), asi que hay que reservar una prueba final para convertir al sano
localizado en utilidad cobrada. Ese es el test acreditador.

**Los nueve checks de trayectoria (§16, G4b).** Cada uno es una afirmacion
falsable sobre el scorer, no una impresion. Se corren contra cualquier candidata
que implemente el protocolo de `augmented.scorers`.

**Que se puede correr y que no.** Los checks de comportamiento necesitan estados
de creencia explicitos, y eso limita n a lo que el evaluador exacto aguanta; se
corren en la sub-familia tratable (G en {2,4}, k en {1,2}). La aritmetica del
ancla es analitica y se verifica en toda la malla declarada, G en {2,4,8,16}
incluido. Los dos alcances se reportan por separado y nunca se mezclan.

Corre con:  python3 -m augmented.acid_test
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from augmented.laminar_benchmarks import ExactPolicyEvaluator
from augmented.scorers import Scorer, get_scorer, select_action

# n maximo para los checks de comportamiento: el evaluador exacto guarda un
# bitmask sobre 2^n mundos y `mass` recorre sus bits.
MAX_N_COMPORTAMIENTO = 8


# ---------------------------------------------------------------- la familia

@dataclass(frozen=True)
class AcidInstance:
    """Instancia de la familia del acid test (§16): homogenea, q pequena."""

    q: float          # probabilidad de estar SANO
    G: int            # tamano de pool raiz
    k: int            # numero de pools raiz disjuntos
    B: int            # presupuesto
    u: float = 1.0

    @property
    def n(self) -> int:
        return self.k * self.G

    @property
    def coverage(self) -> int:
        return self.k * self.G

    def __str__(self):
        return f"(q={self.q}, G={self.G}, k={self.k}, B={self.B})"


def k_from_budget(B: int, G: int) -> int:
    """k = max(0, B - ceil(log2 G) - 1). El -1 es el test acreditador (§5.8)."""
    return max(0, B - math.ceil(math.log2(G)) - 1)


def cbs_lower_bound(inst: AcidInstance) -> float:
    """Cota inferior realizable del plan cover-binary-search: u*(1-(1-q)^(kG))."""
    if inst.k <= 0:
        return 0.0
    return inst.u * (1.0 - (1.0 - inst.q) ** inst.coverage)


def singleton_baseline(inst: AcidInstance) -> float:
    """Valor del baseline de puras pruebas individuales, que es lo que hace S0."""
    return inst.B * inst.q * inst.u


def anchor_instance() -> AcidInstance:
    """El ancla ejecutable declarada en §16."""
    return AcidInstance(q=0.05, G=16, k=k_from_budget(7, 16), B=7)


# ---------------------------------------------------------------- resultados

@dataclass
class CheckResult:
    number: int
    name: str
    passed: bool | None          # None = no aplicable en esta instancia
    detail: str = ""

    def __post_init__(self):
        # Las comparaciones sobre escalares de numpy devuelven np.bool_, que NO
        # es `True` bajo identidad. Sin esta normalizacion los conteos mienten.
        if self.passed is not None:
            self.passed = bool(self.passed)

    def __str__(self):
        marca = {True: "OK  ", False: "FALLA", None: "n/a "}[self.passed]
        return f"  {self.number}. {marca} {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class AcidReport:
    scorer: str
    instance: AcidInstance
    checks: list = field(default_factory=list)

    @property
    def passed(self):
        return [c for c in self.checks if c.passed is True]

    @property
    def failed(self):
        return [c for c in self.checks if c.passed is False]

    @property
    def all_pass(self):
        return not self.failed

    def __str__(self):
        cab = f"{self.scorer} en {self.instance}: {len(self.passed)}/{len(self.checks)} checks"
        return cab + "\n" + "\n".join(str(c) for c in self.checks)


# ---------------------------------------------------------------- utilidades de estado

def _evaluator(inst: AcidInstance, budget: int | None = None):
    n = inst.n
    if n > MAX_N_COMPORTAMIENTO:
        raise ValueError(
            f"n={n} excede el limite de los checks de comportamiento "
            f"({MAX_N_COMPORTAMIENTO}); usa la via analitica para esa instancia"
        )
    p = [1.0 - inst.q] * n           # p = P(activo) = 1 - q
    u = [inst.u] * n
    return ExactPolicyEvaluator(p, u, budget if budget is not None else inst.B, min(inst.G, n))


def _mask(indices):
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def _condition(ev, worlds, pool, count):
    """Estado tras observar `count` en `pool`. Devuelve (worlds', cleared_delta)."""
    return worlds & ev.outcome_worlds[pool][count]


# ---------------------------------------------------------------- los nueve checks

def _check1_virgen(ev, scorer, inst) -> CheckResult:
    """¿Valora abrir territorio virgen en vez de picotear singletons?"""
    raiz = _mask(range(min(inst.G, ev.n)))
    single = _mask([0])
    s_raiz = scorer(ev, ev.all_worlds, 0, ev.B, raiz)
    s_single = scorer(ev, ev.all_worlds, 0, ev.B, single)
    ok = s_raiz > s_single
    return CheckResult(1, "valora abrir territorio virgen", ok,
                       f"pool G={min(inst.G, ev.n)}: {s_raiz:.4f} vs singleton {s_single:.4f}")


def _check2_reentra(ev, scorer, inst) -> CheckResult:
    """Tras un conteo informativo, ¿valora volver al pool util?"""
    G = min(inst.G, ev.n)
    if G < 2 or ev.n <= G:
        return CheckResult(2, "tras conteo informativo vuelve al pool util", None,
                           "requiere un pool de tamano >=2 y territorio virgen restante")
    raiz = _mask(range(G))
    worlds = _condition(ev, ev.all_worlds, raiz, 1)      # conteo 1: uno activo dentro
    dentro = _mask(range(G // 2))                        # subpool del pool util
    fresco = _mask([G])                                  # individuo virgen
    s_dentro = scorer(ev, worlds, 0, ev.B - 1, dentro)
    s_fresco = scorer(ev, worlds, 0, ev.B - 1, fresco)
    ok = s_dentro > s_fresco
    return CheckResult(2, "tras conteo informativo vuelve al pool util", ok,
                       f"reentrada {s_dentro:.4f} vs virgen {s_fresco:.4f}")


def _check3_subdivide(ev, scorer, inst) -> CheckResult:
    """Con presupuesto suficiente, ¿parte a la mitad en vez de probar un singleton?"""
    G = min(inst.G, ev.n)
    if G < 4:
        return CheckResult(3, "subdivide con presupuesto suficiente", None,
                           "requiere G>=4 para que media y singleton difieran")
    raiz = _mask(range(G))
    worlds = _condition(ev, ev.all_worlds, raiz, 1)
    mitad = _mask(range(G // 2))
    single = _mask([0])
    s_mitad = scorer(ev, worlds, 0, ev.B - 1, mitad)
    s_single = scorer(ev, worlds, 0, ev.B - 1, single)
    ok = s_mitad >= s_single
    return CheckResult(3, "subdivide con presupuesto suficiente", ok,
                       f"mitad {s_mitad:.4f} vs singleton {s_single:.4f}")


def _check4_cero_vs_deduccion(ev, scorer, inst) -> CheckResult:
    """¿Distingue un cero observado de una limpieza meramente deducida (§5.8)?"""
    if ev.n < 2:
        return CheckResult(4, "distingue cero observado de limpieza deducida", None, "requiere n>=2")
    par = _mask([0, 1])
    # Estado deducido: conteo 1 en {0,1} y conteo 1 en {0} => el 1 esta sano, sin acreditar.
    worlds = _condition(ev, ev.all_worlds, par, 1)
    worlds = _condition(ev, worlds, _mask([0]), 1)
    s_deducido = scorer(ev, worlds, 0, ev.B - 2, _mask([1]))
    # Mismo individuo, ya acreditado: no puede volver a pagar.
    s_acreditado = scorer(ev, worlds, _mask([1]), ev.B - 2, _mask([1]))
    ok = s_deducido > 0.0 and s_acreditado == 0.0
    return CheckResult(4, "distingue cero observado de limpieza deducida", ok,
                       f"deducido-sin-acreditar {s_deducido:.4f}, ya acreditado {s_acreditado:.4f}")


def _check5_acreditador(ev, scorer, inst) -> CheckResult:
    """Con una prueba restante y un sano deducido, ¿valora la prueba acreditadora?"""
    if ev.n < 2:
        return CheckResult(5, "reserva el test acreditador", None, "requiere n>=2")
    par = _mask([0, 1])
    worlds = _condition(ev, ev.all_worlds, par, 1)
    worlds = _condition(ev, worlds, _mask([0]), 1)       # el 1 esta sano con certeza
    candidatos = [(scorer(ev, worlds, 0, 1, pool), pool) for pool in ev.pools]
    elegido = select_action(candidatos, ev.n)[1]
    ok = bool(elegido & _mask([1]))
    return CheckResult(5, "reserva el test acreditador", ok,
                       f"con 1 prueba elige mascara {elegido:#b}, que {'incluye' if ok else 'excluye'} al sano deducido")


def _check6_no_gasta_todo(ev, scorer, inst) -> CheckResult:
    """¿El ranking reacciona al presupuesto restante?

    Un scorer ciego al presupuesto no puede reservar el test acreditador ni
    abandonar una ruta que ya no cabe: siempre puntua igual. Se detecta
    comparando el ranking con presupuesto amplio contra el de una sola prueba.
    """
    raiz = _mask(range(min(inst.G, ev.n)))
    single = _mask([0])
    amplio = (scorer(ev, ev.all_worlds, 0, ev.B, raiz), scorer(ev, ev.all_worlds, 0, ev.B, single))
    justo = (scorer(ev, ev.all_worlds, 0, 1, raiz), scorer(ev, ev.all_worlds, 0, 1, single))
    ok = amplio != justo
    return CheckResult(6, "no gasta todo explorando (usa el presupuesto)", ok,
                       "el scorer es ciego al presupuesto: mismo ranking con B amplio y con B=1"
                       if not ok else "el ranking cambia con el presupuesto")


def _check7_no_duplica(ev, scorer, inst) -> CheckResult:
    """¿Puntua en cero un pool ya enteramente acreditado (§14.5, sin doble conteo)?"""
    par = _mask([0, 1]) if ev.n >= 2 else _mask([0])
    s = scorer(ev, ev.all_worlds, par, ev.B, par)
    ok = s == 0.0
    return CheckResult(7, "no duplica utilidad acreditada", ok, f"score de pool ya acreditado = {s:.4f}")


_CHECKS_LOCALES = (_check1_virgen, _check2_reentra, _check3_subdivide,
                   _check4_cero_vs_deduccion, _check5_acreditador,
                   _check6_no_gasta_todo, _check7_no_duplica)


def run_local_checks(scorer: Scorer, inst: AcidInstance) -> AcidReport:
    """Checks 1-7 sobre una instancia concreta."""
    ev = _evaluator(inst)
    rep = AcidReport(scorer.spec.name, inst)
    for fn in _CHECKS_LOCALES:
        rep.checks.append(fn(ev, scorer, inst))
    return rep


# ------------------------------------------------------- checks 8 y 9 (meta)

def _tractable_neighborhood():
    """Vecindad de §16 recortada a lo que el evaluador exacto aguanta."""
    out = []
    for q in (0.05, 0.15, 0.30, 0.45):
        for G in (2, 4):
            for k in (1, 2):
                n = k * G
                if n > MAX_N_COMPORTAMIENTO:
                    continue
                B = k + math.ceil(math.log2(G)) + 1
                out.append(AcidInstance(q=q, G=G, k=k, B=B))
    return out


def check8_robustez(scorer: Scorer) -> CheckResult:
    """¿El patron de checks 1-7 se sostiene en toda la vecindad tratable?"""
    # Se compara el veredicto POR NUMERO de check y solo donde el check aplica.
    # Un "n/a" no es una discrepancia: significa que la instancia no tiene la
    # estructura para plantear esa pregunta (p. ej. G=2 no distingue mitad de
    # singleton). Contarlo como diferencia haria fallar el check 8 siempre.
    veredictos: dict[int, set] = {}
    instancias = _tractable_neighborhood()
    for inst in instancias:
        for c in run_local_checks(scorer, inst).checks:
            if c.passed is not None:
                veredictos.setdefault(c.number, set()).add(c.passed)
    inestables = sorted(num for num, vs in veredictos.items() if len(vs) > 1)
    ok = not inestables
    detalle = f"{len(instancias)} instancias"
    detalle += "; veredicto estable en todos los checks aplicables" if ok else \
               f"; checks con veredicto inestable: {inestables}"
    return CheckResult(8, "robusto al variar q, G, k, B", ok, detalle)


def check9_desempate(scorer: Scorer, inst: AcidInstance, eps: float = 1e-6) -> CheckResult:
    """¿El resultado sobrevive a romper los empates perturbando las utilidades?

    Si el veredicto cambia al perturbar u en 1e-6, no estaba midiendo el
    mecanismo: estaba leyendo el orden de iteracion.
    """
    base = tuple(c.passed for c in run_local_checks(scorer, inst).checks)
    ev = _evaluator(inst)
    p = [1.0 - inst.q] * ev.n
    u = [inst.u * (1.0 + eps * (i + 1)) for i in range(ev.n)]
    ev_pert = ExactPolicyEvaluator(p, u, inst.B, min(inst.G, ev.n))
    rep = AcidReport(scorer.spec.name, inst)
    for fn in _CHECKS_LOCALES:
        rep.checks.append(fn(ev_pert, scorer, inst))
    ok = base == tuple(c.passed for c in rep.checks)
    return CheckResult(9, "no depende de un desempate afortunado", ok,
                       "veredicto estable bajo perturbacion de u" if ok else "el veredicto cambia al romper empates")


def run_acid_test(scorer: Scorer | str, inst: AcidInstance | None = None) -> AcidReport:
    """Los nueve checks de G4b sobre una candidata de scorer."""
    s = get_scorer(scorer) if isinstance(scorer, str) else scorer
    inst = inst or AcidInstance(q=0.15, G=4, k=2, B=5)
    rep = run_local_checks(s, inst)
    rep.checks.append(check8_robustez(s))
    rep.checks.append(check9_desempate(s, inst))
    return rep


# ---------------------------------------------------------------- ancla analitica

@dataclass(frozen=True)
class AnchorRow:
    instance: AcidInstance
    cbs: float
    singleton: float

    @property
    def ratio(self):
        return self.cbs / self.singleton if self.singleton > 0 else float("inf")


def anchor_grid():
    """Aritmetica del ancla en toda la malla declarada, G en {2,4,8,16} incluido."""
    filas = []
    for q in (0.05, 0.15, 0.30, 0.45):
        for G in (2, 4, 8, 16):
            for k in (1, 2, 3):
                B = k + math.ceil(math.log2(G)) + 1
                inst = AcidInstance(q=q, G=G, k=k, B=B)
                if k_from_budget(B, G) != k:
                    continue
                filas.append(AnchorRow(inst, cbs_lower_bound(inst), singleton_baseline(inst)))
    return filas


def verify_anchor() -> tuple[bool, str]:
    """Comprueba el ancla exacta declarada en §16: (0.05, 16, 2, 7) -> >=0.806 vs 0.35."""
    inst = anchor_instance()
    cbs, base = cbs_lower_bound(inst), singleton_baseline(inst)
    ok = (inst.k == 2 and inst.coverage == 32
          and abs(base - 0.35) < 1e-12 and cbs >= 0.806)
    return ok, (f"k={inst.k}, kG={inst.coverage}, cbs={cbs:.4f} (>=0.806), "
                f"singleton={base:.4f} (=0.35), razon={cbs / base:.2f}x")


# ---------------------------------------------------------------- demo

def main():
    print("ACID TEST — harness (B-M7, gate G4b)\n")
    ok_ancla, detalle = verify_anchor()
    print(f"Ancla §16 {anchor_instance()}: {'OK' if ok_ancla else 'FALLA'}\n  {detalle}\n")

    filas = anchor_grid()
    gana = sum(1 for f in filas if f.ratio > 1.0)
    print(f"Malla analitica: {len(filas)} celdas, el plan CBS supera al baseline en {gana}\n"
          f"  razon maxima {max(f.ratio for f in filas):.2f}x, minima {min(f.ratio for f in filas):.2f}x\n")

    rep = run_acid_test("S0")
    print(rep)
    print(f"\nS0 falla {len(rep.failed)} de {len(rep.checks)} checks. Eso es el resultado esperado:\n"
          "S0 es el baseline que motiva a S3, no un candidato. Los checks que falla\n"
          "senalan exactamente que tiene que arreglar la candidata siguiente.")
    return 0 if ok_ancla else 1


if __name__ == "__main__":
    raise SystemExit(main())
