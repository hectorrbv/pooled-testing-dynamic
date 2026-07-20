"""Cota superior computable sobre el canal de conteo, para la mision de
separacion certificada en la rama laminar (dapts-autoresearch).

ESTE es el unico archivo editable del carril seguro de esa mision. El
benchmark fijo del harness importa `upper_bound` y la evalua sobre una bateria
pequena con V* exacto (puerta de dominacion) y sobre la familia ancla a escala
(puntaje: hueco contra la cota inferior analitica de la estrategia aumentada).

Contrato de `upper_bound(p, u, B, G)`:

- `p[i]` es la probabilidad de que la persona i este ACTIVA; `u[i] >= 0` su
  utilidad; `B` consultas adaptativas de conteo; pools de tamano <= `G`.
- Debe devolver un float finito que sea cota superior VALIDA del optimo
  dinamico del canal de conteo (sobre TODAS las politicas adaptadas, no solo
  las laminares). La semantica del welfare es la del simulador: se acredita
  `u_i` cuando i pertenece a algun pool testeado con resultado 0.
- Debe evaluar en segundos a n ~ 1000 (la familia ancla llega a n=896).
- La puerta empirica (dominacion sobre V* exacto en n <= 7) atrapa errores;
  NO sustituye el argumento de validez.

DOS cotas, se devuelve el minimo (el minimo de cotas validas es valido):

1. U_PI (informacion perfecta / hindsight), la semilla. Exacta a cualquier
   escala pero floja: en el ancla insignia B*G=96 >= n=32 degenera en
   E[#limpios] = n*q = 3.2 (gap 2.314). Ignora que LOCALIZAR limpios cuesta
   informacion.

2. U_cell (relajacion de informacion por conteos-por-celda), NUEVA. Vale solo
   para instancias HOMOGENEAS (todos los p iguales, todos los u iguales), que
   es donde vive el puntaje (bateria homogenea + familia ancla). Idea: al
   adversario se le REVELA, tras cada test, el conteo exacto de activos dentro
   de CADA celda actual (no solo el total del pool testeado). Eso es una
   relajacion de informacion PURA (el adversario observa un superconjunto de lo
   que ve cualquier politica real) => su valor optimo DOMINA a OPT, para
   cualquier conjunto de acciones. Le damos el conjunto de acciones COMPLETO
   (un test toma a_c personas de cada celda, sum a_c <= G; el pool acredita a
   los tomados sii TODOS son limpios). Como los conteos por celda se revelan,
   el estado factoriza en celdas independientes (m, j=#limpios conocido), y por
   intercambiabilidad la reparametrizacion por #limpios lo hace tratable en el
   regimen limpio-raro: las celdas todo-activas (j=0) nunca acreditan y se
   descartan; solo sobreviven <= g celdas con limpios. Se computa exacto para
   #limpios g <= T y se acota la cola g > T por min(g, B*G) (nunca se acredita
   mas que #limpios), lo que mantiene la validez con truncamiento.

   Por que domina a full-reveal-K (revelar solo el total): revelar los conteos
   por celda es MAS informacion que observar solo el total del pool, luego
   U_cell >= U_revealK >= OPT. Verificado: en TODA la bateria U_cell >= V*
   exacto y >= U_revealK. En el ancla insignia baja el techo de 3.2 a ~3.04
   (T=3), gap 2.314 -> 2.152, en segundos.

COLA POR DILUCION DE SENUELOS (m-monotonia). El termino g>T de la mezcla se
acotaba flojo por min(g, B*G) = "acredita TODOS los limpios" (ignora el costo
de busqueda). Lo apretamos con un techo VALIDO y mas barato:

  Lema (dilucion). En el juego de revelacion-de-conteos, agrandar la poblacion
  m de una celda con su numero de limpios j FIJO solo puede BAJAR el valor
  optimo: dp((m+1,j),B) <= dp((m,j),B). Prueba: revelar la identidad de una
  persona ACTIVA extra es un regalo de informacion al adversario maximizador
  (su optimo solo puede subir); una vez que se sabe activa es un senuelo inutil
  que el adversario descarta (incluirla vuelve cualquier pool no-todo-limpio y
  no aporta informacion), luego el juego se reduce EXACTAMENTE a (m,j). Por
  tanto dp((m,j),B) <= dp((m',j),B) para todo j <= m' <= m.

  Corolario: para el termino g del ancla (m=n), dp((n,g),B) <= dp((m',g),B)
  con m' <= n; ademas dp <= #limpios y dp <= B*G, luego el surrogate satisface
  dp((m',g),B) <= min(g, B*G) (la cola floja) => techo estricto y valido. Menos
  senuelos = menos ramificacion, asi que m' < n es tratable donde dp((n,g),B)
  no lo es (dp((32,4),6) exacto = 3.389 cuesta ~90 s > la puerta de 60 s;
  dp((24,4),6) = 3.664 cuesta ~24 s). Aplicamos el surrogate al termino de
  mayor masa*holgura del ancla insignia (g=4, m'=24): techo 3.0436 -> 2.9804,
  gap insignia 2.152 -> 2.086, ancla en ~34 s. La cola g>=5 sigue floja (su
  masa*holgura no paga su costo: el surrogate util pide m' cerca de n, justo
  donde el DP explota — el costo y la calidad estan acoplados via m').

Direccion para apretar mas (sesiones futuras): full-reveal-K (revelar SOLO el
total, no el perfil por celda) es una cota valida ESTRICTAMENTE mas ajustada
que U_cell (a n=32 U_cell ~ 2x mas floja que su version laminar), pero su
estado es correlacionado (no factoriza) y requiere un DP en espacio de conteos
con estados acoplados; ese es el build pendiente hacia el gap ~1. Sub-lever
abierto: subir m' de la cola g=4 (o cubrir g=5,6) exige comprimir el estado
correlacionado del DP (poda de acciones DEMOSTRABLEMENTE no-dominante; ni el
pooling inter-celda ni los splits informativos son dominados) — multi-sesion.
"""

from __future__ import annotations

from math import comb
from functools import lru_cache

# Instancias mayores a esto usan solo U_PI (el DP de conteos explota en n).
# El ancla insignia (n=32) y fam_G8 (n=24) caben; las familias n>=64 caen a
# U_PI (su gap ~0.58 no manda el puntaje, lo manda la insignia).
_CELL_MAX_N = 40
# Corte de #limpios para el DP exacto; la cola g>T se acota por min(g,B*G).
_CELL_GG_CAP = 3
# Debajo de este n se computan TODOS los #limpios (bateria exacta, domina V*).
_CELL_EXACT_N = 14
# Cola por dilucion de senuelos (m-monotonia, ver docstring): para g>T se
# reemplaza min(g,B*G) por dp((min(n,m'),g),B), un techo valido y mas barato.
# Solo el termino de mayor masa*holgura (g=4) paga su costo; el resto queda
# flojo. m'=24 mantiene el ancla insignia bajo la puerta de 60 s.
_CELL_TAIL_MCAP = {4: 24}


def _u_pi(p, u, B, G):
    """U_PI exacta y escalable: E_Z[top min(B*G, .) utilidades limpias]."""
    n = len(p)
    if n == 0 or B <= 0 or G <= 0:
        return 0.0
    cap = min(B * G, n)
    order = sorted(range(n), key=lambda i: u[i], reverse=True)
    dist = [0.0] * cap
    dist[0] = 1.0
    total = 0.0
    for i in order:
        qi = 1.0 - p[i]
        total += u[i] * qi * sum(dist)
        for j in range(cap - 1, 0, -1):
            dist[j] = dist[j] * (1.0 - qi) + dist[j - 1] * qi
        dist[0] *= 1.0 - qi
    return total


def _u_cell(pa, w, n, B, G):
    """Relajacion de conteos-por-celda para instancia homogenea (p_act=pa,
    u=w, n personas, B tests, pools <= G). Devuelve la cota en unidades de
    utilidad (reward = w por persona limpia acreditada). Vale como cota
    superior de OPT (relajacion de informacion pura + conjunto de acciones
    completo). Ver docstring del modulo."""
    pg = 1.0 - pa  # prob. de estar limpio

    @lru_cache(maxsize=None)
    def hyp(m, j, a):
        # P(gs limpios en un corte de tamano a de una celda (m, j)),
        # gs ~ Hipergeometrica(m, j, a).
        denom = comb(m, a)
        return tuple(
            (gs, comb(j, gs) * comb(m - j, a - gs) / denom)
            for gs in range(max(0, a - (m - j)), min(j, a) + 1)
        )

    @lru_cache(maxsize=None)
    def dp(cells, t):
        # cells: tupla ordenada de (m, j) con j>=1 (celdas todo-activas
        # descartadas: nunca acreditan, no interactuan). t tests restantes.
        # Devuelve E[#limpios acreditados] adicional (en unidades de persona).
        if t == 0 or not cells:
            return 0.0
        best = 0.0  # opcion: detenerse
        ncells = len(cells)

        def evaluate(assign):
            # assign[c] = a_c personas tomadas de la celda c (sum in [1,G]).
            # Ramifica sobre los conteos limpios por celda (independientes);
            # acredita sum(a) sii TODOS los tomados son limpios.
            nonlocal best
            per = []
            for c, a in enumerate(assign):
                m, j = cells[c]
                if a == 0:
                    per.append(((0, 1.0, None, (m, j)),))
                else:
                    outs = []
                    for gs, pr in hyp(m, j, a):
                        taken = (a, gs)
                        rem = (m - a, j - gs) if (m - a > 0 and j - gs > 0) else None
                        outs.append((gs, pr, taken, rem))
                    per.append(tuple(outs))
            total = 0.0
            taken_total = sum(assign)

            def go(c, prob, newcells, allgood):
                nonlocal total
                if c == ncells:
                    nc = tuple(sorted(newcells))
                    reward = taken_total if allgood else 0.0
                    total += prob * (reward + dp(nc, t - 1))
                    return
                a = assign[c]
                for (gs, pr, taken, rem) in per[c]:
                    if pr <= 0.0:
                        continue
                    add = []
                    if a == 0:
                        add.append(rem)  # celda intacta (m,j)
                        ag = allgood
                    else:
                        ag = allgood and (gs == a)
                        if not ag and gs > 0:
                            add.append(taken)  # parte tomada con limpios, no acreditada
                        if rem is not None:
                            add.append(rem)
                    go(c + 1, prob * pr,
                       newcells + [x for x in add if x is not None], ag)

            go(0, 1.0, [], True)
            if total > best:
                best = total

        # Enumera acciones: a_c en [0, m_c] para cada celda, 1 <= sum <= G.
        assign = []

        def gen(idx, budget):
            if idx == ncells:
                if sum(assign) >= 1:
                    evaluate(tuple(assign))
                return
            m = cells[idx][0]
            for a in range(0, min(m, budget) + 1):
                assign.append(a)
                gen(idx + 1, budget - a)
                assign.pop()

        gen(0, G)
        return best

    cap = B * G
    exact = n <= _CELL_EXACT_N
    T = n if exact else _CELL_GG_CAP
    total = 0.0
    for gg in range(n + 1):
        wg = comb(n, gg) * (pg ** gg) * (pa ** (n - gg))
        if wg <= 0.0:
            continue
        if gg > T:
            # Cola: min(g,B*G) es valido (acreditados <= #limpios). Si hay un
            # m' en el schedule, dp((min(n,m'),g),B) es un techo VALIDO y mas
            # ajustado por m-monotonia (dp((n,g),B) <= dp((m',g),B) <= min(g,cap)
            # para m'<=n; ver docstring). Solo se activa donde es barato.
            mcap = _CELL_TAIL_MCAP.get(gg)
            if mcap is not None and gg <= mcap:
                total += wg * dp(((min(n, mcap), gg),), B)
            else:
                total += wg * min(gg, cap)
        else:
            total += wg * dp(((n, gg),), B)
    return w * total


def _homogeneous(p, u):
    p0, u0 = p[0], u[0]
    for pi, ui in zip(p, u):
        if abs(pi - p0) > 1e-12 or abs(ui - u0) > 1e-12:
            return False
    return True


def upper_bound(p, u, B, G):
    """min(U_PI, U_cell) — ambas cotas superiores validas del optimo de conteo.

    U_PI vale siempre y a cualquier escala. U_cell (relajacion de
    conteos-por-celda) solo se usa en instancias HOMOGENEAS tratables (n<=40),
    donde aprieta el techo del ancla insignia; fuera de eso se devuelve U_PI.
    """
    n = len(p)
    if n == 0 or B <= 0 or G <= 0:
        return 0.0
    u_pi = _u_pi(p, u, B, G)
    if n <= _CELL_MAX_N and u[0] >= 0.0 and _homogeneous(p, u):
        u_cell = _u_cell(p[0], u[0], n, B, G)
        if u_cell < u_pi:
            return u_cell
    return u_pi
