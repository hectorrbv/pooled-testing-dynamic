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
  NO sustituye el argumento de validez. Un keep sin argumento de por que la
  construccion domina a toda politica no es un keep.

Semilla: U_PI, la cota de informacion perfecta (hindsight), exacta a cualquier
escala. Con el perfil Z revelado, lo mejor posible es acreditar las top
min(B*G, #limpios) utilidades limpias, asi que

    OPT <= E_Z[ suma de las top B*G utilidades limpias ] = U_PI.

Se computa exacto en O(n * min(B*G, n)) sin enumerar 2^n: ordenando por
utilidad descendente, la persona i entra al top exactamente cuando esta limpia
y hay menos de B*G limpios antes que ella en ese orden, y el numero de limpios
previos es Poisson-binomial (recursion por prefijos). Coincide con
`certificates.u_pi_exact` en n chico (misma cota, otra evaluacion).

Direccion de la mision: reemplazar/apretar esta semilla con una cota
penalizada laminar-descomponible (Brown-Smith-Sun con V-hat que solo use
primitivas laminares escalables), manteniendo validez demostrable y costo
polinomial.
"""

from __future__ import annotations


def upper_bound(p, u, B, G):
    """U_PI exacta y escalable: E_Z[top min(B*G, .) utilidades limpias]."""
    n = len(p)
    if n == 0 or B <= 0 or G <= 0:
        return 0.0
    cap = min(B * G, n)
    order = sorted(range(n), key=lambda i: u[i], reverse=True)
    # dist[j] = P(exactamente j limpios entre los ya procesados), j < cap.
    # La masa que alcanza cap se descarta: con cap o mas limpios previos, la
    # persona actual ya no entra al top aunque este limpia.
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
