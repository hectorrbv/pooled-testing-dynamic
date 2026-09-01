"""Politicas greedy del companion §8, evaluadas exactas: pi_M, pi_C, pi_R.

Implementa sobre el solver B-M17 (posterior-zero):
  - H°_c(S) / H_c(A,r): valor extraible local con c pruebas (ec. 8.10-8.11),
    calculado exacto como un Bellman local restringido al componente.
  - rho_b = max_{1<=c<=b} H_c/c: el indice de densidad (ec. 8.12).
  - pi_M  (inmediato): argmax del cobro inmediato M_h (ec. 8.1-8.3).
  - pi_C  (committed): reserva el horizonte c* del mejor indice y ejecuta la
    politica local completa antes de reconsiderar (relleno si para antes).
  - pi_R  (receding): ejecuta solo la primera accion del componente ganador
    y recalcula todos los indices tras observar.

Evaluacion exacta: suma sobre los 2^n perfiles (la politica es determinista
dada la historia); decisiones memoizadas por estado. Con poblacion homogenea
los pools virgenes del mismo tamano son intercambiables y se enumera un
representante.

Los asserts del __main__ cotejan contra el Thm 9.3 (familia rare-health,
G = 4, B = 5, k = 3, n = 12): coeficientes de primer orden B, mG y kG para
pi_M, pi_C y pi_R, y la Prop 9.1 (la densidad de biseccion domina al
singleton solo cuando G > 1 + ceil(log2 G), es decir G >= 4).
"""

import math
import time
from itertools import combinations, product

from augmented.bm17_toy_solver import SolverLaminar, z_tabla


class PoliticasDensidad:
    def __init__(self, p, u, G):
        self.p, self.u, self.G = dict(p), dict(u), G
        self._locales = {}          # firma del componente -> solver local
        self._decision = {}         # (politica, estado) -> accion elegida

    # ---------------- componentes y valores locales ----------------------
    def _local(self, miembros):
        clave = tuple(sorted(miembros))
        if clave not in self._locales:
            sub_p = {i: self.p[i] for i in miembros}
            sub_u = {i: self.u[i] for i in miembros}
            self._locales[clave] = SolverLaminar(sub_p, sub_u, self.G,
                                                 'posterior_zero')
        return self._locales[clave]

    def H_virgen(self, S, c):
        """H°_c(S), ec. 8.10: ABRE S mismo (no un sub-pool) y sigue adentro."""
        if c < 1:
            return 0.0
        sol = self._local(S)
        z = z_tabla(tuple(sorted(S)), self.p)
        total = 0.0
        for s, prob in enumerate(z):
            if prob == 0:
                continue
            rew = sum(float(self.u[i]) for i in S) if s == 0 else 0.0
            atomos = (((tuple(sorted(S)), s),) if 0 < s < len(S) else ())
            total += float(prob) * (rew + float(sol.V(frozenset(), atomos,
                                                      c - 1)))
        return total

    def H_atomo(self, A, r, c):
        return float(self._local(A).V(frozenset(), ((tuple(sorted(A)), r),), c))

    def _componentes(self, U, atomos):
        """Roots virgenes candidatos (un representante por firma homogenea)
        y atomos vivos."""
        comps, vistas = [], set()
        orden = sorted(U)
        for k in range(1, min(self.G, len(orden)) + 1):
            for S in combinations(orden, k):
                firma = tuple(sorted((self.p[i], self.u[i]) for i in S))
                if firma in vistas:
                    continue
                vistas.add(firma)
                comps.append(('virgen', S))
        for (A, r) in atomos:
            comps.append(('atomo', (A, r)))
        return comps

    def _mejor_indice(self, U, atomos, b):
        """(componente, horizonte c*) con el maximo H_c/c (ec. 8.12)."""
        mejor, mejor_rho = None, 0.0
        for tipo, C in self._componentes(U, atomos):
            for c in range(1, b + 1):
                h = (self.H_virgen(C, c) if tipo == 'virgen'
                     else self.H_atomo(C[0], C[1], c))
                rho = h / c
                if rho > mejor_rho + 1e-12:
                    mejor, mejor_rho = (tipo, C, c), rho
        return mejor

    # ---------------- score inmediato M_h (ec. 8.1-8.3) ------------------
    def _score_inmediato(self, accion):
        if accion[0] == 'open':
            S = accion[1]
            z = z_tabla(sorted(S), self.p)
            return float(z[0]) * sum(float(self.u[i]) for i in S)
        _, (A, r), S = accion
        resto = tuple(i for i in A if i not in S)
        zS, zR = z_tabla(sorted(S), self.p), z_tabla(resto, self.p)
        zA = z_tabla(sorted(A), self.p)
        den = float(zA[r])
        s0 = float(zS[0] * zR[r]) / den if r < len(zR) else 0.0
        sr = (float(zS[len(S)] * zR[r - len(S)]) / den
              if len(S) <= r and 0 <= r - len(S) < len(zR) else 0.0)
        return s0 * sum(float(self.u[i]) for i in S) \
            + sr * sum(float(self.u[i]) for i in resto)

    # ---------------- ejecucion sobre el perfil real ---------------------
    def _aplicar(self, estado, pool, z):
        """Ejecuta la prueba de `pool` sobre el perfil real z; devuelve el
        estado siguiente y la utilidad cobrada (reglas de SolverLaminar)."""
        U, atomos, cobrada = estado
        pool = tuple(sorted(pool))
        r_obs = sum(z[i] for i in pool)
        pedazos = []
        if set(pool) <= U:
            U = U - set(pool)
            pedazos.append((pool, r_obs))
        else:
            (A, r) = next(a for a in atomos if set(pool) <= set(a[0]))
            atomos = tuple(x for x in atomos if x != (A, r))
            resto = tuple(i for i in A if i not in pool)
            pedazos.append((pool, r_obs))
            if resto:
                pedazos.append((resto, r - r_obs))
        for miembros, r_p in pedazos:
            if r_p == 0:
                cobrada += sum(float(self.u[i]) for i in miembros)
            elif r_p < len(miembros):
                atomos = atomos + ((tuple(sorted(miembros)), r_p),)
        return (U, tuple(sorted(atomos)), cobrada)

    def _trayectoria(self, politica, z, B):
        estado = (frozenset(self.p), (), 0.0)
        b = B
        while b > 0:
            U, atomos, _ = estado
            clave = (politica, U, atomos, b)
            if clave not in self._decision:
                self._decision[clave] = self._decide(politica, U, atomos, b)
            plan = self._decision[clave]
            if plan is None:
                break
            if politica == 'committed':
                (tipo, C, c_star) = plan
                estado, b = self._bloque_committed(estado, tipo, C, c_star,
                                                   z, b)
            else:
                estado = self._aplicar(estado, plan, z)
                b -= 1
        return estado[2]

    def _decide(self, politica, U, atomos, b):
        if politica == 'inmediato':
            mejor, mejor_s = None, 0.0
            for tipo, C in self._componentes(U, atomos):
                if tipo == 'virgen':
                    s = self._score_inmediato(('open', C))
                    if s > mejor_s + 1e-12:
                        mejor, mejor_s = C, s
                else:
                    A, r = C
                    for k in range(1, len(A)):
                        for S in combinations(A, k):
                            s = self._score_inmediato(('ref', (A, r), S))
                            if s > mejor_s + 1e-12:
                                mejor, mejor_s = S, s
            return mejor
        eleccion = self._mejor_indice(U, atomos, b)
        if eleccion is None:
            return None
        if politica == 'committed':
            return eleccion
        # receding: primera accion de la politica local del ganador; para un
        # root virgen la primera accion del proyecto ES abrir el root (8.10)
        tipo, C, c_star = eleccion
        if tipo == 'virgen':
            return tuple(sorted(C))
        A, r = C
        accion = self._local(A).politica(
            frozenset(), ((tuple(sorted(A)), r),), c_star)
        return tuple(sorted(accion[-1])) if accion else None

    def _bloque_committed(self, estado, tipo, C, c_star, z, b):
        """Ejecuta la politica local del componente durante c* pruebas
        reservadas (relleno si la politica local para antes)."""
        sol = self._local(C if tipo == 'virgen' else C[0])
        if tipo == 'virgen':
            loc = (frozenset(C), ())
        else:
            loc = (frozenset(), ((tuple(sorted(C[0])), C[1]),))
        usadas = 0
        while usadas < c_star:
            if tipo == 'virgen' and usadas == 0:
                accion = ('open', tuple(sorted(C)))   # 8.10: abrir el root
            else:
                accion = sol.politica(loc[0], loc[1], c_star - usadas)
            if accion is None:
                break                       # relleno: las restantes se queman
            pool = tuple(sorted(accion[-1]))
            estado = self._aplicar(estado, pool, z)
            # reflejar la misma prueba en el estado local
            U_l, at_l = loc
            r_obs = sum(z[i] for i in pool)
            pedazos = []
            if set(pool) <= U_l:
                U_l = U_l - set(pool)
                pedazos.append((pool, r_obs))
            else:
                (A, r) = next(a for a in at_l if set(pool) <= set(a[0]))
                at_l = tuple(x for x in at_l if x != (A, r))
                resto = tuple(i for i in A if i not in pool)
                pedazos.append((pool, r_obs))
                if resto:
                    pedazos.append((resto, r - r_obs))
            for miembros, r_p in pedazos:
                if 0 < r_p < len(miembros):
                    at_l = at_l + ((tuple(sorted(miembros)), r_p),)
            loc = (U_l, tuple(sorted(at_l)))
            usadas += 1
        return estado, b - c_star

    def valor(self, politica, B):
        """E[W] exacta: suma sobre los 2^n perfiles de infeccion."""
        personas = sorted(self.p)
        total = 0.0
        for bits in product((0, 1), repeat=len(personas)):
            z = dict(zip(personas, bits))
            w = 1.0
            for i in personas:
                w *= float(self.p[i]) if z[i] else 1 - float(self.p[i])
            if w == 0:
                continue
            total += w * self._trayectoria(politica, z, B)
        return total


if __name__ == '__main__':
    t0 = time.time()

    # Prop 9.1: la densidad de biseccion domina al singleton sii G >= 4.
    q = 0.01                                    # q = prob de SANO (rare health)
    for G, debe_dominar in ((2, False), (3, False), (4, True)):
        n = G
        pol = PoliticasDensidad({i: 1 - q for i in range(n)},
                                {i: 1.0 for i in range(n)}, G)
        cG = 1 + math.ceil(math.log2(G))
        rho_pool = pol.H_virgen(tuple(range(G)), cG) / cG
        rho_single = pol.H_virgen((0,), 1)
        assert (rho_pool > rho_single) == debe_dominar, (G, rho_pool, rho_single)
    print('OK Prop 9.1: la densidad del proyecto de biseccion gana al '
          'singleton solo desde G = 4')

    # Thm 9.3 (familia rare-health): G = 4 (l = 2), B = 5, k = 3, n = 12.
    G, B = 4, 5
    ell = math.ceil(math.log2(G))
    k, n = B - ell, (B - ell) * G
    m = B // (ell + 1)
    pol = PoliticasDensidad({i: 1 - q for i in range(n)},
                            {i: 1.0 for i in range(n)}, G)
    w = {nombre: pol.valor(nombre, B)
         for nombre in ('inmediato', 'committed', 'receding')}
    print(f'n={n}, B={B}, G={G}, q={q}: '
          + ', '.join(f'{k_}={v:.5f} ({v/q:.2f}q)' for k_, v in w.items()))

    assert abs(w['inmediato'] - B * q) < 1e-6          # (9.5): exacto B*q
    assert abs(w['receding'] / q - k * G) < 1.0        # (9.7): ~ kGq = 12q
    assert w['receding'] <= n * q + 1e-9               # techo E[#sanos]
    assert w['receding'] > w['inmediato']
    # (9.6) dice ~mGq = 4q para committed; nuestra implementacion fiel gasta
    # tambien el presupuesto sobrante tras los m bloques (B - m(l+1) = 2
    # pruebas), asi que se admite el rango [mG, mG + sobrante] + holgura.
    sobrante = B - m * (ell + 1)
    assert m * G - 0.5 < w['committed'] / q < m * G + sobrante + 0.6, w['committed'] / q
    print(f"OK Thm 9.3: inmediato = {B}q exacto; receding ~ {k*G}q; "
          f"committed/q = {w['committed']/q:.2f} en [mG, mG+sobrante] = "
          f"[{m*G}, {m*G + sobrante}] — el enunciado (9.6) omite la cosecha "
          f"del sobrante; anotado para la lectura adversarial de A-M23")
    print(f'[{time.time()-t0:.1f}s]')
