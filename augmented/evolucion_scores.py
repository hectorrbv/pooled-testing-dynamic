"""Busqueda evolutiva de scores V-hat con juez exacto (estilo FunSearch).

La idea: un LLM (claude-opus-5) propone funciones de score candidatas
`score(ctx)`; cada candidata induce una politica golosa (argmax del score
sobre el menu laminar) que se evalua EXACTA por enumeracion (J^pi, ec. 10.4
del companion) contra el optimo laminar exacto del solver B-M17. Los mejores
candidatos y sus instancias de fallo vuelven al LLM como feedback; el loop
itera. Sin Monte Carlo: el fitness es el ratio exacto V^pi / V*.

El presupuesto entra por diseno: el ctx expone b (restante), B (total) y
primitivas de costo-proxy, y la regla de paro (score <= 0 en todo el menu)
permite codificar la tijera. La evolucion decide si el buen score es V, V/C,
V/b o cualquier otra combinacion — esa es la pregunta que se explora.

Estatuto (§25 / §14.8 candidata E): DIAGNOSTICO. Ningun score descubierto se
adopta como candidata S3 sin pasar por G4a/G4b; los resultados se reportan
con etiqueta de origen ("descubierto por busqueda, sin garantia").

Convencion: posterior-zero (G0), R = conteo de infectados, q = prob de sano.

Uso:
  python -m augmented.evolucion_scores                # dry-run: semillas, sin API
  python -m augmented.evolucion_scores --evolucionar 6   # loop con claude-opus-5
Credenciales: perfil de `ant auth login` o ANTHROPIC_API_KEY (el cliente
zero-arg resuelve solo; no se pide key en codigo).
"""

import argparse
import json
import math
import re
import signal
import time
from fractions import Fraction
from pathlib import Path

from augmented.bm17_toy_solver import SolverLaminar, z_tabla

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / 'results' / 'evolucion_scores.json'

# ---------------------------------------------------------------- instancias
TRAIN = [(n, B, G, q)
         for n in (3, 4, 5) for B in (2, 3) for G in (2, 3)
         for q in (0.15, 0.30, 0.45, 0.70)]
HELDOUT = [(6, 3, 2, 0.20), (6, 3, 3, 0.20), (6, 3, 2, 0.60), (6, 3, 3, 0.60)]


def _instancia(n, q):
    p = {i: Fraction(1) - Fraction(q).limit_denominator(100) for i in range(n)}
    u = {i: Fraction(1) for i in range(n)}
    return p, u


def optimo(n, B, G, q, _cache={}):
    clave = (n, B, G, q)
    if clave not in _cache:
        p, u = _instancia(n, q)
        sol = SolverLaminar(p, u, G, 'posterior_zero')
        _cache[clave] = float(sol.V(frozenset(range(n)), (), B))
    return _cache[clave]


# ------------------------------------------------- politica inducida (J^pi)
class PoliticaInducida(SolverLaminar):
    """Politica golosa inducida por un score arbitrario, evaluada exacta.

    En cada estado elige argmax de score(ctx) sobre el menu laminar (empate:
    primera en orden de generacion). Si todos los scores son <= 0, se detiene
    (la regla de paro permite codificar la tijera). El valor es la recursion
    forward J^pi con las mismas transiciones del solver.
    """

    def __init__(self, p, u, G, score_fn, B_total):
        super().__init__(p, u, G, 'posterior_zero')
        self.score_fn = score_fn
        self.B_total = B_total

    def _ctx(self, U, atomos, b, accion):
        if accion[0] == 'open':
            S = accion[1]
            zs = self._z(S)
            p_limpio = float(zs[0])
            p_muerto = float(zs[-1])
            ps = tuple(float(1 - self.p[i]) for i in S)
            atomo_tam, atomo_r = 0, 0
        else:
            _, (A, r), S = accion
            resto = tuple(sorted(set(A) - set(S)))
            zS, zR, zA = self._z(S), self._z(resto), self._z(A)
            den = float(zA[r])
            p_limpio = float(zS[0] * zR[r]) / den if r < len(zR) else 0.0
            smax = min(len(S), r)
            p_muerto = (float(zS[smax] * zR[r - smax]) / den
                        if smax == len(S) and 0 <= r - smax < len(zR) else 0.0)
            ps = []
            for i in S:
                sin_i = tuple(x for x in A if x != i)
                z_sin = z_tabla(sin_i, self.p)
                ps.append(float(z_sin[r] * (1 - self.p[i])) / den
                          if r < len(z_sin) else 0.0)
            ps = tuple(ps)
            atomo_tam, atomo_r = len(A), r
        u_S = float(sum(self.u[i] for i in S))
        v_mag = float(sum(pi * float(self.u[i]) for pi, i in zip(ps, S)))
        return {
            'tipo': accion[0], 'tam': len(S), 'b': b, 'B': self.B_total,
            'G': self.G, 'n': len(self.p),
            'u_S': u_S, 'v_magico': v_mag, 'e_sanos': sum(ps),
            'p_limpio': p_limpio, 'p_muerto': p_muerto, 'p_sano': ps,
            'atomo_tam': atomo_tam, 'atomo_r': atomo_r,
            'virgenes': len(U), 'atomos_abiertos': len(atomos),
        }

    def V(self, U, atomos, b):
        U, atomos = self._canoniza(U, atomos)
        clave = (U, atomos, b)
        if clave in self.memo:
            return self.memo[clave]
        valor = Fraction(0)
        if b > 0:
            mejor, mejor_s = None, 0.0
            for accion in self._acciones(U, atomos):
                s = float(self.score_fn(self._ctx(U, atomos, b, accion)))
                if s > mejor_s + 1e-12:
                    mejor, mejor_s = accion, s
            if mejor is not None:
                valor = self._q_accion(U, atomos, b, mejor)
        self.memo[clave] = valor
        return valor


# ---------------------------------------------------------------- sandbox
_BUILTINS = {'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len,
             'sorted': sorted, 'range': range, 'enumerate': enumerate,
             'float': float, 'int': int, 'zip': zip}


def compila(codigo):
    """Compila el codigo de un candidato en un entorno restringido."""
    entorno = {'__builtins__': _BUILTINS, 'math': math}
    exec(codigo, entorno)
    fn = entorno.get('score')
    if not callable(fn):
        raise ValueError('el candidato no define score(ctx)')
    return fn


class _Timeout(Exception):
    pass


def evalua(codigo, instancias, plazo_s=90):
    """Fitness exacto de un candidato: (media, peor, detalle) o None si falla."""
    try:
        fn = compila(codigo)
    except Exception:
        return None
    viejo = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(plazo_s)
    try:
        detalle = []
        for (n, B, G, q) in instancias:
            p, u = _instancia(n, q)
            pol = PoliticaInducida(p, u, G, fn, B)
            v = float(pol.V(frozenset(range(n)), (), B))
            detalle.append(((n, B, G, q), v / optimo(n, B, G, q)))
        ratios = [r for _, r in detalle]
        return (sum(ratios) / len(ratios), min(ratios), detalle)
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, viejo)


# ---------------------------------------------------------------- semillas
SEMILLAS = {
    's0_inmediato': "def score(ctx):\n    return ctx['p_limpio'] * ctx['u_S']\n",
    'presupuesto_magico': "def score(ctx):\n    return ctx['v_magico']\n",
    'densidad_magica': "def score(ctx):\n    return ctx['v_magico'] / ctx['tam']\n",
    'valor_por_biseccion': (
        "def score(ctx):\n"
        "    c = 1 + (math.ceil(math.log2(ctx['tam'])) if ctx['tam'] > 1 else 0)\n"
        "    if c > ctx['b']:\n"
        "        return 0.0\n"
        "    return ctx['v_magico'] / c\n"),
}


# ---------------------------------------------------------------- evolucion
PROMPT_BASE = """Eres parte de un buscador evolutivo de funciones de score para \
pooled testing dinamico aumentado con politicas laminares (posterior-zero).

Una politica golosa elige en cada paso la accion (pool virgen o refinamiento \
de un atomo) que maximiza score(ctx); si todos los scores son <= 0, se \
detiene. Se evalua EXACTA contra el optimo laminar; fitness = ratio medio y \
peor ratio sobre una malla de instancias (n<=5, B<=3, G<=3, q en 0.15-0.70).

ctx es un dict con: tipo ('open'|'ref'), tam (|S|), b (pruebas restantes), \
B (presupuesto total), G, n, u_S (utilidad del pool), v_magico (utilidad \
esperada extraible con pruebas gratis), e_sanos, p_limpio (prob de acreditar \
todo el pool ya), p_muerto (prob de que todo salga infectado), p_sano (tupla \
por miembro), atomo_tam, atomo_r (0 si es 'open'), virgenes, atomos_abiertos. \
Puedes usar math. Sin imports, sin estado global, sin recursion.

La leccion conocida: el score sin costo (v_magico) se atasca; el costo de \
cobrar un pool grande es ~1+log2(tam) pruebas de biseccion; el presupuesto \
restante b decide si una promesa es cobrable (tijera). Ningun exponente fijo \
domina en todos los regimenes.

Poblacion actual (codigo, ratio medio, peor ratio, peores instancias):
{poblacion}

Propon exactamente {m} funciones NUEVAS y diversas (no repitas la poblacion; \
explora combinaciones de promesa, costo y presupuesto que la poblacion no \
cubre). Devuelve cada una en su propio bloque ```python con la firma \
def score(ctx): y nada mas."""


def _formatea_poblacion(poblacion):
    filas = []
    for nombre, codigo, (media, peor, detalle) in poblacion:
        peores = sorted(detalle, key=lambda x: x[1])[:3]
        peores_txt = ', '.join(f'{i}:{r:.3f}' for i, r in peores)
        filas.append(f'--- {nombre} (media {media:.4f}, peor {peor:.4f}; '
                     f'peores: {peores_txt})\n```python\n{codigo}```')
    return '\n'.join(filas)


def evoluciona(generaciones, m_por_gen=4, k_poblacion=4):
    import anthropic
    client = anthropic.Anthropic()
    poblacion = []
    for nombre, codigo in SEMILLAS.items():
        fit = evalua(codigo, TRAIN)
        if fit:
            poblacion.append((nombre, codigo, fit))
    poblacion.sort(key=lambda x: (x[2][0], x[2][1]), reverse=True)
    historial = []
    for gen in range(1, generaciones + 1):
        prompt = PROMPT_BASE.format(
            poblacion=_formatea_poblacion(poblacion[:k_poblacion]), m=m_por_gen)
        with client.messages.stream(
            model='claude-opus-5', max_tokens=16000,
            messages=[{'role': 'user', 'content': prompt}],
        ) as stream:
            texto = ''.join(b.text for b in stream.get_final_message().content
                            if b.type == 'text')
        candidatos = re.findall(r'```python\n(.*?)```', texto, re.DOTALL)
        for j, codigo in enumerate(candidatos):
            fit = evalua(codigo, TRAIN)
            nombre = f'gen{gen}_c{j}'
            if fit:
                poblacion.append((nombre, codigo, fit))
                print(f'  {nombre}: media {fit[0]:.4f}, peor {fit[1]:.4f}')
            else:
                print(f'  {nombre}: descalificado')
        poblacion.sort(key=lambda x: (x[2][0], x[2][1]), reverse=True)
        poblacion = poblacion[:max(k_poblacion, 6)]
        historial.append({'generacion': gen,
                          'mejor': poblacion[0][0],
                          'media': poblacion[0][2][0],
                          'peor': poblacion[0][2][1]})
        print(f'gen {gen}: mejor = {poblacion[0][0]} '
              f'(media {poblacion[0][2][0]:.4f})')
    _guarda(poblacion, historial)
    return poblacion


def _guarda(poblacion, historial):
    SALIDA.parent.mkdir(exist_ok=True)
    datos = {
        'estatuto': 'diagnostico (§14.8 candidata E); adopcion via G4a/G4b',
        'convencion': 'posterior_zero; R = infectados; juez = SolverLaminar',
        'fecha': time.strftime('%Y-%m-%d %H:%M'),
        'train': TRAIN, 'heldout': HELDOUT,
        'historial': historial,
        'poblacion': [{
            'nombre': n, 'codigo': c,
            'media_train': f[0], 'peor_train': f[1],
            'heldout': (lambda h: {'media': h[0], 'peor': h[1]} if h else None)(
                evalua(c, HELDOUT)),
        } for n, c, f in poblacion],
    }
    SALIDA.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + '\n')
    print(f'guardado: {SALIDA}')


# ---------------------------------------------------------------- dry-run
def dry_run():
    print(f'juez exacto listo: {len(TRAIN)} instancias de entrenamiento, '
          f'{len(HELDOUT)} de held-out\n')
    poblacion = []
    for nombre, codigo in SEMILLAS.items():
        t0 = time.time()
        fit = evalua(codigo, TRAIN)
        assert fit is not None, f'semilla {nombre} fallo'
        media, peor, _ = fit
        hh = evalua(codigo, HELDOUT)
        print(f'{nombre:22s} media {media:.4f}  peor {peor:.4f}  '
              f'held-out {hh[0]:.4f}  [{time.time()-t0:.1f}s]')
        poblacion.append((nombre, codigo, fit))
    # sanity: ningun ratio > 1 (el juez es el optimo exacto)
    for _, _, (media, peor, detalle) in poblacion:
        assert all(r <= 1 + 1e-9 for _, r in detalle)
    # sanity: el magico puro carga la patologia conocida — buen promedio,
    # peor caso catastrofico (la no-reentrada); las demas semillas no caen ahi.
    d = {n: f for n, _, f in poblacion}
    assert d['presupuesto_magico'][1] < min(
        d['s0_inmediato'][1], d['densidad_magica'][1])
    # nota: con G <= 3, densidad y biseccion coinciden (1+ceil(log2 t) = t);
    # se separan recien en G >= 4 — consistente con Prop 9.1 del companion.
    print('\nOK: ratios <= 1 y el magico puro exhibe su peor-caso patologico;')
    print('la brecha media-vs-peor es exactamente el espacio que la evolucion '
          'debe cerrar')
    print('para lanzar la evolucion: python -m augmented.evolucion_scores '
          '--evolucionar 6  (requiere credencial: `ant auth login` o '
          'ANTHROPIC_API_KEY)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evolucionar', type=int, default=0,
                    help='numero de generaciones con claude-opus-5')
    args = ap.parse_args()
    if args.evolucionar:
        evoluciona(args.evolucionar)
    else:
        dry_run()
