"""El tensor condicional de subpools: la maquinaria del greedy laminar.

Pregunta que resuelve este módulo
---------------------------------
Probaste un pool ``T`` de ``m`` personas y salió el conteo exacto ``r``.
Ahora quieres saber, para *cada* subconjunto ``T'`` de ese pool y *cada*
resultado ``k`` posible, la probabilidad

    Q[T'][k] = P( conteo(T') = k  |  conteo(T) = r ).

Con esa tabla decides qué probar después: la fila ``k=0`` es la probabilidad
de que el subconjunto salga limpio, que es cuando se cobra la utilidad.

Convención de coordenadas
-------------------------
El pool se representa por sus priors *locales*: ``p`` es una lista de largo
``m``, donde ``p[i]`` es la probabilidad de que la persona ``i`` del pool sea
positiva.  Un subconjunto es una máscara de bits ``s`` en ``[0, 2**m)``: el
bit ``i`` prendido significa "la persona ``i`` está en el subconjunto".

Es deliberado que este módulo NO use los índices globales de la población.
Una vez que estás dentro de un pool, quién sea la persona 37 del mundo es
ruido; lo único que importa es su prior.  Un solo sistema de coordenadas se
lee mucho mejor que dos.

La identidad que lo hace todo
-----------------------------
Como el prior es un producto (las personas son independientes *antes* de
observar), para cualquier ``S`` dentro del pool::

                        Φ[S][k] · Φ[T∖S][r−k]
    P(conteo(S)=k | ·) = ─────────────────────
                              Φ[T][r]

donde ``Φ[S]`` es la pmf Poisson-binomial del bloque ``S``: la distribución
de "cuántos positivos hay en S" bajo el prior, *sin condicionar en nada*.

Leer esa fórmula con cuidado da la lección central del módulo: **condicionar
no toca los bloques**.  Solo reponderá una familia de pmf que no depende de
ninguna observación.  Por eso el objeto que conviene guardar es esa familia
--- la caché ``Φ`` --- y no la tabla condicional, que cambia con cada ``r``.

Y de ahí sale gratis la respuesta a la pregunta de la sesión del 27-jul: al
partir el pool en dos átomos, todo bloque de un hijo ya es un bloque del
padre, así que los hijos **no cuestan ni una convolución nueva**.

Ejemplo de bolsillo
-------------------
::

    >>> p = [0.2, 0.4, 0.6, 0.8]
    >>> Q = subpool_tensor(p, r=2)      # observamos 2 positivos entre los 4
    >>> Q[0b0011].round(4)              # el subconjunto {0, 1}
    array([0.5353, 0.4498, 0.0149])
    >>> Q[0b0111][0]                    # {0,1,2} limpio: imposible
    0.0

Esa última celda es exactamente cero, y no por redondeo: si hay 2 positivos
entre 4 personas, no caben en la única persona que queda fuera de {0,1,2}.
Multiplicar marginales independientes le daría masa positiva y mentiría.
"""

from numbers import Integral

import numpy as np


__all__ = [
    "subset_pmf_cache",
    "subpool_tensor",
    "subpool_tensor_brute",
    "split_after_test",
    "Atom",
]


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------

def _validated_priors(p):
    """Los priors del pool, como arreglo, o ``ValueError`` explicando por qué no."""

    priors = np.asarray(p, dtype=float)
    if priors.ndim != 1:
        raise ValueError("los priors del pool deben ser una secuencia 1-D")
    if priors.size == 0:
        raise ValueError("el pool está vacío")
    if not np.all(np.isfinite(priors)):
        raise ValueError("todo prior debe ser finito")
    if np.any((priors < 0.0) | (priors > 1.0)):
        raise ValueError("todo prior debe estar en [0, 1]")
    return priors


def _validated_count(r, m, name="el conteo"):
    if isinstance(r, bool) or not isinstance(r, Integral):
        raise ValueError(f"{name} debe ser entero")
    r = int(r)
    if not 0 <= r <= m:
        raise ValueError(f"{name} está fuera del rango [0, {m}]")
    return r


# --------------------------------------------------------------------------
# Φ: la caché de bloques.  El objeto reusable.
# --------------------------------------------------------------------------

def subset_pmf_cache(p):
    """Φ[s] = pmf del conteo de positivos del bloque ``s``, bajo el prior.

    Devuelve una tupla indexada por máscara: ``Φ[s]`` es un arreglo de largo
    ``popcount(s) + 1`` donde ``Φ[s][k] = P(el bloque s tiene k positivos)``.

    Se construye con un programa dinámico sobre subconjuntos.  La clave es
    que cada bloque está a *una sola convolución* del bloque que resulta de
    quitarle su miembro más bajo::

        Φ[s] = Φ[s sin su bit más bajo]  ⊛  (1−p_i, p_i)

    Así la familia entera cuesta ``2**m − 1`` convoluciones en vez de
    construir cada uno de los ``2**m`` bloques por separado.

    No depende de ninguna observación: por eso se calcula una vez por pool y
    sirve para todo conteo ``r`` y para todos los hijos que vengan después.
    """

    priors = _validated_priors(p)
    m = len(priors)

    cache = [None] * (1 << m)
    cache[0] = np.array([1.0])          # bloque vacío: 0 positivos con certeza

    for subset in range(1, 1 << m):
        lowest_bit = subset & -subset               # aísla el bit más bajo
        person = lowest_bit.bit_length() - 1        # a qué persona corresponde
        probability = priors[person]
        cache[subset] = np.convolve(
            cache[subset ^ lowest_bit],             # el bloque sin esa persona
            np.array([1.0 - probability, probability]),
        )
    return tuple(cache)


# --------------------------------------------------------------------------
# El tensor, por las dos vías
# --------------------------------------------------------------------------

def subpool_tensor(p, r, cache=None):
    """El tensor condicional por forma cerrada.

    ``Q[s][k] = P(conteo(s) = k | conteo(pool) = r)`` para todo subconjunto
    ``s``, aplicando la identidad del encabezado del módulo.

    Pasa ``cache`` si ya la tienes (de este pool o de un pool que lo
    contenga): entonces esta función no hace ni una convolución.
    """

    priors = _validated_priors(p)
    m = len(priors)
    r = _validated_count(r, m)
    if cache is None:
        cache = subset_pmf_cache(priors)

    full = (1 << m) - 1
    total_ways = float(cache[full][r])
    if total_ways <= 1e-300:
        raise ValueError(
            f"observar {r} positivos tiene probabilidad nula bajo este prior"
        )

    tensor = {}
    for subset in range(1 << m):
        inside = cache[subset]              # pmf del subconjunto
        outside = cache[full ^ subset]      # pmf de su complemento en el pool

        # Para que el subconjunto tenga k positivos, el complemento debe
        # cargar los r−k restantes.  Fuera del rango donde eso es posible la
        # probabilidad es cero por la identidad misma, no por recorte.
        column = np.zeros(len(inside))
        for k in range(len(inside)):
            rest = r - k
            if 0 <= rest < len(outside):
                column[k] = inside[k] * outside[rest] / total_ways
        tensor[subset] = column
    return tensor


def subpool_tensor_brute(p, r):
    """El mismo tensor, enumerando los ``2**m`` mundos.  El oráculo.

    Lento a propósito y sin una sola idea adentro: recorre cada asignación
    posible de quién es positivo, se queda con las que tienen exactamente
    ``r`` positivos en el pool, y acumula su probabilidad en la casilla que
    le toca a cada subconjunto.

    Existe para no creerle a la forma cerrada por fe.  Dos implementaciones
    independientes que coinciden al decimal 12 es lo que convierte una
    fórmula en un hecho.
    """

    priors = _validated_priors(p)
    m = len(priors)
    r = _validated_count(r, m)

    tensor = {s: np.zeros(s.bit_count() + 1) for s in range(1 << m)}
    total_mass = 0.0

    for world in range(1 << m):
        if world.bit_count() != r:          # incompatible con lo observado
            continue
        probability = 1.0
        for person in range(m):
            probability *= (
                priors[person] if world & (1 << person)
                else 1.0 - priors[person]
            )
        if probability == 0.0:
            continue
        total_mass += probability
        for subset in range(1 << m):
            tensor[subset][(world & subset).bit_count()] += probability

    if total_mass <= 1e-300:
        raise ValueError(
            f"observar {r} positivos tiene probabilidad nula bajo este prior"
        )
    for subset in tensor:
        tensor[subset] /= total_mass
    return tensor


# --------------------------------------------------------------------------
# La división: qué pasa cuando una prueba parte el pool
# --------------------------------------------------------------------------

class Atom:
    """Uno de los dos bloques que deja una prueba dentro del pool.

    ``members`` son los índices *del pool padre* que quedaron en este átomo,
    ``priors`` sus probabilidades a priori, ``count`` el conteo que le tocó, y
    ``tensor`` su propia tabla condicional, ya indexada por máscaras locales
    del átomo (el bit ``j`` es ``members[j]``).
    """

    __slots__ = ("members", "priors", "count", "tensor")

    def __init__(self, members, priors, count, tensor):
        self.members = members
        self.priors = priors
        self.count = count
        self.tensor = tensor

    def __repr__(self):
        return f"Atom(members={self.members}, count={self.count})"


def _restricted_cache(cache, members):
    """La caché de un sub-bloque, prestada del padre.  Cero convoluciones.

    Cada subconjunto del hijo es un subconjunto del padre: basta traducir la
    máscara local del hijo (bit ``j`` = ``members[j]``) a la del padre y
    devolver la misma pmf que ya estaba calculada.
    """

    child_size = len(members)
    parent_bits = [1 << person for person in members]

    restricted = [None] * (1 << child_size)
    for child_subset in range(1 << child_size):
        parent_subset = 0
        remaining = child_subset
        while remaining:
            lowest_bit = remaining & -remaining
            parent_subset |= parent_bits[lowest_bit.bit_length() - 1]
            remaining ^= lowest_bit
        restricted[child_subset] = cache[parent_subset]
    return tuple(restricted)


def split_after_test(p, r, tested, tested_count, cache=None):
    """Los dos átomos que deja probar ``tested`` dentro del pool.

    El pool traía el conteo ``r``; se prueba el subconjunto ``tested`` (una
    máscara local, subconjunto propio y no vacío) y sale ``tested_count``.
    Quedan dos bloques disjuntos: el probado, con su conteo, y el residuo,
    con ``r − tested_count`` por la resta de conteos.

    Devuelve ``(átomo probado, átomo residual)``, cada uno con su tensor ya
    armado desde la caché del padre --- sin una sola convolución nueva.

    Que cada átomo se condicione **solo en su propio conteo**, ignorando el
    del hermano, no es un descuido: es la factorización entre átomos.  Los
    dos bloques son disjuntos y el prior es producto, así que lo que pasa en
    uno no informa sobre el otro.
    """

    priors = _validated_priors(p)
    m = len(priors)
    r = _validated_count(r, m, "el conteo del pool")

    if isinstance(tested, bool) or not isinstance(tested, Integral):
        raise ValueError("el subconjunto probado debe ser una máscara entera")
    tested = int(tested)
    full = (1 << m) - 1
    if tested <= 0:
        raise ValueError("el subconjunto probado no puede ser vacío")
    if tested & ~full:
        raise ValueError("el subconjunto probado se sale del pool")
    if tested == full:
        raise ValueError(
            "probar el pool entero no lo parte; no hay división que hacer"
        )

    residual = full ^ tested
    tested_count = _validated_count(
        tested_count, tested.bit_count(), "el conteo del subconjunto probado"
    )
    residual_count = r - tested_count
    if not 0 <= residual_count <= residual.bit_count():
        raise ValueError(
            f"un conteo de {tested_count} en el subconjunto es incompatible "
            f"con {r} en el pool: al residuo le tocarían {residual_count}"
        )

    if cache is None:
        cache = subset_pmf_cache(priors)

    atoms = []
    for mask, count in ((tested, tested_count), (residual, residual_count)):
        members = [person for person in range(m) if mask & (1 << person)]
        child_priors = priors[members]
        child_cache = _restricted_cache(cache, members)
        atoms.append(Atom(
            members=members,
            priors=child_priors,
            count=count,
            tensor=subpool_tensor(child_priors, count, cache=child_cache),
        ))
    return tuple(atoms)
