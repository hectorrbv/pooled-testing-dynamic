"""Build hand-drawn style notes for notebook 22 as an Excalidraw scene.

Produces ``22_apuntes.excalidraw``, openable at excalidraw.com or with the
Excalidraw VS Code extension.  Regenerating is deterministic (fixed seeds), so
the file diffs cleanly in git.

    python augmented/notebooks/build_apuntes_excalidraw.py
"""

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent / "22_apuntes.excalidraw"

BLACK = "#1e1e1e"
BLUE = "#1971c2"
RED = "#e03131"
GREEN = "#2f9e44"
ORANGE = "#f08c00"
VIOLET = "#6741d9"

FILL_BLUE = "#a5d8ff"
FILL_GREEN = "#b2f2bb"
FILL_RED = "#ffc9c9"
FILL_YELLOW = "#ffec99"
FILL_VIOLET = "#d0bfff"
FILL_GRAY = "#e9ecef"

HAND = 1        # Virgil
CODE = 3        # Cascadia

elements = []
_counter = [0]


def _next_id():
    _counter[0] += 1
    return f"el{_counter[0]:04d}", 1000 + _counter[0] * 7919


def _base(kind, x, y, width, height, stroke, background, **extra):
    element_id, seed = _next_id()
    element = {
        "id": element_id,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": background,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": seed,
        "version": 1,
        "versionNonce": seed + 13,
        "isDeleted": False,
        "boundElements": [],
        "updated": 1753600000000,
        "link": None,
        "locked": False,
    }
    element.update(extra)
    elements.append(element)
    return element


def box(x, y, width, height, stroke=BLACK, background="transparent"):
    return _base("rectangle", x, y, width, height, stroke, background)


def text(x, y, body, size=20, color=BLACK, family=HAND):
    lines = body.split("\n")
    width = max(len(line) for line in lines) * size * 0.55
    height = len(lines) * size * 1.25
    return _base(
        "text", x, y, width, height, color, "transparent",
        roundness=None,
        text=body,
        fontSize=size,
        fontFamily=family,
        textAlign="left",
        verticalAlign="top",
        containerId=None,
        originalText=body,
        lineHeight=1.25,
        strokeWidth=1,
    )


def arrow(x, y, dx, dy, color=BLACK):
    return _base(
        "arrow", x, y, abs(dx), abs(dy), color, "transparent",
        roundness={"type": 2},
        points=[[0, 0], [dx, dy]],
        lastCommittedPoint=None,
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead="arrow",
    )


def panel(x, y, width, height, title, body, fill, stroke, size=17):
    """Panel with a title and a body, checked so the text cannot overflow."""

    lines = body.split("\n")
    needed_height = 58 + len(lines) * size * 1.25 + 20
    needed_width = 44 + max(len(line) for line in lines) * size * 0.55
    if needed_height > height:
        raise ValueError(
            f"{title!r}: el cuerpo necesita {needed_height:.0f}px de alto "
            f"y el panel mide {height}"
        )
    if needed_width > width:
        raise ValueError(
            f"{title!r}: el cuerpo necesita {needed_width:.0f}px de ancho "
            f"y el panel mide {width}"
        )
    box(x, y, width, height, stroke, fill)
    text(x + 22, y + 18, title, size=22, color=stroke)
    text(x + 22, y + 58, body, size=size, color=BLACK)


# ---------------------------------------------------------------- título ---
text(60, 40, "NOTEBOOK 22 — conteos agrupados, familias laminares", size=36)
text(60, 92, "y el atlas de razones", size=36)
text(60, 148,
     "lo esencial en una hoja  ·  cada número viene de augmented/data/laminar_week/",
     size=16, color="#868e96")


# Cada fila declara su alto; panel() falla si el texto no cabe.
ROW1, ROW1_H = 200, 400
ROW2, ROW2_H = 640, 450
ROW3, ROW3_H = 1130, 380
ROW4, ROW4_H = 1550, 450


# ------------------------------------------------------- fila 1: problema ---
panel(60, ROW1, 620, ROW1_H, "1 · EL PROBLEMA", """N personas.  Cada una trae dos números:
   p_i = prob. de estar positiva
   u_i = lo que vale liberarla

Presupuesto: B pruebas, cada una
sobre un pool de a lo más G personas
(G = límite físico de dilución).

REGLA CLAVE
Cobras u_i solo si un pool que la
contiene sale con conteo CERO.

Dinámico: eliges cada prueba después
de ver el resultado de la anterior.""", FILL_BLUE, BLUE)

panel(720, ROW1, 620, ROW1_H, "2 · POR QUÉ CONTEOS Y NO BINARIO", """Prueba binaria:  "¿hay alguno?"
Prueba aumentada: el CONTEO exacto.

Ejemplo con 3 personas:
   {0,1,2}  ->  1 positivo
   {1,2}    ->  1 positivo

Con conteos DEDUCES que 0 está limpia:
el único positivo ya está en {1,2}.

Con binario no concluyes nada: ambos
pools salieron "positivo".

Esa deducción es utilidad que cobras
sin haber probado a nadie solo.""", FILL_GREEN, GREEN)

panel(1380, ROW1, 620, ROW1_H, "3 · EL MURO", """Para la posterior exacta hay que sumar
la probabilidad de TODOS los mundos
compatibles con lo observado.

Un mundo = quién está positivo.
Con n personas hay 2^n mundos.
n = 50  ->  10^15 mundos.

Eso es #P-DIFÍCIL.
   NP  pregunta: ¿existe solución?
   #P  pregunta: ¿CUÁNTAS hay?
Contar es más duro que decidir.

Sin estructura, no hay atajo.""", FILL_RED, RED)


# ------------------------------------------------------ fila 2: laminar ----
panel(60, ROW2, 940, ROW2_H, "4 · LA SALIDA: FAMILIAS LAMINARES", """LAMINAR = todo par de pools es disjunto, o uno contiene al otro.
             Nunca se cruzan a medias.  Es un árbol.

Si las pruebas están anidadas puedes RESTAR conteos:

      {1,2,3,4} = 2 positivos
      {1,2}     = 1 positivo
      ------------------------------
      {3,4}     = 1     <- aritmética, no estimación

Los residuos son los ÁTOMOS. No comparten a nadie.
La posterior FACTORIZA ENTRE ÁTOMOS  ->  convoluciones baratas.

El muro #P desaparece: en vez de sumar sobre 2^n mundos,
resuelves cada bloque por separado y los combinas.""", FILL_YELLOW, ORANGE)

panel(1040, ROW2, 960, ROW2_H, "5 · LA TRAMPA QUE EL NOTEBOOK CAZÓ", """Factoriza ENTRE átomos.  NO dentro de uno.

Dentro de {3,4} sabes que hay exactamente 1 positivo:
si 3 es positivo, entonces 4 es negativo. Están ATADOS.

Marginal exacta  =/=  independencia.

Ejemplo:  p = (.2, .4, .6, .8), observas 2 positivos.
Distribución del conteo de {0,1}:

   exacto    [ .5353   .4498   .0149 ]
   producto  [ .5675   .3855   .0470 ]     TV = .064
                                  ^ tres veces inflado

En historias reales el error llegó a TV = 0.6  (§7).
Para el rollout: pasa los ÁTOMOS con sus conteos,
NUNCA el vector de marginales.""", FILL_RED, RED)


# ------------------------------------------- fila 3: cuatro cantidades -----
panel(60, ROW3, 940, ROW3_H, "6 · EL INSTRUMENTO: LAS CUATRO CANTIDADES", """V = utilidad total esperada al final (value function).
Lo que cambia es SOBRE QUÉ ESTRATEGIAS maximizas:

   V*              todas.  Cualquier pool, adaptándose.
   V^L             solo pools de la mejor familia laminar.
   V^greedy_L      un árbol fijo heurístico, decisiones miopes.
   V^static_bin    pools fijos de antemano, resultado sí/no.
                   (el ancla con la literatura previa)

Cadena:     V^greedy_L   <=   V^L   <=   V*

Mientras menos libertad, menor el valor.
"0.928" quiere decir  V^L / V* = 92.8%.""", FILL_VIOLET, VIOLET)

panel(1040, ROW3, 960, ROW3_H, "7 · EL RESULTADO INSIGNIA", """                              media      peor caso
   ser laminar   V^L/V*        99.3%       92.8%
   el árbol      Vgr/V^L       97.3%       74.7%   <--

EL ENEMIGO NO ES LA RESTRICCIÓN LAMINAR.
ES ELEGIR MAL EL ÁRBOL.

El árbol heurístico coincide con el óptimo
solo en 28.8% de las 2,592 instancias.

Perder poco por ser laminar (~7% peor caso) pero
mucho por el árbol (~25%) reordena el problema:
la pregunta deja de ser "¿cuánto cuesta laminar?"
y pasa a ser "¿cómo se elige el árbol?".""", FILL_GREEN, GREEN)


# ------------------------------------------------- fila 4: resultados -----
panel(60, ROW4, 620, ROW4_H, "8 · LOS OTROS HALLAZGOS", """B <= 2 homogéneo:  V^L = V*
en los 420 puntos de la malla.
Es el blanco teórico más limpio
-> eje del notebook 23.
(B=1 es teorema; B=2, conjetura)

Búsqueda adversaria:
0.928 -> 0.9069, sin bajar de 0.9.

Pipeline n=40 (resultado incómodo):
   MILP miope       11.46  <- gana
   greedy plano     11.27
   pipeline laminar 10.07
Fijar el árbol costó más de lo que
el rollout recuperó.""", FILL_GRAY, BLACK)

panel(720, ROW4, 620, ROW4_H, "9 · DÓNDE SÍ PAGA EL LAMINAR", """El rollout laminar gana al mejor
diseño estático binario en:

   prevalencia ALTA  ->  98.2%
   prevalencia BAJA  ->  50.9%
   tasas homogéneas  ->  87.5%
   tasas dispersas   ->  61.8%

La prevalencia es la variable que
decide.  Confirma la intuición
de la sesión.

Matiz: la mejor instancia tiene
utilidades PLANAS.  La ganancia
no está en el primer paso, sino
en los pasos 2...B.""", FILL_BLUE, BLUE)

panel(1380, ROW4, 620, ROW4_H, "10 · LA TABLA Y LA CACHÉ", """LA TABLA (sesión 27 jul)
columnas = subconjuntos t' (2^|t|)
filas    = conteos posibles r'
cada columna suma 1
última columna: masa en lo observado

decisión greedy = fila r'=0
                  x utilidad

LA CACHÉ = pmf de cada bloque SIN
condicionar. No depende de lo
observado -> sirve para cualquier
conteo Y para los hijos al partir:
CERO convoluciones nuevas.

Pero solo ~1.2x en tiempo real:
no materialices la tabla.""", FILL_YELLOW, ORANGE)


# ------------------------------------------------------------- flechas ----
arrow(690, ROW1 + 200, 20, 0, BLACK)
arrow(1350, ROW1 + 200, 20, 0, BLACK)
arrow(1690, ROW1 + ROW1_H + 12, -600, 15, ORANGE)
arrow(530, ROW1 + ROW1_H + 10, 0, 20, BLACK)
arrow(1010, ROW2 + 225, 20, 0, RED)
arrow(530, ROW2 + ROW2_H + 5, 0, 25, VIOLET)
arrow(1010, ROW3 + 190, 20, 0, GREEN)
arrow(1520, ROW3 + ROW3_H + 5, 0, 25, GREEN)

text(1130, ROW1 + ROW1_H + 18, "sin estructura no se puede calcular",
     size=14, color=ORANGE)
text(300, ROW2 + ROW2_H + 8, "con el instrumento ya se puede medir",
     size=14, color=VIOLET)


# ------------------------------------------------------------- cierre -----
box(60, ROW4 + ROW4_H + 40, 1940, 110, BLACK, FILL_GRAY)
text(90, ROW4 + ROW4_H + 60, "EN UNA FRASE", size=20, color=BLACK)
text(90, ROW4 + ROW4_H + 95,
     "Los conteos dan poder de deducción que el binario no tiene, pero la inferencia exacta es #P-difícil; "
     "las familias laminares la vuelven barata\n"
     "a un costo pequeño (~7% peor caso) — y lo caro resultó ser elegir el árbol (~25%), no la restricción laminar.",
     size=15, color=BLACK)


scene = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://github.com/excalidraw/excalidraw",
    "elements": elements,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

OUT.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT} ({len(elements)} elements)")
