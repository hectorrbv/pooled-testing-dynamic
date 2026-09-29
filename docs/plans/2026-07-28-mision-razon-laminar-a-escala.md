# Misión de autoresearch: la razón laminar a escala

Propuesta para una segunda misión en el harness `dapts-autoresearch`, con la
misma arquitectura que la misión de separación certificada: un solo archivo
editable, puertas duras que descartan el cambio, un escalar a minimizar y una
familia ancla que escala.

**Pregunta que ataca.** El atlas midió `V^L / V*` solo en `n ∈ {4,5,6}`, y el
peor caso empeora al crecer n (0.9350, 0.9355, 0.9280). Eso es exactamente lo
que uno no querría ver si espera que la restricción laminar sea barata a
escala, y con tres puntos no se puede distinguir entre ruido de malla y una
tendencia real.

---

## Por qué la misión actual no responde esto

La misión de separación certifica `V*` **por arriba**: produce `U ≥ V*` y
puntúa por el ancho del intervalo `[LB, U]`.

Esta misión necesita otra cosa: una cota **por abajo** de la razón
`V^L / V*`. La diferencia importa porque la razón se puede acotar sin acotar
bien ninguna de las dos cantidades por separado, y porque el numerador ahora
también hay que producirlo (V^L tampoco es computable a escala).

---

## Tres rutas, y por qué el orden importa

### Ruta 0 — más datos exactos (barata, hacer primero)

Antes de cualquier maquinaria: empujar el DP exacto a `n = 7, 8`. Pasar de 3
a 5 puntos no es extrapolación, pero **sí distingue ruido de tendencia**. Si
la caída 0.9350 → 0.9355 → 0.9280 se revierte en n=7, la alarma se apaga y
esta misión baja de prioridad.

No requiere harness. Es un barrido con memoización agresiva sobre la clase de
instancias que produjo el peor caso (n=6, B=3, G=2, prevalencia 0.7, tasas
dispersas, utilidades planas).

**Criterio de decisión:** si con n=7,8 el mínimo sigue bajando de forma
monótona, la misión se justifica. Si no, no.

### Ruta 1 — el emparedado certificado (rigurosa, hoy floja)

La cadena de desigualdades ya conocida:

```
V^π_L  ≤  V^L  ≤  V*  ≤  U
```

donde `V^π_L` es el valor de **cualquier** política laminar factible —
computable a escala precisamente porque la estructura laminar hace barata la
inferencia — y `U` es la cota de la misión actual. Entonces

```
V^π_L / U   ≤   V^L / V*
```

y el lado izquierdo se computa a cualquier n. Si ese cociente certificado se
mantiene alto al crecer n, **queda demostrado** que la restricción laminar
sigue siendo barata.

**El problema, dicho de frente.** Las cotas actuales son flojas. En la
sesión del 9 de julio: greedy/OPT real ≈ 0.98 contra certificado ≈ 0.70. Con
esa holgura el enunciado sería "laminar recupera al menos el 70%", no "cuesta
7%". Cierto y escalable, pero débil.

**Dónde puede apretar.** El cuello no es `U` en abstracto sino `U` **contra
una política laminar**. Una cota superior construida por relajación de
información sobre la *misma* jerarquía que usa la política inferior debería
apretar mucho más que `U_PI`, porque comparte la estructura. Ésa es la idea
editable de esta ruta.

### Ruta 2 — la escalera de treewidth (la apuesta principal)

En vez de comparar contra `V*` (incomputable), comparar contra rivales
**cada vez menos restringidos**:

```
V^L  ≤  V^{L+1}  ≤  V^{L+2}  ≤  ...  ≤  V*
```

donde `V^{L+k}` permite `k` pools que cruzan la familia laminar. Con `k=1` la
inferencia sigue siendo tratable (el historial deja de ser un árbol pero
mantiene treewidth acotado), y escala **mucho** más lejos que el DP exacto.

**El argumento de fondo:** si permitir un cruce recupera casi nada, y permitir
dos tampoco, entonces la restricción laminar es casi gratis — y esa evidencia
vive en n mucho mayores que 6.

Ya está anotada como meta de estiramiento en el plan del 21 de julio
("permitir un único pool cruzado sobre la biblioteca fija y medir cuánta razón
recupera en la región mala del atlas"). Esta misión la convierte en el objeto
central.

---

## Contrato de la misión (Ruta 2)

Espejo del contrato de `laminar_cert.upper_bound`.

**Archivo editable:** `augmented/laminar_ladder.py`, único del carril seguro.

**Función:**

```python
def ladder_value(p, u, B, G, crossings):
    """Valor óptimo permitiendo `crossings` pools no laminares.

    crossings=0 debe devolver exactamente V^L (o una cota inferior válida
    de él si la enumeración de bibliotecas no cabe).
    Debe evaluar en segundos hasta n ~ 64 con crossings <= 1.
    """
```

**Puntaje (minimizar):**

```
ladder_gap_max = max_ancla  (V^{L+1} - V^L) / V^L
```

Cuanto más plana la escalera, menor el puntaje, y más fuerte la evidencia de
que laminar es casi óptimo. **Ojo con el signo:** aquí un puntaje bajo es un
resultado *positivo* para la tesis del proyecto, al revés que en la misión de
separación, donde bajar el gap es apretar un certificado.

**Puertas duras (violación = descarte):**

- `ladder_not_monotone`: `V^{L+k}` debe crecer con `k`. Si baja, la
  implementación está mal.
- `ladder_below_laminar`: `ladder_value(..., 0) < V^L` del atlas en n≤6.
- `ladder_above_star`: `V^{L+k} > V*` exacto en la batería n≤7. Un valor que
  supere el óptimo libre no es un valor alcanzable.
- `anchor_timeout`: más de 60 s en una sola ancla.
- `non_finite`, `eval_error`.
- `tests_status`: deben pasar `tests_laminar_tables.py` y
  `tests_laminar_benchmarks.py`.

**Familia ancla.** Distinta a la de la misión de separación, y elegida a
propósito donde el atlas duele: tasas dispersas (beta bimodal), prevalencia
media-alta, utilidades planas, con `n ∈ {8, 12, 16, 24, 32, 48, 64}`,
`B ∈ {2,3}`, `G ∈ {2,3}`. La instancia insignia es la extensión a n grande de
la que produjo el peor caso del atlas.

**Métricas de referencia (no deciden keep/discard):**

- `ladder_gap_med`: ¿toda la familia es plana o solo el ancla?
- `slope_vs_n`: la pendiente de `ladder_gap_max` contra n. **Ésta es la
  métrica que responde la pregunta original.** Si es plana o negativa, la
  restricción laminar no se degrada con la escala.
- `tight_batt_med`: `(V^{L+1} - V^L)/V^L` en la batería n≤7, donde sí hay
  ground truth contra `V*`. Sirve para calibrar cuánto del gap real captura
  un solo cruce.
- `crossing_used_rate`: en qué fracción de instancias la política óptima con
  un cruce disponible efectivamente lo usa. Si casi nunca lo usa, la escalera
  es plana por una razón estructural y no por falta de presupuesto.

---

## Lo que hay que construir antes de arrancar el harness

1. **Inferencia con un cruce.** Hoy `laminar_inference` asume historial
   laminar. Con un pool cruzado el historial deja de factorizar en átomos
   disjuntos, pero sigue siendo tratable: los bloques quedan acoplados de a
   pares. Es el trabajo técnico real de la misión.

2. **Un `V^L` de referencia a escala.** Enumerar bibliotecas maximales no
   escala más allá de n≈6. A escala hay que fijar una jerarquía (la
   balanceada) y reportar `V^{opt}` dentro de ella, aceptando que es una cota
   inferior de `V^L`. Documentar la degradación, no esconderla.

3. **La familia ancla**, con su generador determinista.

---

## Riesgos

**La escalera puede no ser informativa.** Si `V^{L+1}` ya recupera casi todo
el gap a `V*` en n≤6, entonces medir la escalera equivale a medir el gap
completo y no se gana escalabilidad. **Verificar esto primero en n≤6, antes
de invertir en la implementación general.** Es una prueba de una tarde.

**El cruce puede volver intratable la inferencia posterior.** Un cruce en el
paso 1 acopla bloques; dos cruces pueden acoplar todo. Puede que solo `k=1`
sea viable, lo que limita la escalera a un peldaño.

**El resultado puede ser el contrario al deseado.** Si la escalera es
empinada, el hallazgo es que laminar sí cuesta a escala. Eso también es un
resultado publicable, y hay que estar dispuesto a reportarlo.

---

## Recomendación de secuencia

1. **Ruta 0** (una tarde): empujar a n=7,8 y ver si la tendencia existe.
2. **Prueba de informatividad** (una tarde): en n≤6, medir `V^{L+1}` y
   comparar contra `V*`. Si `V^{L+1} ≈ V*`, la escalera no compra
   escalabilidad y hay que replantear.
3. Solo entonces, **montar la misión de Ruta 2**.

La Ruta 1 queda como carril paralelo: es la única que produce un enunciado
rigurosamente válido a cualquier n, aunque hoy sea flojo.

**Y el recordatorio que ordena todo esto:** un teorema por régimen —como la
conjetura `B≤2` homogénea— es **independiente de n** y vuelve esta misión
innecesaria en ese régimen. Antes de invertir semanas de cómputo en
extrapolar, conviene gastar días en intentar demostrar.
