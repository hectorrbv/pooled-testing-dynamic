# Plan de 3 días: tensor condicional, falsificador de anidamiento y greedy laminar

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task.

**Meta:** Entregar a Francisco mañana el objeto que pidió — el tensor
`Q[T'][R'] = P(r(T')=R' | r(T)=R)` para todo `T' ⊆ T` — corriendo, validado por
dos vías independientes, con sus tres sanity checks como tests; más la primera
evidencia real de si el greedy exacto anida (el falsificador); y dejar el código
en `augmented/laminar_inference.py` para no volver a construirlo (es el
entregable 3 del plan semanal, nunca empezado).

**Arquitectura:** Un módulo nuevo `augmented/laminar_inference.py` con el oráculo
bruto (enumeración de perfiles del pool) y la forma cerrada
`Q[s][r'] = f_s[r'] · f_{T∖s}[r−r'] / f_T[r]` sobre una caché de
Poisson-binomiales por subconjunto (DP `O(2^G·G)`). Un experimento independiente
`augmented/experiments_nesting.py` que camina el árbol de decisiones del greedy
exacto y mide qué fracción de decisiones son anidadas/mixtas/vírgenes. Teoría en
paralelo (Persona A, modo propedéutico): nota técnica de 2 páginas + guion de
sesión.

**Tech stack:** Python 3.13, numpy 2.4, pytest (`pytest.ini` colecciona
`tests*.py`). Sin scipy (hipergeométrica vía `math.comb`). Máscaras de bits como
en el resto de `augmented/`.

---

## Reconciliación tras el pull del 27-jul (LEER PRIMERO)

El commit `6ac1569` de Héctor (24-jul, `feat(augmented): completa atlas laminar
y pipeline MILP`) entregó los pendientes del plan semanal ANTES de escribirse
este plan; el repo local estaba 3 días atrasado. Consecuencias:

- `augmented/laminar_inference.py` **YA EXISTE** (interfaz con jerarquía
  suministrada, sin parser cúbico). Las tareas 1–2 lo EXTIENDEN, no lo crean.
  No tocar las funciones existentes; extender `__all__`.
- `laminar_pool_pmf(p, atoms, pool_mask)` ya implementa la ley predictiva
  general (convolución sobre átomos + prior en territorio virgen, pools no
  compatibles incluidos) y está validada contra `exact_pool_pmf`
  (`tests_laminar_benchmarks.py:69`). **B8 del día 2 queda hecha**; se
  convierte en revisión. Además da una TERCERA vía de validación del tensor.
- `scenario_milp.py`, `laminar_benchmarks.py` (las cuatro cantidades con firma
  común), `laminar_pipeline.py`, atlas v1 (2,592 filas), búsqueda adversaria
  (mínimo local $V^{\mathcal L}/V^*=0.9069$), barrido homogéneo $B\le 2$ con
  igualdad numérica en toda la malla, y `proposicion_b_policy_improvement.md`
  — todo existe. El día 3 rama A reusa `laminar_benchmarks` en vez de
  construir la evaluación de políticas desde cero.
- **Sigue sin existir** (y es el valor nuevo de este plan): el tensor de
  Francisco para TODOS los subconjuntos con caché Φ, sus tres sanity checks
  como tests nombrados, la demo del pizarrón, el falsificador de anidamiento,
  la nota técnica y el guion de sesión.
- Para la sesión: la igualdad $B\le 2$ homogéneo de Héctor y el 0.9069
  adversario entran al guion (bloque A5) como material fresco.

---

## Reglas de cierre (aplican a TODO el plan)

Un ítem está **terminado** solo si cumple las tres:

1. **Dos vías independientes coinciden**: forma cerrada vs. enumeración bruta
   (tolerancia `1e-12`), o enunciado demostrado vs. prueba de fuego numérica.
2. **Mapeo 1:1 enunciado ↔ test**: cada afirmación de la nota tiene un test con
   su nombre. Sin test, se degrada a "[VERIFICADO n≤X]" o "[PREGUNTA]" con
   etiqueta explícita.
3. **Revisión cruzada**: quien demostró revisa los tests; quien programó revisa
   los enunciados. Es gate, no costumbre.

Etiquetas de estatus en todo el paquete: **[DEMOSTRADO]**, **[VERIFICADO n≤X]**,
**[PREGUNTA]**. Nunca mezcladas.

Reglas de sesión con Francisco: (a) "lema" se usa como pieza interna, no como
claim de novedad, hasta que el barrido de literatura regrese; (b) expectativas
calibradas en voz alta: greedy-con-tensor solo ≈ 1–2 puntos (el peldaño de
scoring es ~¼ del gap según `independence_gap.py`); el premio real es habilitar
rollout/lookahead (~¾ del gap, bloqueado sin esperanzas exactas — notebook 21
§7); (c) la pregunta incremental padre→hijos se responde con números: la tabla
de `G=10` pesa ~90 KB y se llena en milisegundos — bonita, no bloqueante.

---

# DÍA 1 — HOY (8–9 h antes de la sesión)

Tres carriles en paralelo:

- **Persona B (código):** Tareas 0–6, en orden estricto. Nada de la tarea N+1
  sin la N en verde.
- **Persona A (teoría):** Bloques A1–A6.
- **Claude (paralelo, costo humano cero):** Tarea 7 (barrido de literatura QGT).

Cronograma orientativo y gates:

| Hora | Persona B | Persona A | Gate |
|---|---|---|---|
| 0:00–0:15 | T0 higiene | A1 ejercicio a mano | |
| 0:15–1:15 | T1 oráculo bruto | A1 (sigue) | **G1** (h1): números de A1 → `test_ejercicio_ancla` |
| 1:15–2:45 | T2 forma cerrada + Φ | A2 borrador nota | |
| 2:45–3:45 | T3 checks de Francisco | A2 (sigue) | |
| 3:45–4:45 | T4 hipergeom + pizarrón | A3 lupa de revisor | **G2** (h4.5): mapeo nota ↔ tests |
| 4:45–5:45 | T5 demo | A4 versión final | |
| 5:45–7:45 | T6 falsificador | A5 guion de sesión | |
| 7:45–8:30 | T8 congelación | A6 revisión cruzada | **G3**: suite verde desde checkout limpio |

---

## Tarea 0: Higiene del repo (10 min)

**Files:**
- Modify: `.gitignore`

Los modelos whisper (`ggml-*.bin`, ~9 GB), `transcription_work/`,
`pasted-text.txt` y `TRANSCRIPTION_README.md` están sin trackear en la raíz y no
pertenecen a este repo. NO borrar nada — solo ignorar.

**Step 1:** Añadir al final de `.gitignore`:

```
# Artefactos de transcripcion (sesiones con Francisco)
ggml-*.bin
transcription_work/
pasted-text.txt
TRANSCRIPTION_README.md
```

**Step 2:** `git status` — los binarios ya no aparecen.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignora modelos whisper y artefactos de transcripcion"
```

---

## Tarea 1: Oráculo bruto del tensor

**Files:**
- Modify: `augmented/laminar_inference.py` (existe — commit `6ac1569` de
  Héctor; AÑADIR al final, sin tocar lo existente, y extender `__all__`)
- Create: `augmented/tests_laminar_inference.py`

**Step 1: Escribir los tests que fallan**

```python
"""Tests del tensor condicional de subpools (pedido de Francisco, 27-jul).

Convencion: el pool T se representa por sus priors locales p_pool (lista de
longitud m); los subconjuntos T' son mascaras locales s en [0, 2^m). El tensor
es un dict {s: np.ndarray de longitud popcount(s)+1} con
Q[s][r'] = P(r(T')=r' | r(T)=r).
"""
import numpy as np
import pytest

from augmented.laminar_inference import (
    subpool_tensor,
    subpool_tensor_brute,
    subset_pmf_cache,
)


def _instancia(seed, m_lo=2, m_hi=8):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(m_lo, m_hi + 1))
    p = rng.uniform(0.05, 0.95, size=m).tolist()
    return m, p


def test_bruto_columnas_suman_uno():
    for seed in range(10):
        m, p = _instancia(seed)
        for r in range(m + 1):
            tensor = subpool_tensor_brute(p, r)
            for s, col in tensor.items():
                assert abs(col.sum() - 1.0) < 1e-12, (seed, r, s)


def test_bruto_valida_entradas():
    with pytest.raises(ValueError):
        subpool_tensor_brute([0.5, 0.5], 3)      # conteo fuera de rango
    with pytest.raises(ValueError):
        subpool_tensor_brute([0.5, 0.5], -1)
    with pytest.raises(ValueError):
        subpool_tensor_brute([0.0, 0.5], 1)      # prior degenerado
    with pytest.raises(ValueError):
        subpool_tensor_brute([], 0)              # pool vacio
```

**Step 2:** `python3 -m pytest augmented/tests_laminar_inference.py -v`
Esperado: FAIL (ImportError).

**Step 3: Implementación mínima**

Añadir al FINAL de `augmented/laminar_inference.py` (el módulo y sus imports
ya existen; solo se agregan las funciones nuevas más una sección comentada
`# --- Tensor condicional de subpools (sesion 27-jul) ---` y las entradas
`"subpool_tensor_brute"`, `"subpool_tensor"`, `"subset_pmf_cache"` en
`__all__`):

```python
# --- Tensor condicional de subpools (sesion 27-jul) ---
# Input (T, r): pool con priors p_pool y conteo observado r. Output:
# Q[s][r'] = P(r(T')=r' | r(T)=r) para todo T' ⊆ T (mascaras locales).
# Fundamento: Corolario A.2(a) de lemma_A_laminar_inference.tex con L={T}.


def _validar(p_pool, r):
    m = len(p_pool)
    if m == 0:
        raise ValueError('pool vacio')
    if not all(0.0 < pi < 1.0 for pi in p_pool):
        raise ValueError(
            'prior degenerado: reduzca la instancia primero '
            '(Remark de priors degenerados del Lema A)')
    if not (0 <= r <= m):
        raise ValueError('conteo fuera de rango [0, %d]' % m)
    return m


def subpool_tensor_brute(p_pool, r):
    """Oraculo: enumera los 2^m perfiles del pool y condiciona a suma = r.

    Devuelve {mascara_local s: np.ndarray de longitud popcount(s)+1}.
    Tonto a proposito: es la referencia contra la que se valida todo.
    """
    m = _validar(p_pool, r)
    consistentes = []
    total = 0.0
    for z in range(1 << m):
        if z.bit_count() != r:
            continue
        w = 1.0
        for i in range(m):
            w *= p_pool[i] if (z >> i) & 1 else 1.0 - p_pool[i]
        consistentes.append((z, w))
        total += w
    tensor = {}
    for s in range(1 << m):
        col = np.zeros(s.bit_count() + 1)
        for z, w in consistentes:
            col[(z & s).bit_count()] += w
        tensor[s] = col / total
    return tensor
```

(`total > 0` está garantizado: con priors en `(0,1)` todo conteo `0 ≤ r ≤ m` es
factible — es exactamente la caracterización de coherencia del Lema A.)

**Step 4:** `python3 -m pytest augmented/tests_laminar_inference.py -v`
Esperado: 2 PASS.

**Step 5: Commit**

```bash
git add augmented/laminar_inference.py augmented/tests_laminar_inference.py
git commit -m "feat(laminar_inference): oraculo bruto del tensor condicional de subpools"
```

---

## Tarea 2: Caché Φ + forma cerrada + identidad contra el oráculo

**Files:**
- Modify: `augmented/laminar_inference.py`
- Modify: `augmented/tests_laminar_inference.py`

**Step 1: Test que falla** (añadir):

```python
def test_cerrado_igual_bruto():
    """Identidad de las dos vias sobre instancias aleatorias, TODOS los r."""
    for seed in range(40):
        m, p = _instancia(seed)
        for r in range(m + 1):
            ref = subpool_tensor_brute(p, r)
            got = subpool_tensor(p, r)
            assert set(got) == set(ref)
            for s in ref:
                assert np.allclose(got[s], ref[s], atol=1e-12), (seed, r, s)


def test_cache_phi_es_poisson_binomial():
    """f_S de la cache == convolucion directa, por subconjunto."""
    m, p = _instancia(3, m_lo=5, m_hi=5)[0], _instancia(3, m_lo=5, m_hi=5)[1]
    cache = subset_pmf_cache(p)
    for s in range(1 << m):
        f = np.array([1.0])
        for i in range(m):
            if (s >> i) & 1:
                f = np.convolve(f, [1.0 - p[i], p[i]])
        assert np.allclose(cache[s], f, atol=1e-13)


def test_cerrado_igual_laminar_pool_pmf():
    """TERCERA via: identidad contra laminar_pool_pmf de Hector con el
    atomo unico (T, r, T), submascara por submascara. Tres implementaciones
    independientes coincidiendo cierran el triangulo de validacion."""
    from augmented.laminar_inference import laminar_pool_pmf
    for seed in range(10):
        m, p = _instancia(seed)
        full = (1 << m) - 1
        for r in range(m + 1):
            tensor = subpool_tensor(p, r)
            atoms = [(full, r, full)]
            for s in range(1, 1 << m):
                ref = laminar_pool_pmf(p, atoms, s)
                assert np.allclose(tensor[s], ref, atol=1e-12), (seed, r, s)
```

**Step 2:** Correr — FAIL (`subpool_tensor` no existe).

**Step 3: Implementación** (añadir):

```python
def subset_pmf_cache(p_pool):
    """Phi: pmf Poisson-binomial prior f_S para CADA submascara S del pool.

    DP sobre subconjuntos: f_S = f_{S sin su bit mas bajo} conv (q_i, p_i).
    Costo O(2^m · m) total; despues, cada entrada del tensor es O(1).
    Esta cache se hereda a todo el subarbol del pool: cualquier atomo
    descendiente D ⊆ T y cualquier candidato S ⊆ D ya tienen su f en Phi.
    """
    m = len(p_pool)
    cache = [None] * (1 << m)
    cache[0] = np.array([1.0])
    for s in range(1, 1 << m):
        i = (s & -s).bit_length() - 1        # bit mas bajo de s
        prev = cache[s & (s - 1)]            # s sin ese bit
        pr = p_pool[i]
        out = np.zeros(len(prev) + 1)
        out[:-1] += prev * (1.0 - pr)
        out[1:] += prev * pr
        cache[s] = out
    return cache


def subpool_tensor(p_pool, r, cache=None):
    """Forma cerrada: Q[s][r'] = f_s[r'] · f_{T∖s}[r−r'] / f_T[r].

    Derivacion (ruta de Bayes de la sesion): la verosimilitud
    P(r(T)=r | r(T')=r') es f_{T∖s}[r−r'] por independencia del prior.
    Identico al Corolario A.2(a) del Lema A con L={T}.
    """
    m = _validar(p_pool, r)
    if cache is None:
        cache = subset_pmf_cache(p_pool)
    full = (1 << m) - 1
    denom = cache[full][r]
    tensor = {}
    for s in range(1 << m):
        fs = cache[s]
        fc = cache[full ^ s]
        col = np.zeros(s.bit_count() + 1)
        for rp in range(len(col)):
            j = r - rp
            if 0 <= j < len(fc):
                col[rp] = fs[rp] * fc[j] / denom
        tensor[s] = col
    return tensor
```

**Step 4:** Correr toda la suite — PASS. (El test de identidad tarda unos
segundos; es el precio del oráculo.)

**Step 5: Commit**

```bash
git commit -am "feat(laminar_inference): forma cerrada con cache de subconjuntos, identidad contra oraculo"
```

---

## Tarea 3: Los tres checks de Francisco + ley de soporte + casos triviales

Estos tests llevan LOS NOMBRES de las propiedades que Francisco dictó en sesión
([06:36–07:48] audio 2). Son la parte del paquete que se enseña en pantalla.

**Step 1: Tests que fallan… solo si hay bug** (añadir; deben pasar ya — su
valor es de regresión y de demostración):

```python
def _soporte(m, s_size, r):
    lo = max(0, r - (m - s_size))
    hi = min(r, s_size)
    return lo, hi


def test_francisco_columnas_suman_uno():
    """'Por cada columna la suma de elementos es igual a 1.'"""
    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            for s, col in subpool_tensor(p, r).items():
                assert abs(col.sum() - 1.0) < 1e-12


def test_francisco_columna_total_es_onehot():
    """'El posterior de T' cuando T'=T es nada mas r: unos y ceros.'"""
    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            col = subpool_tensor(p, r)[(1 << m) - 1]
            esperado = np.zeros(m + 1)
            esperado[r] = 1.0
            assert np.allclose(col, esperado, atol=1e-12)


def test_francisco_ley_de_soporte():
    """'Algunas entradas son vacias': cero FUERA de
    [max(0, r−|T∖T'|), min(r, |T'|)] y positivo DENTRO.
    La cota inferior es 'por que no puedes tener un conteo que no concuerde
    con el pool grande' — la parte 2 de la formalizacion.
    """
    for seed in range(20):
        m, p = _instancia(seed)
        for r in range(m + 1):
            tensor = subpool_tensor(p, r)
            for s, col in tensor.items():
                lo, hi = _soporte(m, s.bit_count(), r)
                for rp in range(len(col)):
                    if lo <= rp <= hi:
                        assert col[rp] > 0.0, (seed, r, s, rp)
                    else:
                        assert col[rp] == 0.0, (seed, r, s, rp)


def test_casos_triviales_r0_y_r_lleno():
    """'Si r=0 siempre va a ser 0; si estan todos, siempre |T'|.'"""
    m, p = 6, [0.3, 0.7, 0.2, 0.9, 0.4, 0.55]
    t0 = subpool_tensor(p, 0)
    tm = subpool_tensor(p, m)
    for s in range(1 << m):
        k = s.bit_count()
        assert t0[s][0] == 1.0 and t0[s].sum() == 1.0
        assert tm[s][k] == 1.0 and tm[s].sum() == 1.0
```

**Step 2:** Correr — PASS. Si algo falla, PARAR: es un bug real, no se maquilla.

**Step 3: Commit**

```bash
git commit -am "test(laminar_inference): checks de Francisco (columnas, one-hot, soporte)"
```

---

## Tarea 4: Hipergeométrica, pizarrón 11/10/5 y ejercicio ancla de Persona A

**Step 1: Tests** (añadir):

```python
from math import comb


def test_hipergeometrica_caso_homogeneo():
    """p_i ≡ p ⟹ Q[s][r'] = C(k,r')·C(m−k, r−r') / C(m,r), sin depender de p.

    'El heuristico usa una binomial donde la verdad es una hipergeometrica.'
    """
    m, r = 9, 4
    for p0 in (0.15, 0.37, 0.8):
        tensor = subpool_tensor([p0] * m, r)
        for s, col in tensor.items():
            k = s.bit_count()
            for rp in range(len(col)):
                if 0 <= r - rp <= m - k:
                    esperado = comb(k, rp) * comb(m - k, r - rp) / comb(m, r)
                else:
                    esperado = 0.0
                assert abs(col[rp] - esperado) < 1e-12


def test_pizarron_11_10_5():
    """El ejemplo con el que Francisco abrio la sesion: 11 personas, r=10,
    subprueba de 5. Soporte exacto {4,5}; homogeneo da (5/11, 6/11).
    El producto de marginales posteriores pone masa FUERA del soporte
    ('es imposible que R'=1' — y la heuristica dice que no lo es).
    """
    m, r = 11, 10
    p = [0.3] * m
    cache = subset_pmf_cache(p)
    s = (1 << 5) - 1                      # las primeras 5 personas
    col = subpool_tensor(p, r, cache)[s]
    assert np.allclose(col[:4], 0.0)
    assert abs(col[4] - 5 / 11) < 1e-12
    assert abs(col[5] - 6 / 11) < 1e-12

    # Marginal posterior dentro del pool: p~_i = p_i · f_{T∖i}[r−1] / f_T[r]
    full = (1 << m) - 1
    marg = [p[i] * cache[full ^ (1 << i)][r - 1] / cache[full][r]
            for i in range(m)]
    # Ley de independencia sobre las 5 personas: Poisson-binomial de p~
    indep = np.array([1.0])
    for i in range(5):
        indep = np.convolve(indep, [1.0 - marg[i], marg[i]])
    assert indep[:4].sum() > 1e-6         # masa donde la verdad es cero
    assert indep[0] > 0.0                 # P(limpio) heuristica > 0 = exacta


def test_ejercicio_ancla_persona_A():
    """Numeros del ejercicio a mano de Persona A (bloque A1, 27-jul).
    n=3, p=(1/2,1/2,1/2), r=2. Mundos consistentes: 110, 101, 011 —
    uniforme 1/3. GATE G1: si estos numeros difieren de los de A,
    PARAR y reconciliar a mano; no confiar en ninguno de los dos lados.
    """
    tensor = subpool_tensor([0.5, 0.5, 0.5], 2)
    assert np.allclose(tensor[0b001], [1 / 3, 2 / 3])          # {0}
    assert np.allclose(tensor[0b011], [0.0, 2 / 3, 1 / 3])     # {0,1}
    assert np.allclose(tensor[0b111], [0.0, 0.0, 1.0])         # T completo
    assert np.allclose(tensor[0b000], [1.0])                   # vacio
```

**Step 2:** Correr — PASS.

**Step 3 (GATE G1):** Persona A entrega sus números del ejercicio a mano.
Persona B los compara contra `test_ejercicio_ancla_persona_A` SIN mostrar antes
los del test. Discrepancia ⟹ sesión de reconciliación inmediata.

**Step 4: Commit**

```bash
git commit -am "test(laminar_inference): hipergeometrico, pizarron 11/10/5 y ejercicio ancla"
```

---

## Tarea 5: Demo reproducible para la sesión

**Files:**
- Create: `augmented/demo_tensor.py`

**Step 1: Implementar** (determinista, sin argumentos, imprime y sale):

```python
"""Demo del tensor condicional para la sesion con Francisco (28-jul).

Corre con: python3 -m augmented.demo_tensor
Tres actos: (1) el ejemplo del pizarron 11/10/5, exacto vs heuristica;
(2) el tensor de bolsillo n=3 en formato tabla; (3) tiempos y memoria
para G=5 y G=10.
"""
import time

import numpy as np

from augmented.laminar_inference import subpool_tensor, subset_pmf_cache


def acto_1_pizarron():
    m, r = 11, 10
    p = [0.3] * m
    cache = subset_pmf_cache(p)
    s = (1 << 5) - 1
    col = subpool_tensor(p, r, cache)[s]
    full = (1 << m) - 1
    marg = [p[i] * cache[full ^ (1 << i)][r - 1] / cache[full][r]
            for i in range(m)]
    indep = np.array([1.0])
    for i in range(5):
        indep = np.convolve(indep, [1.0 - marg[i], marg[i]])
    print('=== Acto 1: pizarron — 11 personas, r=10, subprueba de 5 ===')
    print("r'   exacta        heuristica-producto")
    for rp in range(6):
        print(f'{rp}    {col[rp]:.6f}      {indep[rp]:.6f}')
    print(f'La heuristica pone masa {indep[:4].sum():.4f} en resultados')
    print('imposibles (soporte exacto: {4,5} = hipergeometrica 5/11, 6/11).')


def acto_2_bolsillo():
    print('\n=== Acto 2: tensor completo de bolsillo — n=3, p=1/2, r=2 ===')
    tensor = subpool_tensor([0.5, 0.5, 0.5], 2)
    print("T'      r'=0    r'=1    r'=2    r'=3")
    for s in range(8):
        etiqueta = '{' + ','.join(str(i) for i in range(3)
                                  if (s >> i) & 1) + '}'
        celdas = ['  --  '] * 4
        for rp, v in enumerate(tensor[s]):
            celdas[rp] = f'{v:.3f} '
        print(f'{etiqueta:8s}' + '  '.join(celdas))
    print('Columna por fila suma 1; fila T completo es one-hot en r=2;')
    print('-- marca las entradas estructuralmente vacias (ley de soporte).')


def acto_3_escala():
    print('\n=== Acto 3: costo real de la tabla (la pregunta de G=10) ===')
    rng = np.random.default_rng(0)
    for m in (5, 10):
        p = rng.uniform(0.1, 0.9, size=m).tolist()
        t0 = time.perf_counter()
        cache = subset_pmf_cache(p)
        t1 = time.perf_counter()
        tensor = subpool_tensor(p, m // 2, cache)
        t2 = time.perf_counter()
        celdas = sum(len(c) for c in tensor.values())
        kb = celdas * 8 / 1024
        print(f'G={m:2d}: cache {1000*(t1-t0):7.2f} ms, tensor '
              f'{1000*(t2-t1):7.2f} ms, {celdas} celdas ≈ {kb:.0f} KB')
    print('La cache se hereda a todo el subarbol: tablas posteriores son O(1)')
    print('por celda. El costo es independiente de N (todo atomo cabe en G).')


if __name__ == '__main__':
    acto_1_pizarron()
    acto_2_bolsillo()
    acto_3_escala()
```

**Step 2:** `python3 -m augmented.demo_tensor` — corre sin errores, números
coherentes con los tests (5/11 ≈ 0.4545, 6/11 ≈ 0.5455).

**Step 3: Commit**

```bash
git add augmented/demo_tensor.py
git commit -m "feat(demo): demo del tensor con ejemplo del pizarron y tiempos G=5/G=10"
```

---

## Tarea 6: Falsificador de anidamiento (la evidencia que cambia decisiones)

**Files:**
- Create: `augmented/experiments_nesting.py`
- Modify: `augmented/tests_laminar_inference.py`

**Pregunta:** ¿el greedy EXACTO (score `P(r=0|H)·ganancia` sobre perfiles
supervivientes — el greedy que el tensor habilita) elige alguna vez re-testear
dentro de un pool con conteo positivo? Si la respuesta es "casi nunca salvo p
alto", el greedy laminar ≈ greedy estático y el día 3 pivota.

**Método exacto, sin Monte Carlo:** caminar el árbol de decisiones del greedy
(una acción por nodo, ramificación solo por resultados), ponderando cada
decisión por la probabilidad de su rama. Clasificación exhaustiva de cada
decisión: `virgen` (disjunta de todo lo testeado), `anidada` (⊆ algún pool
testeado — necesariamente positivo: los pools con r=0 quedan acreditados y dan
ganancia 0), `mixta` (toca territorio testeado sin caber en un solo pool).

**Step 1: Tests del experimento** (añadir a `tests_laminar_inference.py`):

```python
from augmented.experiments_nesting import greedy_nesting_stats


def test_nesting_b1_formula_cerrada():
    """Con B=1 el arbol tiene un nodo: valor = max_t ganancia·prod(q_i)
    (prior independiente ⟹ P(r=0) exacta = producto). Chequeo cerrado.
    """
    from itertools import combinations
    p = [0.2, 0.4, 0.1, 0.3, 0.25]
    u = [1.0, 2.0, 1.5, 1.0, 3.0]
    n, G = len(p), 3
    mejor = 0.0
    for size in range(1, G + 1):
        for c in combinations(range(n), size):
            prob = 1.0
            for i in c:
                prob *= 1.0 - p[i]
            mejor = max(mejor, prob * sum(u[i] for i in c))
    stats = greedy_nesting_stats(p, u, B=1, G=G)
    assert abs(stats['valor'] - mejor) < 1e-12
    assert abs(stats['virgen'] - 1.0) < 1e-12    # primer paso siempre virgen


def test_nesting_probabilidades_cierran():
    """Las categorias particionan las decisiones y las ramas suman 1."""
    p = [0.5, 0.6, 0.3, 0.7, 0.45, 0.55]
    u = [1.0] * 6
    stats = greedy_nesting_stats(p, u, B=3, G=3)
    partes = stats['virgen'] + stats['anidada'] + stats['mixta']
    assert abs(partes - stats['decisiones']) < 1e-12
    assert stats['decisiones'] <= 3.0 + 1e-12


def test_nesting_arbol_igual_simulacion_por_perfil():
    """Oraculo: el valor esperado del arbol == simular el mismo greedy
    perfil por perfil (misma politica, contabilidad reorganizada).
    """
    from augmented.experiments_nesting import greedy_value_per_profile
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 6
        p = rng.uniform(0.15, 0.85, size=n).tolist()
        u = rng.uniform(0.5, 3.0, size=n).tolist()
        a = greedy_nesting_stats(p, u, B=2, G=3)['valor']
        b = greedy_value_per_profile(p, u, B=2, G=3)
        assert abs(a - b) < 1e-10, seed
```

**Step 2:** Correr — FAIL (módulo no existe).

**Step 3: Implementación**

```python
"""¿El greedy exacto anida? — falsificador previo a construir greedy laminar.

Camina el arbol de decisiones del greedy miope con esperanzas EXACTAS
(P(r=0|H) por conteo de perfiles supervivientes, no producto de marginales)
y mide la fraccion, ponderada por probabilidad, de decisiones anidadas /
mixtas / virgenes. Exacto: sin Monte Carlo.

Correr la curva: python3 -m augmented.experiments_nesting
"""
from itertools import combinations

import numpy as np


def _profile_weights(p):
    n = len(p)
    w = np.empty(1 << n)
    for z in range(1 << n):
        prob = 1.0
        for i in range(n):
            prob *= p[i] if (z >> i) & 1 else 1.0 - p[i]
        w[z] = prob
    return w


def _pools(n, G):
    out = []
    for size in range(1, G + 1):
        for c in combinations(range(n), size):
            m = 0
            for i in c:
                m |= 1 << i
            out.append(m)
    return out


def _mejor_pool(pools, u, n, remaining, weights, mass, cleared):
    """Score exacto: (masa de r=0 / masa) x ganancia no acreditada.
    Desempate determinista por mascara menor."""
    best_score, best_t = -1.0, None
    for t in pools:
        gain = sum(u[i] for i in range(n)
                   if (t >> i) & 1 and not (cleared >> i) & 1)
        if gain <= 0.0:
            continue
        mass0 = sum(weights[z] for z in remaining if (z & t) == 0)
        score = (mass0 / mass) * gain
        if score > best_score + 1e-15 or (
                abs(score - best_score) <= 1e-15
                and best_t is not None and t < best_t):
            best_score, best_t = score, t
    return best_t


def greedy_nesting_stats(p, u, B, G):
    n = len(p)
    weights = _profile_weights(p)
    pools = _pools(n, G)
    stats = {'virgen': 0.0, 'anidada': 0.0, 'mixta': 0.0,
             'decisiones': 0.0, 'valor': 0.0}

    def walk(k, remaining, cleared, tested, prob):
        if k == B:
            return
        mass = sum(weights[z] for z in remaining)
        if mass <= 0.0:
            return
        t = _mejor_pool(pools, u, n, remaining, weights, mass, cleared)
        if t is None:
            return
        stats['decisiones'] += prob
        union = 0
        for ta in tested:
            union |= ta
        if (t & union) == 0:
            stats['virgen'] += prob
        elif any((t & ta) == t for ta in tested):
            stats['anidada'] += prob
        else:
            stats['mixta'] += prob
        buckets = {}
        for z in remaining:
            buckets.setdefault((z & t).bit_count(), []).append(z)
        gain = sum(u[i] for i in range(n)
                   if (t >> i) & 1 and not (cleared >> i) & 1)
        for r, zs in buckets.items():
            bm = sum(weights[z] for z in zs)
            if r == 0:
                stats['valor'] += prob * (bm / mass) * gain
            walk(k + 1, tuple(zs),
                 cleared | t if r == 0 else cleared,
                 tested + (t,), prob * bm / mass)

    walk(0, tuple(range(1 << n)), 0, (), 1.0)
    return stats


def greedy_value_per_profile(p, u, B, G):
    """Oraculo del arbol: simula el MISMO greedy perfil por perfil.

    Para cada perfil z, la historia es determinista; en cada paso se
    reconstruyen los supervivientes desde la historia y se elige con el
    mismo scorer. Contabilidad distinta, mismo valor esperado.
    """
    n = len(p)
    weights = _profile_weights(p)
    pools = _pools(n, G)
    total = 0.0
    for z in range(1 << n):
        historia = []
        cleared = 0
        valor_z = 0.0
        for _ in range(B):
            remaining = tuple(
                zz for zz in range(1 << n)
                if all((zz & t).bit_count() == r for t, r in historia))
            mass = sum(weights[zz] for zz in remaining)
            t = _mejor_pool(pools, u, n, remaining, weights, mass, cleared)
            if t is None:
                break
            r = (z & t).bit_count()
            if r == 0:
                valor_z += sum(u[i] for i in range(n)
                               if (t >> i) & 1 and not (cleared >> i) & 1)
                cleared |= t
            historia.append((t, r))
        total += weights[z] * valor_z
    return total


def curva_anidamiento(n=10, B=3, G=3, ps=(0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9)):
    print(f'n={n}, B={B}, G={G}, u=1 homogeneo')
    print('p      virgen   anidada  mixta    decisiones  valor')
    filas = []
    for p0 in ps:
        s = greedy_nesting_stats([p0] * n, [1.0] * n, B, G)
        d = s['decisiones']
        filas.append((p0, s))
        print(f"{p0:.2f}   {s['virgen']/d:.3f}    {s['anidada']/d:.3f}    "
              f"{s['mixta']/d:.3f}    {d:.3f}       {s['valor']:.4f}")
    return filas


if __name__ == '__main__':
    curva_anidamiento()
    print()
    # Variante con utilidades heterogeneas (la ganancia concentrada
    # puede empujar a re-testear donde vive la utilidad):
    rng = np.random.default_rng(1)
    n = 10
    u = rng.uniform(0.5, 5.0, size=n).tolist()
    print('u heterogeneo (seed 1):')
    print('p      virgen   anidada  mixta')
    for p0 in (0.2, 0.4, 0.6, 0.8):
        s = greedy_nesting_stats([p0] * n, u, 3, 3)
        d = s['decisiones']
        print(f"{p0:.2f}   {s['virgen']/d:.3f}    {s['anidada']/d:.3f}    "
              f"{s['mixta']/d:.3f}")
```

**Step 4:** Tests en verde, luego `python3 -m augmented.experiments_nesting`.
Guardar la salida (copiarla al paquete). **La curva se reporta tal como salga**
— si el anidamiento es raro, ese ES el resultado y se le presenta a Francisco
como tal.

**Step 5: Commit**

```bash
git add augmented/experiments_nesting.py augmented/tests_laminar_inference.py
git commit -m "feat(experiments): falsificador de anidamiento del greedy exacto"
```

**Stretch (solo si sobra tiempo):** misma curva con el DP óptimo (n≤6) para
distinguir "greedy no anida" de "anidar no sirve". Si no hay tiempo, queda como
primera tarea del día 3.

---

## Tarea 7 (Claude, paralelo): barrido de literatura QGT

**Files:**
- Create: `docs/notes/2026-07-27-revision-QGT.md`

Deep-research sobre: quantitative group testing adaptativo (canal de conteo),
conditional Bernoulli distribution (Chen–Liu 1997), asociación negativa
(Joag-Dev–Proschan 1983), binary splitting / Dorfman generalizado y su relación
con familias laminares, inferencia exacta en historiales anidados de conteos,
group testing con utilidades/bienestar. Pregunta central: ¿el contenido del
Lema A (átomos residuales + factorización + ley predictiva) existe ya con otro
nombre? Salida: nota con citas, qué es folclor, qué parece nuestro, y el
lenguaje seguro para usar con Francisco. Persona A la revisa el día 2.

---

## Tarea 8: Congelación (GATE G3)

**Step 1:** Suite completa desde el estado limpio:

```bash
python3 -m pytest augmented/tests_laminar_inference.py -v
```

Esperado: TODOS PASS. Cualquier fallo bloquea la sesión de demo de ese ítem.

**Step 2:** Ensayo de la demo completa, cronometrado:

```bash
python3 -m augmented.demo_tensor && python3 -m augmented.experiments_nesting
```

**Step 3:** Revisión cruzada final (con Persona A): checklist del mapeo
enunciado ↔ test (ver bloque A6). Congelar: no se toca código después de G3.

---

## Persona A — bloques de teoría (en paralelo, mismo día)

**A1 (1 h) — Ejercicio numérico a mano.** (i) Caso de bolsillo: n=3,
p=(½,½,½), T={0,1,2}, R=2 — enumerar los 3 mundos consistentes, llenar el
tensor completo a mano (las 8 columnas), verificar a mano: columnas suman 1,
columna T'=T one-hot, entradas vacías. (ii) Pizarrón: |T|=11, R=10, |T'|=5 —
deducir el soporte {4,5} SOLO con la ley de soporte, sin calcular; luego, con
p homogéneo, los valores 5/11 y 6/11 por hipergeométrica. Entregar los números
a B para el GATE G1.

**A2 (3 h) — Borrador propio de la "Nota técnica: la tabla condicional de
subpruebas"** (`augmented/paper/nota_tensor_subpruebas.md`, 2 páginas máximo):
1. El objeto, tal como Francisco lo dictó (input (T,R), input (T',R'), output).
2. Lema de soporte: `max(0, R−|T∖T'|) ≤ R' ≤ min(R, |T'|)` — con demostración
   de dos renglones (contar por complemento). Es "la parte 2 de la
   formalización" que pidió.
3. La forma cerrada, demostrada POR SU ruta de Bayes (la verosimilitud es
   `f_{T∖T'}[R−R']` por independencia del prior).
4. Corolario hipergeométrico (p homogéneo) — "binomial donde la verdad es
   hipergeométrica".
5. Corolario fila R'=0: score exacto = score de independencia × el cociente
   `f_{T∖T'}[R]/f_T[R]` (el gap de independencia aislado en un número).
6. Observación (no proposición): el producto de marginales sobreestima
   P(limpio) siempre — corolario de asociación negativa, Joag-Dev–Proschan
   (1983). CITA, no demostración propia. [VERIFICADO 0/2000]
7. Observación de costos (no teorema): caché Φ `O(2^G·G)` una vez, `O(1)` por
   celda después, heredada al subárbol; todo átomo cabe en G ⟹ costo
   independiente de N. Tabla de G=10 ≈ 90 KB.
8. Una línea de linaje interno: Corolario A.2(a) del Lema A con L={T}.
9. Sección final: "Relación con literatura: EN REVISIÓN (barrido en curso)".

**A3 (1 h) — Lupa de revisor** (con Claude): cada frase contra el ejercicio A1
y contra los tests de B. Las moralejas de siempre: enunciar sin argumentar,
tipos consistentes, cada término definido antes de usarse.

**A4 (1.5 h) — Versión final** de la nota (limpia, 2 páginas).

**A5 (1 h) — Guion de sesión (10 min de exposición):**
1. La tabla corriendo (demo actos 1–3), con sus tres checks como tests.
2. Alineación de vocabulario: "tus hojas y nuestros átomos son lo mismo — el
   Lema A le pone demostración a tu dibujo" (alineación, no corrección).
3. La corrección de escala: G=10 son 90 KB y milisegundos; la pregunta
   incremental es bonita, no bloqueante (herencia de Φ ya responde la mitad).
4. La curva de anidamiento, tal como haya salido, con su lectura honesta.
5. Calibración de expectativas: greedy-solo ≈ 1–2 pts; el premio es habilitar
   rollout (¾ del gap, bloqueado sin exactitud — notebook 21 §7).
6. Qué queda [PREGUNTA]: candidatos mixtos, tablas incrementales, robustez al
   prior, B&B (nota futura con la objeción de holgura de cotas anotada).

**A6 (0.5 h) — Revisión cruzada (GATE G2/G3):** leer los nombres de los tests
de B y verificar el mapeo 1:1 contra la nota; B lee la nota y verifica que no
afirme nada sin test o etiqueta.

---

# DÍA 2 (post-sesión; se ajusta con el feedback de Francisco)

## Persona A

- **A7 — Lema A parte (iii), propedéutico** (el único hueco del lema):
  ejercicio numérico (n=4, dos átomos disjuntos, posterior a mano, ver
  aparecer la factorización) → borrador propio (la indicadora factoriza sobre
  átomos por (ii); prior producto ⟹ bloques independientes) → lupa →
  integrar a `augmented/paper/lema_A_construccion.md`. Fallback si se atora:
  demostrar solo el caso de dos bloques disjuntos (suficiente para el día 3).
- **A8 — Casos degenerados y coherencia**: los `ValueError` de
  `laminar_atoms` como hipótesis del enunciado (sección pendiente del doc).
- **A9 — Leer el barrido de literatura** (Tarea 7) y decidir el lenguaje del
  paquete: qué es folclor citable, qué es empaque nuestro.

## Persona B

- **B7 — ~~Extracción de la celda 12~~ HECHA por Héctor** (`6ac1569`):
  `laminar_inference.py` recibe la jerarquía (sin parser cúbico) y
  `tests_laminar_milp.py` cubre identidad contra conteo y los `ValueError`.
  Tarea residual: LEER su módulo y sus tests con lupa (media hora), y anotar
  cualquier hueco de cobertura como test nuevo, no como opinión.
- **B8 — ~~Ley predictiva general~~ HECHA por Héctor**: `laminar_pool_pmf`
  ya convoluciona átomos + territorio virgen y está validada contra
  `exact_pool_pmf` (`tests_laminar_benchmarks.py:69`). Tarea residual:
  verificar que la batería incluya pools ADVERSARIOS (atravesando átomos,
  unión de átomos, mixtos virgen+átomo); si falta alguno, añadirlo (≤1 h).
- **B9 — Wrapper por hoja**: `atom_tensor(p, history, hierarchy, atom_mask)`
  = tensor del átomo con su conteo residual (la "tabla por hoja" del dibujo
  de Francisco): componer `laminar_atoms` de Héctor + `subpool_tensor` del
  día 1. Pocas líneas + test contra `exact_pool_pmf` en historiales
  ramificados n≤12.

## Sincronía día 2

Decidir si el Corolario A.2 del `.tex` se reescribe sin la hipótesis de
compatibilidad para la ley predictiva (la verificación numérica del 27-jul
dice que sobra; solo la clausura la necesita). Si sí, Persona A lo redacta con
lupa el día 3.

---

# DÍA 3 (condicionado al falsificador y a la sesión)

**Punto de decisión (30 min, ambas personas), según la curva de anidamiento:**

| Resultado del falsificador | Rumbo del día 3 |
|---|---|
| Anida de forma no trivial en régimen relevante (p medio-alto y/o u heterogéneo) | **Rama A**: greedy laminar v1 completo |
| Anida solo en p muy alto o casi nunca | **Rama B**: re-centrar en alta prevalencia + la comparación "creencias exactas / acciones restringidas vs. creencias aproximadas / acciones libres" |

## Rama A — greedy laminar v1

- **B10**: dos subrutinas, como las nombró Francisco: territorio virgen →
  selector existente con producto de priors (ahí es EXACTO, no heurístico;
  se reusa `greedy._myopic_best_pool` / `pool_solvers.py`); dentro de un
  átomo → candidatos `T' ⊆ D_A`, score `u(T')·Q[T'][0]` del tensor. v1
  EXCLUYE candidatos mixtos — decisión documentada, no omisión.
- **B11**: validación sin Monte Carlo: valor exacto de la política por
  enumeración de perfiles REUSANDO `laminar_benchmarks.py` de Héctor (las
  cuatro cantidades con firma común ya existen) en n≤10; verificar la cadena
  `greedy_laminar ≤ V^L ≤ V*` en un lote; comparar contra el greedy de
  producción (producto de marginales) — **el número del paquete: cuánto
  recupera el tensor.**

## Rama B — alta prevalencia / comparación informativa

- **B10'**: curva fina de anidamiento en p∈[0.5,0.9] con u heterogéneo +
  óptimo DP n≤6 (¿el óptimo anida donde greedy no?).
- **B11'**: el experimento "creencias exactas + acciones laminares vs.
  creencias producto + acciones libres" en n≤10 — la comparación que nadie ha
  corrido y es genuinamente informativa, con la misma maquinaria de B11.

## Persona A (ambas ramas)

- **A10**: análisis escrito del falsificador (la curva comentada, con su
  lectura por régimen — conecta con "laminar es útil cuando las tasas son
  altas" de Francisco).
- **A11**: ensamblar paquete #2: nota del tensor (final), Lema A completo
  (con (iii)), ley predictiva general, curva de anidamiento, número de
  recuperación (rama A) o comparación (rama B), y la lista [PREGUNTA]:
  tablas incrementales (enunciadas, no implementadas), candidatos mixtos,
  robustez al prior (diseño: greedy exacto con prior perturbado vs.
  heurística con prior verdadero), Conjetura C (agenda de la próxima
  semana), B&B (nota futura + objeción de holgura).

---

# Qué NO hacemos (recortes deliberados)

- **Atlas y búsqueda adversaria**: ya NO están pospuestos — Héctor los
  entregó (`6ac1569`). Lo que este plan no hace es EXTENDERLOS; revisarlos
  y absorber sus números (0.9069 adversario, igualdad B≤2 homogéneo) es
  material del guion de sesión y del día 2, no trabajo nuevo.
- **Tablas incrementales padre→hijos**: se enuncia la herencia de Φ como
  observación con números; no se construye maquinaria incremental (90 KB no
  ameritan optimización prematura).
- **Branch & Bound**: nota futura; las cotas actuales (0.63–0.84) no podan
  diferencias del 5%.
- **Optimización G>10, MILP por escenarios, pipeline n=40**: fuera del
  horizonte.
- **Proposiciones nuevas con nombre propio**: nada se llama "teorema" ni
  "proposición" en el paquete si es corolario de literatura citable o
  ingeniería de caché.

# Riesgos y mitigaciones

1. **El test de identidad bruto/cerrado es lento** (oráculo 4^m): acotado a
   m≤8 y 40 semillas (~segundos). No subir m en el oráculo; la forma cerrada
   ya queda validada.
2. **La parte (iii) se atora**: fallback declarado — dos bloques disjuntos,
   que es lo que consume el día 3.
3. **El falsificador da "nunca anida"**: no es fracaso, es EL resultado; la
   sesión lo presenta como tal y el día 3 tiene rama B lista.
4. **La sesión con Francisco cambia prioridades**: los días 2–3 son módulos
   independientes; se reordenan sin romper el día 1, que queda congelado.
5. **Oráculo general limitado a n≤18** (`exact_pool_pmf`): se declara en la
   nota; no se extrapola validación más allá.
