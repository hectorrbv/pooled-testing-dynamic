# La constante empírica del portafolio best-of-k (2026-09-02)

Encargo de la sesión 2026-09-01 [48:10–49:53]: *"corre estos greedy y toma el
mejor de los tres… si hay algún caso donde a todos les va mal, tenemos que
buscar en otra parte; si no lo encuentran, vamos bien"*. **Estatuto §25:
diagnóstico.** Reproducible: misión `constante` de dapts-autoresearch
(`results_constante.tsv`).

## 1. Resultado: no hay contraejemplo del portafolio de cinco

Tras ~7,000 instancias tamizadas y 10 candidatas registradas, el mínimo del
ratio del portafolio contra el óptimo laminar exacto es **0.9307**. Ninguna
instancia hunde a las cinco políticas a la vez.

Batería: π_M (inmediato), π_C (committed density), π_R (receding density) del
§8 del companion, C3 (keep de la misión V̂) y π_L (índice Lagrangiano, ec.
8.13). Portafolio **ex ante** en el sentido del Thm 10.3: el valor de cada
política se calcula exacto bajo el prior, sin mirar el óptimo.

| Instancia | π_M | π_C | π_R | C3 | π_L | portafolio |
|---|---|---|---|---|---|---|
| peor hallada (n=7, B=3, G=4) | 0.7870 | 0.7870 | 0.7870 | 0.6426 | **0.9307** | **0.9307** |
| contraejemplo del 1-sep | 0.6576 | 0.6576 | 0.6576 | 0.6567 | **0.9641** | 0.9641 |
| B-M16 | 0.7752 | 0.7752 | 0.7752 | **1.0000** | 1.0000 | 1.0000 |
| rare-health G=4 | 0.5087 | 0.6359 | 0.9976 | **0.9999** | 0.9976 | 0.9999 |

Obsérvese que **la política ganadora cambia en cada instancia**: el portafolio
no es redundancia, es cobertura complementaria — exactamente el argumento del
Thm 10.3. Nótese también que el contraejemplo que refutaba al portafolio de
cuatro (0.6576) deja de serlo al añadir π_L.

## 2. La peor familia hallada, y su mecanismo

n=7, B=3, G=4, en tres bloques: tres personas **seguras** (p≈0, u≈1), tres de
**lotería** (p=0.975, u≈40) y una de **relleno** (p≈0.51, u≈1.08).

El óptimo **abre primero el pool de la lotería**: esa prueba revela si hay
algún sano entre los tres premios; en la rama mala (r=3, prob 0.927) todavía
le quedan dos pruebas para el bloque seguro y el relleno. Las políticas de
índice son locales por componente y toman primero el pool seguro, que paga más
por prueba ex ante. Es el obstáculo de **costos efectivos aleatorios** (§10.3,
punto 2): una exploración que casi siempre falla barato pero cuyo premio es
enorme.

## 3. El hallazgo metodológico: la constante depende de cómo se calibre λ

La primera medición de esa instancia dio **0.7940**, y resultó ser un
artefacto: la rejilla de λ era gruesa (seis puntos, razón ≈2) y la ventana
buena —λ ∈ [1.1, 1.5]— caía **entre** dos puntos de la rejilla (1.070 y
2.141). Verificado por barrido fino independiente:

| λ | 0.001–1.0 | **1.1–1.5** | 2.0+ |
|---|---|---|---|
| ratio de π_L | 0.7940 | **0.9307** | 0.7870 |

Con la rejilla corregida (log-espaciada, razón 1.4, dos décadas alrededor de
Σqᵢuᵢ/B) π_L encuentra la ventana y el portafolio sube a 0.9307.

**Consecuencia para el enunciado:** la calibración de λ **es parte de la
especificación de la política**, no un detalle de implementación. Una rejilla
gruesa subestima al portafolio en 14 puntos. Cualquier claim de la forma "el
portafolio es una α-aproximación" tiene que declarar la rejilla; si no, α no
está definido. Es la pregunta abierta 10.5 del companion, y la misión le pone
un número al costo de ignorarla.

## 4. Mapa de la búsqueda

**Regiones que bajan el ratio:** presupuesto apretado (B=3) con G=4; contraste
extremo de infección (bloque p≈0 contra bloque p≈0.975); premio grande en el
bloque casi seguro infectado; una persona de infección intermedia como
relleno. n=7 es el punto dulce.

**Regiones que no bajan:** B=4–5 (todo →1: el horizonte rodante de 3 de π_L no
se paga); G=2 (nada bajo 0.90 en 400 corridas); infección intermedia uniforme
(0.35–0.75); n=8 (diluye); búsqueda aleatoria amplia (~7,000 instancias,
mínimo 0.96 sin estructura).

π_L nunca cayó por debajo de 0.844 en todo lo tamizado y nunca fue mucho peor
que las otras cuatro: no se le encontró una debilidad complementaria que
permita componer bloques que hundan a todas.

## 5. Consecuencias

- **Para el paper:** α ≈ 0.93 es una conjetura empírica razonable para el
  portafolio de cinco en n ≤ 8, G ≤ 4, B ≤ 5, bajo la rejilla declarada. Es el
  desenlace bueno de los dos que fijó Francisco: *"si no logran encontrar un
  ejemplo, nos da ok, vamos más o menos bien"*.
- **Para A:** el blanco a demostrar tiene número. Y la peor familia hallada
  (bloque seguro / lotería / relleno) es el caso de prueba que cualquier
  intento de demostración debe sobrevivir.
- **Cautelas:** rango chico (n ≤ 8, G ≤ 4, B ≤ 5), sin garantía probada, y el
  número depende de la rejilla de λ. Ninguna política se adopta sin G4a/G4b.
