# Cuentas de B para la ratificación de G0 (posterior-zero) — 2026-08-31

Material para el bloque conjunto del lunes (plan de semana, §LUNES). Son las
tres cuentas que cambian con la convención, re-derivadas de forma independiente
por B; los números coinciden con los del despacho de A (acta 2026-08-25, D2).
La convención en juego: bajo **posterior-zero** una persona queda acreditada en
cuanto su posterior de sana vale 1 (la deducción acredita); bajo la variante
**estricta** solo acredita una prueba observada limpia (la regla con la que se
construyeron los notebooks 24–26).

## Cuenta 1 — el ancla del acid test: 0.806u → 0.9147u

Instancia del ancla (§16): q = 0.05 de sana, G = 16, B = 7, u ≡ 1. La política
canónica cubre k pools raíz de 16 y, si algún pool muestra sanos, localiza una
por bisección de conteos en ⌈log₂ 16⌉ = 4 pruebas. El presupuesto se reparte
como B = k + 4 + (prueba acreditadora), y ahí está todo el cambio:

- Estricta: la persona localizada por descarte necesita su prueba limpia
  propia, así que k = 7 − 4 − 1 = **2** pools raíz, y el bienestar garantizado
  es 1 − 0.95³² = **0.806289u**.
- Posterior-zero: la bisección deja a la persona con posterior 1 y eso ya
  acredita; k = 7 − 4 = **3**, y el bienestar sube a 1 − 0.95⁴⁸ =
  **0.914742u**.

El baseline de 7 singletons no toca deducción alguna y queda igual en ambas
convenciones: 7 × 0.05 = 0.35u.

## Cuenta 2 — la reentrada de B-M16: 0.5 → 1.0

Estado del contraejemplo: par {a,b} probado con conteo 1, cada miembro con
posterior 1/2, utilidad viva 1, queda 1 prueba. La acción es reentrar con {a};
las dos ramas pesan 1/2:

| Rama | Estricta | Posterior-zero |
|---|---|---|
| a sana (prob 1/2) | acredita a: +1 | acredita a: +1 |
| a activa (prob 1/2) | b queda deducida sana, sin acreditar: +0 | posterior de b = 1, acredita: +1 |
| **Valor de la reentrada** | **0.5** | **1.0** |

Consecuencia directa: bajo posterior-zero la reentrada individual pasa a
dominar el menú del contraejemplo incluso en valor realizable, y la mitad
"cara" del costo local del par (las 2 subpruebas de la rama con deducción) se
vuelve 1. El costo local de un par fresco baja de 3q(1−q) a 2q(1−q)·1 = 2q(1−q)
subpruebas.

## Cuenta 3 — el colapso S₁ = S₀ queda adscrito a la variante estricta

En el mismo estado, la acción {a}: S₀({a}) = P(a sana)·u = 0.5. El lookahead a
un paso con clearing estricto cobra exactamente lo mismo (solo la rama
observada limpia): S₁ʰᵃʳᵈ({a}) = 0.5 = S₀ — el colapso de §14.3. Con
posterior-zero la rama de deducción también cobra y S₁ᵖᶻ({a}) = 1.0 ≠ S₀: el
colapso deja de ser un hecho de la convención normativa y pasa a ser una
propiedad de la variante estricta.

## Elementos para la decisión de B

1. Nada de lo entregado se invalida: los notebooks 24–26, el costo local y el
   barrido α quedan correctos **como variante estricta**, que es exactamente lo
   que la pregunta (21) propone conservar como columna del harness.
2. La migración es barata si B-M17 nace con el flag `posterior_zero | strict`
   (nota de diseño del mismo día): los números duales de B-M18 salen del mismo
   solver.
3. El punto a vigilar, si se objeta algo: posterior-zero acerca el modelo al
   companion (Remark 4.2 lo exige para que Thm 4.1 transfiera literal) pero
   aleja la utilidad del evento observable "prueba limpia"; si el modelo
   práctico de referencia exige acreditación por prueba, la variante estricta
   debe seguir viva en el harness, no solo en el texto. Eso es una fila de
   seguimiento, no una objeción bloqueante.

Verificación: las tres cuentas recomputadas numéricamente el 2026-08-31
(0.806289, 0.914742, 0.5→1.0) coinciden con el acta y el plan de semana.
