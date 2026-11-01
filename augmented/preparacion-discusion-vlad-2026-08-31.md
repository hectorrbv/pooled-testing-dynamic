# Preparación para la discusión con A (bloque conjunto, lunes 31-ago)

Agenda del bloque según el plan de semana: (1) A presenta G0 y tú ratificas u
objetas, (2) confirmas tu estado heredado, (3) se congela el guion del martes
(la sesión es mañana 1-sep, 11:00), (4) A declara su pieza. Media hora.

## 1. Tu posición en G0: ratificar, con una condición registrada

Llegas con cuentas propias, no solo con las de A
(`docs/notes/2026-08-31-ratificacion-G0-cuentas-B.md`), y además el solver ya
las verificó por máquina: reentrada 0.5 → 1.0 exacta, y la brecha completa en
la instancia de su spec (0.6 → 0.774 con cambio de primera acción). Ratifica.
La condición para la fila de §32: la variante estricta vive como columna del
harness (es la pregunta 21 del guion — tu ratificación y esa pregunta son
consistentes, díselo así). Punto a vigilar sin bloquear: si el modelo práctico
exigiera acreditación por prueba observada, estricta es la que lo modela.

## 2. Lo que tú reportas (estado heredado + lo de hoy)

- Notebook 26 completo y pusheado (B-M6 ext, B-M16 con CSV, barrido α).
- Nota de diseño B-M17 cotejada contra el companion real (ec. 3.5, 5.1–5.5,
  Prop 6.1) — `docs/notes/2026-08-31-diseno-BM17.md`.
- **El toy corrió y pasa sus tests de aceptación** (los de su spec del 20-ago):
  óptimo estricto 3/5 sin agrupar, par primero 0.564 exacto. El guion decía
  "[+ demo si corrió]": corrió — que A lo deje en afirmativo al congelar.
- El regalo para su pieza: el **número dual** 387/500 = 0.774 con la primera
  acción óptima cambiando de singleton a par. Su Proposición de brecha de
  convención nace con evidencia computacional (cuidado §25: evidencia por
  enumeración, no prueba — así se etiqueta).
- Consistencia cruzada: B=3 estricto reproduce el 1.011 del caso de sesión.

## 3. Verificaciones que te tocan a ti sobre la parte de A

- **A-M24 núcleo:** pídele la re-derivación a mano del ejemplo ABCD (test
  {a,b} conteo 1, luego {a}: ambas ramas pagan 1.0, nadie paga dos veces) y
  cotéjala contra el solver en vivo — es un assert de un minuto.
- **El ancla:** su k=3 y 1−0.95⁴⁸ ≈ 0.9147 coinciden con tus cuentas y con el
  Ej 7.5 del companion (mismo objeto, otros parámetros). Confírmalo dicho.
- **El guion congelado:** revisa que los cinco puntos del paquete sigan
  siendo ciertos tras el día — en particular el punto 1 ("ratificado por
  ambos el lunes" será cierto solo si cierran G0 hoy) y el punto 3 (demo).

## 4. Tus preguntas para A

1. La semántica de 5.1 en cadenas de deducción (complemento de complemento):
   mi implementación acredita al crearse el átomo con conteo extremo, vía ν —
   ¿coincide con su lectura del PDF? (Es el riesgo abierto de la nota de
   diseño.)
2. El enumerador pathwise de referencia (C-M3) para la validación del
   miércoles: ¿lo produce el soporte IA hoy o lo escribo yo? Sin él no hay
   "dos vías".
3. Pregunta (22) antes de validar: ¿comparamos el solver contra qué clase?
   Like-with-like: pathwise contra pathwise; nada contra el atlas ex ante sin
   etiqueta.
4. El esqueleto `laminar_formulation.tex` quedó fuera del repo por el
   `.gitignore` de `*.tex`: ¿lo subo yo al Overleaf o lo integra él?

## 5. Fricciones posibles, con tu postura

- **Convención de conteos:** notebooks 24–26 cuentan sanos; companion, spec y
  solver cuentan infectados. Postura: el solver habla la del companion y cada
  documento declara la suya una vez; no se reescriben los notebooks.
- **Estatuto del companion:** nada de §8+ se cita como nuestro ni como
  demostrado hasta A-M23 (y el reparto con Nick, D10, sigue pendiente).
- **La sorpresa de B=3:** bajo posterior-zero, con B=3 el par y el singleton
  empatan exactos (537/500) — la ventaja estricta de agrupar de B=2 (+0.174)
  se disuelve cuando el presupuesto sobra. No debilita la brecha: muestra que
  la ventaja no es monótona en B. Mejor decirlo nosotros antes de que
  Francisco lo encuentre.

## 6. Checklist de cierre del bloque

- [ ] G0 ratificado por B → fila en §32 (con la condición de la columna estricta).
- [ ] Guion congelado con demo en afirmativo y punto 1 actualizado.
- [ ] Acordado quién corre el enumerador pathwise mañana por la tarde.
- [ ] Confirmado: sesión mañana 11:00, se graba, acta el miércoles.
