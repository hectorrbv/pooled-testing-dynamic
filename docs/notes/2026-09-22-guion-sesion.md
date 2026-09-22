# [BORRADOR, se congela en el sync de las 16:00] Guion de sesión 2026-09-22, 19:00 Praga / 11:00 CDMX

**Protocolo §34-bis:** paquete y preguntas congelados antes de la sesión (A-M20). Última sesión real: 1-sep (acta `2026-09-01-sesion-francisco.md`, despacho aplicado). Las del 8 y 15-sep no ocurrieron. Reparto: **Héctor 10 min** (su guion: `2026-09-22-guion-francisco-hector.md`, notebook 26 §5→§7→§8→§10) · **A 7 min** (Proposición de brecha de convención, `augmented/paper/proposicion_brecha_convencion.tex`) · **Héctor 3 min** (solver validado, ancla, C1) · preguntas. Todo número citado está en el pie del `.tex` o en los `.meta.json` de B; etiquetas §25 donde aplica. Grabar; acta y despacho el miércoles.

## Frase de apertura (A)

> "Cumplimos el encargo del 1-sep con sobre-entrega y lo presenta Héctor: contraejemplo donde todos los greedy fallan, el ingrediente que lo repara y una constante empírica del portafolio. Yo traigo la primera pieza propia del paper bajo posterior-zero: la brecha de convención, con el ejemplo de cuatro personas que te pareció interesante, ya como proposición con prueba y verificación. Traemos cinco preguntas de decisión."

## Paquete

1. **Héctor, 10 min:** contraejemplo universal (n=6, B=3, G=4: las cuatro políticas a 0.6576 del óptimo) → π_L con índice Lagrangiano (0.9641) → constante del portafolio best-of-5, 0.9307 reproducido en la candidata registrada. *Frase defendible:* "el 0.9307 es conjetura empírica reproducida con esta especificación; no tenemos garantía universal del 93%".
2. **A, 7 min, en el pizarrón:** las siete frases (sección "A's version" del `.tex`) sobre el árbol de cuatro personas. Orden: dominación (1 línea) → el evento único, par con conteo 1 y refinamiento: paga 1 vs ½ → los tres números 0.774 / 0.564 / 0.6 → la política óptima se voltea → el umbral ½ de la Prop 1 de Nick aparece bajo estricta y desaparece bajo posterior-zero, en el juego de dos pruebas → Mapa 1 de Héctor como Figura 1 (665/665). Cierre: "la brecha es un fenómeno de conteos: con pruebas binarias las dos reglas coinciden". Parte 2 se declara enunciada, con el testigo 0.774 ≤ 1.011.
3. **Héctor, 3 min:** el juez validado por dos enumeradores independientes (626 tests exactos, n≤5, B≤3, ambas convenciones); k del ancla corregida a 3 (0.9147); C1 recontada gana/empata/pierde 10.9/56.1/33.0. **Mapa 3:** el plan cover-then-bisect no es óptimo (frontera n=24, G=8, B=6: 0.8654 vs 0.7080; ancla restringida a G≤8: ≥1.1233 vs 0.9147); el ancla completa G=16 sigue abierta computacionalmente.

## Preguntas congeladas (§34), en orden de decisión

- **(19) Reparto de territorios:** proponemos no-aumentado dinámico = paper de Nick (factor 2 de adaptabilidad, greedy constante, singletons si q_sano ≤ ½); aumentado laminar posterior-zero (modelo, Bellman exacto, brecha de convención, separación, portafolio) = nuestro. ¿De acuerdo? ¿Ya salió el paper reescrito (arXiv sigue en v4)?
- **(18) Estatuto del companion tras el reparto:** ¿se funde en nuestro paper contigo como coautor, o queda como nota técnica citada? ¿Qué fecha vemos para un draft arXiv-able?
- **(23) La barrera:** en tu argumento perfil/antiperfil, ¿en qué paso concreto se rompe el coupling cuando la prueba devuelve un conteo y el crédito es por deducción? ¿Es ese el obstáculo para pasar de 1/G a constante?
- **(24) Regla de λ para π_L:** ¿λ = OPT/B estimado, dual del Thm 10.2, o q̄·ū? ¿Vale como especificación de política para un enunciado de factor? (Pregunta 3 de Héctor: fijar horizonte, no-parálisis y rejilla antes de volver a medir la batería.)
- **Logística:** hora estable del semestre con nosotros en Praga (19:00 aquí = 11:00 CDMX; tú dijiste que miércoles por la mañana era posible).

Preguntas de Héctor que van dentro de su bloque: (i) ¿qué continuación incluye cover-then-bisect cuando encuentra sanos antes de agotar B? (el "CBS" necesita definición antes de un único valor); (ii) hipótesis de población de la proposición de primera acción (homogénea, n = B·G).

## Extras (mantra; solo si hay tiempo, nunca como claims)

- Reconstrucción escrita del argumento perfil/antiperfil de Francisco (una página de A; entra solo si existe a las 18:00). Él pidió que revisáramos su prueba: "cuando tenga el resultado todo escrito se los paso".
- Hallazgo: con B=3 la ventaja de agrupar no es monótona (empate exacto par/singleton bajo posterior-zero en q=0.3, 537/500).

## Qué NO se dice

- "El ancla es exacta" o "cover-then-bisect es óptimo": Mapa 3 dice lo contrario. El 0.9147 es el valor de un plan fijo bajo posterior-zero.
- "0.9307 es una garantía". Es conjetura empírica reproducida.
- Nada del companion como validado ni como nuestro antes de (18)/(19).
- Las fórmulas de la Proposición para q > ½ (son cotas inferiores; el enunciado vive en q ≤ ½).
- El umbral ½ bajo estricta fuera del juego de dos pruebas (con B=3 la estricta ya agrupa en q=0.3).
- Thm 9.3 y su término sobrante: fuera hasta que A lo re-derive.

## Checklist para el sync de las 16:00

- [ ] A: siete frases aprobadas; `.tex` compila (`git add -f`, está ignorado por `*.tex`).
- [ ] Héctor: su guion y el notebook 26 ejecutado; Mapa 1 como figura lista para compartir pantalla.
- [ ] Los dos: ensayo de 7 + 10 min; quién comparte pantalla; grabación.
- [ ] Commit y push de guion y `.tex` antes de las 17:30.
