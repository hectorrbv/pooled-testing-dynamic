# [BORRADOR, se congela en el sync de las 16:00] Guion de sesión 2026-09-29, 19:00 Praga / 11:00 CDMX

**Protocolo §34-bis:** paquete y preguntas congelados antes de la sesión (A-M20). Acta previa: `2026-09-22-sesion-francisco.md` (despacho aplicado 28-sep). Regla nueva desde el 22-sep: **lo que no se alcance a decir se manda por escrito antes o después de la sesión** (el PDF de la brecha y este guion). Reparto: **A 8 min** (la separación como teorema propio) · **Héctor 8 min** (Mapa 4: el candidato de Q1) · **A 2 min** (el paper existe) · preguntas. Todo número con fuente: `results/mapa4_razones.csv`, `results/mapa_homogeneo_3.csv`, §16 del plan, `cota_clasica_construccion.md`.

## Frase de apertura (A)

> "Tomamos tus dos preguntas como el norte del paper y las convertimos en un plan de tres hitos. Hoy traemos tres cosas: la separación ya no como ejemplo sino como teorema con prueba propia de la cota clásica, la evidencia de cuál regla con planeación es el candidato para tu pregunta 1, y el esqueleto del paper con cada enunciado en su sección. Y las cuatro preguntas que el ruido no nos dejó hacer."

## Paquete

1. **A, 8 min — Teorema de separación.** (a) Lema de la cota clásica: en el modelo sin conteos, con probabilidad de sano ≤ ½, ninguna política dinámica supera B·q; prueba en tres pasos: una prueba sobre m personas rinde m·q^m sin historia; m·q^m ≤ q si q ≤ ½; la historia no ayuda porque los positivos son malas noticias (Harris) y los negativos sacan a su gente. Es tu Thm 7.1, validado por nuestra cuenta; si tu §4–5 lo contiene, te citamos. **Estatus a decir: [DEMOSTRADO] solo si A cerró las Partes 1–4 antes de las 16:00; si no, "prueba en curso, partes 1–2 cerradas".** (b) Con el lema, la separación se mide contra el óptimo clásico verdadero, no contra una cota: ancla (0.05, 16, 7): 0.35 clásico contra 0.9147 (CBS) y 1.1233 (exacto restringido a G ≤ 8, Mapa 3): factores 2.61 y 3.21; frontera (0.05, 8, 6): 0.30 contra 0.8654, factor 2.88. (c) Familia G = 2^j, B = j+1, q = c/G: la razón crece como G / log G, sin cota. (d) La brecha de convención no toca la separación: tu observación del martes es correcta, y el ejemplo escrito ya te lo mandamos en PDF.
2. **Héctor, 8 min — Mapa 4.** 36 celdas exactas (n = 6 homogéneo, q ≤ ½, G 2–4, B 2–3). Razón valor/óptimo, mínimo sobre la malla: **π_ratio 0.9524** (óptima en 32 de 36), π_L 0.9500, C3 0.6330, y el greedy clásico y las dos densidades 0.5417 sin acertar ninguna celda. Tus tres encargos: (i) la instancia donde las reglas difieren: q = 0.1, G = 4, B = 2, tres decisiones distintas, 0.367 de dispersión; (ii) el clásico abre singleton en las 36 celdas y el óptimo abre pool en las 36: la decisión subóptima que describiste, medida; (iii) λ ordena las decisiones por tamaño con umbral entre 0.2 y 0.4 (0.9578 contra 0.6588). **Frase defendible:** "en esta malla π_ratio no baja de 0.95; no es una cota, es el candidato que la evidencia señala".
3. **A, 2 min — El paper existe.** `augmented/paper/main.tex`: ocho secciones, cada enunciado que tenemos en su sitio con estatus, y una tabla de qué existe y qué falta. Hitos: octubre, modelo y algoritmo exacto escritos más el intento de garantía para π_ratio en salud rara; noviembre–diciembre, teorema o caracterización, draft, arXiv.

## Preguntas congeladas (§34), en orden de decisión

- **(26) El candidato.** La evidencia señala a π_ratio (utilidad esperada entre pruebas esperadas, tu sugerencia). ¿Te parece el candidato correcto para un teorema dentro de laminar? Ruta que vemos: la relajación Lagrangiana del presupuesto esperado (tu Thm 10.2) hace de π_L y π_ratio la misma regla con el multiplicador óptimo; ¿atacamos la garantía en salud rara por ahí?
- **(27) La cota clásica.** ¿Tu §4–5 contiene el resultado "q ≤ ½ ⇒ el óptimo dinámico es individual" con prueba? Nosotros lo probamos vía Harris para la sección de separación; si está, te citamos y comparamos pruebas.
- **(19) Territorios.** No-aumentado dinámico = paper de Nick; aumentado laminar posterior-zero (modelo, algoritmo exacto, brecha, separación, portafolio) = nuestro. ¿De acuerdo?
- **(18) Estatuto del companion** tras el reparto: ¿se funde en nuestro paper contigo como coautor, o queda como nota técnica citada? ¿Fecha para un draft arXiv-able?
- **(23) La barrera.** ¿En qué paso concreto se rompe el coupling perfil/antiperfil cuando la prueba devuelve un conteo y el crédito es por deducción?
- **Logística:** hora estable; el paper §4–5 cuando puedas.

Respuesta lista si vuelve a preguntar "¿esto depende de laminar?": no; la brecha es un fenómeno de conteos y existe con pools arbitrarios; laminar la hace computable. Pendiente de pedir: la revisión de tu Parte 2 por Héctor se hace hoy o mañana.

## Extras (solo si hay tiempo, nunca como claims)

- Parte 2 de la brecha con su cota B−1, y que Francisco enunció el mecanismo solo el 22-sep.
- Con B = 3 la ventaja de agrupar no es monótona (empate exacto par/singleton en q = 0.3).

## Qué NO se dice

- "π_ratio tiene garantía 0.95": es el mínimo de una malla n = 6, G ≤ 4, B ≤ 3; estatuto diagnóstico.
- "El lema clásico está demostrado" si A no cerró la Parte 3 (Harris) y la Parte 4 (suma sin doble conteo).
- "El ancla es exacta" o "CBS es óptimo": Mapa 3 dice lo contrario; 1.1233 es exacto solo restringido a G ≤ 8.
- Nada del companion como validado ni como nuestro antes de (18)/(19).
- Las fórmulas de la brecha para q > ½; el umbral ½ fuera del juego de dos pruebas.

## Checklist del sync de las 16:00

- [ ] A: Partes 1–4 de `cota_clasica_construccion.md` cerradas o estatus honesto; tabla de números apretados llena; PDF de la brecha enviado a Francisco.
- [ ] Héctor: notebook 27 ejecutado; figura del Mapa 4 lista; revisión cruzada de la Parte 2 agendada (hoy o mañana).
- [ ] IA: `seccion_separacion.tex` con la prueba de A; `main.tex` actualizado con el candidato π_ratio.
- [ ] Los dos: ensayo 8 + 8 + 2; quién comparte pantalla; grabación con dos dispositivos otra vez.
