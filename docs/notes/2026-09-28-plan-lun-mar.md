# Plan lun 28 → mar 29-sep — recuperar el hilo: del ejemplo al teorema, del contraejemplo al candidato

**Sesión con Francisco:** martes 29-sep, 19:00 Praga / 11:00 CDMX. **Sync A+B:** martes 16:00. **Presupuesto:** A 5 h hoy + 8 h mañana; B 5 h + 8 h. **Autocontenido:** Héctor puede leerlo sin el chat. Fuentes: acta del 22-sep (`2026-09-22-sesion-francisco.md`, dos transcripciones), plan maestro §1 y §32 (filas 2026-09-22).

## 1. Diagnóstico honesto (A, 28-sep)

Tres semanas sin sesión, mudanza y enfermedad; el trabajo de A vive en chats y no en el repo; lo que avanzó fue instrumentación y contraejemplos, no teoremas ni texto del paper. La garantía (Q1 de Francisco) es genuinamente difícil: no se ataca en general, se ataca en el régimen donde ya sabemos qué pasa (salud rara, q_sano ≤ ½), que es además la familia de la separación.

## 2. Norte y meta

- **Norte (Francisco, 22-sep):** Q1, algoritmo laminar eficiente aproximadamente óptimo dentro de laminar, con greedies **con planeación** como candidatos (el inmediato queda descartado); Q2, precio de lo laminar. "Cualquier resultado incremental es valioso."
- **Meta (A):** paper de primer nivel, SODA (julio 2027).
- **Tres hitos:** (1) esta semana: esqueleto del paper con enunciados (`augmented/paper/main.tex`), separación como teorema propio, Mapa 4 elige al candidato de Q1. (2) octubre: secciones de modelo y algoritmo exacto; garantía para el candidato en salud rara; precio laminar por tipos. (3) nov–dic: teorema o caracterización honesta; draft; arXiv.

## 3. Qué compra cada persona en estos dos días

**A — la separación como teorema, no como números.** Francisco lo llamó pendientillo porque él lo ve como aritmética sobre sus resultados no publicados. Nosotros lo hacemos autónomo: probar en casa que en el modelo clásico (sin conteos), con toda probabilidad de sano ≤ ½, ninguna política dinámica supera a las pruebas individuales: OPT = B·q·u. Prueba en tres pasos: (1) una prueba sobre m personas rinde a lo más m·q^m en esperanza sin historia; (2) m·q^m ≤ q para todo m si q ≤ ½ (igualdad en m=1 y, si q=½, en m=2); (3) la historia no ayuda: los resultados positivos son eventos decrecientes en los indicadores de salud, luego por Harris/FKG bajan la probabilidad de que un grupo esté limpio; los negativos acreditan y sacan a su gente. Es el Thm 7.1 del companion validado (A-M23) y el lema de la sección de separación. Con él: ancla 0.9147u (CBS) y 1.1233u (exacto restringido a G≤8, Mapa 3) contra 0.35u → factores 2.61 y 3.21 contra el óptimo clásico verdadero; frontera n=24, G=8, B=6: 0.8654u contra 0.30u → 2.88; familia G=2^j, B=j+1, q=c/G: razón ≥ (1−(1−c/G)^G)/((j+1)c/G) ~ G/log G → ∞. Entregable: la prueba en el cuaderno de A hoy, `seccion_separacion.tex` (IA a partir de la prueba de A) mañana. Además: PDF de la brecha a Francisco con la petición de §4–5 y el aviso de que si su §4–5 contiene la cota clásica, lo citamos.

**B — el Mapa 4 con razones, no solo primeras acciones.** Sobre la malla del Mapa 1 restringida a q_sano ≤ ½: para π_C, π_R, π_L, C3 y π_ratio (= E[U]/E[T] sobre proyectos locales, con y sin λ; implementar), la razón valor/óptimo por celda (exacto donde alcance el solver; Monte Carlo ≥ 10k corridas donde no) y la primera acción contra el óptimo. Pregunta que responde: ¿alguna regla mantiene una constante uniforme en salud rara? Esa sería la garantía a intentar en octubre, y es la respuesta a los tres encargos de Francisco (dónde difieren entre sí; cuál decide bien donde el clásico falla; la regla de cociente). Extras: la instancia donde los greedies difieren (Francisco la pidió), la tabla de qué primera acción toma cada λ. Mañana: revisión cruzada de la Parte 2 de `proposicion_brecha_convencion.tex` (firma u objeción, 45 min), figura del Mapa 4, CSVs con sidecar, push.

**IA.** Hoy: acta y despacho aplicados; `main.tex` como esqueleto real del paper con cada enunciado que ya existe en su sección; lupa sobre la prueba de A. Mañana: `seccion_separacion.tex`, guion del 29, lista de gaps del Thm 8.2.

## 4. Calendario

| Bloque | A | B |
|---|---|---|
| lun 13:00–17:00 | Cota clásica pasos 1–3 (a mano, con lupa de IA) · números del ancla y la frontera | Mapa 4: malla de salud rara, razones exactas donde alcance; arranque de π_ratio |
| lun 17:00–18:00 | PDF a Francisco · aprobar acta/despacho · leer nota del Mapa 4 y escribir 5 preguntas | π_ratio en la batería del contraejemplo y el scoreboard |
| mar 09:00–13:00 | Paso 3 de la prueba (Harris) · familia paramétrica · revisar `main.tex` | Revisión cruzada Parte 2 · Monte Carlo del Mapa 4 · tabla de decisiones por λ |
| mar 13:00–16:00 | Revisar `seccion_separacion.tex` y guion · ensayo | Figura Mapa 4 · sidecars · push · revisar guion |
| mar 16:00 | Sync: congelar guion, ensayo 5+5 min, commit y push 17:30 | |
| mar 19:00 | Sesión | |

## 5. Qué se presenta el martes (sustancia primero)

1. **Un teorema propio:** la separación aumentado-laminar vs clásico con el óptimo clásico exacto (lema de Harris) y la familia no acotada. "Ya no es un ejemplo, es un teorema con prueba."
2. **El esqueleto del paper** con seis secciones y los enunciados en su sitio.
3. **Mapa 4:** qué regla con planeación se acerca al óptimo en salud rara y cuál es el candidato a garantía; π_ratio como respuesta a su sugerencia; la instancia donde los greedies difieren.
4. **Las preguntas que no se hicieron:** (19) territorios, (18) companion, (23) barrera del coupling, (24) regla de λ, más (27) si su §4–5 contiene la cota clásica. Respuesta lista a "¿esto depende de laminar?": no, es un fenómeno de conteos.

## 6. Válvula y qué no se hace

Si A cae a 3 h hoy: cota clásica pasos 1–3 y PDF a Francisco; la familia paramétrica y la lectura del Thm 8.2 pasan a la semana siguiente. Si B cae a 4 h: Mapa 4 solo en primeras acciones; razones por Monte Carlo mañana. No se hace: más contraejemplos nuevos, barridos de λ por sí mismos, Thm 9.3, mapas para presentar, sección de modelo (extra de IA solo si todo lo demás cerró).
