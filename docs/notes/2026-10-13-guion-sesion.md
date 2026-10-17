# Guion — sesión con Francisco, martes 2026-10-13, 19:00 Praga / 11:00 CDMX

**Estado:** borrador 2026-10-12; se congela el 13-oct a las 16:00 (§34-bis, punto 3). **Enviado la víspera (lun 12, antes de 20:00):** `proposicion_individuales_nunca_optimas.pdf` + `proposicion_brecha_convencion.pdf` + pregunta (30). **Etiquetas:** todo número de este guion está en `docs/notes/2026-10-12-encargos-6-oct.md` o en la nota; lo que B no haya verificado por segunda vía se dice como "diagnóstico, una implementación".

Francisco suele abrir con lo suyo 20–25 min. Nuestro bloque cabe en 25 min; si solo hay 10, se hace el bloque 1 y el testigo del bloque 2.

## Bloque 1 (5 min, A) — la nota: individuales nunca óptimas

- Una frase: con conteos y soft clearing, si $B\ge2$, $G\ge2$, $n\ge B+1$, ningún plan de puras individuales es óptimo. El intercambio en dos líneas: par, y la segunda prueba va al refinamiento (conteo 1, deducción) o a una persona fresca (0 o 2); ganancia $(q_1q_2+p_1p_2)\,q_3u_3>0$.
- Las dos fronteras son sus dos intuiciones del 29-sep: $B=1$ (singleton óptimo ⟺ $\Lambda\le1$) y $n\le B$.
- Corolario con su Thm 3.3(i): en $\Lambda\le1$ el binario dinámico es $T_B$ y el laminar es estrictamente mayor.
- **Pregunta (30):** ¿es el enunciado que le parecía sorprendente? ¿va en la sección de separación junto a Thm 3.3(i)?

## Bloque 2 (8 min, B) — encargo 1: binario óptimo > laminar existe

- Testigo $n=3$, $B=2$, $G=3$, $u\equiv1$, $q=(0.9,0.7,0.7)$: binario 2.079 (tras el positivo cruza $\{a,c\}$, paga 2 con prob. 0.511) > laminar 2.062 (refina $\{a\}$, paga 1 seguro) < irrestricto 2.121. Mostrar los tres árboles.
- Búsqueda: 6,800 instancias con $\Lambda>1$, $n\le5$: binario gana en 63, razón mínima 0.978; nunca con $q$ homogénea. "Las dos fuerzas", medidas: el cruce apuesta por dos personas, la deducción asegura una.
- Observación (V4): con $\Lambda\le1$ su Thm 3.3(i) más "no-solapado ≤ laminar" da Laminar ≥ Binary sin el ¼; el ejemplo que pedía solo puede tener $\Lambda>1$, y así salió. **Pregunta (31):** ¿de acuerdo?
- Si el loop nocturno corrió: "un loop de búsqueda corrió N experimentos; el peor cociente que encontró es X en la familia Y, re-resuelto por B"; mostrar sus árboles si son más claros que el testigo $n=3$.
- Si B cerró la segunda vía: "dos implementaciones coinciden"; si no: "diagnóstico, una implementación".

## Bloque 3 (7 min, B) — encargo 2: el precio de la laminaridad en $q=\tfrac12$

- Tabla exacta, $u\equiv1$: peor razón laminar/irrestricto 0.875 / 0.857 / 0.833 para $n=4/5/6$ en $q=\tfrac12$, contra 0.895 / 0.874 / 0.860 en $q=0.3$ y 0.937 / 0.895 / 0.879 en $q=0.7$; siempre en $G=3$, $B=3$–4; con $B=2$ la razón es 1. (Si $n=7$ corrió: añadir la fila.)
- Si el loop nocturno corrió: la familia y el cociente mínimo que encontró para `precio`, re-resuelto por B.
- Lo que dice: $q=\tfrac12$ es el peor de los tres en todo $n$ medido y la pérdida crece con $n$ y $B$. Lo que no dice: nada del límite ni de $G$ grande.
- Observación (V5): con tope $G$, cada prueba tiene $G+1$ resultados, luego identificar a todos cuesta $\ge n/\log_2(G+1)$ pruebas para cualquier política; el $n/\log n$ de las matrices decodificadoras es sin tope. **Pregunta (32):** ¿el precio en $q=\tfrac12$ es $O(\log G)$ y no $G$? **Pregunta (33):** referencia exacta del paper que compartió.

## Bloque 4 (5 min, A) — el escalón $T_{BG}$ en nuestras palabras

- A dice el argumento de la cota superior tal como lo reescribió (tocar ≤ $BG$ personas; acreditado ⟹ sano; $T_x$ cóncava ⟹ $T_{BG}\le G\,T_B$; $T_B$ es binaria). Cierra Cor 8.3 del companion por identificación.
- **Pregunta (34, nueva):** el segundo factor 2 (estático ≤ 2·no-solapado), ¿dónde está? No lo encontramos en arXiv:2601.22419v2 (Thm 3.3(iii) es otra cosa).

## Si queda tiempo: (29) reparto explícito y estatuto del companion; (28) la frase de la introducción.

## Qué no se dice

- No "pooling siempre es mejor" (empates en la primera acción). No "probado el peor caso en $q=\tfrac12$". No constante para Q2. No "autoresearch encontró". No cifras fuera de la nota y de `2026-10-12-encargos-6-oct.md`. No "verificado" para números del solver nuevo si B no cerró la segunda vía.
- No se abre el memo de la §5 ni el experimento del primer cruce: quedaron fuera de esta semana (el testigo $n=3$ es su primera fila).

## Después

Acta y despacho en 48 h (§34-bis, punto 5). WhatsApp de A con las dudas de $\Lambda$ si no salió antes.
