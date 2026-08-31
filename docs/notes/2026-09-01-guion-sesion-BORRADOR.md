# [BORRADOR — A edita y congela el lunes] Guion de sesión 2026-09-01

**Protocolo §34-bis:** paquete y preguntas congelados antes de la sesión (A-M20). Este borrador lo preparó el soporte IA (2026-08-30); A lo edita, recorta y congela el lunes. Todo número citado aquí está verificado; etiquetas §25 donde aplica.

## Frase de apertura

> "Aplicamos el despacho completo de la sesión pasada: adoptamos posterior-zero como nos dijiste — ya está en el modelo, con la variante estricta etiquetada para comparación —, tenemos el companion inventariado resultado por resultado como checklist de verificación, y Bellman ya está en diseño [/ corriendo en juguete]. Traemos cinco preguntas de decisión y una declaración de trabajo."

## Paquete (mostrar, en este orden)

1. **G0 ejecutado** — "soft clearing es mejor" → §5 reescrito (posterior-zero normativo; estricta como variante nombrada). Ratificado por ambos el lunes. Cascada re-derivada: ancla del acid test $k{=}3$, $1-0.95^{48}\approx 0.9147u$ (antes $0.806u$); en el contraejemplo de no-reentrada la reentrada pasa de 0.5 a **1.0**.
2. **Inventario del companion** (`docs/notes/2026-08-30-inventario-companion.md`): las ~18 piezas con estatus; verificado ya: aritmética de Prop 6.2, Ejemplo 7.5, testigo de Prop 8.5 bajo ambas convenciones ($q<1/2$). Prioridad de lectura §8–§10, como pediste; tú tomas §8+, nosotros vamos detrás con lupa.
3. **B-M17 (Bellman)**: nota de diseño siguiendo Prop 6.1 Steps 1–6, con flag de convención (posterior-zero / estricta) [+ demo del prototipo $n\le4$, si corrió].
4. **Confluencia**: tu §3 y nuestro tensor/Lema A son el mismo objeto por rutas independientes (ec. 3.5 ≡ nuestra forma cerrada; la caché por convolución ya está implementada y testeada en `laminar_tables.py`) — B-M17 la reutiliza tal cual.
5. **En curso esta semana**: escritura del documento formal bajo posterior-zero (mié–jue), outline del paper (vie), y la pieza de abajo.

## Preguntas congeladas (§34; en orden de decisión)

- **(18) Estatuto de escritura:** ¿el outline del paper (SODA) se construye sobre tu companion, sobre nuestro documento laminar, o se fusionan — y qué secciones lleva cada quien? Sub-pregunta de calendario: ¿qué fecha vemos para un draft arXiv-able (dijiste que arXiv puede ir antes)?
- **(19) Reparto con el paper de Nick:** ¿qué migra allá (¿Thm 7.1?) y cuál es el "factor interesante que entra en la aproximación" de ese argumento?
- **(20) Para congelar el barrido del índice:** ¿el eje decisivo es el exponente $\alpha$ o committed-vs-receding? (Tu Thm 9.3 separa $1/\log G$ de $\to 1$ justo por ese eje.)
- **(21) Harness:** bajo posterior-zero, ¿conservamos la variante estricta (costo del test acreditador) como columna de comparación, o se retira?
- **(22) Clase normativa:** tu companion trabaja con pathwise (ex post); nuestro atlas era ex ante. ¿Pasamos la normativa a pathwise, o reportamos ambas etiquetadas? (Toca cómo validamos Bellman: like-with-like.)

## Declaración de trabajo ("yo tomo X") — A, en sus palabras

- **Esta semana:** enuncio y pruebo la **Proposición de brecha de convención** (estricta vs posterior-zero: costo exacto en tests y bienestar; la semilla son mis cálculos a mano del inventario), verificada por enumeración con el solver de Héctor.
- **Siguiente:** lectura adversarial de Thm 9.3 (reparar lo que encuentre).
- **Mediano plazo, declarado:** quiero atacar 10.4 / la Conjetura 10.7 cuando cerremos la verificación de §8–§10.

## Extras (mantra — solo si hay tiempo, nunca como claims)

- Demo del toy Bellman (si corre) sobre la instancia del contraejemplo.
- Hallazgo de lectura: la ventaja de densidad exige $G>1+\lceil\log_2 G\rceil$ ⟹ $G\ge4$ — nuestras mallas chicas ($G\in\{2,3\}$) no pueden exhibir el mecanismo; lo incorporamos a las matrices.
- Pregunta relámpago si sobra minuto: ¿el "ejemplo del presupuesto-infinito" que anunciaste ya tiene forma?

## Logística

- Confirmar la hora estable del semestre (mar/jue das clase).
- Recordatorio interno: grabar; el transcript va al protocolo §34-bis (acta + despacho el miércoles).
