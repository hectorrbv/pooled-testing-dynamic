# Sesión con Francisco (2026-09-01)

**Fuente:** `TRANSCRIPCION_FRANCISCO_1_SEPTEMBER.md` (50:43; habla 00:00–50:40). Francisco confirmado por nombre en el audio. Participante 1 consistente con A (visa de Praga; Francisco le dice "Vlad" en los ejemplos [08:26]; lleva las preguntas), **confirmado por A al aprobar esta acta (2026-09-13)**. Participante 2 consistente con B ("Creo que estás en mute" [00:24]; Francisco nombra a "Héctor" en el ejemplo [07:35]). Disciplina: a Francisco solo lo inequívoco; lo demás, "el equipo reporta".
**Fecha:** 2026-09-01, 11:00 CDMX (guion congelado `2026-09-01-guion-sesion-BORRADOR.md`; Francisco: "hoy fue el primer día de clases" [02:36]).
**Guion previo:** paquete de 5 puntos, preguntas (18)–(22) y declaración "yo tomo X". Francisco ocupó unos 40 minutos con sus resultados nuevos; del paquete se alcanzó a reportar la brecha de convención y el estado de Bellman y del Lema A; ninguna pregunta congelada se hizo de forma explícita. La sesión se cortó a los 47 minutos por otra reunión de Francisco [47:15–47:44].

## Resumen ejecutivo

Francisco dedica la sesión a los resultados nuevos con Edwin en **dinámico no-aumentado**: el factor de adaptabilidad (óptimo adaptativo sobre óptimo estático no solapado) baja a **2**, vía muestreo del prior, thinning al 50% y un coupling perfil/antiperfil; el greedy estático queda a factor constante del óptimo dinámico; y con q_sano ≤ 1/2 el óptimo dinámico no-aumentado es individual, lo que cierra el lado no-aumentado del ejemplo de separación ("más allá de un factor de 2"). Todo va al paper de Nick, reescrito casi sin material empírico, con envío previsto el 14-sep. Dice que el argumento **"no va a funcionar así en nuestro caso"** aumentado y que en aumentado "necesitas dinámico". El equipo reporta la brecha de convención (n=4, q_sano=0.3: bajo estricta nunca agrupa; bajo posterior-zero abre par) y Francisco responde "**Qué interesante**". Enuncia el **resultado soñado**: laminar dentro de un factor constante del óptimo dinámico, y dentro de laminar un algoritmo eficiente aproximadamente óptimo. Encargo para la siguiente sesión: **análisis empírico de varios greedy en el régimen de separación, buscar un contraejemplo donde todos fallen, "corre tres greedy y toma el mejor"**. Mandará su prueba escrita para revisión del equipo.

## D1. Resultados nuevos de Francisco y Edwin en dinámico no-aumentado [04:18–12:39, 13:23–41:07]

Prioridad de Francisco en 2–3 semanas: un capítulo (entrega 1-sep) y "otro paper que quiero mandar para el 14... me urge hacer esos cambios en el documento del paper con Nick, para ponerlos en línea... antes de que alguien más lo descubra" [04:18–05:05]. El paper de Nick "lo vamos a cambiar completamente... casi no va a tener material empírico de Nick... más teoría de dinámico non-augmented" [05:05–06:46].

Resultados dichos (cifras habladas, provisionales hasta la prueba escrita):
- Factor de adaptabilidad = **2** [05:05–05:54]; antes "como B" y luego "2^B − 1 sobre B" [06:46].
- Greedy exacto "dentro de un factor de como 3 de lo óptimo dinámico" [06:46–07:35]; más tarde: greedy estático dentro de (1+e) del óptimo solapado (Prop 4 de v4) y óptimo solapado dentro de 2 del óptimo adaptativo, "dentro de un factor de 6, mínimo" [39:29–40:16]; antes había dicho "tres por dos por dos, como doce" [11:45–12:35]. Los factores exactos los fija la prueba escrita.
- Solapado vs no solapado: 2, antes 4 [07:05–07:35]. Es el Thm 1 de v4 (extraído por C-M1 el 1-sep).
- **Singletons óptimos en dinámico no-aumentado cuando "la probabilidad de infección es menor a 0.5 para todo el mundo"** [40:16–41:07]. Lectura: la consistente con Prop 1 de v4, con Thm 7.1 del companion y con la sesión del 25-ago ("suficientemente altas de infección, no usa pruebas grupales") es **q_sano ≤ 1/2**; P1 lo entiende así y responde con el ejemplo q_sano = 0.3 [41:07]. Se confirma contra el PDF (cotejo de A, C-M1). "Eso termina la última parte de nuestro ejemplo... lo óptimo dinámico es lo que habíamos dicho anteriormente... y nosotros ya tenemos el algoritmo laminar que demuestra una separación más allá de un factor de 2" [40:16–41:07].

El argumento, reconstruido por el equipo a partir de [08:26–10:54, 13:23–39:13] (A escribe su propia reconstrucción como extra):
1. Dada cualquier política dinámica "regular" (no vuelve a probar a quien ya salió sano [33:11–34:03]), se muestrea un perfil ficticio del prior y se sigue por el árbol hasta una hoja; la rama es una lista de pruebas, es decir, una asignación estática posiblemente solapada [09:13–10:04, 34:03–34:53].
2. Thinning: por cada incidencia persona–prueba, una moneda al 50% decide si se queda [34:53–36:54]. Lo que queda es estática; en promedio saca al menos la mitad del dinámico, luego existe una estática con al menos la mitad (método probabilístico, no constructivo) [10:04–11:45, 38:50–39:13].
3. Paso sutil: la desigualdad de conteo (perfiles que acreditan a una persona distinguida bajo la dinámica ≤ 2 veces la suma de probabilidades de acreditarla usando el perfil para escoger pruebas y el **antiperfil** para aplicarlas, con thinning) se prueba con distribución uniforme en las K coordenadas libres, por inducción ("el conteo es como una inducción" [32:25]) [22:22–29:53]. Los priors reales no son uniformes: se toman dos muestras, una escoge pruebas y otra las aplica; condicionando en las coordenadas en desacuerdo, la posterior perfil/antiperfil es simétrica (pq = qp, luego 1/2) y aplica el lema; después esperanza total [13:23–20:56, 29:53–31:34]. Dos roles de una prueba: cambia la rama del árbol, o es la última prueba de la persona distinguida [16:04–18:33].
4. Estado: "Igual y todavía está mal; todavía lo estamos finalizando. Estamos casi seguros" [13:23]; "casi seguro que siga, tengo algunos casos... más pequeños" [32:25].

Uso como certificado (P1 lo nombra así [12:39]): la cadena de factores da una cota superior computable del óptimo dinámico no-aumentado a partir del greedy; una estrategia aumentada que la supere "con certeza es mejor que el otro, sin haber hecho el cómputo explícito" [11:45–12:39]. Francisco: el coupling "pueden ser técnicas interesantes para comprobar alguna comparación entre laminar y [inaudible]" [21:01–21:12].

## D2. Territorio: el argumento no pasa a aumentado; aumentado exige dinámico [37:04–38:32]

"Estoy casi seguro que esto funciona en el caso non-augmented porque las pruebas positivas hacen mucho daño... eliminar personas es muy benéfico... en augmented no es necesariamente el caso... **creo que este argumento no va a funcionar así en nuestro caso**... definitivamente en el caso augmented necesitas tener un ejercicio dinámico; de nada te sirve tener augmented si no es dinámico" [37:04–38:15]. P1 lo parafrasea; Francisco: "exacto" [38:15–38:32].
Lectura para (19): lo que migra al paper de Nick es el dinámico no-aumentado (factor 2, greedy constante, singletons con q_sano ≤ 1/2); el "factor interesante" es el 2 de adaptabilidad vía thinning. El acuerdo explícito "aumentado laminar = nuestro" **no se pidió**: queda como propuesta para el 15-sep.

## D3. El equipo reporta la brecha de convención; "Qué interesante" [41:07–45:39]

P1: bajo hard clearing, en la instancia de 4 personas con q_sano = 0.3, u uniforme, B=2, G=2, "no había ningún movimiento que justificara una primera agrupación... el mejor movimiento siempre era jugar individual" [41:49–42:27]; bajo soft clearing "no hay prueba individual que le gane" a abrir grupo y seguir óptimo [42:27–43:10]; individual = 0.6 [43:10]; la búsqueda binaria bajo hard clearing "siempre vas a necesitar de una prueba extra" para acreditar [44:32–45:09]. Francisco: "**Qué interesante.** ¿Y sí pudieron implementar el algoritmo dinámico?" [45:09–45:19]. P1: el borrador de Bellman "sí está funcionando" [45:19]; Francisco: "para casos pequeños... mejor que la enumeración" [45:30–45:39]. P1: el lema de factorización posterior por átomos "quizá mañana o el miércoles ya queda" [45:39–46:24].
Cifras: en sesión se dijeron "una pérdida como de 0.12 puntos" [42:27] y "0.74 44... o sea 0.75" [43:52, cifra incierta]. **Rigen las verificadas por dos vías (adenda del guion):** óptimo estricto 3/5 = 0.60, nunca agrupa; óptimo posterior-zero 387/500 = 0.774, abre el par. Nota: 0.12 coincide con 0.60 − 0.48, el valor de par-primero seguido de un virgen en la rama R=1 bajo estricta, no con la mejor continuación (0.564, pérdida 0.036). Se corrige en el guion del 15-sep si se vuelve a citar.
Estado al 13-sep: Lema A no cerrado en el repo (la factorización está en `lemma_A_laminar_inference.tex` desde julio); Bellman: solver B-M17 con review adversarial (33c1687, 5889672); `tests_bm17.py` pendiente.

## D4. Resultado soñado [46:24–47:36]

"En un mundo perfecto, un resultado padrísimo sería, aunque va a estar difícil, pero pues tenemos tiempo para hacerlo, **demostrar que lo laminar está dentro de un factor de óptimo a lo óptimo dinámico**. O sea, no necesariamente laminar. **Y, dentro del espacio laminar, demostrar que alguna variante, algún algoritmo eficiente, nos permite hacer algo laminar que sea aproximadamente óptimo**, ya sea el greedy con hard clearing, soft clearing o alguna regla" [46:24–47:15]. "Ya tenemos resultados ahí con esto. Con eso la podemos meter como a toda la cadenita" [47:15–47:36].
Lectura: (ii) es la Conjetura 10.7 del companion, meta viva de la pregunta (1); (i) es la comparación laminar vs irrestricto, hoy enunciada solo como factor G (Thm 8.2). Confirma la espina.

## D5. Encargo: greedies en el régimen de separación; contraejemplo; "corre tres y toma el mejor" [48:10–49:53]

"Más análisis empírico de estos diferentes tipos de greedy... especialmente en el régimen en donde ya tenemos una separación... ver si hay algún caso en donde a todos les va mal, eso quiere decir que completamente descarta lo que tenemos a la mano" [48:10–48:57]. "En el peor de los casos podemos decir: corre estos tres greedy y toma el que es mejor de los tres... si uno de los tres greedy es mejor que cierto factor, ese es un algoritmo eficiente con factor" [48:57–49:24]. "Tratar de encontrar un contraejemplo. Y si no sale uno, no lo hemos comprobado, pero es como que evidencia de que vamos en buena dirección" [49:41–49:53].
**Estado al 13-sep: cumplido por B con sobre-entrega (1–9 sep):** π_M, π_C, π_R exactas sobre el solver (50eb431); contraejemplo universal n=6, B=3, G=4 heterogéneo, las cuatro (π_M, π_C, π_R, C3) a 0.6576 del óptimo (3bb9415; `2026-09-01-contraejemplo-universal-greedy.md`); π_L con índice Lagrangiano I_λ (ec. 8.13) lo rescata a 0.9641 (c3070ee, 5889672); constante empírica del portafolio best-of-5 α ≈ 0.9307 en ~7,000 instancias n≤8, G≤4, B≤5, sin contraejemplo, con λ como parte de la especificación (db9712c; `2026-09-02-constante-empirica-portafolio.md`). Etiqueta: hallazgo diagnóstico [VERIFICADO n≤8]; no es garantía.

## D6. Compromiso: la prueba escrita, a revisión del equipo [49:58–50:24]

"Cuando tenga el resultado todo escrito se los paso, para que le echen un ojo también. Si ven algún error, también que nos digan" [49:58–50:11]. P1: "es un argumento muy sutil" [50:11–50:24].

## D7. Logística [02:08, 03:18–05:05]

La hora se mantiene "estas próximas dos semanas, tres semanas"; cuando el equipo esté en Praga (semestre desde el 14-sep [02:08]) se busca otra: "hasta por miércoles por la mañana, de acá... puedo cambiar con la reunión que ya tengo con KK" [03:33–04:18]. Francisco cierra dos papers en 2–3 semanas [04:18–05:05]. **No hubo sesión el 8-sep** (vuelo de A a Praga); la siguiente es el 15-sep, con A ya en Praga: la hora debe confirmarse.

## Preguntas de §34

- **(15)** enunciado del teorema estático p>1/2: **RESPONDIDA, condicionada al cotejo de A del PDF v4**. Prop 1 de v4 (estático) más la versión dinámica dicha en sesión [40:16–41:07] (Thm 7.1 del companion): singletons óptimos en no-aumentado si q_sano ≤ 1/2. Ojo al signo hablado ("probabilidad de infección menor a 0.5").
- **(19)** reparto con el paper de Nick: **PARCIAL**. Qué va allá y por qué (D1, D2); el acuerdo explícito de territorios se pide el 15-sep.
- **(1)** ¿V^{*,L} o V^*?: **PARCIAL, reforzada**. D4 fija el comparador en dos capas: laminar vs dinámico irrestricto, y eficiente vs óptimo laminar.
- **(12)** contraejemplo de referencia: **PARCIAL**. El lado no-aumentado del ejemplo de separación cierra por singletons-óptimos, condicionado a la prueba escrita; "más allá de un factor de 2".
- **(4)** garantía valorada: **PARCIAL**. Aproximación; "greedy con hard clearing, soft clearing o alguna regla" [46:24].
- **(18), (20), (21), (22):** no se hicieron; la sesión se cortó. Siguen.
- **Nuevas:** (23) ¿en qué paso concreto se rompe el coupling perfil/antiperfil cuando la prueba devuelve un conteo y el crédito es por deducción?; (24) regla de λ para π_L; (25) término sobrante del Thm 9.3, solo si A lo re-deriva antes del martes. En buzón sin número: P1 (dirección) y P3 (oráculo y benchmark del piloto) de la nota C-M1 del 1-sep.

## Compromisos y pendientes

- **De Francisco (2026-09-01):** prueba escrita del factor 2 y del greedy, para revisión del equipo [49:58] · paper de Nick reescrito, envío previsto 14-sep [04:18–05:05] · nueva hora de sesión cuando el equipo esté en Praga [03:33–04:18]. **Siguen del 25-ago:** ejemplo propio de la falla del presupuesto-infinito; update del paper de Nick (ahora en curso).
- **Del equipo (2026-09-01):** análisis empírico de greedies, contraejemplo y portafolio: **CUMPLIDO por B** (D5) · Lema A cerrado ("mañana o el miércoles" [45:39]): **pendiente** · Bellman "al cien": solver con review; `tests_bm17.py` pendiente · revisar la prueba de Francisco al recibirla; A llega con reconstrucción propia como extra.
- **Siguen vigentes:** A-M22 (documento formal; la Proposición de brecha de convención como primer resultado propio) · A-M19 (outline SODA) · A-M23 (companion §8–§10) · A-M24/B-M18 (migración) · B-M6 · cotejo del PDF v4 (A) · reformulación de la Conjetura C (P21-A8).
- Despacho §34-bis aplicado el 2026-09-13 (acta y despacho con retraso por el traslado de A a Praga; sin sesión el 8-sep).
