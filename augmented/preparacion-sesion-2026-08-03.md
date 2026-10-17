# Preparación de la sesión con Francisco — 3 de agosto de 2026

Dos tareas de dos horas, una por persona, para llegar a la sesión con el encargo
resuelto y un caso concreto que traer a la mesa. Los números del caso fueron
verificados con código independiente contra la maquinaria del repo
(`ExactPolicyEvaluator`, `greedy_laminar_value`, `rollout_laminar_value`,
`dynamic_augmented_value`); las cuatro afirmaciones salieron confirmadas.

## Vladimir (~2h): la evaluación de V(T), a mano, más las correcciones

**1. La derivación del colapso (núcleo).** Escribir en limpio los dos hechos
sobre la V(T) propuesta en la sesión (promediar sobre conteos la utilidad
localizada):

- *Se cancela la información.* Al promediar las probabilidades posteriores sobre
  todos los conteos posibles se recuperan las probabilidades previas: el promedio
  de lo que se creerá mañana es lo que se cree hoy. Entonces V(T) = suma de
  u_i·q_i con las previas, un número que no depende de nada que la prueba pueda
  enseñar. Dos líneas de álgebra.
- *Nunca cobra.* Tras observar un conteo, subdividir el pool no cambia V en
  promedio. El score jamás premia volver a entrar a extraer a los sanos que ya
  localizó.

Con la lectura doble: la propuesta **arregla el arranque** (crece con el tamaño,
elige el pool grande que el greedy nunca se atreve a dar) y **pierde la cosecha**
(nunca regresa). No es "está mal": es "hace exactamente la mitad del trabajo".

**2. Extra (~20 min): las V's que creemos que sí funcionan.**

- *Utilidad extraíble:* contar solo lo que de verdad se puede cobrar con el
  presupuesto restante (encontrar al sano y pagarle sus pruebas de extracción).
  Se apaga sola cuando el presupuesto no alcanza; satura en vez de crecer sin
  techo.
- *Utilidad descontada:* la localizada dividida entre las ~log G pruebas que
  cuesta sacar al individuo — conecta con el argumento de log G que el propio
  Francisco dio [18:45–19:32].

Cierre propuesto: "tu idea se colapsa por esto, y esto es lo que apunta a
funcionar — ¿cuál de las dos conserva mejor tu intuición original?"

## Héctor (~2h): resultados del notebook 23 + el caso a detalle

**1. Elegir y ensayar 2–3 resultados del notebook 23:** E1 como respuesta
directa al encargo (el cómputo que confirma el colapso, respaldando la
derivación de Vladimir), y E4/R1 como hallazgo: en prevalencia alta homogénea el
greedy se vuelve idéntico al diseño estático — dígito por dígito — y aun así el
mejor árbol le saca 11–20%, porque el óptimo vuelve a entrar a pools ya
observados y el greedy nunca lo hace.

**2. El caso, rama por rama:** 5 personas, B=3, pools de hasta 2, p=0.7
homogéneo, u=1. Greedy: tres individuales, cobra 0.90. El óptimo abre con un
**par**: conteo 0 → cobra dos de golpe; conteo 1 → sabe que exactamente uno está
sano y reentra a cobrarlo; conteo 2 → par muerto, territorio nuevo. Total 1.011,
**+12%**. Llevar el árbol dibujado con los números por rama.

**Por qué traerlo a la mesa:** responde punto por punto la objeción de Francisco
(D2 del acta) dándole la razón en dos de tres — greedy sí hace puras
individuales en p>0.5, y su teorema estático se sostiene (verificado incluso
heterogéneo, con demostración); lo que falla es "todos deberían coincidir,
incluido Opt": el óptimo queda 11–20% arriba en todo el régimen p ≥ 0.5 (432
instancias, no una anécdota), porque su teorema cubre el pooling como jugada
estática, no como recolección de información con reentrada. Además el ejemplo
exhibe en tres pruebas las dos mitades del tema de la sesión — dar el primer
paso grande *y* volver a cobrar — que son justo la que V(T) hace y la que
pierde. Y como Francisco se comprometió a checar el teorema para el paper, el
tema va a salir sí o sí: mejor llegar con la respuesta lista.

## Verificación de los números del caso

Instancia n=5, B=3, G=2, p=0.7 homogéneo, u=1, strict hard clearing:

1. Greedy laminar = greedy estático = 0.900 exacto (= 3q).
2. Óptimo = rollout = 1.011 = q(3q²−3q+4); mejora +12.3% sobre greedy.
3. Política óptima extraída: abre con el par (0,1); conteo 0 → cobra ambos y
   sigue a territorio nuevo; conteo 1 → reentra probando dentro del par;
   conteo 2 → territorio nuevo.
4. En la rejilla p ∈ {0.50,…,0.90}: rollout coincide con la fórmula en toda la
   rejilla y coincide con el óptimo exactamente desde p ≥ 2−√2 ≈ 0.5858
   (estrictamente abajo en p = 0.50 y 0.55).
