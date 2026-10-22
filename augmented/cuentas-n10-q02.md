# n = 10, q = 0.2 — las cuatro variantes, exactas

Población de 10 personas idénticas e independientes, cada una sana con
probabilidad q = 0.2 (prevalencia alta: 80% infectados), utilidad u = 1. La
utilidad de una política es el número esperado de personas que quedan
certificadas sanas con certeza. Todo lo que sigue es enumeración exacta del
espacio de creencias, no simulación: `scratchpad/solver_n10.py` y
`scratchpad/laminar.py`.

Prueba binaria: devuelve si el pool tiene al menos un sano. Prueba aumentada:
devuelve cuántos sanos tiene. Estático: los pools se fijan de antemano.
Dinámico: cada pool puede depender de lo observado.

| B | B·q | est. binario | din. binario | est. aumentado | din. aum. laminar | din. aumentado |
|---|-----|--------------|--------------|----------------|-------------------|----------------|
| 1 | 0.20 | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| 2 | 0.40 | 0.400 | 0.488 | 0.408 | 0.536 | 0.536 |
| 3 | 0.60 | 0.600 | 0.790 | 0.800 | 0.928 | 1.000 |

Con B = 3 la escalera separa cinco niveles y cada peldaño aísla un ingrediente:
contar vale 0.600 → 0.800, adaptarse vale 0.600 → 0.790, y los dos juntos valen
0.600 → 1.000. Contar solo y adaptarse solo valen casi lo mismo por separado;
combinados dan más que la suma de sus partes.

## El diseño estático aumentado no son pruebas individuales

La premisa de que el óptimo estático consiste en puras pruebas individuales vale
en el modelo binario (el solver da exactamente B·q en B = 1, 2, 3) pero es falsa
en el aumentado. Con B = 3 el óptimo estático vale 0.800, un 33% arriba de
3q = 0.600, y lo logra con un diseño sobre cuatro personas: cada persona entra
en un subconjunto distinto de las tres pruebas, con incidencias 011, 101, 110 y
111. Llamando a, b, c, d a las cuatro personas, las tres pruebas devuelven
a+b+d, a+c+d y b+c+d. La suma de las tres es 2(a+b+c) + 3d, cuya paridad
determina d; conocido d, las tres ecuaciones determinan a, b y c. Las tres
pruebas identifican exactamente a las cuatro personas, así que la utilidad es
4q = 0.800.

Esto no es un accidente de B = 3: es el problema clásico de pesar monedas, donde
B pesadas con balanza que devuelve sumas identifican del orden de B·log(B)
objetos. En el modelo aumentado la cota estática crece superlinealmente en el
presupuesto, no como B·q. Cualquier argumento de separación que use B·u·q como
el óptimo estático está comparando contra el rival equivocado.

## La laminaridad no es gratis aquí

Con B = 3 la política óptima vale 1.000 y la mejor política laminar vale 0.928:
la restricción cuesta 7.2%. La política óptima cruza explícitamente. Abre un
grupo de 3; si el conteo sale 1, su segunda prueba toma dos personas de ese
grupo junto con una persona fresca, un pool que ni contiene ni está contenido ni
es disjunto del grupo ya probado.

La razón de que cruzar pague: tras el conteo 1 el grupo de tres tiene una unidad
de información sobrante — se sabe que hay exactamente un sano, pero no cuál. Un
pool que mezcla dos miembros del grupo con alguien fresco cobra las dos cosas a
la vez. Si el conteo sale 0, el tercer miembro del grupo queda certificado
inmediatamente y además se aprendió que el fresco está infectado. La prueba
cruzada recicla el residuo de información del grupo mientras tantea territorio
nuevo.

En el modelo aumentado la clase laminar tiene una propiedad que la hace
tratable: como cada grupo probado termina con su conteo conocido exactamente,
agregar un grupo entero a un pool nuevo solo suma una constante conocida y no
informa. Las creencias se factorizan en urnas hipergeométricas independientes y
el estado se reduce a un multiconjunto de pares (tamaño, conteo). Eso es lo que
la laminaridad compra: el solver laminar llega a B = 8 en segundos, mientras que
el irrestricto no terminó B = 4 en diez minutos. Es un beneficio computacional y
analítico, no un beneficio de valor.

De ahí que la manera honesta de usarla sea como cota inferior. Una política
laminar explícita que le gane al óptimo estático demuestra la separación sin
salir de la clase tratable, y el enunciado resultante es más fuerte que uno que
necesite políticas arbitrarias. Pero la laminaridad no puede presentarse como
inocua: en n = 10, q = 0.2, B = 3 ya pierde 7.2%.

## Sobre el ejemplo de población infinita

El argumento de la población infinita compara B·u·q contra
u·(1 − (1−q)^(kG)) y concluye separación cuando B·q < 1 − (1−q)^B, tomando
kG ≈ B. Esa desigualdad no tiene solución: por Bernoulli, (1−q)^B ≥ 1 − B·q,
así que 1 − (1−q)^B ≤ B·q siempre. La separación aparece precisamente al no
colapsar kG con B, porque k pruebas grupales de tamaño G tocan kG personas y no
B. Con q = 0.01, B = 10 y G = 64 se tiene k = 4 y kG = 256: el lado estático da
0.100 y la estrategia adaptativa da 1 − 0.99^256 = 0.924, casi 9.3 veces más. El
techo es 1/(B·q) = 10, porque una sola búsqueda binaria cosecha una sola
persona.

Ese ejemplo sigue siendo válido como cota inferior contra el diseño de puras
pruebas individuales. Contra el verdadero óptimo estático aumentado hay que
rehacerlo, porque el rival correcto crece como B·log(B)·q.
