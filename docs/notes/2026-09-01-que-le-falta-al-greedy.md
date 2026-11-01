# Qué le falta al greedy: lo que enseña la región del contraejemplo (2026-09-01)

Continuación del contraejemplo universal (`2026-09-01-contraejemplo-universal-greedy.md`).
La pregunta: la región donde las cuatro heurísticas caen a 0.6576, ¿solo
refuta, o dice cómo repararlas? Dice cómo. **Estatuto §25: diagnóstico**;
reproducible con `python -m augmented.indice_lagrangiano`.

## 1. La región, delimitada por perturbación

Partiendo del keep (n=6, B=3, G=4) y moviendo un parámetro a la vez:

| Variante | score | ¿sigue engañando? |
|---|---|---|
| keep original | 0.658 | sí |
| infección +0.02 (más alta) | 0.683 | sí |
| premio de E: 4 → 3 | 0.690 | sí |
| **utilidades planas u = 1** | 0.754 | apenas |
| infección −0.05 (más baja) | 0.951 | no |
| premio de E: 4 → 6 | 0.961 | no |

La familia vive en **infección muy alta (p ≳ 0.85) con premio moderado**, y
tiene dos fronteras nítidas. Que sobreviva casi intacta con utilidades planas
dice que el motor es la **estructura de deducciones**, no la heterogeneidad de
utilidad: ésta solo lo agrava.

## 2. Lo que la región señala como faltante

En ese régimen casi toda exploración muere en la primera prueba, y el valor
llega por cadenas de deducción que solo existen dos pasos adelante. Las cuatro
heurísticas fallan por dos razones distintas, ambas visibles en la región:

1. **Cobro inmediato (π_M, y el primer término de C3)** no ve nada: el trío
   ganador tiene P(limpio) ≈ 0.001.
2. **Densidad (π_C, π_R, y la promesa de C3)** sí mira proyectos de varias
   pruebas, pero los divide entre un **horizonte reservado** c. Cuando el
   proyecto casi siempre muere en la primera prueba, ese denominador cobra
   pruebas que jamás se gastarán, y hunde al proyecto bueno bajo un singleton
   mediocre.

El ingrediente faltante no es "más lookahead" sino **contabilizar el costo
esperado, con derecho a abandonar** — que es exactamente el índice
Lagrangiano que el companion ya define y deja sin probar (ec. 8.13):

$$I_\lambda(C) = \sup_{\pi \text{ local a } C} \mathbb{E}[\text{utilidad acreditada} - \lambda \cdot \text{pruebas usadas}]$$

con el supremo incluyendo la política nula (abandono gratis).

## 3. La prueba: π_L contra la batería

Política: en cada paso, argmax de I_λ sobre componentes; ejecuta la primera
acción y recalcula. Más una **regla de no-parálisis** (si nada supera el
precio, se toma el mejor cobro inmediato en vez de detenerse) — la misma
lección que ya había dejado el barrido de α.

| Instancia | óptimo | π_L | ratio | mejor de la batería |
|---|---|---|---|---|
| **contraejemplo universal** | 1.0645 | 1.0263 | **0.9641** | **0.6576** |
| B-M16 (no-reentrada) | 0.7740 | 0.7740 | 1.0000 | 1.0000 (C3) |
| rare-health G=4 | 0.0786 | 0.0784 | 0.9976 | 0.9976 (π_R) |
| baja prevalencia q=0.7 | 2.9414 | 2.8742 | 0.9772 | — |

π_L iguala a la mejor de la batería en todas y **gana +0.31 exactamente donde
las cuatro fracasan**. El contraejemplo dejó de ser solo una refutación: es el
test que discrimina entre familias de heurísticas.

## 4. La grieta que queda: calibrar λ

λ es el precio sombra de una prueba y **no es libre**:

| λ | contraejemplo | rare-health |
|---|---|---|
| 0.001 – 0.01 | 0.964 | 0.998 |
| 0.05 | 0.964 | 0.509 |
| 0.30 | 0.964 | 0.509 |

Con λ grande la exploración deja de pagar y la política degenera. El λ útil
depende de la escala de la instancia (≈ OPT/B): en rare-health el valor total
es 0.08 y λ = 0.05 ya lo mata. Es literalmente la **pregunta abierta 10.5** del
companion ("¿existe un índice basado en I_λ … con garantía de factor
constante?"), ahora con evidencia de que el candidato funciona cuando λ está
bien puesto y de dónde se rompe cuando no.

## 5. Consecuencias

- **Para la teoría (A / Francisco):** el contraejemplo acota la Conjetura 10.7
  por abajo — ninguna política de densidad pura puede alcanzar constante — y
  señala a I_λ como el candidato con chance. La sub-pregunta concreta que
  desbloquea el intento: **cómo fijar λ** (¿λ = OPT/B estimado? ¿dual del LP de
  presupuesto esperado, Thm 10.2?).
- **Para B:** π_L es caro (resuelve Bellman locales, como π_R). Lo barato sería
  destilar su comportamiento en una fórmula cerrada tipo C3 — un cuarto término
  de "valor extraíble con abandono" — y volver a pasar la misión V̂ y la
  misión contraejemplo con esa candidata.
- **Cautelas:** cuatro instancias, población homogénea salvo el
  contraejemplo, G ≤ 4, B ≤ 4. Sin garantía probada. π_L no se adopta como
  candidata S3 sin G4a/G4b.
