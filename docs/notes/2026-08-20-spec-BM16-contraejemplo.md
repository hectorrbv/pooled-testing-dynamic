# Especificación B-M16 — Artefacto del contraejemplo de no-reentrada (2026-08-20)

**Para:** Héctor (B-M16, día 2 de la tabla §1). **Origen:** sesión 2026-08-18 (acta D2) + corrida a mano de A (2026-08-20, dos vías: estructura y enumeración de 16 mundos). Los números de abajo son los **tests de aceptación**: el artefacto debe reproducirlos por enumeración exacta — sin Monte Carlo, sin semillas.

## Instancia

n = 4 ({a,b,c,d}) · q = P(sano) = 0.3 i.i.d. · u ≡ 1 · B = 2 · G = 2 · hard clearing estricto · laminar (los pares mixtos tipo {a,c} cruzan {a,b} y quedan fuera del menú) · desempate declarado (lexicográfico) · **supuesto**: V̂ descuenta acreditados (la versión dictada en sesión los cuenta — pregunta (8) de §34; documentar como flag).

## Números objetivo

**Predictiva de un par virgen:** P(R=0)=0.09 · P(R=1)=0.42 · P(R=2)=0.49.

**Posterior tras R=1 en {a,b}:** P(a sano)=P(b sano)=1/2 · P(ambos sanos)=0 (≠ 1/4 del producto de marginales — check del tensor) · P(c sano)=0.3 intacta.

**Tabla de inversión (estado: R=1 en {a,b}, b=1 restante):**

| Acción | V̂ (presupuesto mágico) | Realizable con b=1 |
|---|---|---|
| Retest {a,b} | 1.0 | 0 |
| Par virgen {c,d} | 0.6 | 0.18 |
| Reentrada {a} | 0.5 | 0.5 |
| Singleton virgen {c} | 0.3 | 0.3 |

Inversión exacta en las tres primeras (las dictadas en sesión). Regret local de V̂: **0.5** con retest permitido; **0.32** con retest podado.

**Bienestares esperados de políticas completas (desde el estado inicial):**

| Política | Valor |
|---|---|
| V̂ cruda (retest permitido) | 0.2844 |
| V̂ + poda de repetición idéntica | 0.36 |
| S₀ (dos singletons frescos) | 0.6 |
| Par primero con continuación perfecta | 0.564 |
| **Óptimo** | **0.6 — nunca agrupa en estos parámetros** |

## Asserts mínimos

1. Predictivas y posteriores exactos (incluido P(ambos sanos | R=1) = 0).
2. La inversión de ranking en las tres acciones de la sesión.
3. Los cinco bienestares de política, exactos.
4. La poda mejora a V̂ (0.2844 → 0.36) pero no la salva (0.36 < 0.6).

## Extensión opcional (sobre-entrega)

Variante B = 3 para **exhibir la absorción**: V̂ cruda retestea {a,b} dos veces seguidas ("se quedaba retesteando el mismo par"). CSV por decisión con formato del falsificador (§17), clase "repetida".

## Alcance del claim (§25 — copiar al artefacto)

El artefacto demuestra la patología de scoring (inversión + absorción + no-reentrada). **No** es evidencia a favor del pooling: en estos parámetros el óptimo nunca agrupa; la evidencia de valor del conteo+agrupación vive en la escalera (n=10, q=0.2) y el ancla del acid test (q=0.05, G=16, B=7).
