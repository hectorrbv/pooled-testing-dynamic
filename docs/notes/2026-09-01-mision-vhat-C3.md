# Misión V-hat — keep candidato C3 (portado al repo principal 2026-09-01)

**Procedencia:** misión `vhat` del harness dapts-autoresearch (local; commit `7b5299b` de ese repo, juez medido contra este repo en `dc1ec2e`). Scoreboard completo: `results/vhat_mission_scoreboard.tsv`. Reproducción: `dapts-autoresearch/run_vhat.py` con el juez `augmented/evolucion_scores.py` de este repo. La corrida evolutiva por API murió por créditos en gen 4 sin persistir (parcheado: guardado incremental); estos resultados vienen del carril de agente.

## Código exacto del candidato C3

```python
# Candidato editable de la mision V-hat (ver program_vhat.md).
# Solo se permite editar este archivo. Firma obligatoria: score(ctx).
# Sandbox: builtins restringidos + math; sin imports, sin estado global.
# C3: C2 + reserva de virgenes — valor de oportunidad de los virgenes que la
# accion NO consume (min(b-1, virgenes restantes) * sano medio * u medio).
# Corrige la sobrevaloracion del pool grande cuando el presupuesto restante
# alcanza para explotar virgenes sueltos.

def score(ctx):
    u_S = ctx['u_S']
    tam = ctx['tam']
    imm = ctx['p_limpio'] * u_S
    if ctx['tipo'] == 'ref':
        at, r = ctx['atomo_tam'], ctx['atomo_r']
        resto = at - tam
        if resto > 0 and 0 < r <= tam:
            p_resto_limpio = math.comb(tam, r) / math.comb(at, r)
            imm += p_resto_limpio * (u_S / tam) * resto
    promesa = ctx['v_magico'] - ctx['p_limpio'] * u_S
    total = imm
    if promesa > 0:
        c_extra = math.ceil(math.log2(tam)) if tam > 1 else 1
        total += min(1.0, (ctx['b'] - 1) / c_extra) * promesa
    virg_restantes = ctx['virgenes'] - (tam if ctx['tipo'] == 'open' else 0)
    if virg_restantes > 0 and ctx['b'] > 1:
        sano_medio = ctx['e_sanos'] / tam
        total += min(ctx['b'] - 1, virg_restantes) * sano_medio * (u_S / tam)
    return total
```


**Etiqueta de origen: descubierto por búsqueda, diagnóstico §25, pendiente G4a/G4b.**

## Qué es

`score(ctx)` = inmediato G0 + promesa factible + reserva de vírgenes
(contenido final de `vhat_candidato.py`):

1. **Inmediato G0**: `p_limpio * u_S`, y para 'ref' se suma el crédito del
   resto deducido limpio — si S contiene los `atomo_r` infectados, el
   complemento se acredita bajo posterior-zero (ratificación G0). La
   probabilidad es hipergeométrica `comb(tam, r)/comb(atomo_tam, r)`, exacta
   por intercambiabilidad dentro del átomo (prior homogéneo del benchmark).
2. **Promesa factible**: `(v_magico - p_limpio*u_S)` ponderada por
   `min(1, (b-1)/ceil(log2 tam))` — la promesa solo vale si el presupuesto
   restante alcanza para cobrarla por bisección (tijera suave).
3. **Reserva de vírgenes**: `min(b-1, virgenes_no_consumidos) * sano_medio *
   u_medio` — valor de oportunidad de los vírgenes que la acción deja vivos.
   Corrige la preferencia espuria por el pool grande cuando quedan tests para
   explotar vírgenes sueltos.

## Números (juez exacto B-M17, posterior-zero; ratio = V^pi / V*)

| candidato | media_train | peor_train | media_heldout | peor_heldout |
|---|---|---|---|---|
| barra de keep (mejores baselines) | 0.8780 | 0.6877 | — | — |
| C1 inmediato + crédito resto | 0.8574 | 0.6877 | 0.8346 | 0.6588 |
| C2 = C1 + promesa factible | 0.9971 | 0.9302 | 0.9917 | 0.9668 |
| **C3 = C2 + reserva de vírgenes (KEEP)** | **0.9993** | **0.9647** | **1.0000** | **1.0000** |
| C4 = C3 con reserva − costo de cobro | 0.8426 | 0.6877 | 0.7744 | 0.6588 |

0 violaciones en todas las corridas (tests del juez pass, ningún ratio > 1).
Held-out (n=6) perfecto — sin señal de sobreajuste; al contrario, generaliza
mejor que en train. Única instancia imperfecta en train: (5,3,3,0.70) con
ratio 0.9647 (la política abre un triple donde el óptimo abre un par; el
modelo aditivo no captura la concavidad presupuesto-vs-cobro; el intento C4
de descontar el costo de cobro de la reserva colapsa la media, así que se
deja documentado y descartado).

## Por qué funciona

- La patología del mágico puro (no-reentrada, peor 0.4651) se cura con el
  crédito del resto deducido: el score ve que refinar un átomo con
  `tam == atomo_r` cobra por ambos lados (ratificación G0, notebook 26 §4).
- La miopía del inmediato puro (media 0.8492) se cura con la promesa, pero
  solo la parte cobrable con el presupuesto restante — la tijera suave evita
  heredar la patología del mágico.
- El término de reserva alinea la elección de tamaño de pool con el
  presupuesto: abrir el pool máximo deja de dominar cuando quedan tests que
  rendirían más sobre vírgenes sueltos.

## Advertencias

- La hipergeométrica del crédito del resto y el `sano_medio` de la reserva
  asumen intercambiabilidad/homogeneidad — exacta en este benchmark, no en
  general. Antes de G4a/G4b hay que decidir la forma general (con `p_sano`
  por miembro se puede hacer sin esa hipótesis).
- Benchmark con G ≤ 3: densidad y bisección coinciden; en G ≥ 4 el término
  `ceil(log2 tam)` empieza a separarse (Prop 9.1) y habría que revalidar.
- Estatuto §25: diagnóstico. No es candidata S3 hasta pasar G4a/G4b.

Filas correspondientes en `results_vhat.tsv` (la última es el keep-candidato
final, status `candidate` a la espera de la revisión humana que autorice
`--status keep`).
