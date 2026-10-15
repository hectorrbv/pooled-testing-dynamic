# Búsqueda en Hermeneuta — 15 de octubre de 2026

Héctor pidió ejecutar en el VPS la búsqueda nocturna preparada. Se inició a las 15:20 de Praga (13:20 UTC), como servicio independiente de la conexión SSH. El arranque se comprobó con el servicio activo, su proceso consumiendo CPU y el registro del piloto. Todavía no hay una noche terminada ni un resultado nuevo que presentar.

## Qué está ejecutándose

- Host SSH: `hermeneuta-vps`.
- Directorio aislado: `/home/hermeneuta/agent-workspace/groupcount-night-20261015`.
- Servicio: `groupcount-night-20261015.service`; PID inicial 1720917.
- Controlador: `augmented/autoresearch_ejemplos/night_runner.py`.
- Primero, control de viabilidad de la semilla n=7, B=4, G=3, razón 41/52. Si alcanza el límite, la búsqueda de precio continúa con n≤6. No se repite el mismo control limitado en cada experimento.
- Después, hasta 40 experimentos alternando binario y precio. Cada uno dispone de 180 s de búsqueda y un límite de 360 s. Un fallo del juez detiene la sesión; un límite de recursos se registra como `limited`.
- Un solo proceso de cálculo a la vez, prioridad baja, una CPU como máximo. El juez se corta al superar 1024 MiB; el servicio completo tiene un límite de 1200 MiB y cinco horas.

Se usan las semillas certificadas actuales: laminar/binario≈0.90000075 y laminar/general=41/52 (5/6 para n≤6). No se repite la malla homogénea q=1/2 ya cerrada. Se perturban probabilidades y utilidades en la misión binaria; en precio se conserva u=1 y se distinguen familias homogéneas, una coordenada perturbada y heterogéneas cerca de q=1/2.

El controlador cambia la familia y la semilla aleatoria entre experimentos, conserva un generador cuando mejora el récord comprobado y restaura el anterior en los demás casos. Es una búsqueda programada de familias, sin llamadas a un modelo de lenguaje. El juez y los solvers permanecen intactos y se comprueban sus hashes antes y después de cada ejecución.

## Verificaciones del arranque

Los 23 archivos desplegados coincidieron con sus hashes locales; pasaron las ocho anclas exactas del juez en el VPS. Se comprobaron localmente 700 candidatos distintos y válidos entre las siete familias. Python 3.11 y sus dependencias viven en `.venv311` dentro del directorio aislado. La copia inicial incompleta no llegó a lanzar ningún experimento.

Manifiesto: `results/sesion_2026-10-15/noche_hermeneuta/deployment.json`. El código de referencia parte del commit `072e0d9`; el controlador nuevo queda versionado con esta nota. La comprobación humana e independiente de nuevos récords sigue pendiente.

## Dónde consultar y recuperar

En el directorio remoto, `launcher.log` registra la ejecución general. Dentro de `augmented/autoresearch_ejemplos/`, los TSV `results_binario.tsv` y `results_precio.tsv` contienen exclusivamente filas escritas por el juez. La carpeta `night_20261015-hermeneuta/` guarda `status.json`, `experiments.jsonl`, notas y un generador y log por experimento, incluidos los limitados.

Estado del servicio:

```sh
ssh hermeneuta-vps 'systemctl --user status groupcount-night-20261015 --no-pager'
```

Para detenerlo por petición del usuario:

```sh
ssh hermeneuta-vps 'systemctl --user stop groupcount-night-20261015'
```

Al terminar: recuperar los registros, separar controles de descubrimientos, comparar con las semillas y re-resolver los mejores candidatos por la vía independiente. El inicio de la ejecución no cierra esas tareas ni constituye un teorema.
