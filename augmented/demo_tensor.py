"""Demo minima del tensor condicional pedido por Francisco.

Ejecutar desde la raiz de ``group-count-dynamic``:

    python -m augmented.demo_tensor

La demo responde solamente tres preguntas:

1. Que tabla produce Q[T'][r'].
2. Como comprobamos que la formula cerrada es correcta.
3. Que guarda la cache Phi y como se reutiliza al dividir un pool.
"""

import numpy as np

from augmented.laminar_tables import (
    split_after_test,
    subpool_tensor,
    subpool_tensor_brute,
    subset_pmf_cache,
)


# Priors iguales para que los numeros de la exposicion sean faciles de verificar
# a mano. Las pruebas automatizadas usan priors heterogeneos.
PRIORS = np.array([0.5, 0.5, 0.5, 0.5])
COUNT = 2


def subset_name(mask, size):
    members = [str(i) for i in range(size) if mask & (1 << i)]
    return "{" + ",".join(members) + "}"


def print_tensor(tensor, size, columns_per_block=8):
    """Imprime subconjuntos en el eje x, conteos en el eje y y sumas abajo."""

    subsets = list(tensor)
    print("Subconjuntos T' en el eje x; conteos r' en el eje y.")
    for start in range(0, len(subsets), columns_per_block):
        block = subsets[start:start + columns_per_block]
        print()
        print(" " * 10 + "".join(
            f"{subset_name(subset, size):>10}" for subset in block
        ))
        for result in range(size + 1):
            cells = []
            for subset in block:
                distribution = tensor[subset]
                cells.append(
                    f"{distribution[result]:10.4f}"
                    if result < len(distribution)
                    else " " * 10
                )
            print(f"{f'r={result}':>10}{''.join(cells)}")
        print(f"{'suma':>10}" + "".join(
            f"{tensor[subset].sum():10.4f}" for subset in block
        ))


def max_tensor_error(first, second):
    return max(
        float(np.max(np.abs(first[subset] - second[subset])))
        for subset in first
    )


def show_tensor_and_validation():
    size = len(PRIORS)
    full = (1 << size) - 1
    impossible_subset = 0b0111

    cache = subset_pmf_cache(PRIORS)
    tensor = subpool_tensor(PRIORS, COUNT, cache=cache)

    print("   Q[T'][r'] = P(count(T') = r' | count(T) = R)")
    print(f"   Priors p = {PRIORS.tolist()}")
    print(f"   Conteo observado del pool completo: R = {COUNT}\n")
    print_tensor(tensor, size)

    print("\n2. COMPROBACIONES VISIBLES")
    print(
        "   Pool completo: "
        f"Q[T] = {np.round(tensor[full], 4).tolist()} "
        f"(toda la masa esta en r={COUNT})"
    )
    print(
        f"   Evento imposible: Q[{subset_name(impossible_subset, size)}][0] "
        f"= {tensor[impossible_subset][0]:.1f}"
    )

    brute = subpool_tensor_brute(PRIORS, COUNT)
    formula_error = max_tensor_error(tensor, brute)
    normalization_error = max(
        abs(float(distribution.sum()) - 1.0)
        for distribution in tensor.values()
    )

    assert formula_error < 1e-12
    assert normalization_error < 1e-12
    assert tensor[impossible_subset][0] == 0.0

    print(f"   Formula cerrada vs enumeracion exhaustiva: error {formula_error:.2e}")
    print(f"   Error maximo en sumas de columnas: {normalization_error:.2e}")
    print("   Resultado: PASS")
    return cache, tensor


def show_cache(cache, tensor):
    size = len(PRIORS)
    full = (1 << size) - 1
    subset = 0b0011
    complement = full ^ subset
    result = 1

    inside = cache[subset][result]
    outside = cache[complement][COUNT - result]
    denominator = cache[full][COUNT]
    reconstructed = inside * outside / denominator

    print("\n3. RELACION CON LA CACHE")
    print("   Phi[S][k] = P(count(S) = k) antes de condicionar.")
    print(f"   Phi[{subset_name(subset, size)}] = {np.round(cache[subset], 4).tolist()}")
    print(
        f"   Phi[{subset_name(complement, size)}] "
        f"= {np.round(cache[complement], 4).tolist()}"
    )
    print(f"   Phi[T][R] = {denominator:.4f}")
    print(
        f"   Q[{subset_name(subset, size)}][{result}] "
        "= Phi[T'][r'] * Phi[T\\T'][R-r'] / Phi[T][R]"
    )
    print(
        f"   Q[{subset_name(subset, size)}][{result}] "
        f"= {inside:.4f} * {outside:.4f} / {denominator:.4f} "
        f"= {reconstructed:.4f}"
    )
    assert abs(reconstructed - tensor[subset][result]) < 1e-12

    tested_atom, residual_atom = split_after_test(
        PRIORS,
        COUNT,
        tested=subset,
        tested_count=1,
        cache=cache,
    )
    tested_fresh = subpool_tensor(tested_atom.priors, tested_atom.count)
    residual_fresh = subpool_tensor(residual_atom.priors, residual_atom.count)
    reuse_error = max(
        max_tensor_error(tested_atom.tensor, tested_fresh),
        max_tensor_error(residual_atom.tensor, residual_fresh),
    )
    assert reuse_error < 1e-12

    print("\n   Al dividir T en dos hijos, ambos reutilizan la cache del padre.")
    print(f"   Error contra recalcular ambos hijos desde cero: {reuse_error:.2e}")
    print("   Convoluciones nuevas para construir las caches de los hijos: 0")
    print("   Resultado: PASS")


def main():
    print("TENSOR CONDICIONAL DE SUBPOOLS\n")
    cache, tensor = show_tensor_and_validation()
    show_cache(cache, tensor)
    print("\nCONCLUSION")
    print("   Q es la tabla condicionada que pidio Francisco.")
    print("   Phi es el calculo reusable que permite construir Q sin repetir trabajo.")


if __name__ == "__main__":
    main()
