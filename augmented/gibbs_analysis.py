"""
Standalone Gibbs convergence analysis for fixed small augmented instances.

Run with: python augmented/gibbs_analysis.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.bayesian import bayesian_update_by_counting, gibbs_update
from augmented.core import mask_from_indices


ITERATION_COUNTS = [50, 100, 200, 500, 1000, 2000, 5000]
SEEDS = list(range(10))


def _analyze_instance(n, pool_indices):
    p = [0.15] * n
    history = ((mask_from_indices(pool_indices), 1),)
    exact = bayesian_update_by_counting(p, history, n)

    print(f"Instance: n={n}, pool={pool_indices}, result=1")
    print("iterations | mean_max_error | std_max_error")

    for iterations in ITERATION_COUNTS:
        errors = []
        for seed in SEEDS:
            approx = gibbs_update(
                p, history, n, num_iterations=iterations, seed=seed)
            max_abs_error = max(abs(approx[i] - exact[i]) for i in range(n))
            errors.append(max_abs_error)

        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        print(f"{iterations:10d} | {mean_error:14.6f} | {std_error:13.6f}")

    print()


def main():
    _analyze_instance(n=8, pool_indices=[0, 1, 2, 3])
    _analyze_instance(n=6, pool_indices=[0, 1, 2])


if __name__ == "__main__":
    main()
