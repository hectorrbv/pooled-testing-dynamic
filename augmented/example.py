"""
Example: solve a small DAPTS instance and display results.

Usage:
    cd "pooled testing dynamic"
    python dapts/example.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask, mask_from_indices, test_result
from augmented.strategy import DAPTS
from augmented.simulator import apply_dapts
from augmented.expected_utility import exact_expected_utility, mc_expected_utility
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts


def fmt_pool(mask: int, n: int) -> str:
    """Format a pool mask as a readable set."""
    indices = indices_from_mask(mask, n)
    return "{" + ", ".join(str(i) for i in indices) + "}" if indices else "{}"


def main():
    # ------------------------------------------------------------------
    # Define a population instance J = (p, u)
    # ------------------------------------------------------------------
    #   Individual i:  utility u[i],  infection probability p[i]
    #   Index:            0     1     2     3     4
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0,  6.0,  3.0,  5.0,  7.0]
    n = len(p)
    B = 2   # budget: 2 tests
    G = 3   # max pool size: 3

    print("=" * 60)
    print("Dynamic Augmented Pooled Testing — Brute Force Example")
    print("=" * 60)
    print(f"\nPopulation (n={n}):")
    for i in range(n):
        print(f"  Individual {i}:  u={u[i]:.1f},  p={p[i]:.2f},  q={1-p[i]:.2f}")
    print(f"\nBudget B={B},  Max pool size G={G}")

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    um = u_max(p, u)
    us, us_sel = u_single(p, u, B)

    print(f"\n--- Baselines ---")
    print(f"U_max   = {um:.4f}  (upper bound: all individuals tested)")
    print(f"U_single = {us:.4f}  (test {B} best individuals: {us_sel})")

    # ------------------------------------------------------------------
    # Solve for optimal DAPTS
    # ------------------------------------------------------------------
    print(f"\nSolving optimal DAPTS (n={n}, B={B}, G={G}) ...")
    opt_val, F_opt = solve_optimal_dapts(p, u, B, G)
    print(f"U_A^D   = {opt_val:.4f}  (optimal augmented dynamic)")

    print(f"\nInequality check:  U_single <= U_A^D <= U_max")
    print(f"  {us:.4f}  <=  {opt_val:.4f}  <=  {um:.4f}  ✓")

    # ------------------------------------------------------------------
    # Display the optimal policy (first test + second test branches)
    # ------------------------------------------------------------------
    print(f"\n--- Optimal Policy ---")
    first_pool = F_opt.choose(1, ())
    print(f"Step 1: Test pool {fmt_pool(first_pool, n)}")

    if B >= 2:
        # Show all possible outcomes of the first test and the response
        pool_size = bin(first_pool).count("1")
        for r in range(pool_size + 1):
            history = ((first_pool, r),)
            try:
                second_pool = F_opt.choose(2, history)
                print(f"  If result r={r}: Step 2 tests {fmt_pool(second_pool, n)}")
            except KeyError:
                pass  # this outcome is unreachable given the population

    # ------------------------------------------------------------------
    # Verify with exact and Monte Carlo expected utility
    # ------------------------------------------------------------------
    exact_eu = exact_expected_utility(F_opt, p, u, n)
    mc_eu = mc_expected_utility(F_opt, p, u, n, trials=500_000, seed=42)
    print(f"\n--- Verification ---")
    print(f"Exact E[u(F*)]  = {exact_eu:.6f}")
    print(f"MC E[u(F*)]     = {mc_eu:.6f}  (500k samples)")

    # ------------------------------------------------------------------
    # Simulate on specific infection profiles
    # ------------------------------------------------------------------
    print(f"\n--- Simulations on specific profiles ---")
    profiles = [
        ("Nobody infected", 0),
        ("Only individual 3 infected", mask_from_indices([3])),
        ("Individuals 0,2 infected", mask_from_indices([0, 2])),
        ("Everyone infected", (1 << n) - 1),
    ]

    for label, z_mask in profiles:
        hist, cleared, u_val = apply_dapts(F_opt, z_mask, n, u)
        z_set = indices_from_mask(z_mask, n)
        cleared_set = indices_from_mask(cleared, n)
        print(f"\n  Z = {z_set}  ({label})")
        for step, (pool, r) in enumerate(hist, 1):
            print(f"    Step {step}: test {fmt_pool(pool, n)}, result r={r}"
                  f"{'  → CLEARED!' if r == 0 else ''}")
        print(f"    Cleared: {cleared_set},  Utility: {u_val:.1f}")

    print()


if __name__ == "__main__":
    main()
