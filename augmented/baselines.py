"""
Baseline benchmarks from Section 2.1 of the paper.

U_max  — upper bound: sum of u_i * q_i for all individuals.
U_single — optimal strategy that tests individuals one at a time.
"""

from __future__ import annotations

from typing import List, Tuple


def u_max(
    p: List[float],
    u: List[float],
) -> float:
    """U^max(J) = sum_i u_i * q_i.

    Upper bound on expected utility — achieved only if every individual
    could be tested individually (infinite budget).
    """
    return sum(ui * (1.0 - pi) for ui, pi in zip(u, p))


def u_single(
    p: List[float],
    u: List[float],
    B: int,
) -> Tuple[float, List[int]]:
    """U^single(J, B) = sum of top min(B, n) values of u_i * q_i.

    Optimal strategy when each test can only contain a single individual
    (non-augmented, non-pooled).

    Returns
    -------
    value : float
        Expected utility of the optimal single-individual strategy.
    selected : list[int]
        Indices of the selected individuals (sorted by u_i*q_i descending).
    """
    n = len(p)
    scores = [(u[i] * (1.0 - p[i]), i) for i in range(n)]
    scores.sort(reverse=True)
    k = min(B, n)
    selected = [idx for _, idx in scores[:k]]
    value = sum(s for s, _ in scores[:k])
    return value, selected
