"""
DAPTS strategy representation.

A DAPTS F = (F1, ..., FB) maps histories to pools.
  - Fk : H_{k-1} -> P_G([n])
  - History h_k is a tuple of k (pool_mask, result) pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# A History is a tuple of (pool_mask, test_result) pairs.
# Length-0 history is the empty tuple ().
History = Tuple[Tuple[int, int], ...]


class DAPTS:
    """Dynamic Augmented Pooled Testing Strategy.

    Stores a policy table: for each step k (1-indexed), a mapping from
    histories of length k-1 to a pool mask.

    Attributes
    ----------
    B : int
        Budget (number of test rounds).
    policy : list[dict[History, int]]
        policy[k] maps a History of length k to the pool chosen at step k+1.
        So policy[0] maps () -> pool for the first test, etc.
    """

    def __init__(self, B: int, policy: Optional[List[Dict[History, int]]] = None):
        self.B = B
        if policy is None:
            self.policy = [{} for _ in range(B)]
        else:
            assert len(policy) == B
            self.policy = policy

    def choose(self, k: int, history: History) -> int:
        """Return the pool mask for step *k* (1-indexed) given *history*.

        Parameters
        ----------
        k : int
            Step number, 1-indexed.  Must satisfy 1 <= k <= B.
        history : History
            Tuple of (pool_mask, result) pairs of length k-1.

        Returns
        -------
        int
            Pool bitmask chosen by the policy.
        """
        assert 1 <= k <= self.B, f"k={k} out of range [1, {self.B}]"
        assert len(history) == k - 1, (
            f"Expected history length {k - 1}, got {len(history)}"
        )
        return self.policy[k - 1][history]

    def set_action(self, k: int, history: History, pool_mask: int) -> None:
        """Set the pool chosen at step *k* for a given *history*."""
        assert 1 <= k <= self.B
        assert len(history) == k - 1
        self.policy[k - 1][history] = pool_mask

    def __repr__(self) -> str:
        entries = sum(len(d) for d in self.policy)
        return f"DAPTS(B={self.B}, entries={entries})"
