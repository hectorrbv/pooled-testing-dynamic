# Design: Five Follow-Up Items for Augmented Pooled Testing

**Date**: 2026-03-09

## Item 1: B >= 3 Instances

Show divergence between sequential and counting greedy when pools overlap with B>=3.
- New experiments with B in {3,4,5}, n in {5,6,7}, G in {2,3}
- Compare sequential, counting, gibbs greedy + optimal (if n<=14)
- Graph: divergence vs B

## Item 2: Semi-Utility Meta-Parameter

Francisco's extension (NOT in the paper, which uses binary clearance):
```
U_semi(alpha) = sum_i u_i * [alpha * P(healthy_i | H_k) + (1-alpha) * 1_{cleared}(i)]
```
- alpha=0: current binary model
- alpha=1: posterior-based
- New file `semi_utility.py` with evaluation functions
- Integrate with greedy for comparison

## Item 3: High Infection Rate Experiments at Scale

500+ instances, p in [0.3, 0.8], n in {5,8,10}, B in {2,3}.
Confirm Francisco's hypothesis that augmented benefit grows with infection rate.

## Item 4: Cross-Verification with Nick's Code

Export synthetic instances as JSON for `nrlopez03/pooled-testing`.
Document comparison protocol.

## Item 5: Tree Pruning for Large n

- `prune_tree(tree, max_depth)`: cut at depth
- `summarize_tree(tree, n)`: aggregate stats
- Updated DOT export with pruning option
