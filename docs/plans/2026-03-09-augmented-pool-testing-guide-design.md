# Augmented Pool Testing Guide Implementation Design

Date: 2026-03-09
Source: "Guide for Augmented Pool Testing" (Francisco's meeting prep)

## Overview

Six improvements to the DAPTS codebase based on Francisco's feedback.

## Item 1: Preprocessing — Exclude cleared/confirmed-infected from pools

**Problem:** Pool selection considers all individuals, even those whose status is already known.

**Solution:**
- `core.py`: Add `compute_active_mask(p, cleared_mask, n, threshold)` — returns bitmask of uncertain individuals
- `core.py`: Add `all_pools_from_mask(active_mask, G)` — pools only from active individuals
- `greedy.py`: Update `_myopic_best_pool` and `_lookahead_best_pool` to use filtered pools
- Track `confirmed_infected_mask` when p_i approaches 1 after Bayesian updates

## Item 2: Full-history Bayesian update by counting

**Problem:** Sequential Bayesian updates obscure the joint computation. Francisco wants explicit counting over consistent worlds.

**Solution:** Add `bayesian_update_by_counting(p, history, n)` to `bayesian.py`:
- Enumerate all 2^n infection profiles
- Filter to profiles consistent with ALL test results
- P(Z_i=1 | h_k) = weighted count of consistent profiles with Z_i=1 / total weighted count
- O(2^n * k * n), feasible for n <= ~14

## Item 3: Eliminate restriction on parameter "p"

**Problem:** Edge cases when p_i = 0 or p_i = 1.

**Solution:**
- Handle p_i = 0 and p_i = 1 gracefully in Bayesian update (deterministic individuals)
- Add `estimate_p_from_history(history, n, prior_p)` for data-driven estimation
- Ensure solver and greedy don't break with extreme p values

## Item 4: Extract the decision tree

**Problem:** Can't see what strategy the solver is actually using.

**Solution:** Add `tree_extractor.py`:
- `extract_tree(policy, n)` — nested tree structure from DAPTS policy
- `print_tree(tree, n)` — human-readable text output
- `export_tree_dot(tree, n)` — DOT format for Graphviz
- Each node shows: pool chosen, outcomes, cleared individuals, posterior beliefs

## Item 5: Greedy with full-history Bayesian update

**Problem:** Need both sequential and counting-based Bayesian greedy variants.

**Solution:** Add to `greedy.py`:
- `greedy_myopic_counting_simulate(p, u, B, G, z_mask)` — uses `bayesian_update_by_counting` on full accumulated history
- `greedy_myopic_counting_expected_utility(p, u, B, G)` — expected utility of counting variant
- Keep existing greedy functions unchanged

## Item 6: Random instance exploration

**Problem:** Need systematic exploration across different parameter regimes.

**Solution:** Add `experiments.py`:
- `random_instance(n, B, G, p_range, u_range, seed)` — generate random instances
- `run_experiment(instances, strategies)` — compare all strategies
- `summarize_results(results)` — statistics and performance comparison
- Focus on high infection rate regimes (Francisco's hypothesis)

## Files modified

| File | Changes |
|------|---------|
| `core.py` | `compute_active_mask`, `all_pools_from_mask` |
| `bayesian.py` | `bayesian_update_by_counting`, `estimate_p_from_history`, edge-case handling |
| `greedy.py` | Filtered pool selection, counting-based greedy variants |
| `tree_extractor.py` | New file: tree extraction and visualization |
| `experiments.py` | New file: random instance exploration |
| `tests.py` | Tests for all new functionality |
