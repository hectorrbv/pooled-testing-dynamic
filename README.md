# Pooled Testing: Classical & Augmented Dynamic Strategies

Research codebase for welfare-maximizing pooled testing strategies, implementing both **classical** (binary outcome) and **augmented** (count-based outcome) dynamic models.

## Repository Structure

```
classical/          Classical binary pooled testing (prior work)
  solvers/          MILP, greedy dynamic, exact DP, Bayes updates
  rl_training/      PPO reinforcement learning training
  rl_evaluation/    PPO model evaluation
  training/         PyTorch RL training scripts
  notebooks/        Jupyter development & visualization notebooks
  data/             Experimental results and training datasets
  models/           Saved model weights
  figures/          Output visualizations

augmented/          Augmented pooled testing (new, January 2026)
  core.py           Bitmask helpers, pool enumeration PG(S), test result r(t,Z)
  strategy.py       DAPTS strategy representation F = (F1,...,FB)
  simulator.py      Simulate strategy on fixed infection profile
  expected_utility.py  Exact and Monte Carlo expected utility
  baselines.py      U_max and U_single benchmarks
  solver.py         Brute-force DP solver for optimal DAPTS
  tests.py          Unit tests (17 tests)
  example.py        Demo script
```

## Quick Start

```bash
# Run augmented model tests
python augmented/tests.py

# Run augmented model example (n=5, B=2, G=3)
python augmented/example.py
```

## References

- "Dynamic-Augmented Pooled Testing" (Vladimir, Hector, Francisco, January 2026)
- S. Finster et al., "Welfare-maximizing pooled testing," arXiv:2206, 2022
