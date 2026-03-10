# Dynamic Augmented Pooled Testing (DAPTS)

Welfare-maximizing pooled testing strategies with **augmented** test outcomes: each pooled test returns the exact count of infected individuals, not just a binary positive/negative result.

This repository implements the framework described in *"Dynamic Augmented Pooled Testing"* (V. Dvorkin, H. Becerril, F. Lopez, 2026).

## Key Features

- **Exact DP solver** for optimal augmented and classical dynamic strategies
- **Three Bayesian posterior update methods**: Sequential (fast), Counting (exact), and Gibbs sampling with swap moves (scalable)
- **Greedy heuristics** that run in polynomial time and closely approximate the optimal
- **Semi-utility framework** interpolating between binary clearing and posterior-weighted decisions
- **Cross-verification protocol** for validating results against independent implementations
- **Comprehensive experiments**: B-divergence analysis, high infection rate regimes, augmented vs. classical benefit

## Repository Structure

```
pooled-testing-dynamic/
├── README.md
├── requirements.txt
├── .gitignore
│
├── augmented/                  # Augmented pooled testing (main contribution)
│   ├── core.py                 # Bitmask helpers, pool enumeration, test result r(t,Z)
│   ├── bayesian.py             # Sequential, counting, and Gibbs posterior updates
│   ├── greedy.py               # Greedy myopic strategies (sequential, counting, Gibbs)
│   ├── solver.py               # Exact DP solver for optimal DAPTS
│   ├── classical_solver.py     # Classical (binary) dynamic solver for comparison
│   ├── comparison.py           # Compare all strategies on a given instance
│   ├── semi_utility.py         # Semi-utility with alpha parameter
│   ├── csv_experiments.py      # Large-scale experiment runner (B-divergence, high infection)
│   ├── cross_verification.py   # Cross-verification protocol and synthetic instance generator
│   ├── tree_extractor.py       # Decision tree extraction, pruning, Graphviz export
│   ├── strategy.py             # DAPTS strategy representation F = (F1,...,FB)
│   ├── simulator.py            # Simulate strategy execution on fixed infection profile
│   ├── expected_utility.py     # Exact and Monte Carlo expected utility computation
│   ├── baselines.py            # U_max and U_single benchmark bounds
│   ├── experiments.py          # Experiment definitions and CSV output
│   ├── example.py              # Quick demo script
│   ├── tests.py                # Unit tests (17 tests)
│   ├── data/                   # Experimental result CSVs
│   ├── figures/                # Generated plots and visualizations
│   ├── notebooks/              # Jupyter notebooks with illustrated examples
│   │   └── examples_notebook.ipynb
│   └── paper/                  # LaTeX source for the paper
│       ├── results.tex         # Main paper
│       └── findings_report.tex # Internal technical report
│
├── classical/                  # Classical binary pooled testing (prior work)
│   ├── solvers/                # MILP, greedy dynamic, exact DP solvers
│   ├── rl_training/            # PPO reinforcement learning training
│   ├── rl_evaluation/          # Trained model evaluation
│   ├── training/               # PyTorch RL training scripts
│   ├── models/                 # Saved model weights (.pth)
│   ├── data/                   # Training datasets and experimental results
│   ├── figures/                # Output visualizations
│   ├── notebooks/              # Analysis notebooks
│   └── slurm_scripts/          # HPC cluster job scripts
│
└── docs/                       # Internal documentation and design notes
```

## Installation

```bash
git clone <repo-url>
cd pooled-testing-dynamic
pip install -r requirements.txt
```

For decision tree visualization, also install the Graphviz system binary:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

## Quick Start

### Run tests

```bash
python -m pytest augmented/tests.py -v
```

### Run the demo

```bash
python augmented/example.py
```

### Interactive notebook

```bash
jupyter notebook augmented/notebooks/examples_notebook.ipynb
```

The notebook covers:
1. Basic concepts (pools, bitmasks, augmented test results)
2. Three Bayesian update methods compared
3. Gibbs sampling convergence
4. Strategy comparison and the inequality chain
5. Decision tree visualization
6. Greedy simulation step-by-step
7. Scalability: Gibbs vs Counting
8. Augmented benefit over classical
9. Gibbs swap moves for ergodicity
10. B>=3 sequential vs counting divergence
11. Semi-utility alpha sweep
12. High infection rate analysis
13. Tree pruning
14. Cross-verification protocol

## Core Concepts

The **inequality chain** guarantees:

```
U_single <= U_s_NO <= U_s_O <= U_D <= U_D_A <= U_max
```

where `U_D_A` (Dynamic Augmented) dominates `U_D` (Dynamic Classical) because the exact count provides strictly more information than binary positive/negative.

### Bayesian Update Methods

| Method | Complexity | Cross-Test Info | Best For |
|--------|-----------|----------------|----------|
| Sequential | O(G) per test | Approximate | Real-time, large n |
| Counting | O(2^n) | Exact | Small n (<=20), validation |
| Gibbs (MCMC) | O(n*G*iters) | Approximate | Medium-large n (20-50+) |

## Reproducing Paper Results

Generate the main experimental results:

```bash
# Run B-divergence comparison (Section 3)
python -c "from augmented.csv_experiments import run_b_comparison; print(run_b_comparison(n=5, B_values=[2,3,4], G=3))"

# Run high infection rate analysis (Section 4)
python -c "from augmented.csv_experiments import run_high_infection_experiment; print(run_high_infection_experiment(n_values=[5], B_values=[2], G=3))"
```

## Authors

- Vladimir Dvorkin
- Hector Becerril
- Francisco Lopez

## References

- V. Dvorkin, H. Becerril, F. Lopez, "Dynamic Augmented Pooled Testing," January 2026
- S. Finster et al., "Welfare-maximizing pooled testing," arXiv:2206, 2022
