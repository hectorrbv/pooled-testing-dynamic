# Augmented adaptive group counting

`augmented/` contains the Dynamic Augmented Adaptive Group Counting Strategies (DAPTS)
code. In the augmented model, a grouped test returns the exact count
`r = |t cap Z|` of active individuals in the tested pool.

## How to read this folder

1. Start with `notebooks/README.md` for the recommended notebook order.
2. Use the notebooks for explanation, figures, and experiment narratives.
3. Use the Python modules here as the source of reusable algorithms.

## Module map

| Area | Files | Role |
| --- | --- | --- |
| Core primitives | `core.py`, `strategy.py`, `simulator.py` | Bitmasks, pool/test primitives, strategy representation and simulation. |
| Bayesian updates | `bayesian.py`, `gibbs_analysis.py`, `laminar_inference.py` | Exact, counting, Gibbs-style and exact laminar posterior updates; `laminar_pool_pmf` preserves within-atom dependence. |
| Solvers | `solver.py`, `static_solver.py`, `classical_solver.py`, `pool_solvers.py`, `scenario_milp.py`, `hybrid_solver.py` | Optimal dynamic, static, classical, scenario-MILP and hybrid solver variants. |
| Laminar benchmarks | `laminar_benchmarks.py`, `laminar_pipeline.py` | Exact four-quantity atlas for small n and the n=40 particles → MILP → atoms → rollout pipeline. |
| Greedy methods | `greedy.py`, `state_reward_greedy.py`, `semi_utility.py`, `expected_utility.py` | Myopic policies and scoring variants. |
| Experiments | `experiments.py`, `csv_experiments.py`, `experiments_laminar_week.py`, `sprint3_experiments.py`, `overnight_experiments.py`, `comparison.py`, `cross_verification.py` | Reproducible experiment runners and comparisons. The weekly laminar suite writes reanalyzable CSVs to `data/laminar_week/`. |
| VW / super-nodes | `vw_demo.py`, `vw_restrict.py`, `vw_restrict_sweep.py` | Experiments around the VW super-node reformulation. |
| RL | `rl_env.py`, `rl_train.py`, `rl_examples.py`, `rl_examples_demo.py`, `rl_models/` | Reinforcement-learning environment, training and saved exploratory models. |
| Visualization | `tree_extractor.py`, `tree_visualizer.py`, `figures/` | Tree extraction, plotting helpers and generated figures. |
| Tests | `tests*.py` | Regression and solver checks. |

## Notebook convention

The notebooks follow the experimental style used in the dynamic part of the
project: small parameter blocks (`n`, `B`, `G`, `p`, `u`), direct calculations,
short comments near the code, and printed results next to each experiment.

Each main notebook now starts with the same header:

- objective,
- guiding question,
- expected reading path,
- setup cell,
- numbered sections with a short explanation before each code block.

LaTeX source files and LaTeX build auxiliaries were removed from `augmented/`.
The folder should now focus on notebooks, code, figures, data and PDFs.

## Reproduce notebook 22

```bash
python -m augmented.experiments_laminar_week all --workers 4
python augmented/notebooks/build_milp_laminar_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
  augmented/notebooks/22_milp_laminar.ipynb
```
