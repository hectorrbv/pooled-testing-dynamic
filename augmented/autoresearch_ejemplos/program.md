# Mission: the worst example for Francisco (autoresearch loop, started 2026-10-12)

You are running an autonomous experiment loop, in the style of karpathy/autoresearch, on an exact combinatorial search. No GPU, no training; the metric is exact arithmetic.

## What we are looking for

Francisco (our advisor) asked on 2026-10-06 for two concrete examples:

1. **binario** — an instance where the optimal *dynamic binary* policy (tests return positive/negative, overlapping pools allowed) beats the optimal *laminar augmented* policy (tests return exact infection counts, pools nested or disjoint). The worst one possible: minimize `laminar / binary`. Known: it needs Λ > 1 (some healthy probabilities above ½); the judge skips instances with Λ ≤ 1. Best so far 0.978 at n = 4 (random search).
2. **precio** — the price of laminarity: minimize `laminar_augmented / unrestricted_augmented`. Francisco conjectures the worst case is near q = ½. Best so far 0.833 at n = 6, q = ½, G = 3, B = 4 (grid search).

## The rules (do not break them)

- You edit **only** `generator.py` (the function `sample(rng)`). Never edit `judge.py`, never edit solvers under `augmented/`, never write anywhere else except this folder.
- One experiment = one run of the judge with a fixed wall-clock budget. From the repository root:
  `PYTHONPATH=. /opt/miniconda3/bin/python augmented/autoresearch_ejemplos/judge.py binario 180 6 "<one-line description of the family you tried>"`
  and the same with `precio`. Budget 180 s, n ≤ 6. Alternate missions or focus on one; both results files matter.
- Read the printed line. **Keep** the generator if `status=ok` and `ratio_min` improved on the best kept value for that mission; otherwise **revert** `generator.py` to the last kept version (keep a copy as `generator_best_<mission>.py`). Status `fail` means a gate tripped: revert and do not retry the same idea.
- Every experiment is logged by the judge in `results_<mission>.tsv` with the full argmin instance. Never edit those files.
- Ideas worth trying, in order of promise: one very healthy person plus several at ½–0.7 (the cross {a,c} mechanism found on 2026-10-12); two-type populations; utilities concentrated on one or two people; B equal to n − 1 or n − 2; for `precio`: all q = ½ with structure in u, G = 3, B = 3–4, n = 6; mixtures of q = ½ and q near 0 or 1.
- Write a short note of what you learned after every 5 experiments in `notes.md` (append only): which families moved the minimum, which did nothing.
- Stop after 40 experiments or when told. Do not commit.

## What the humans will do with it

On Tuesday morning they re-solve the top three instances of each mission by hand-run of the solvers, draw the optimal trees, and explain the mechanism to Francisco. Nothing you find is a theorem; it is a diagnostic on small exact cases.
