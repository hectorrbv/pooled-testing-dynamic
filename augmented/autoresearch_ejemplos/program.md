# Mission: the worst example for Francisco (autoresearch loop, started 2026-10-12)

You are running an autonomous experiment loop, in the style of karpathy/autoresearch, on an exact combinatorial search. No GPU, no training; the metric is exact arithmetic.

## What we are looking for

Francisco (our advisor) asked on 2026-10-06 for two concrete examples:

1. **binario** — minimize optimal `laminar / binary`. The starting family is n=4, B=G=2, q=(s,½,½,s), u=(1,3,3,2), s=999999/1000000. The certified finite ratio is about 0.90000075; its family tends to 0.9. This limit is NOT a universal lower bound. Do not restart from the obsolete 0.978 baseline. Full fractions and source are in `warm_start.json`.
2. **precio** — minimize `laminar_augmented / unrestricted_augmented`. Start from the smallest verified ratio in `warm_start.json` that meets NMAX. The current n=7 seed is 41/52 at q=½, G=3, B=4; if NMAX=6, use 5/6 at n=6 with the same q,G,B. The expanded grid is in `results/precio_q05_G/`. Compare within the same (n,G,B) as well as globally.

## The rules (do not break them)

- You edit **only** `generator.py` (the function `sample(rng)`). Never edit `judge.py`, never edit solvers under `augmented/`, never write anywhere else except this folder.
- One experiment = one run of the judge with a fixed wall-clock budget. From the repository root:
  `PYTHONPATH=. ../venv/bin/python -m augmented.autoresearch_ejemplos.run_bounded binario --seconds 180 --nmax 6 --description "<one-line description of the family you tried>"`
  For `precio`, use `precio --seconds 180 --nmax 7` so the new best seven-person seed is included. On another machine replace `../venv/bin/python` with its Python environment. The wrapper sets the mission for the generator and stops a single slow run after 360 s. Budget 180 s; n≤6 for binario and n≤7 for precio. Alternate missions; both results files matter. A timeout is LIMITED, not a successful experiment and not an optimum.
- Read the printed line. **Keep** the generator if `status=ok` and `ratio_min` improves on the best certified seed or later kept value for that mission; otherwise **revert** `generator.py` to the last kept version (keep a copy as `generator_best_<mission>.py`). Replaying a seed is a smoke check, not a discovery. Status `fail` means a gate tripped: revert and do not retry the same idea.
- Every experiment is logged by the judge in `results_<mission>.tsv` with the full argmin instance. Never edit those files.
- Ideas: perturb the four-person 10/9 family in probabilities and utilities; compare different budgets without assuming the mechanism persists; retain some fresh heterogeneous trials. For `precio`, perturb probabilities around the verified q=½ examples, retaining u=1, and report homogeneous and heterogeneous priors separately. The q=½, u=1 grid within this judge's n≤7, B≤4, G≤4 limits is already covered (plus trivial full-budget cases); repeating 40 runs there cannot improve it. This supersedes the original nights 3–5 schedule. `--q-half-only` stops before launching for that reason. G=5 belongs to the separate completed grid. Any larger homogeneous experiment needs a separately specified domain; do not change the judge to obtain a better result.
- The generator suppresses duplicates within one run. The judge deliberately resets its random seed each run: do not run an unchanged generator 40 times. Change the proposed family between experiments and record the hypothesis; otherwise stop the loop.
- Write a short note of what you learned after every 5 experiments in `notes.md` (append only): which families moved the minimum, which did nothing.
- Stop after 40 experiments or when told. Do not commit.

## What the humans will do with it

Before Tuesday 20 October, re-solve the top three instances of each mission through the independent path, draw their trees and explain the mechanism. Search outputs remain finite-instance diagnostics. The previously certified 10/9 family has a separate symbolic argument; a new search record does not inherit that proof.

Preparation status, 14 October: short binario and n≤6 precio controls passed. The attempt to replay the n=7 record through this fixed judge hit the short pilot's 60 s / 1536 MiB limits; it is LIMITED, not an `ok` TSV row. Its value remains independently checked in the grid. The production wrapper permits up to 360 s, but completing that seed under the production judge budget is still pending. No overnight loop has started.
