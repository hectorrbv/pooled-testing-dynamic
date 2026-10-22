# Mission: an efficient laminar policy that stays close to the laminar optimum (loop Q1, started 2026-10-17)

You are running an autonomous experiment loop, in the style of karpathy/autoresearch, on an exact evaluation. No GPU, no training; every value is exact rational arithmetic.

## What we are looking for

Setting: n people, person i healthy with probability q_i, utility u_i when accredited healthy. A test takes a pool of at most G people and returns the exact number of infected. Policies here are *laminar*: along every path, two pools are disjoint or one contains the other. Soft clearing: a person is accredited when every profile compatible with the counts so far has them healthy. The laminar optimum of each cell is known exactly (versioned CSVs); computing it is exponential, and the research question Q1 asks for a cheap rule that gets close.

The judge evaluates your rule on a fixed battery of 125 cells: 32 homogeneous cells with q = 1/2 (n = 6–7, B ≤ 4, G = 2..5), 81 homogeneous cells with q ∈ {0.3, 0.5, 0.7} (n = 4–6), and 12 heterogeneous cells (n = 5, random q and u). Per cell it reports value / optimum. The **score** is the worst ratio over the battery; the mean ratio and the number of exact cells break ties. Higher is better.

Reference: the baseline scorer is the myopic greedy (largest exact expected immediate gain). The repository's earlier measurement found the greedy loses most at the opening move and that a one-step rollout recovers most of it (worst 18/19 on the q = 1/2 grid, exact on 26 of 32), but rollout costs one full greedy evaluation per candidate pool. The prize is a rule with greedy cost and rollout quality, or better.

## The rules (do not break them)

- You edit **only** `scorer.py` (the function `choose(state)`). Never edit `judge.py`, `q1_tools.py`, `run_bounded.py`, `run.sh`, the solvers under `augmented/`, the CSVs, or anything outside this folder.
- Interface: `choose(state)` returns a pool from `state.candidates` (an n-bit mask) or an iterable of people, or `None` to stop testing. `state` carries `n`, `G`, `tests_left`, `q`, `u` (tuples of Fractions), `support` (compatible profiles as a 2^n-bit mask), `cleared` (mask of accredited people), `past` (pool masks on this path), `candidates` (laminar-compatible pools of size ≤ G), and `inst` (precomputation with `mass`, `healthy`, `utility`, `outcomes`). Helpers in `q1_tools`: `branches(state, pool)`, `expected_gain(state, pool)`, `members(mask)`, `pool_mask(people)`. The rule must be deterministic and must not look at the optimum or at the CSVs.
- One experiment = one run of the judge from the repository root:
  `bash augmented/autoresearch_q1/run.sh "<one-line description of the idea>"`
  It prints `status`, `worst_ratio`, `mean_ratio`, `exact_cells`, `worst_cell`, `scorer_calls`, `seconds`, appends the row to `results_q1.tsv`, and writes the per-cell ratios of that run to `last_run_cells.tsv` (sorted, worst first). A wall-clock limit of 900 s applies; exit code 124 is LIMITED (your rule is too slow for the battery), not a result.
- Read the printed line. **Keep** `scorer.py` if `status=ok` and the score improves (higher `worst_ratio`; if equal, higher `mean_ratio`; if equal, more `exact_cells`); otherwise **revert** to the last kept version. Keep a copy of the best as `scorer_best.py` after every improvement. `status=fail` means a gate tripped (a non-candidate pool, a crash, or a value above the optimum); revert and do not retry the same idea. If the error says "above optimum", stop the loop and write it in `notes.md`: humans must look.
- Cost matters. Report `seconds` and `scorer_calls` in your notes. A rule that needs more than ~300 s on the battery is not the efficient rule we want, even if its ratio is good.
- Never edit `results_q1.tsv`, `last_run_cells.tsv` or `scorer_best.py` by hand except to copy the kept scorer.
- After every 5 experiments append a short note to `notes.md`: which ideas moved the worst cell, which cells remain hard (use `last_run_cells.tsv`), the current best rule in one paragraph.
- Stop after 30 experiments or when told. Do not commit. Do not call any network service.

## Ideas worth trying (in no particular order)

- Look-ahead only where it is cheap: a one-step rollout restricted to the top-k pools by immediate gain, or only at the opening move (where the greedy loses most).
- Information per test: expected number of accredited people plus a bonus for expected entropy reduction (or expected number of "bits fixed"), weighted by tests left.
- Index rules: expected gain per person tested, or expected gain divided by the probability the test is wasted (count equal to the pool size).
- Prefer pools that keep future options open: a pool nested in or disjoint from the ones already used leaves the most laminar candidates.
- Budget-aware pool sizes: with few tests left take smaller pools; with many, larger ones.
- Tie-breaking matters on heterogeneous cells: break ties toward higher utility or higher q.

## What the humans will do with it

The best rule is re-evaluated through an independent path (`augmented/laminar_benchmarks.py`, Monte Carlo at n = 50 against the rollout) before anything is said about it; a battery score is a finite-grid diagnostic, not a guarantee.
