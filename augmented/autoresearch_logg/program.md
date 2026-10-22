# Mission: how many healthy people can B count-tests of size ≤ G accredit? (loop log-G, started 2026-10-17)

You are running an autonomous experiment loop, in the style of karpathy/autoresearch, on an exact combinatorial evaluation. No GPU, no training; the metric is exact arithmetic over all 2^n infection profiles.

## What we are looking for

Setting: n people, each healthy with probability 1/2 independently, utility 1 per accredited healthy person. A test takes a pool of at most G people and returns the exact number of infected in it. After B tests, a person is accredited healthy when every profile compatible with all observed counts has that person healthy (soft clearing). Pools may overlap and cross freely: this is the *general* (non-laminar) class.

Francisco (our advisor) believes the best general policy accredits about c·B·log₂G people, while any laminar policy (pools pairwise disjoint or nested along each path) stays below C·B. We already know the upper bound: no policy accredits more than B·log₂(G+1). What nobody has measured is how high explicit designs actually get at finite G. That curve, with the designs that attain it, is the result.

Targets (fixed in the judge): n = 16 with (B, G) ∈ {(4,2), (4,4), (4,8), (4,16), (8,8), (8,16)}. The judge reports, per target, the expected number accredited divided by B (`per_test_*`) and divided by B·log₂(G+1) (`bits_*`). The **score** is the sum of the six `per_test_*` values; higher is better. Improving any target counts. Keep a separate eye on the shape: we want `per_test` to grow with G.

Reference points: individuals give 0.5 per test at every G. The baseline design (blocks of four solved by the detector ABC, ABD, ACD, leftover tests on individuals) gives 2/3 per test inside a block. The laminar optimum for n = 7, B = 4 was 0.64–0.67 per test for G = 2..5 (exact grid, `results/precio_q05_G/`).

## The rules (do not break them)

- You edit **only** `design.py` (the function `policy(n, B, G)`). Never edit `judge.py`, `run_bounded.py`, `run.sh`, the solvers under `augmented/`, or anything outside this folder. Do not write files elsewhere.
- Interface: `policy(n, B, G)` returns `next_pool(history)`. `history` is a tuple of `(pool, count)` pairs along the current path. Return the next pool as an iterable of 1..G distinct people in `range(n)`, or `None` to stop. The judge calls `next_pool` once per distinct history, so adaptive designs cost nothing extra. The policy must be deterministic (no randomness, no clocks) and pure Python with the standard library; numpy is available but not needed.
- One experiment = one run of the judge from the repository root:
  `bash augmented/autoresearch_logg/run.sh "<one-line description of the idea>"`
  It prints one line with `status`, `score` and the per-target values, and appends the same row to `results_logg.tsv`. A wall-clock limit of 600 s and 1 GiB apply; exit code 124 is LIMITED (too slow), not a result.
- Read the printed line. **Keep** `design.py` if `status=ok` and `score` improves on the best kept score; otherwise **revert** it to the last kept version. Keep a copy of the best as `design_best.py` after every improvement. `status=fail` means a gate tripped (invalid pool, too many tests, non-deterministic policy, or a value above the information bound); revert and do not retry the same idea. If the error says "information bound violated", stop the loop and write it in `notes.md`: that would be a bug or a theorem, and humans must look.
- Never edit `results_logg.tsv` or `design_best.py` by hand except to copy the kept design.
- After every 5 experiments append a short note to `notes.md`: which ideas moved which target, which did nothing, the current per-target best and the design that attains it (pools listed explicitly for the static ones, the rule for the adaptive ones).
- Stop after 30 experiments or when told. Do not commit. Do not call any network service.

## Ideas worth trying (in no particular order)

- Detecting matrices / coin-weighing designs: a 0-1 matrix whose columns have distinct column-sum signatures identifies everyone in the block. Known constructions (Lindström 1964; Cantor–Mills 1966) identify m people with about 2m/log₂m tests when pools are unrestricted; inside a block of size G the pool cap does not bind.
- Partial identification: with B tests fewer than a full detecting matrix needs, choose rows that accredit many healthy people in expectation even when the profile is not fully resolved. Soft clearing credits a person as soon as every compatible profile has them healthy.
- Adaptive refinement: a first wide pool, then pools chosen from the count (a count of 0 clears the whole pool at once; a count equal to the pool size reveals all infected and frees nothing; intermediate counts call for crossing follow-ups).
- Mixing block sizes: for (B, G) = (4, 16) one block of 16 with four crossing tests may beat four blocks of four.
- Per-target designs: `policy` receives (n, B, G) and may branch on them.
- Compare against the information bound: `bits_*` near 1 means little room is left at that target.

## What the humans will do with it

The best design per target is re-evaluated by an independent path (the exact Bellman solver where n ≤ 7, a separate enumerator otherwise), its pools are drawn, and the per-test curve against G goes into the discussion with Francisco as a finite measurement, not as an asymptotic claim.
