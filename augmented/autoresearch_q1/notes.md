# Notes, loop Q1

Round 1 (this session). Found only the baseline row in results_q1.tsv and no notes.md/scorer_best.py
at start, so the baseline greedy was the starting best. Experiments 2-7 below (cells = 125).

| idea | worst | mean | exact | s |
|---|---|---|---|---|
| baseline greedy | 0.7204 | 0.8731 | 31 | 2.2 |
| 2-step lookahead, top-3 greedy pools | 0.7204 | 0.9383 | 65 | 3.6 |
| 2-step lookahead, all pools | 0.9231 | 0.9929 | 97 | 13.8 |
| 2-step, top-8 pools | 0.8596 | 0.9887 | 92 | 5.4 |
| 3-step exhaustive, greedy-style leaf (KEPT) | 0.9495 | 0.9981 | 115 | 202 |
| 3-step only at opening, 2-step after | 0.9495 | 0.9974 | 109 | 183 |
| 3-step, inner levels pruned to top-6 by gain | 0.9495 | 0.9980 | 114 | 64 |

Lessons: lookahead over ALL pools matters (pruning the outer choice to top-k hurts worst cell badly);
pruning only the inner levels costs almost nothing and is ~3x cheaper. Hard cells: q0.3 with n=5, B=4 (0.9495),
then q0.5 n=6 B=4 (0.975). Depth-3 near the 300 s budget.

Later in round 1 (experiments 8-10; 10 new experiments in total, round done):

| idea | worst | mean | exact | s |
|---|---|---|---|---|
| depth-4, inner levels top-4, greedy leaf (KEPT, scorer_best.py) | 0.9829 | 0.99954 | 120 | 131 |
| depth-5, inner top-3 | 0.9829 | 0.99952 | 119 | 88 |
| depth-4, inner top-5 | 0.9829 | 0.99954 | 120 | 184 |
| depth-4, inner top-3 | 0.9829 | 0.99952 | 119 | 88 |

Current best rule: exhaustive choice over all candidate pools at the root; each is scored by its exact
expected immediate gain plus the expectation over outcomes of a 3-more-step lookahead in which each
inner node tries only the 4 pools with highest immediate gain (last level: greedy gain), limited by
tests_left. 131 s on the battery, worst cell q0.3:6,4,3 at 0.9829. Wider/deeper inner search did not move
the worst cell; top-3 inner is within 2e-6 mean at 88 s (cheaper alternative, not strictly better so not kept).
Untried: tie-breaking toward larger pools, reusing a cache of child values across candidates, hetero-specific tweaks.
Note: results_q1.tsv rows so far were all produced in this round; no "above optimum" errors seen.

Round 2, experiments 1-5 (start: scorer_best.py, worst 0.9829 / mean 0.99954 / exact 120 / 131 s). None kept.

| idea | worst | mean | exact | s |
|---|---|---|---|---|
| top-8 at first inner level, top-4 below | 0.9829 | 0.99954 | 120 | 242 |
| inner ties broken toward larger pools | 0.9829 | 0.99934 | 118 | 133 |
| inner top-4 + max-entropy remaining pool | 0.9495 | 0.99946 | 123 | 212 |
| inner top-4 + 2 max-entropy pools | 0.9495 | 0.99946 | 123 | 210 |
| entropy extra only at first inner level | 0.9495 | 0.99916 | 120 | 174 |
| inner top-4 + best gain-per-person pool | 0.9829 | 0.99954 | 120 | 198 |

Lessons: entropy-based extra candidates fix q0.3:6,4,3 and raise exact cells to 123, but break q0.3:5,4,3 (0.9495),
so the worst-cell score drops. Widening or re-ranking by size/gain-per-person changes nothing at the worst cell.
Remaining hard: q0.3:6,4,3 (0.9829), q0.5:6,4,4 / q05:6,4,4 / q05:6,4,5 (0.9877).

Round 2, experiments 6-10 (10 of 10 used; stop). No improvement; scorer_best.py unchanged (still the round-1 depth-4, inner top-4 rule).

| idea | worst | mean | exact | s |
|---|---|---|---|---|
| top-8 at depth2, top-4 at depth3 | 0.9829 | 0.99941 | 119 | 206 |
| top-4 at depth3, top-2 at depth2 | 0.9829 | 0.99954 | 120 | 94 |
| top-6 at depth3, top-2 at depth2 | 0.9829 | 0.99954 | 120 | 133 |
| root ties toward larger pools (>=) | 0.9495 | 0.99912 | 118 | 90 |
| root ties by higher immediate gain (top-4/top-2) | 0.9829 | 0.99954 | 120 | 94 |

Lessons: the score is on a plateau (worst 0.9829, mean 0.99954, 120 exact). Widening is non-monotone and sometimes hurts.
Best cost finding: top-4 at the first inner level and top-2 below it gives the identical score at 94 s instead of 131 s
(not kept under the strict-improvement rule, but it is the cheaper equal rule). Root tie-breaking to larger pools hurts.
Entropy-added candidates reach 123 exact cells but lose the worst cell (see experiments above).
Untried: caching child values across root candidates, hetero-specific tweaks, pruning key other than immediate gain
(e.g. 1-step rollout value) for the inner levels. No "above optimum" errors seen.

Round 3, experiments 1-10 (10 of 10 used; stop). KEPT: scorer_best.py now worst 0.9972 / mean 0.999978 / exact 124 / 150 s.

| idea | worst | mean | exact | s |
|---|---|---|---|---|
| memo of inner values keyed by id(inst) (BUG: id reuse across cells, stale values) | 0.9772 | 0.99936 | 119 | 84 |
| memo keyed by inst object (same rule as round-2 top-4/top-2, cheaper) | 0.9829 | 0.99954 | 120 | 83 |
| memo, depth-5, top-4/top-2 | 0.9829 | 0.99954 | 120 | 86 |
| memo, top-4 first inner then pure greedy | 0.9829 | 0.99954 | 120 | 67 |
| memo, top-6 then top-3 | 0.9829 | 0.99954 | 120 | 130 |
| memo, top-3 then greedy | 0.9829 | 0.99952 | 119 | 56 |
| memo, top-2 then greedy | 0.9829 | 0.99939 | 118 | 45 |
| first inner: top-4 + best-gain pool of each other pool size, greedy below (KEPT) | 0.9877 | 0.99962 | 119 | 89 |
| size-diverse extras at every inner level >1, top-4 first inner (KEPT, scorer_best.py) | 0.9972 | 0.999978 | 124 | 150 |
| same, top-1 base + best pool per size (equal score, cheaper; not kept: no strict gain) | 0.9972 | 0.999978 | 124 | 107 |

Lessons: widening by gain rank is a dead plateau (worst 0.9829 whatever the width); diversifying by POOL SIZE
(best-gain pool of each size at inner levels) is what moved the worst cell, from 0.9829 to 0.9972, and fixed
q0.3:6,4,3 and the q0.5/q05 6,4,4-5 cells. Remaining worst: q0.7:6,3,4 (0.99719), one non-exact cell besides it.
Cheaper equal rule: top-1 by gain plus best pool per size, 107 s (same ratios). The memo must key on the Instance
object, not id(): id reuse across cells silently corrupted values (exp 1; exp 2 repaired it).
Best rule: exhaustive root over all candidates; inner nodes (depth>1) try the best-gain pool of each size plus the
top-4 (first inner level) by gain, greedy leaf, memoised. No "above optimum" errors. Untried: size-diverse at the leaf-adjacent
level only with top-1 vs more sizes at root-level pruning, a size-diverse rule with n=50 transfer check by humans.
