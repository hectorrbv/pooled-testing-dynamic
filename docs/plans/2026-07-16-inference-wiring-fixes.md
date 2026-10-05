# Inference Wiring Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 13 verified "wiring bugs" (inferior inference/scoring used where exact machinery exists), make the n=50 fleet demo honest and reproducible, and re-measure the two contaminated headline results (lookahead law, decomposition attribution).

**Architecture:** Three phases. Phase 1 hardens the flagship demo (closed-form U_PI, backend visibility, exact belief carrying, sample-based joint scoring). Phase 2 ports the canonical exact-branch-weight pattern (`greedy._branch_pmf`) into the four modules that never received it (hybrid, semi_utility, state_reward, sprint3/overnight). Phase 3 re-measures and corrects the documented science. Every task is TDD: failing test first, minimal fix, suite green, commit.

**Tech Stack:** Python 3 (stdlib only), pytest (installed), repo modules under `augmented/`. All tests go in a new `augmented/tests_wiring_fixes.py` (picked up by `pytest.ini`'s `tests*.py` pattern) unless stated otherwise.

**Verified evidence backing this plan:** workflow `wf_a80894fb-a46` (13 confirmed findings, adversarially verified with reproductions) and workflow `wf_a06f8205-5ec` (variant decomposition: exact scoring subsumes all three greedy weaknesses, V1==V3 on 177/177 instances; joint-frequency estimator dominates product at S≥200).

**The canonical pattern** (already correct, in `augmented/greedy.py:36-42`) that Phase 2 ports everywhere:

```python
def _branch_pmf(prior, history, current_p, pool, pool_idx, n):
    """Outcome distribution P(r | history) used to weight the EU recursion's
    branches: exact (over consistent profiles) when n is small enough, else the
    Poisson-Binomial of the posterior marginals."""
    if n <= EXACT_PMF_MAX_N:
        return exact_pool_pmf(prior, history, pool, n)
    return _poisson_binomial_pmf([current_p[i] for i in pool_idx])
```

Rule of thumb throughout: **the policy may be cheap (that is the object of study), but every *reported number* must be exact or an unbiased estimate with a standard error.**

---

## Task 0: Branch and baseline

**Files:** none modified.

**Step 1:** Confirm clean starting point. The repo currently has untracked files (`augmented/group_solvers.py`, `augmented/paper/conceptos_fundamentales.md`, `augmented/paper/lema_swaps_camino_alternante.md`, `classical/figures/...`, `pytest.ini`). Leave them untracked; do NOT commit them as part of this plan (they belong to other workstreams).

**Step 2:** Create the branch.

```bash
cd /Users/mac/Desktop/ASE/group-count-dynamic
git checkout -b fix/inference-wiring
```

**Step 3:** Record the pre-fix baseline for later comparison:

```bash
PYTHONPATH=. python3 augmented/demo_fleet_certification.py > /tmp/demo_before.txt 2>&1
tail -5 /tmp/demo_before.txt
```

Expected: certificate line prints ~71% (heuristic fallback active — that is finding #11; the published 78% needed the Mosek license). Keep the file; Task 5 compares against it.

**Step 4:** Verify the suite is green before touching anything:

```bash
PYTHONPATH=. python3 -m pytest augmented/ -q 2>&1 | tail -3
```

Expected: all passing except the known `gymnasium`/`graphviz` import skips.

---

## Phase 1 — Demo integrity (highest visibility, lowest risk)

### Task 1: Closed-form U_PI in the saturated regime (cap ≥ n)

Finding: at B·G ≥ n, `_pi_welfare` sums ALL clean utilities for every Z, so
U_PI = Σ u_i·(1−p_i) exactly, by linearity. The demo burns 200k MC samples
(~2 s) estimating this constant and prints it with no error bar as an
"absolute" bound.

**Files:**
- Modify: `augmented/certificates.py` (functions `u_pi_exact`, `u_pi_mc`)
- Test: `augmented/tests_wiring_fixes.py` (create)

**Step 1: Write the failing tests**

```python
"""Regression tests for the 2026-07 inference wiring fixes."""
import math
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_u_pi_saturated_closed_form_matches_enumeration():
    # cap = B*G = 6 >= n = 5: U_PI must equal sum u_i (1-p_i) exactly
    from augmented.certificates import u_pi_exact
    rng = random.Random(42)
    p = [rng.uniform(0.05, 0.6) for _ in range(5)]
    u = [rng.uniform(1.0, 5.0) for _ in range(5)]
    closed = sum(ui * (1.0 - pi) for pi, ui in zip(p, u))
    assert abs(u_pi_exact(p, u, B=2, G=3) - closed) < 1e-12


def test_u_pi_mc_saturated_is_exact_and_instant():
    from augmented.certificates import u_pi_mc, u_pi_exact
    rng = random.Random(43)
    p = [rng.uniform(0.05, 0.6) for _ in range(30)]
    u = [rng.uniform(1.0, 5.0) for _ in range(30)]
    closed = sum(ui * (1.0 - pi) for pi, ui in zip(p, u))
    # must return the closed form, not an MC estimate (no seed sensitivity)
    assert u_pi_mc(p, u, B=6, G=5, num_draws=10, seed=0) == closed
    assert u_pi_mc(p, u, B=6, G=5, num_draws=10, seed=99) == closed


def test_u_pi_unsaturated_unchanged():
    # cap < n: behavior identical to before the guard
    from augmented.certificates import u_pi_exact
    val = u_pi_exact([0.5, 0.5], [3.0, 1.0], B=1, G=1)
    assert abs(val - 1.75) < 1e-12  # hand-computed case from tests_certificates
```

**Step 2: Run to verify they fail**

```bash
PYTHONPATH=. python3 -m pytest augmented/tests_wiring_fixes.py -q -k u_pi
```

Expected: `test_u_pi_mc_saturated_is_exact_and_instant` FAILS (MC estimate ≠ closed form; seed-sensitive). The first test may pass by luck of enumeration — that is fine, it pins the invariant.

**Step 3: Implement.** At the top of BOTH `u_pi_exact` and `u_pi_mc` in `augmented/certificates.py`, right after computing `n = len(p)`:

```python
    if B * G >= n:
        # Regimen saturado: _pi_welfare(z) = suma de TODAS las utilidades
        # limpias para todo z, asi que por linealidad U_PI = sum u_i (1-p_i),
        # exacta y O(n). (El MC de antes estimaba esta constante.)
        return sum(u[i] * (1.0 - p[i]) for i in range(n))
```

**Step 4: Run tests**

```bash
PYTHONPATH=. python3 -m pytest augmented/tests_wiring_fixes.py -q -k u_pi
PYTHONPATH=. python3 augmented/tests_certificates.py
```

Expected: all PASS, and the pre-existing 9/9 certificate tests still pass.

**Step 5: Commit**

```bash
git add augmented/certificates.py augmented/tests_wiring_fixes.py
git commit -m "fix(certificates): forma cerrada de U_PI en regimen saturado cap>=n"
```

### Task 2: Backend visibility in pool_solvers (no more silent fallback)

Finding: Mosek's license is expired; `mosek_best_pool` silently falls back to
`_heuristic_best_pool`, and the demo's published numbers (31 clean, 78%) are
irreproducible with no trace beyond one deduplicated RuntimeWarning.

**Files:**
- Modify: `augmented/pool_solvers.py`
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Write the failing test**

```python
def test_pool_solvers_records_backend():
    import augmented.pool_solvers as ps
    p = [0.1] * 6
    u = [1.0] * 6
    ps.mosek_best_pool(p, u, G=2, n=6, cleared_mask=0)
    # After any call, LAST_BACKEND must say what actually ran
    assert ps.LAST_BACKEND in ("mosek", "heuristic")
    # In this environment the license is expired -> heuristic
    # (if the license gets renewed this assertion should be relaxed to the
    #  membership check above; keep both lines documented)
```

**Step 2: Run to verify it fails** (`AttributeError: LAST_BACKEND`)

```bash
PYTHONPATH=. python3 -m pytest augmented/tests_wiring_fixes.py -q -k backend
```

**Step 3: Implement.** In `augmented/pool_solvers.py`:
- Add module-level `LAST_BACKEND = None` after the imports.
- In `mosek_best_pool`: set `global LAST_BACKEND; LAST_BACKEND = "mosek"` on the success path, and `LAST_BACKEND = "heuristic"` plus a `warnings.warn("mosek no disponible: usando _heuristic_best_pool (calidad inferior)", RuntimeWarning)` in the fallback branch. Same pattern in `gurobi_best_pool` with `"gurobi"`.

**Step 4: Run tests** — expected PASS.

**Step 5: Commit**

```bash
git add augmented/pool_solvers.py augmented/tests_wiring_fixes.py
git commit -m "fix(pool_solvers): registra el backend real y avisa el fallback heuristico"
```

### Task 3: Exact belief carrying in `greedy_myopic_simulate`

Finding: the n=50 pipeline carries beliefs with `bayesian_update_single_test`
(misses every cross-test deduction) while `gibbs_update`
(deduction propagation → component decomposition → exact enumeration ≤16 →
MCMC only above) sits unused.

**Files:**
- Modify: `augmented/greedy.py:87-120` (`greedy_myopic_simulate`)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Write the failing test.** The Station-1 pattern: after {2,3,4}→1 and {2,4}→0, individual 3 is cross-deducibly active; exact carrying must never test it again, sequential carrying does not know.

```python
def test_simulate_with_exact_belief_carrying_bans_deduced_active():
    from augmented.greedy import greedy_myopic_simulate
    from augmented.bayesian import bayesian_update_by_counting
    # 5 people; person 3 is the only active. Force first pools via selector.
    p = [0.3, 0.3, 0.3, 0.3, 0.3]
    u = [1.0, 1.0, 1.0, 3.0, 1.0]
    z = 0b01000  # person 3 active
    forced = [0b11100, 0b10100]  # {2,3,4} then {2,4}

    def scripted_selector(cur_p, uu, G, n, cleared):
        if forced:
            return forced.pop(0)
        # afterwards: default product scoring on the carried marginals
        from augmented.greedy import _myopic_best_pool
        return _myopic_best_pool(cur_p, uu, G, n, cleared)

    hist, cleared, val = greedy_myopic_simulate(
        p, u, B=3, G=3, z_mask=z, pool_selector=scripted_selector,
        belief_update=bayesian_update_by_counting)
    # the third pool must NOT contain person 3 (exact marginal = 1.0)
    third_pool = hist[2][0]
    assert not (third_pool >> 3 & 1), f"deduced-active tested: {hist}"


def test_simulate_default_behavior_unchanged():
    # belief_update=None must reproduce the old sequential behavior bit-for-bit
    import random
    from augmented.greedy import greedy_myopic_simulate
    rng = random.Random(7)
    p = [rng.uniform(0.05, 0.6) for _ in range(6)]
    u = [rng.uniform(1.0, 5.0) for _ in range(6)]
    for z in (0, 5, 33):
        a = greedy_myopic_simulate(p, u, 3, 3, z)
        b = greedy_myopic_simulate(p, u, 3, 3, z, belief_update=None)
        assert a == b
```

Note the exact `greedy_myopic_simulate` signature must be read first
(`augmented/greedy.py:87`) — the z-argument name above (`z_mask`) must match
the real one; adjust the test to the actual signature, do not adjust the
signature to the test.

**Step 2: Run to verify failure** (`TypeError: unexpected keyword 'belief_update'`).

**Step 3: Implement.** In `greedy_myopic_simulate`, add keyword-only parameter `belief_update=None`:

```python
        if belief_update is None:
            current_p = bayesian_update_single_test(current_p, pool, r, n)
        else:
            # belief_update(prior, full_history, n) -> exact/gibbs marginals
            current_p = belief_update(prior, history, n)
```

where `prior` is a `list(p)` captured before the loop and `history` is the tuple built so far (it already exists in the function). Keep the docstring explicit: `belief_update` receives the ORIGINAL prior and the FULL history, matching `bayesian_update_by_counting` / `gibbs_update` signatures (`gibbs_update` needs a lambda to pin its extra kwargs).

**Step 4: Run tests + full greedy suite**

```bash
PYTHONPATH=. python3 -m pytest augmented/tests_wiring_fixes.py -q -k simulate
PYTHONPATH=. python3 augmented/tests.py
```

Expected: PASS; the 79 legacy tests untouched (default path unchanged).

**Step 5: Commit**

```bash
git add augmented/greedy.py augmented/tests_wiring_fixes.py
git commit -m "feat(greedy): belief_update opcional en simulate (posteriores exactos/gibbs)"
```

### Task 4: Posterior sampling API + joint-frequency pool selector

Finding: product-of-marginals scoring has an irreducible bias floor
(RMSE 0.043; assigns 29% clean-probability to provably-dirty pools). The
joint-frequency estimator on the same samples dominates it at every S tested.
The Mosek exponential-cone selector is mathematically product-form and cannot
be upgraded — this is a NEW selector, not a patch.

**Files:**
- Modify: `augmented/bayesian.py` (new function `posterior_draws`)
- Modify: `augmented/pool_solvers.py` (new function `sample_best_pool`)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Read first.** `augmented/bayesian.py` internals to reuse:
`_connected_components` (bayesian.py:304), `_exact_component_marginals`
(bayesian.py:345), the deduction preprocessing inside `gibbs_update`
(bayesian.py:630-667), `EXACT_ACTIVE_THRESHOLD = 16` (bayesian.py:259), and the
`mcmc_stats` collector (its per-component dict includes a `"draws"` key —
verify whether those are full assignments; if yes reuse, if no extend the MCMC
loop to optionally collect thinned assignments).

**Step 2: Write the failing tests**

```python
def test_posterior_draws_match_exact_marginals():
    import random
    from augmented.bayesian import posterior_draws, bayesian_update_by_counting
    rng = random.Random(11)
    n = 8
    p = [rng.uniform(0.1, 0.5) for _ in range(n)]
    history = ((0b00000111, 1), (0b00011100, 1))  # overlapping, correlated
    exact = bayesian_update_by_counting(p, history, n)
    draws = posterior_draws(p, history, n, num_draws=20000, seed=3)
    assert len(draws) == 20000
    for i in range(n):
        freq = sum(1 for z in draws if z >> i & 1) / len(draws)
        assert abs(freq - exact[i]) < 0.02, (i, freq, exact[i])


def test_posterior_draws_joint_frequency_beats_product_on_known_case():
    # history {0,1}=1, {1,2}=1 with p=0.15: worlds are (0,1,0) w/ 0.85 and
    # (1,0,1) w/ 0.15. P(0 and 2 both clean) = 0.85; the product says 0.7225.
    from augmented.bayesian import posterior_draws
    p = [0.15, 0.15, 0.15]
    history = ((0b011, 1), (0b110, 1))
    draws = posterior_draws(p, history, 3, num_draws=20000, seed=5)
    joint = sum(1 for z in draws if (z & 0b101) == 0) / len(draws)
    assert abs(joint - 0.85) < 0.02


def test_sample_best_pool_picks_true_argmax_on_small_case():
    from augmented.bayesian import posterior_draws
    from augmented.pool_solvers import sample_best_pool
    p = [0.15, 0.15, 0.15]
    u = [1.0, 1.0, 1.0]
    history = ((0b011, 1), (0b110, 1))
    draws = posterior_draws(p, history, 3, num_draws=5000, seed=8)
    pool = sample_best_pool(draws, u, G=2, n=3, cleared_mask=0)
    # True best pool is {0,2}: P(clean)=0.85, gain 2 -> score 1.70;
    # any pool containing 1 has P(clean)<=0.15.
    assert pool == 0b101, bin(pool)
```

**Step 3: Run to verify failures** (`ImportError: posterior_draws`).

**Step 4: Implement `posterior_draws(p, history, n, num_draws, seed)`** in `augmented/bayesian.py`, reusing the `gibbs_update` pipeline stages:

1. Run the same deterministic-deduction preprocessing → forced-clean /
   forced-active sets and the residual tests.
2. Decompose residual agents into connected components.
3. Per component ≤ `EXACT_ACTIVE_THRESHOLD`: enumerate its consistent
   assignments with prior weights (same enumeration `_exact_component_marginals`
   performs) and draw `num_draws` iid samples from that categorical via
   `random.Random(seed).choices(assignments, weights, k=num_draws)`.
4. Per oversized component: run the existing MCMC and collect one thinned
   assignment per kept iteration (extend the collector if `mcmc_stats`'s
   `"draws"` are not full assignments); recycle if fewer than `num_draws`,
   with a `warnings.warn` noting sample reuse.
5. Unconstrained agents (in no residual test): iid Bernoulli(p_i) per draw.
6. Compose each draw into a full n-bit mask (forced bits fixed).

Return `list[int]` of length `num_draws`. Docstring must state: draws are iid
exact for instances whose components all fit the enumeration threshold; MCMC
components introduce autocorrelation.

**Step 5: Implement `sample_best_pool(draws, u, G, n, cleared_mask)`** in `augmented/pool_solvers.py` — greedy construction over members using the draws as bitmask columns:

```python
def sample_best_pool(draws, u, G, n, cleared_mask):
    """Seleccion de pool por frecuencia conjunta muestral.

    score(t) = (#draws con t completamente limpio / S) * sum(u_i, i en t no
    acreditado). Construccion greedy: en cada paso agrega el miembro que
    maximiza el score del pool extendido; se detiene si ningun miembro lo
    mejora. Mata el sesgo de independencia del producto de marginales
    (los draws ya cargan las correlaciones del posterior).
    """
    S = len(draws)
    if S == 0:
        return 0
    candidates = [i for i in range(n) if not (cleared_mask >> i & 1)]
    pool, alive, gain, best_score = 0, list(range(S)), 0.0, 0.0
    for _ in range(G):
        best = None
        for i in candidates:
            if pool >> i & 1:
                continue
            bit = 1 << i
            alive_i = [k for k in alive if not (draws[k] & bit)]
            g = gain + (u[i] if not (cleared_mask >> i & 1) else 0.0)
            score = (len(alive_i) / S) * g
            if score > best_score + 1e-15:
                best, best_score, best_alive, best_gain = i, score, alive_i, g
        if best is None:
            break
        pool |= (1 << best)
        alive, gain = best_alive, best_gain
    return pool
```

**Step 6: Run tests**

```bash
PYTHONPATH=. python3 -m pytest augmented/tests_wiring_fixes.py -q -k "draws or sample_best"
```

Expected: PASS (the marginal-match test at 20k draws has ±0.02 headroom; if flaky, raise draws, never the tolerance).

**Step 7: Commit**

```bash
git add augmented/bayesian.py augmented/pool_solvers.py augmented/tests_wiring_fixes.py
git commit -m "feat(bayesian,pool_solvers): posterior_draws + selector por frecuencia conjunta"
```

### Task 5: Rewire the demo (exact inference, honest certificate, scarcity regime)

**Files:**
- Modify: `augmented/demo_fleet_certification.py`
- Test: manual run + output capture (this is a demo; its correctness is pinned by Tasks 1-4's unit tests)

**Step 1: Implement.** Changes to the demo:

1. `selector`: replace the Mosek call with a history-aware selector using
   `posterior_draws(p, history, N, num_draws=1000, seed=step_seed)` +
   `sample_best_pool`. Since the `pool_selector` callback receives no history,
   wrap state in a small class (the demo already re-runs simulate per MC draw;
   give each simulation its own selector instance). Keep
   `belief_update=lambda prior, h, n: gibbs_update(prior, h, n, seed=0)` in the
   `greedy_myopic_simulate` calls.
2. Print the inference/selector configuration and `pool_solvers.LAST_BACKEND`
   in the header — no more silent environment dependence.
3. Parameters: keep the current `B=10, G=5` (saturated) run for continuity,
   and ADD a scarcity run `B=6, G=5` (cap 30 < n=50) as the headline: this is
   the regime where the certificate is meaningful (U_PI no longer collapses to
   U_max) and where the penalized bound bites (slack finding, 2026-07-07).
4. The U_PI line now uses the closed form automatically at cap ≥ n (Task 1);
   at cap < n print `u_pi_mc ± SE` (compute the SE inside the demo from the
   existing sample machinery or a second-moment accumulator).
5. `NUM_SIMS`: measure one simulate call's wall time first; if the new
   selector makes 300 sims slow, drop to 100 and report the SE (which the demo
   already prints).

**Step 2: Run and capture**

```bash
PYTHONPATH=. python3 augmented/demo_fleet_certification.py | tee /tmp/demo_after.txt
diff /tmp/demo_before.txt /tmp/demo_after.txt || true
```

Expected: header states backend + inference mode; scarcity-run certificate
computed against a non-degenerate U_PI. Record both certificates in the commit
message. **Do NOT edit the published numbers in `masterplan_una_pagina.md` /
`conceptos_fundamentales.md` yet** — that is Task 12, a user-reviewed change.

**Step 3: Commit**

```bash
git add augmented/demo_fleet_certification.py
git commit -m "feat(demo): inferencia exacta + selector conjunto + regimen de escasez, backend visible"
```

---

## Phase 2 — Port the exact-branch-weight pattern (reported numbers become true)

### Task 6: hybrid_solver branch weights + K=B invariant

Finding (HIGH): PB weights at `hybrid_solver.py:351, 436, 595`; published
appendix values biased (up to +48% on alpha=1.0; rankings flip). The invariant
`hybrid(K=B) == greedy_myopic_expected_utility` is broken; `tests_hybrid.py`'s
n=3 case passes only because that instance has no overlap.

**Files:**
- Modify: `augmented/hybrid_solver.py:351,436,595` (+ thread `prior` through `_full_greedy_tree`, `_hybrid_recurse`, `_greedy_fallback` — `history` is already a parameter at all three sites)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Failing test**

```python
def test_hybrid_kB_equals_greedy_on_overlap_heavy_instances():
    import random
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    from augmented.greedy import greedy_myopic_expected_utility
    for seed in range(10):
        rng = random.Random(400 + seed)
        n = 5 if seed % 2 else 6
        p = [rng.uniform(0.2, 0.7) for _ in range(n)]  # high p -> overlap-rich histories
        u = [rng.uniform(1.0, 5.0) for _ in range(n)]
        g = greedy_myopic_expected_utility(p, u, 3, 3)
        h = hybrid_greedy_bruteforce(p, u, 3, 3, K=3)   # match real signature
        assert abs(g - h) < 1e-9, (seed, g, h)
```

(Read `hybrid_greedy_bruteforce`'s real signature first and adapt the call — K parameter name and value semantics must match the module.)

**Step 2: Run — expect FAIL** on most seeds (documented divergence up to ~7%).

**Step 3: Implement.** At each of the three sites replace

```python
pmf = _poisson_binomial_pmf([current_p[i] for i in pool_indices])
```

with the greedy pattern:

```python
pmf = _branch_pmf(prior, history, current_p, pool, pool_indices, n)
```

importing `_branch_pmf` from `augmented.greedy` and threading the original `prior` down the recursions (one extra positional arg per function; callers all originate from the public entry point where `prior = list(p)`).

**Step 4: Run** the new test + `augmented/tests_hybrid.py` (14 legacy tests must stay green). Expected: PASS.

**Step 5: Commit** `fix(hybrid): pesos de rama exactos via _branch_pmf; invariante K=B restaurado`

### Task 7: hybrid_solver DP-phase belief

Finding (HIGH): `_dp_phase` (hybrid_solver.py:505-538) hands the exact DP an
*independent* prior built from sequential marginals — the DP then optimizes
against a belief that cross-test deductions have already falsified.

**Files:**
- Modify: `augmented/solver.py` (`solve_optimal_dapts`: optional initial profile-weight vector), `augmented/hybrid_solver.py` (`_dp_phase`)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Read first.** `solve_optimal_dapts`'s state representation (it already operates on consistent-profile sets per the recon; confirm how the root prior enters).

**Step 2: Failing test** — hybrid K=0 must equal the exact DP optimum:

```python
def test_hybrid_k0_equals_exact_dp():
    import random
    from augmented.hybrid_solver import hybrid_greedy_bruteforce
    from augmented.solver import solve_optimal_dapts
    for seed in range(5):
        rng = random.Random(500 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        opt, _ = solve_optimal_dapts(p, u, 3, 3)
        h = hybrid_greedy_bruteforce(p, u, 3, 3, K=0)
        assert abs(opt - h) < 1e-9, (seed, opt, h)
```

and a mid-K sanity: `greedy EU <= hybrid(K) <= OPT + 1e-9` for K in {1,2} on the same instances.

**Step 3: Implement.** Extend `solve_optimal_dapts` with an optional
`initial_weights` (map/list over profiles consistent with a passed-in history)
OR — simpler if the DP is history-based — an optional `history=()` parameter
that pre-filters the root profile set. Then `_dp_phase` passes the actual
greedy-phase history instead of fabricating an independent `sub_p`. Choose the
variant that matches the DP's internal representation after Step 1; keep the
default path bit-identical (legacy callers unaffected).

**Step 4: Run** new tests + `tests_hybrid.py` + `tests_solvers.py`. Expected: PASS.

**Step 5: Commit** `fix(hybrid): fase DP condicionada en la historia real, no en un prior independiente`

### Task 8: semi_utility branch weights

Finding (MEDIUM): PB weights at `semi_utility.py:235, 268`; sequential mode
inflated up to +4%; "counting" mode crashes on ~30% of instances (PB puts mass
on history-infeasible r, then `bayesian_update_by_counting` raises). No
published number is wrong (verified), but the API is a landmine.

**Files:**
- Modify: `augmented/semi_utility.py:235,268` (+ thread history through the sequential-mode recursion)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Failing tests**

```python
def test_semi_utility_matches_bruteforce_all_modes():
    import random
    from augmented.semi_utility import greedy_myopic_semi_expected_utility
    # brute force: enumerate all z, run the same policy via simulate, weight by prior
    # (write a 15-line local helper _bruteforce_semi(p,u,B,G,alpha,mode))
    for seed in range(6):
        rng = random.Random(600 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        for mode in ("sequential", "counting"):
            eu = greedy_myopic_semi_expected_utility(p, u, 3, 3, alpha=0.5,
                                                     update_method=mode)
            bf = _bruteforce_semi(p, u, 3, 3, 0.5, mode)
            assert abs(eu - bf) < 1e-9, (seed, mode, eu, bf)


def test_semi_utility_counting_mode_no_crash():
    import random
    from augmented.semi_utility import greedy_myopic_semi_expected_utility
    for seed in range(20):
        rng = random.Random(700 + seed)
        p = [rng.uniform(0.2, 0.7) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        greedy_myopic_semi_expected_utility(p, u, 3, 3, alpha=0.3,
                                            update_method="counting")  # must not raise
```

(Adapt names/kwargs to the real signature after reading `semi_utility.py`; the policy scorer `_semi_best_pool` stays untouched — it is the policy definition.)

**Step 2: Run — expect FAIL** (inflation on ~half the seeds; ValueError crashes in counting mode).

**Step 3: Implement.** Replace both pmf computations with `_branch_pmf(p_prior, history, current_p, pool, pool_idx, n)`; thread `history` through the sequential recursion (it currently carries none). The exact pmf assigns zero mass to infeasible r, which eliminates the crash class.

**Step 4: Run** new tests + `python3 augmented/tests.py`. Expected: PASS.

**Step 5: Commit** `fix(semi_utility): pesos de rama exactos; modo counting ya no revienta en historias infactibles`

### Task 9: state_reward_greedy — exact branch to n≤18, honest MC above

Finding (HIGH): the exact branch is PB-weighted without history and capped at
n≤12; above that, "expected utility" is a **4-sample** MC mean (~25% relative
SE) compared in the same tables against deterministic values — the published
"beta 33% worse than mosek" row is a noise artifact (true ≈7.06 vs 7.83).

**Files:**
- Modify: `augmented/state_reward_greedy.py` (`greedy_myopic_beta_expected_utility`, `_LARGE_N_MC_TRIALS`)
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Failing tests**

```python
def test_beta_eu_exact_matches_bruteforce_small_n():
    # same _bruteforce pattern as Task 8, with beta=0 it must ALSO equal
    # greedy_myopic_expected_utility (existing tests_state_reward_greedy
    # invariant, now to 1e-9 on overlap-heavy instances)
    ...

def test_beta_eu_large_n_reports_se_and_uses_enough_trials():
    from augmented.state_reward_greedy import greedy_myopic_beta_expected_utility
    # n=20 path: returns (mean, se) when return_se=True; se below 5% of mean
    # with the new default trials; seeded reproducibility
    ...
```

**Step 2-3:** Implement: (a) thread history + `_branch_pmf`, raising the exact frontier from 12 to `EXACT_PMF_MAX_N`; (b) replace `_LARGE_N_MC_TRIALS = 4` with parameter `num_trials=200`, add `return_se=False`, seeded RNG parameter. Keep the default *return type* backward compatible (scalar mean) so legacy callers don't break.

**Step 4: Run** new tests + `python3 augmented/tests_state_reward_greedy.py` (5 legacy tests green).

**Step 5: Commit** `fix(state_reward): rama exacta hasta n=18 y MC honesto (200 trials + SE) a escala`

### Task 10: sprint3/overnight experiment runners — unbiased columns

Finding (HIGH + MEDIUM): all `U_greedy_*` columns at n>18 in
`results/sprint3_*.csv` / `results/overnight_*.csv` are biased PB recursions
(+3-9% measured); 6 committed rows at n=19-20 are silently PB despite the
"enum" label (gates `len(p) > 20` / `n <= 20` disagree with
`EXACT_PMF_MAX_N=18`).

**Files:**
- Modify: `augmented/sprint3_experiments.py:121`, `augmented/overnight_experiments.py:81` (gates), plus their EU-measurement helpers
- Test: `augmented/tests_wiring_fixes.py` (gate unit test), regeneration deferred

**Step 1: Failing test** — pin the gate to the single source of truth:

```python
def test_experiment_gates_match_exact_pmf_frontier():
    from augmented.greedy import EXACT_PMF_MAX_N
    import augmented.sprint3_experiments as s3
    # expose the gate as a function or constant during the fix, then:
    assert s3.exact_eu_feasible(18) and not s3.exact_eu_feasible(19)
    assert EXACT_PMF_MAX_N == 18
```

**Step 2-3:** Implement: (a) for n > `EXACT_PMF_MAX_N`, replace the EU
recursion call with a seeded MC over `greedy_myopic_simulate` (the
`mc_value` pattern from the demo), emitting `*_mean` and `*_se` columns;
(b) fix both gates to compare against `EXACT_PMF_MAX_N`; (c) add a
`estimator` column ("exact" | "mc") so no future CSV row is ambiguous.
**Do not regenerate the long overnight CSVs in this task** — add a
`DEPRECATED_biased_pmf_n_gt_18` note row/README stub next to the old CSVs and
schedule regeneration as Task 13's optional long run.

**Step 4: Run** the gate test + a smoke run of one small sprint3 config.

**Step 5: Commit** `fix(experiments): columnas EU insesgadas (MC+SE) sobre n>18 y gates alineados a EXACT_PMF_MAX_N`

---

## Phase 3 — Re-measure and correct the documented science

### Task 11: Exact-wired lookahead + re-measurement of the recovery law

Finding (HIGH): the headline law 99%/40%/16% is dominantly an artifact of PB +
sequential updates inside `_lookahead_best_pool`/`_greedy_future`
(greedy.py:173-240); exact-wired recovery on identical instances ≈
100%/86%/76%.

**Files:**
- Create: `exact_lookahead_expected_utility` in `augmented/greedy.py` (or a new `augmented/lookahead_exact.py` if greedy.py grows past taste) — depth-1 lookahead on the frozenset-of-consistent-profiles representation, reusing `_exact_best_pool` and the bucket-partition pattern from `independence_gap.py:248-279`
- Create: `augmented/experiments_lookahead_exact.py` — the law table, both wirings side by side, n=6, G=4, B∈{1,2,3,4}, ≥30 seeded instances, CSV to `augmented/data/lookahead_law_rewired.csv`
- Modify (docs, AFTER numbers exist): `augmented/paper/lineas_research_francisco.md` §2, `augmented/paper/conceptos_fundamentales.md` B4 and D5
- Test: `augmented/tests_wiring_fixes.py`

**Step 1: Failing test** — at B=2, exact depth-1 lookahead IS full optimization:

```python
def test_exact_lookahead_B2_equals_optimum():
    import random
    from augmented.greedy import exact_lookahead_expected_utility
    from augmented.solver import solve_optimal_dapts
    for seed in range(5):
        rng = random.Random(800 + seed)
        p = [rng.uniform(0.1, 0.8) for _ in range(5)]
        u = [rng.uniform(1.0, 5.0) for _ in range(5)]
        opt, _ = solve_optimal_dapts(p, u, 2, 3)
        la = exact_lookahead_expected_utility(p, u, 2, 3)
        assert abs(opt - la) < 1e-9
```

**Step 2-3:** Implement the function, then the experiment script (seeded; report per-B: myopic gap, legacy-lookahead recovery, exact-lookahead recovery). The verify agent's scratchpad had a working reference implementation; re-derive it cleanly rather than copying.

**Step 4:** Run the experiment; put the two-wirings table in the CSV + stdout.

**Step 5: Docs.** Rewrite `lineas_research_francisco.md` §2 with the corrected law and an explicit erratum note ("la version 99/40/16 media la degradacion del cableado, no la miopia"); update `conceptos_fundamentales.md` B4 (and D5's cross-reference, which should now cite the slack finding as the certificate-side phenomenon, not the lookahead echo). **Present the before/after wording to the user before committing docs — this changes what gets said to Francisco.**

**Step 6: Commit** (code+CSV first, docs as a second commit after user review)
`feat(lookahead): lookahead exacto + re-medicion de la ley de recuperacion` /
`docs(paper): corrige la ley del lookahead (era artefacto de cableado)`

### Task 12: Decomposition third rung + demo-facing doc corrections

Finding (HIGH): the "costo de la independencia" column conflates two causes;
measured with the repo's own middle rung (`greedy_myopic_counting_expected_utility`),
sequential-update propagation error dominates (99.5% at n=5, 70% at n=6).
"La palanca chica y barata es el scoring exacto" points at the wrong primitive.

**Files:**
- Modify: `augmented/independence_gap.py` (or the decomposition experiment entry point found at `notebooks/build_avances_post_sesion_notebook.py:608-643`) — three-way split per instance: miopia = OPT − exact_greedy; propagacion = counting_greedy − greedy; scoring puro = exact_greedy − counting_greedy
- Modify (docs): `augmented/paper/lineas_research_francisco.md` §1
- Modify (docs): `augmented/paper/masterplan_una_pagina.md` + `conceptos_fundamentales.md` D7 (demo numbers — after Task 5's new run)
- Test: covered by Task 8/9's bruteforce-parity pattern; the experiment is seeded and asserts `greedy <= counting <= exact <= OPT + 1e-9` per instance

**Steps:** regenerate the §1 table with the third rung (same n∈{5,6,7} configs, seeded, ≥30 instances); update the "lectura operativa" to name exact marginal PROPAGATION as the cheap lever; fold in the Task 5 demo numbers and the Mosek-fallback disclosure. **User reviews all doc wording before commit** (same reason as Task 11).

**Commit:** `docs(paper): descomposicion en tres peldanos y numeros de demo reproducibles`

### Task 13: Hygiene batch (LOW findings, one commit)

**Files & one-line fixes:**
- `augmented/vw_demo.py:143-146,162` — stale "(inflated by indep. PMF approx)" comment/label: now false at n=6 (exact ≤18). Reword per finding.
- `augmented/bayesian.py:739-754` (`estimate_p_from_history`) — delete the dead Beta-smoothing params (`prior_strength`/alpha/beta computed then discarded); docstring says what it actually does. Keep `tests.py:651-667` green.
- `augmented/certificates.py` `_EXACT_POSTERIOR_MAX_N=16` + `augmented/vhat.py:13-29` — import `EXACT_PMF_MAX_N` from greedy as single source of truth (both at 18); correct vhat.py's contract note (greedy_value is exact only ≤18, PB above, cost O(C(n,≤G)) per step). Re-run `tests_certificates.py` + `tests_solvers.py` (no fixture sits at n=17-18, verified).
- Optional long run (user decides): regenerate `results/sprint3_*.csv` / `results/overnight_*.csv` with Task 10's unbiased estimators.

**Test:** existing suites + one new test pinning `certificates` and `greedy` share the frontier constant.

**Commit:** `chore: etiquetas obsoletas, parametros muertos y frontera de exactitud unificada`

### Task 14: Final verification

**Steps:**
1. `PYTHONPATH=. python3 -m pytest augmented/ -q` — full suite green.
2. Re-run the Phase-1 demo, the Task 11 law table, the Task 12 decomposition — attach outputs.
3. Run the variant-decomposition harness from `scratchpad/greedy_analysis/validate_harness.py` as an external cross-check that `greedy_myopic_expected_utility` semantics did not drift (max diff vs library must stay < 1e-9).
4. Summary table: finding → task → commit hash → number-before/number-after, for the session log and for Francisco.

---

## Deferred backlog (verified-adjacent, not in this plan)

21 lower-priority candidates were found but not verified (fan-out cap):
`group_solvers.py` duplicate-of-`pool_solvers`, `tree_extractor.py` sequential
posteriors in DP trees, `greedy_myopic_gibbs_simulate` wiring,
`masterplan_una_pagina.md` horizon-comparison wording, `vw_restrict.py`
run_trial, and the rest listed in workflow `wf_a80894fb-a46` logs. Also: the
`state_reward` verify agent stalled (its finding was confirmed via a second
lens); and renewing the Mosek license is an ops task outside the repo.
