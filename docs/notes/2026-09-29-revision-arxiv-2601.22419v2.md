# Review of arXiv:2601.22419v2 — "Dynamic Welfare-Maximizing Pooled Testing"

Lock (KCL), Lopez (Harvard), Marmolejo-Cossío (Boston College), Tello Ayala (Harvard), Parkes (Harvard). v2, 25 Sep 2026. Link sent by A on 2026-09-29; this is "the paper Francisco was telling us about" (acta 22-sep: "paper de Nick, secciones 4 y 5").

**Status: IA extraction (2026-09-29, automated fetch of the HTML version). Pending A's validation against the PDF before any citation enters the paper (master plan §21).** Quotes below are as returned by the fetch; exact forms of T_x, φ_G and the hypotheses of 3.3(iii)–(iv) must be read from the primary source.

## What the paper is

The binary (non-augmented) model, exactly our classical comparator: a pool tests negative iff all members are healthy; an agent is cleared when it first appears in a negative pool; heterogeneous u_i, q_i; budget B; pool cap G; welfare = expected utility of cleared agents. Benchmarks: OPT_dyn (dynamic), OPT* (static overlapping), OPT (static non-overlapping). Main result: OPT_dyn ≤ 2·OPT* for every instance. **No mention of counts, quantitative group testing, augmented tests, or laminar families.**

## Claim table

| # | Their result | One-line statement | Our object | Match | Action |
|---|---|---|---|---|---|
| 1 | Lemma 3.1, posterior domination | Pr(all of A healthy \| H) ≤ Π_{i∈A} q_i for A disjoint from the cleared set N(H). Proof: condition on negatives (pins N to healthy), residual product measure, positives = one decreasing event E, Harris–FKG increasing–decreasing form | Part 3 of `augmented/paper/cota_clasica_construccion.md` | identical, same proof route | A proves Part 3 first, then compares; cite |
| 2 | Λ(J,G) := max_{\|A\|≤G} Σ_{i∈A} Π_{j∈A∖{i}} q_j | homogeneous: Λ = max_{m≤G} m·q^{m−1} | Part 2 (m·q^m ≤ q ⟺ m·q^{m−1} ≤ 1) | identical | Part 2 = "Λ ≤ 1 iff q ≤ ½ (G ≥ 2)" |
| 3 | Thm 3.2, master bounds | V(π) ≤ T_{BΛ} and V(π) ≤ B·u*. Proof: ρ_i := Pr(i cleared)/q_i ∈ [0,1]; each test adds ≤ Λ to Σ_i ρ_i (via Lemma 3.1); Σρ_i ≤ BΛ; welfare = Σ m_i ρ_i ≤ fractional knapsack T_{BΛ} | Part 4 + the heterogeneous extension flagged "out of scope" | the ρ_i trick is the companion's "normalized potential 7.3" and removes the repeated-leader obstacle | A reads and rewrites the ρ_i argument in own words (30 min) |
| 4 | Thm 3.3(i) | Λ ≤ 1 ⟹ OPT_dyn = OPT* = OPT = Σ_{b≤min(B,n)} m_(b), m_i = u_i q_i; in particular if every q_i ≤ ½ | `lem:classical` of `main.tex`; companion Thm 7.1 | identical, and stronger (heterogeneous, dynamic) | **cite as the classical side of the separation**; A-M23 entry 7.1 closes by identification; §34 question (27) answered: it is in §3, not §4–5 |
| 5 | Thm 3.3(ii)–(iv) | B=1 or B≥n ⟹ gap 1; near-optimal disjoint pools ⟹ gap ≤ 1/(1−γ); q_i ≥ θ and n ≥ BG ⟹ gap ≤ min{2, φ_G(θ)}, φ_G(θ) = min_{1≤s≤G} G/(sθ^{s−1}) | — | new to us | read later (high-health regime) |
| 6 | Prop 3.6 | path-disjoint dynamic policies (pools disjoint along every path) are worth exactly OPT; any gain over static non-overlapping requires re-pooling an agent from an earlier positive test | augmented cover-only (no refinement) policies: only count-0 pools clear, so same value as binary static non-overlapping; all of our separation comes from refining atoms with positive counts | [DERIVACIÓN — to verify] | candidate one-liner for the intro; question (28) for Francisco |
| 7 | Thm 5.4 | OPT_dyn ≤ 2·OPT*, all instances, budgets, caps | the "factor 2" Francisco cited on 1-sep and 22-sep | now published | the DERIVACIÓN CONDICIONAL remark in `main.tex` §3 becomes a citation |
| 8 | Thm 4.2, Cor 5.6 | Exact-Joint Greedy ≥ OPT*/(e+1); both greedies ≥ OPT_dyn/[2(e+1)] | their Q1 in the binary model | — | contrast for our Q1: constant-factor greedy guarantee exists in binary; open with counts |
| 9 | Example 2.10 | 3 agents, B=2, adaptivity gap 1.154; (u,q) = (0.129,0.556), (0.175,1.0), (0.569,0.12) | our brecha minimal game | different phenomenon (adaptivity vs convention) | — |

## Anchor check (homogeneous q = 0.05, G = 16, B = 7)

Λ = max_{m≤16} m·0.05^{m−1} = 1 (attained at m = 1; m = 2 gives 0.1). T_7 = 7 × 0.05 = 0.35. The anchor lies in the Λ ≤ 1 regime of Thm 3.3(i), so 0.35 is the exact classical dynamic optimum, not a bound. Same at the frontier (0.30) and along the family (q = c/G ≤ ½ whenever G ≥ 2c).

## Consequences for the plan

- §32 row 2026-09-22 / A's decision of 28-sep ("prove at home so the paper does not depend on an unpublished result"): the dependency is now on a published arXiv paper → **cite**. A's home proof stays as the mastery exercise and as a "for completeness" remark in the homogeneous case of `seccion_separacion.tex`.
- §34 (27): answered by the paper. §34 (19), reparto: the binary-model results live in this paper; ours is the augmented laminar model; the companion's Thm 7.1 is their Thm 3.3(i).
- New questions for 29-sep: **(28)** Prop 3.6 + Thm 5.4 against counts: "re-pooling after a positive is worth at most a factor 2 in the binary model and an unbounded factor with counts" — acceptable as the one-line statement of our separation? **(29)** reparto confirmed as above?
