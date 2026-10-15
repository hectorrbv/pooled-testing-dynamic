# Week plan — Wed 14 Oct → Tue 20 Oct, 19:00 Prague (session with Francisco)

**Self-contained for Héctor.** Supersedes `2026-10-12-plan-hasta-sesion.md` (written for the 13-Oct session, which was cancelled). Sources: master plan §1 (the three parts of 6 Oct) and §34-bis; minutes of 6 Oct (`2026-10-06-sesion-francisco.md`); results note (`2026-10-12-encargos-6-oct.md`); Francisco's WhatsApp of 13 Oct. **Budget:** A ≈ 2 h/day, Héctor ≈ 2–3 h/day in daytime. The search loop runs **only at night** on Héctor's machine: 5 min to start it in the evening, 5 min to commit its results in the morning. **Rule of the week:** this week has to be big in written, verified advances; every piece below is either a written result in its author's words or a table with a second-way check. Nothing aspirational is said on Tuesday.

## 0. Direction (master plan §1, the three parts of 6 Oct) and what changed

Execution update, 15 Oct: Héctor requested the night search on Hermeneuta. It started at 15:20 Prague in an isolated VPS directory, with a separate n=7 feasibility pilot followed by up to 40 bounded experiments. The judge and solvers are unchanged; no completed night or new record is claimed yet. See [deployment, resource limits and retrieval instructions](2026-10-15-busqueda-hermeneuta.md). This supersedes only the "not started" status below; the independent review of future records remains pending.

Research direction reference, saved at Héctor's request on 15 Oct: [Francisco's message and the intended scale of results](2026-10-15-direccion-y-alcance-francisco.md). It preserves the original wording, separates the proposed log G result from an established proof, and records how small examples and grids should contribute toward structural bounds and separating families.

Update from Héctor's workspace, 14 Oct: [the results note, §4](2026-10-12-encargos-6-oct.md#4-actualización-de-héctor-14-oct) records the completed computational second-way check (81 grid cells, 20 binary rows, 20 B=1 controls; no discrepancies) and the family with laminar/binary approaching 0.9. Use that as the binary search baseline instead of 0.978. The expanded Q2 grid and Q1 columns are now complete: all 32 cells, five independent laminar spot checks and full-profile policy replays; see §5 of that note. Human review/sign-off and A's written proof remain pending. The notebook is aligned; the seeded night loop is prepared and smoke-tested, not started.

Francisco's three parts: **(a)** laminar and augmented against the best binary policy; **(b)** Q2, what is lost by forcing laminar, worst case conjectured at q = ½; **(c)** Q1, efficient laminar policies against the best laminar policy — the backbone of the master plan (§0: greedy with planning, rollout, acid test). Last week's plan covered (a) and (b) only; this one covers all three.

**New on 13 Oct.** Francisco wrote: with everyone worth 1 and healthy with probability ½, he believes the gap between augmented overlapping and laminar is about log G — laminar bounded by a constant times B, detection matrices reach B·log G. Our side, derived by the AI support and validated by A this week: with pool cap G the history of B tests has at most (G+1)^B values, so at most B·log₂(G+1) bits; a cleared person is one fixed bit; at q = ½, u = 1 this gives E[cleared] ≤ B·log₂(G+1) for **any** policy, laminar singletons already give B/2, so the price of laminarity is at most 2·log₂(G+1). That is the "at most log G" half. His two halves (laminar ≤ cB; a construction reaching B·log G) stay his and are not our work this week.

## 1. What we bring on Tuesday — five pieces, three parts covered

| # | Part | Piece | Owner | Status today | Label on Tuesday |
|---|---|---|---|---|---|
| 1 | (b) | **Proposition "price of laminarity ≤ 2·log₂(G+1) at q = ½, u = 1"**, one page, A's proof (counting lemma), with the q = ½ table as numeric check; sent Sun | A (proof), Héctor (table), AI (typeset) | argument derived, not yet in A's words | DERIVED, A's wording; table "two implementations agree" if the check passes |
| 2 | (b) | **q = ½ table as G grows:** u = 1, G = 2…5, n = 6–7, B ≤ 4, laminar vs unrestricted, every ratio checked against 1 / (2·log₂(G+1)) | Héctor | Complete, 32/32; min ratio 41/52 at n=7, B=4, G=3; all 32 information checks pass | Exact finite grid; worst witness independently re-solved and replayed |
| 3 | (c) | **First Q1 measurement:** same grid, soft clearing, greedy/rollout and fixed-hierarchy optimum vs pathwise laminar optimum; worst ratio per (G, B) | Héctor | Complete; five independent laminar checks and ten full-profile policy replays; rollout min 18/19 and exact in 26/32 | Checked finite-grid comparison; specify hierarchy and tie-break |
| 4 | (a) | **Binary beats laminar:** original n=3 witness plus notebook witness (2.432 vs 2.366) and certified four-person family approaching laminar/binary=0.9 | Héctor validates, A explains | 20 binary rows and B=1 controls pass; family has symbolic certificate | Computationally verified; human reading/sign-off still pending |
| 5 | (a) | **A's reply to the log G message** (Thu) and **A's own version of T_BG ≤ G·T_B ≤ G·Binary** (Mon) | A | not written | A's wording; closes companion Cor 8.3 by identification |

Extras only if the base above is closed (§1 over-delivery rule, verified artefacts never claims): the "individuals never optimal" note with steps 3–4; two sentences on Section 3 of his paper.

Not this week: Francisco's lower half (laminar ≤ cB) or any B·log G construction; proving q = ½ is the worst case; any constant for Q2; the §5 memo; a greedy *with planning* (the Q1 measurement is of the existing greedy and rollout; designing the planned scorer stays on the master-plan chain).

## 2. Start right now (Wed 14)

**A, today (2 h).** Tutoring on Francisco's message, steps 1–2 (what a singleton pays at q = ½: B tests → B/2; what a pool of 2 pays: 2·¼ = 0.5, the same). Then WhatsApp Héctor the link to this plan.

**Héctor, today (2–3 h).**
1. `git pull` on `augmented/correctness-and-paper` (commit `189d71c` or later).
2. **Sign the note** (30 min): read `augmented/paper/proposicion_individuales_construccion.md`; run `pytest augmented/tests_brecha_convencion.py` (934 pass) and `PYTHONPATH=. python augmented/barrido_proposicion_individuales.py <seed> 120 6 4 4` (expects `fails 0`). Object or sign in the results note.
3. **Second-way check of `augmented/dp_perfiles_binario_conteo.py`** (2 h, nobody but the AI has run it): (a) `BellmanPerfiles(laminar=False)` vs the `unrestricted` column of the two grid CSVs in `results/encargos_2026-10-06/` (81 cells); (b) your own enumerator on 20 rows of `binario_vs_laminar_seed*.csv`, binary channel; (c) `static_binary_value` where B = 1. Any mismatch stops everything else and becomes the result of the week. Write the verdict as a row in `2026-10-12-encargos-6-oct.md`.
4. **Tonight:** start night 1 of the loop (§4).

## 3. Day by day

| Day | A (≈ 2 h) | Héctor, daytime (≈ 2–3 h) | Héctor, night |
|---|---|---|---|
| **Wed 14** | §2 | §2 | **Night 1:** loop as in `program.md`, both missions |
| **Thu 15** | Steps 3–4: the (G+1)-answers counting fact; "a cleared person = one fixed bit". Write the reply to Francisco yourself; **send by 20:00** with the n = 3 witness and its label (from Héctor's Wed check) | Morning: commit night 1. **Grid script:** extend `augmented/malla_precio_laminar_q.py` to take q, G list and n list from the command line and to add three columns — `greedy_laminar`, `rollout_laminar` (from `augmented/laminar_benchmarks.py`, same p, u convention: p = infection probability = 1 − q) and `exact_laminar` (already there). Time one cell at n = 7, G = 5, B = 4 before launching anything | **Night 2:** same as night 1 |
| **Fri 16** | Counting lemma, fact 1 in your words: ≤ (G+1)^B histories. Draw the three trees of the n = 3 witness from memory | Morning: commit night 2. **Run the q = ½ grid** (piece 2 + piece 3 columns), u = 1, G = 2…5, n = 6–7, B ≤ 4; if G = 5 at n = 7 is slow, drop to n = 6 for G = 5 and say so. Save `results/precio_q05_G/*.csv`. Edit `program.md` for nights 3–5 (§4) | **Night 3:** `precio`, q = ½, u = 1 only |
| **Sat 17** | Fact 2: at q = ½ each person is one full bit of uncertainty; clearing someone fixes that bit | **Re-solve the top 3 of each mission** from nights 1–2 with the independent path (`BellmanPerfiles` + your enumerator); trees of the best witness; one paragraph on the mechanism → row in the results note. Spot-check 5 grid cells of piece 3 with `BellmanPerfiles(laminar=True)` | **Night 4:** same as night 3 |
| **Sun 18** | Fact 3: B·log₂(G+1) vs laminar's B/2 ⟹ price ≤ 2·log₂(G+1). Write the proposition in your words (statement + proof, one page, like the "individuals" note); AI typesets; **send by 20:00** with the table | By 14:00: table final, every ratio checked against 1 / (2·log₂(G+1)) (a violation is a bug in the solver or in the lemma — stop and report). Worst Q1 ratio per (G, B) as a second table | **Night 5:** same, or back to `binario` if `precio` has stalled |
| **Mon 19** | Your own version of T_BG ≤ G·T_B ≤ G·Binary (four facts; minutes of 6 Oct, D3, have Francisco's wording); check it on the two-test table: B = 2, G = 2, four people at q = 0.3: T₄ = 1.2 ≥ 0.774 ≥ 0.6 = T₂ (1.074 belongs to B = 3) | Re-solve top 3 of nights 3–5; one line per mission for the session script; one paragraph on what the Q1 table says (where greedy loses most, whether rollout recovers it). Push everything | Loop off |
| **Tue 20** | 16:00 freeze the script `docs/notes/2026-10-13-guion-sesion.md` (re-dated); rehearse 5 + 5 min; commit and push 17:30; **19:00 session** | Rehearsal; commit and push 17:30 | — |

## 4. The night loop (Héctor)

Folder `augmented/autoresearch_ejemplos/`: `judge.py` FIXED (exact solvers, 8 anchor values checked at start, every best instance re-solved from scratch, full instance logged); `generator.py` the ONLY file the agent edits; `program.md` the agent's instructions (humans edit it); `results_binario.tsv` / `results_precio.tsv` append-only. Two missions: **binario** = minimize laminar/binary over Λ > 1 (request 1); **precio** = minimize laminar/unrestricted (request 2). 180 s per experiment, 40 experiments per night ≈ 2–3 h unattended. Anchors checked on A's machine 14 Oct: pass.

**Before night 1 (once):** in `program.md`, replace `/opt/miniconda3/bin/python` with the Python of your environment (the one where the tests pass).

**Start, every night, from the repository root, in a terminal you leave open:**

```bash
claude --dangerously-skip-permissions -p "Read augmented/autoresearch_ejemplos/program.md and run the mission until you reach 40 experiments. Follow its rules exactly."
```

**Every morning (5 min):** `git add augmented/autoresearch_ejemplos/ && git commit -m "loop: night N" && git push`.

**Execution update, 14 Oct; supersedes the q=½ search in nights 3–5 above:** the homogeneous q=½, u=1 domain within this judge's n≤7, B≤4, G≤4 limits is now covered by the old and expanded grids, the eight n=2–3 controls and trivial B=1/full-budget cases. Repeating random runs there cannot improve the record. Keep the best examples as initial controls, then explore probability perturbations around them (homogeneous and heterogeneous reported separately), or continue the binary mission. To study larger homogeneous instances, specify a new experiment first; do not silently change the fixed judge. The generator rejects duplicates within a run; an unchanged generator must not be rerun 40 times, because the judge resets its random seed.

Updated starting points: binario 0.900000750000975 (finite member of the 10/9 family); precio 41/52 at n=7, or 5/6 when capped at n=6. The guarded wrapper in `program.md` uses the local Python and imposes time and memory limits without modifying `judge.py`. Two short smoke runs passed; they rediscovered seeds and are not new findings or a completed night.

The additional n=7 seed replay through the fixed night judge hit the short pilot's 60 s / 1536 MiB limits. Its grid value is independently verified, but an `ok` judge run at n=7 under the production 360 s cap remains pending. Do not label that limited pilot a successful search.

What it can give: a lower ratio than these verified starting points and the *family* that produces it, in `notes.md`. What it cannot give: a theorem. Every new number stays "diagnostic" until Héctor re-solves it (Sat and Mon). If it stalls or breaks, the week still stands on §1.

## 5. WhatsApps to Francisco (A sends, in A's words)

- **Thu 15, by 20:00:** reply to his log G message (our "at most log G" half, in words) + the n = 3 witness with its label.
- **Sun 18, by 20:00:** the proposition (PDF) + the q = ½ table + the Q1 table with its label.
- **Mon 19, by 20:00 (§34-bis, eve of the session):** one line with what Tuesday shows, only if something changed since Sunday.

## 6. AI support (applied only with A's approval)

Typeset the proposition from A's text (Sun); re-date the session script to 20 Oct with the log G message opening block 3 and a new block for the Q1 table; re-date master plan §1 and add §32 rows for Francisco's 13-Oct WhatsApp and for this plan; extend the grid script if Héctor prefers to specify rather than code; final tables and the three-tree figure. Nothing goes into the repo or to Francisco that A has not read.

## 7. Valves

- Héctor's Wed check fails → Thursday's WhatsApp carries only the reply to the log G message; the discrepancy is the week's result.
- A's reply is not ready Thursday → it goes Sunday with the proposition; nothing else is sent Thursday.
- The proposition is not in A's words by Sunday 16:00 → send the table alone, say "lemma being written", keep the proposition for Tuesday's block.
- G = 5 too slow → table up to G = 4 at n = 7 and G = 5 at n = 6; say so.
- Q1 columns too slow at n = 7 → run them at n = 6 only; the table is still the first Q1 measurement.

*Note: `.tex` and `.pdf` are gitignored; the "individuals" note PDF reaches Héctor by WhatsApp from A; its proof text is in `proposicion_individuales_construccion.md`.*
