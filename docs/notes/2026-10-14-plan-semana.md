# Week plan — Wed 14 Oct → Tue 20 Oct, 19:00 Prague (session with Francisco)

**Self-contained for Héctor.** Supersedes `2026-10-12-plan-hasta-sesion.md` (written for the 13-Oct session, which was cancelled). Sources: minutes of 6 Oct (`2026-10-06-sesion-francisco.md`), results note (`2026-10-12-encargos-6-oct.md`), master plan §34-bis, Francisco's WhatsApp of 13 Oct. **Budget:** A ≈ 2 h/day, Héctor ≈ 2–3 h/day in daytime; the search loop runs **only at night** on Héctor's machine and never takes his daytime hours (5 min in the morning to save the results, 5 min at night to start it).

## 0. What changed since Monday

Francisco wrote on 13 Oct: with everyone worth 1 and healthy with probability ½, he believes the gap between augmented overlapping and laminar is about log G: laminar is bounded by a constant times B, and detection matrices reach B·log G. Our side (derived by the AI support, A validates this week): for any policy with pool cap G the test history has at most (G+1)^B values, so at most B·log₂(G+1) bits; a cleared person is one bit fixed; at q = ½, u = 1 this gives E[cleared] ≤ B·log₂(G+1), and laminar singletons already give B/2, so the price of laminarity is at most 2·log₂(G+1). That is the "at most log G" half; his two halves (laminar ≤ cB, a construction reaching B·log G) stay his and are **not** our work this week.

So the q = ½, u = 1 table **as G grows** becomes the main numeric piece of the week, and the night loop shifts towards it from Friday.

## 1. What we bring on Tuesday (the whole list)

| # | Result | Owner | Label on Tuesday |
|---|---|---|---|
| 1 | A's reply to Francisco's log G message, in A's words (sent Thu) | A | argument, derived; his halves cited as his |
| 2 | Binary beats laminar: witness n = 3, B = 2, G = 3, q = (0.9, 0.7, 0.7), 2.079 vs 2.062; 63 of 6,800 Λ > 1 instances; min ratio 0.978; plus the loop's best, re-solved | Héctor validates, A explains | "two implementations agree" only if Héctor's check passes; else "diagnostic, one implementation" |
| 3 | Price of laminarity at q = ½, u = 1 as G grows (G = 2…5, n up to 7), plus the loop's best | Héctor | same as 2 |
| 4 | The counting lemma (≤ (G+1)^B histories ⟹ price ≤ 2·log₂(G+1)) in A's words (sent Sun) | A | derived, A's wording |
| 5 | A's version of Francisco's T_BG ≤ G·T_B ≤ G·Binary argument | A | closes companion Cor 8.3 by identification |

Not this week: Francisco's lower half (laminar ≤ cB) or any B·log G construction; proving q = ½ is the worst case; any constant for Q2; the §5 memo; steps 3–4 of the "individuals never optimal" construction (stay "stated, proof being written").

## 2. Day by day

| Day | A (≈ 2 h) | Héctor, daytime (≈ 2–3 h) | Héctor, night (loop) |
|---|---|---|---|
| **Wed 14** | Approve this plan. Tutoring, Francisco's message steps 1–2: what a singleton pays at q = ½ (B tests → B/2), what a pool of 2 pays (2·¼ = 0.5, the same) | `git pull`. **Sign the note:** read `augmented/paper/proposicion_individuales_construccion.md`, run `pytest augmented/tests_brecha_convencion.py` (934 pass) and `PYTHONPATH=. python augmented/barrido_proposicion_individuales.py <seed> 120 6 4 4` (expects `fails 0`) | **Night 1:** loop as written in `program.md` (both missions) |
| **Thu 15** | Steps 3–4: the (G+1)-answers counting fact; "a cleared person = one fixed fact". Write the reply to Francisco yourself; **send by 20:00** with the n = 3 witness (label from Héctor's check) | Morning: commit `results_*.tsv` + `notes.md` of night 1. **Second-way check** of `augmented/dp_perfiles_binario_conteo.py`, done by 17:00 so A can use the label: (a) `BellmanPerfiles(laminar=False)` vs the `unrestricted` column of the two grid CSVs in `results/encargos_2026-10-06/` (81 cells); (b) your own enumerator on 20 rows of `binario_vs_laminar_seed*.csv`, binary channel; (c) `static_binary_value` where B = 1. Any mismatch stops everything else and becomes the result | **Night 2:** same as night 1 |
| **Fri 16** | Counting lemma, fact 1 in your words (≤ (G+1)^B histories). Learn the n = 3 witness: draw its three trees from memory | Morning: commit night 2. Re-solve the **top 3 of each mission** from nights 1–2 with the independent path (`BellmanPerfiles` + your enumerator); draw the trees of the best witness; one paragraph on the mechanism → row in the results note. Edit `program.md` for nights 3–5 (see §3) | **Night 3:** `precio` focused on q = ½, u = 1 |
| **Sat 17** | Lemma, fact 2 (at q = ½ each person is one full bit of uncertainty; clearing someone fixes that bit) | **q = ½ table as G grows:** u = 1, q = ½, G = 2, 3, 4, 5, n = 6–7, B ≤ 4, laminar vs unrestricted, with `augmented/malla_precio_laminar_q.py` extended to take G and q from the command line (n = 6, G = 4, B = 4 took 27 s; time G = 5 on one cell first). Write rows in the results note | **Night 4:** same as night 3 |
| **Sun 18** | Lemma, fact 3 (the bound B·log₂(G+1) vs laminar's B/2 ⟹ price ≤ 2·log₂(G+1)). **Send by 20:00:** the q = ½ table + the lemma in your words | Finish the table by 16:00 so A can send it; check the table against the lemma (no laminar/unrestricted ratio may fall below 1 / (2·log₂(G+1))) | **Night 5:** same, or back to `binario` if `precio` has stalled |
| **Mon 19** | Your own version of T_BG ≤ G·T_B ≤ G·Binary (four facts; minutes of 6 Oct, D3, have Francisco's wording); check it on the two-test table: B = 2, G = 2, four people at q = 0.3: T₄ = 1.2 ≥ 1.074 ≥ 0.6 = T₂ | Re-solve top 3 of nights 3–5; one line per mission for script blocks 2–3 ("the loop ran N experiments; the worst it found is …, mechanism …"). Push everything | Loop off |
| **Tue 20** | 16:00 freeze the script `docs/notes/2026-10-13-guion-sesion.md`; rehearse 5 + 5 min with Héctor; commit and push 17:30; **19:00 session** | Rehearsal; commit and push 17:30 | — |

## 3. The night loop (Héctor)

Folder `augmented/autoresearch_ejemplos/`: `judge.py` FIXED (exact solvers, 8 anchor values checked at start, every best instance re-solved from scratch, full instance logged); `generator.py` the ONLY file the agent edits; `program.md` the agent's instructions (humans edit it); `results_binario.tsv` / `results_precio.tsv` append-only. Two missions: **binario** = minimize laminar/binary over Λ > 1 (request 1); **precio** = minimize laminar/unrestricted (request 2). 180 s per experiment, 40 experiments per night ≈ 2–3 h unattended. Anchors checked on A's machine 14 Oct: pass.

**Before night 1 (once):** in `program.md`, replace `/opt/miniconda3/bin/python` with the Python of your environment (the one where `pytest augmented/tests_brecha_convencion.py` passes).

**Start, every night, from the repository root, in a terminal you leave open:**

```bash
claude --dangerously-skip-permissions -p "Read augmented/autoresearch_ejemplos/program.md and run the mission until you reach 40 experiments. Follow its rules exactly."
```

**Every morning (5 min):** `git add augmented/autoresearch_ejemplos/ && git commit -m "loop: night N" && git push`, so A sees the results without asking.

**From Friday (nights 3–5), focus on Francisco's setting:** add to `program.md` under "Ideas": "Nights 3–5: mission `precio` only, q = ½ and u = 1 for everyone, vary n (pass NMAX = 7 to the judge), B = 3–4, G = 2–4 (the judge caps G at 4); report the minimum ratio separately for each G." The judge does not change.

What it can give: a lower minimum ratio than random search (0.978 / 0.833) and, more useful, the family that produces it, written in `notes.md`. What it cannot give: a theorem or a worst-case proof. Every number stays "diagnostic" until Héctor re-solves it (Fri and Mon). If the loop stalls or breaks, the week still stands on rows 1–5 of §1.

## 4. WhatsApps to Francisco this week (A sends, in A's words)

- **Thu 15, by 20:00:** reply to his log G message (our "at most log G" half) + the n = 3 witness with its label.
- **Sun 18, by 20:00:** the q = ½ table as G grows + the counting lemma.
- **Mon 19, by 20:00 (§34-bis, eve of the session):** one line with what Tuesday will show, only if something changed since Sunday.

## 5. AI support (applied only with A's approval)

Re-date the session script to 20 Oct and add the log G message as block 3's opening; re-date master plan §1 to 20 Oct and add a §32 row for Francisco's 13-Oct WhatsApp; extend the grid script for G = 5 if Héctor wants it; final tables. Nothing goes into the repo or to Francisco that A has not read.

## 6. Valves

- Héctor's second-way check fails → Thursday's WhatsApp carries only the reply to the log G message; the discrepancy becomes the result of the week.
- A's reply is not ready Thursday → it moves to Sunday's WhatsApp together with the table; nothing else is sent Thursday.
- G = 5 too slow → the table goes up to G = 4, n = 7, and we say so.

*Note: `.tex` and `.pdf` are gitignored; the one-page note `proposicion_individuales_nunca_optimas.pdf` reaches Héctor by WhatsApp from A, its proof text is in `proposicion_individuales_construccion.md`.*
