# Plan from now to the session — Mon 12 Oct → Tue 13 Oct, 19:00 Prague

> **Superseded on 2026-10-14 by `2026-10-14-plan-semana.md`** (the 13-Oct session was cancelled; next session Tue 20 Oct).

**Self-contained for Héctor.** Sources: minutes of 6 Oct (`2026-10-06-sesion-francisco.md`), results note (`2026-10-12-encargos-6-oct.md`), master plan §1 and §34-bis (protocol revised today). **Budget:** 10 h each. **Rule of the week (§34-bis, point 1):** one written piece per person, closed and sent, before anything else.

## 1. What we show on Tuesday (the whole list; nothing else)

| # | Result | Status today | Owner | Label on Tuesday |
|---|---|---|---|---|
| 1 | One-page note: *with counts, individual-only testing is never optimal* (PDF, sent Monday with the convention-gap PDF) | typeset, A's proof text; steps 3–4 and B's signature missing | A, B | PROVED for B = 2; general: DERIVED + VERIFIED (934 tests + 32,195-instance sweep) once B signs |
| 2 | Francisco's request 1: an instance where optimal dynamic **binary beats laminar**. Witness n = 3, B = 2, G = 3, q = (0.9, 0.7, 0.7): 2.079 > 2.062; 63 of 6,800 instances with Λ > 1; min ratio 0.978 | computed, one implementation (new solver) | B validates | "diagnostic, one implementation" → "two implementations agree" if B's check passes |
| 3 | Francisco's request 2: **price of laminarity at q = ½**. Worst laminar/unrestricted ratio 0.875 / 0.857 / 0.833 at n = 4 / 5 / 6, worse than q = 0.3 and 0.7 at every n | computed, same caveat | B validates, extends to n = 7 if cheap | same as 2 |
| 4 | A's rewrite of Francisco's upper-bound argument V(π) ≤ T_BG ≤ G·T_B ≤ G·Binary | not written | A | closes companion Cor 8.3 by identification |
| 5 | Two remarks as questions, not claims: (31) binary > laminar needs Λ > 1; (32) with pool cap G, identifying everyone costs ≥ n/log₂(G+1) tests, so the q = ½ price may be O(log G) | derived by the AI support | A reads them, asks | questions (31), (32), (33) |

Not realistic by Tuesday, and not on the list: proving that q = ½ is the worst case; any constant for Q2; the §5 transfer memo; the first-crossing experiment (the n = 3 witness is its first row and that is all we say); validating the information bound as a welfare bound; the Section 3 comments beyond two sentences.

## 2. Who does what

**A (10 h).**
1. (2 h) Steps 3 and 4 of the proof in your own words: the two boundaries with their numbers (B = 1: a fresh pair pays 2q²; singleton wins iff Λ ≤ 1; n = B: 0.6 vs 0.6, the freed test has nothing to buy) and the corollary (Λ ≤ 1 ⟹ binary optimum = T_B < laminar optimum). Then read `augmented/paper/proposicion_individuales_nunca_optimas.pdf` and mark every sentence that is not yours; the AI recompiles.
2. (1 h) Monday 20:00 at the latest: WhatsApp to Francisco with both PDFs and question (30).
3. (2 h) Rewrite the T_BG argument in your own words (four facts; the minutes D3 have Francisco's wording). Check it against the two-test table: with B = 2, G = 2, four people at q = 0.3, T_4 = 1.2 ≥ 1.074 ≥ 0.6 = T_2.
4. (3 h) Learn the n = 3 witness until you can draw the three trees from memory, and on Tuesday the best witness of each overnight mission the same way and say why the cross wins (two bets of 0.511 × 2 vs one sure 1), and the q = ½ table. Read remarks (31) and (32) and decide whether you can defend asking them.
5. (2 h) Tuesday 16:00: freeze the script `docs/notes/2026-10-13-guion-sesion.md`, rehearse 5 + 5 min with Héctor, commit and push 17:30.

**Héctor (10 h).**
1. (2 h) Sign the note: read the swap, run `pytest augmented/tests_brecha_convencion.py` (934 pass) and `PYTHONPATH=. python augmented/barrido_proposicion_individuales.py <seed> 120 6 4 4` with a seed of your choice (expects `fails 0`). Object or sign in the minutes.
2. (4 h) Second implementation check of `augmented/dp_perfiles_binario_conteo.py`, which nobody but the AI has run: (a) `BellmanPerfiles(laminar=False)` against the `unrestricted` column of the two grid CSVs in `results/encargos_2026-10-06/` (81 cells); (b) your own enumerator (or a fresh one) on 20 rows of `binario_vs_laminar_seed*.csv`, binary channel; (c) `static_binary_value` where B = 1. Any mismatch stops everything else and becomes the result. If all match, items 2–3 of the table go from "diagnostic" to "two implementations agree".
3. (3 h) **If the overnight loop ran:** re-solve the top three instances of each `results_*.tsv` with the repository solvers (independent code path: `BellmanPerfiles` + your enumerator), draw the trees of the best witness per mission, one paragraph on the mechanism. **If it did not:** extend the q = ½ grid to n = 7 with G ≤ 3, B ≤ 4 (`augmented/malla_precio_laminar_q.py`; n = 6, G = 4, B = 4 took 27 s, so n = 7 is minutes, not hours); binary-vs-laminar worst search at n = 6 with free u: a plain exact hill-climb over (q, u) starting from the 63 witnesses, 1 h of CPU, keep the minimum ratio. Write both as rows in the results note.
4. (1 h) Push notebook 28 and Francisco's shared paper reference to the repo; rehearsal.

**AI support.** Recompiles the note; the three-tree figure of the witness; runs whatever Héctor specifies; final tables; the frozen script.

## 3. The overnight loop (decided by A, 2026-10-12 evening)

A wants Francisco's "concrete, worst possible example" searched by an autoresearch-style loop overnight, and Tuesday spent interpreting. **Héctor runs the loop** (decided 2026-10-12, late). Set up in `augmented/autoresearch_ejemplos/`: `judge.py` (FIXED: exact solvers, 8 anchor values asserted at startup, every argmin re-solved from scratch, full instance logged), `generator.py` (the ONLY file the agent edits: `sample(rng)` draws instances), `program.md` (the agent's instructions), `results_binario.tsv` / `results_precio.tsv` (append-only log). Two missions: **binario** = minimize laminar/binary over Λ > 1 instances (request 1); **precio** = minimize laminar/unrestricted (request 2). Budget 180 s per experiment, n ≤ 6, 40 experiments max.

Héctor starts it tonight from the repository root, in a terminal, and leaves it running (bypass permissions is needed because it edits a file and runs Python unattended):

```bash
cd /Users/mac/Desktop/ASE/group-count-dynamic && claude --dangerously-skip-permissions -p "Read augmented/autoresearch_ejemplos/program.md and run the mission until you reach 40 experiments. Follow its rules exactly."
```

What it can and cannot give: a lower minimum ratio than random search found (0.978 / 0.833) and, more useful, the *family* that produces it, written in `notes.md`. It cannot produce a theorem or a worst-case proof, and every number it logs stays "diagnostic" until Héctor re-solves the top instances on Tuesday morning (three per mission, by hand-run of `arbol_testigo_binario_vs_laminar.py` adapted to the instance). Risk: the loop stalls or breaks the generator — then Tuesday's material is what we already have (section 1), which is enough.

**Tuesday morning, what changes:** Héctor's block 3 becomes "re-solve the top three of each `results_*.tsv`, draw the trees of the best witness, write the mechanism in one paragraph"; A's block 4 becomes "learn the best witness of each mission the way you learned the n = 3 one". The session script gets, in blocks 2 and 3, one extra line each: "the loop ran N experiments overnight; the worst it found is …, mechanism …".

What Karpathy's loop is, for reference: an agent edits one file (`train.py`), runs a fixed 5-minute experiment, keeps the change if the metric improves, logs it; the human edits only `program.md`. Ours keeps that shape with an exact metric and no GPU.

## 4. Expected state at 19:00 Tuesday

Sent Monday: the note and the convention-gap PDF. In the repo, committed: the minutes of 6 Oct and the dispatch, the results note with the two tables (n = 7 row if it ran), the solver with Héctor's second-way check recorded, the frozen script. Said in the session: five items of the table in section 1, four questions (30)–(33), and nothing that is not in the repo.

Valve: if B's second-way check fails, we present the note only and report the discrepancy as the finding. If A's steps 3–4 do not close by Monday 18:00, the note goes with steps 1–2 and the boundaries marked "stated, proof being written".
