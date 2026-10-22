# Plan Mon 5 → Tue 6 Oct — one finished piece each, sent before the session

**Session with Francisco:** Tuesday 6 Oct, 19:00 Prague / 11:00 CDMX / 13:00 Boston. **Sync A+B:** today at end of day (sign and send), tomorrow 16:00. **Budget:** A and B full time both days. **Self-contained:** Héctor can read this without the chat. Sources: minutes of 29 Sep (`2026-09-29-sesion-francisco.md`), master plan §1 and §32 (rows dated 2026-09-29).

## 1. Where we stand

On 29 Sep Francisco asked for one thing: read his paper (arXiv:2601.22419v2) and send comments, because they submit it in early November. He said its Section 5 is the part most likely to apply to us. He liked Map 4 and called the π_L / π_ratio style "the direction" for Q1.

He also doubted that pooling is always better with counts. His words: proving that individual tests are never optimal "would be very interesting, and not something I would imagine offhand" [D 19:59]. We answered that we had not been able to prove it. That was wrong: the two-test case is already proved in A's convention-gap proposition, part 3(b), and the PDF was never sent.

This is the second session in a row where the package was delivered halfway. The rule of 20 Aug applies: we cut ambition. This week the base is **one written, closed piece per person, sent before the session**.

## 2. The thesis for this week

> Your Theorem 3.3(i) says that when healthy probabilities are at most ½, the best binary dynamic policy is individual tests. We show that with counts, individual tests are never the best policy as soon as there are two tests and one spare person. The proof is a six-line swap, and it is verified exactly.

**Proposition.** Counts, soft clearing (posterior-zero), pool cap G ≥ 2, budget B ≥ 2, n ≥ B + 1 people, every q_i strictly between 0 and 1, every u_i > 0. Then the policy that uses only individual tests is not optimal.

**The swap.** Take the best individual-only policy and two of its people, a and b. Replace the tests {a} and {b} by the pair {a, b} followed by one more test. If the count is 1, test {a}: whoever is healthy is now known, by deduction. If the count is 0 or 2, both are already resolved, and the last test goes to a fresh person c. The gain is (q_a·q_b + p_a·p_b)·q_c·u_c > 0.

**The two exact boundaries**, which are Francisco's own two intuitions:

- B = 1: counts are useless, the model equals the binary one. This is his counterexample.
- n ≤ B: individual tests are optimal, everyone's state gets known.

**Corollary, using his Theorem 3.3(i).** In the whole regime where every q_i ≤ ½, the laminar optimum with counts is strictly above the binary dynamic optimum for every instance with B ≥ 2 and n > B.

## 3. What is verified today

| Check | Size | Result |
|---|---|---|
| Two-test formula, by hand (A, September) | every q | pair first minus two individual tests = q(q² + p²) |
| Exact solver, new test in `tests_brecha_convencion.py` | 456 cases, n ≤ 5, B ≤ 3, G ≤ 3, different q and u per person | optimum ≥ swap bound > individual value whenever B ≥ 2 and n > B; equality at both boundaries |
| Second independent program (`enumerador_historias.py`) | 69 heterogeneous instances | same optimum, same inequality |
| Map 1 (Héctor, types solver) | 665 cells, n up to 64, B up to 8, G up to 8, q from 0.05 to 0.95 | optimum above B·q in all 665; ratio at least 1.25; up to 3.30 at q = 0.05 |
| Hard clearing | 650 cases | fails in 156: the statement needs soft clearing |
| Unrestricted optimum (Héctor, `results/precio_laminar_2026-09-21/`) | two-test game, n = 4, q = 0.3 | pair first is optimal among all policies, laminar or not: 0.774 = 0.774 |

**Labels.** Two tests, equal q: PROVED (convention-gap PDF, part 3(b)). General case: DERIVED + VERIFIED n ≤ 5 until A's written proof is refereed and Héctor signs it. Nothing else gets claimed.

## 4. The deliverable: one page to Francisco, sent today

`augmented/paper/proposicion_individuales_nunca_optimas.tex` → PDF, one page: statement, the two-test table, the swap, the two boundaries, the corollary, the evidence table, what it does not claim, one question. Attached with it: `proposicion_brecha_convencion.pdf`, which we owed him since 22 Sep.

Draft of the WhatsApp message, to edit:

> Hola Francisco. Te mandamos una página antes de la sesión de mañana. El martes dudaste de que con conteos siempre convenga agrupar, y dijiste que probar que las pruebas individuales nunca son óptimas sería muy interesante. Creemos que sí se puede probar, con las dos excepciones que tú mismo señalaste: una sola prueba, o tantas pruebas como personas. Es un intercambio de seis líneas y va en la nota con la verificación exacta. Adjuntamos también el PDF de la brecha soft/hard que te debíamos; el caso de dos pruebas sale de ahí. Sobre Λ: ya quedó claro, es la mejor prueba medida en pruebas individuales, y vale 1 exactamente cuando q ≤ ½ en el caso homogéneo. Mañana llevamos comentarios de las secciones 3 y 4 de tu paper.

## 5. Who does what

| Block | A (Vladimir) | B (Héctor) | AI support |
|---|---|---|---|
| Mon, block 1, about 3 h | The proof in his own words: (1) redo the two-test table at q = 0.3; (2) write the general swap, naming where each hypothesis is used; (3) the two boundaries; (4) the corollary | Section 5 of the paper, his current work: a one-page transfer memo. The proof of Theorem 5.4 in numbered steps, and for each step one verdict: transfers as is, needs a new idea with counts, or breaks, with one sentence of why | Referee each piece; typeset the one-pager as A closes |
| Mon, block 2, about 2 h | Read Section 3 of the paper against the PDF: Lemma 3.1, Λ, Theorems 3.2 and 3.3. Write at least three comments for Francisco | First-crossing experiment on the eight existing price-of-laminarity cases, after fixing the three definitions of section 5-bis. Comments on Section 5 for Francisco as they come up | Reading guide for Section 3 tied to A's exercises |
| Mon, end of day, 30 min | **Sync: Héctor's two sign-offs, about 75 min in total: the one-page note (read the swap, run `pytest augmented/tests_brecha_convencion.py`) and the overdue cross-review of Part 2. Then send the WhatsApp with both PDFs. Hard deadline 20:00 Prague.** | | |
| Tue 09:00–12:00 | Theorem 3.2, the ρ_i argument, and Francisco's exercise: why the bound with Λ = G is obvious. That exercise is the top rung of the ladder in section 5-bis | Finish the transfer memo. Extend the experiment to a small exhaustive grid if the eight cases behave | Draft of the session script |
| Tue 12:00–13:00 | Merge comments into `docs/notes/2026-10-06-comentarios-paper.md` and send them to Francisco before the session | | |
| Tue 13:00–15:30 | Extras, only if the base is closed: `seccion_separacion.tex` v0 with AI | Extras, only if the base is closed: a figure for the first-crossing table. The recursive (E[U], E[T]) evaluator moves to next week | |
| Tue 16:00 | Sync: freeze the script, rehearse 5 + 5 min, commit and push at 17:30 | | |
| Tue 19:00 | Session | | |

## 5-bis. Héctor's line: the price of laminarity

Adapted on 5 Oct from Héctor's own description of what he is working on. His notebook 28 and the review he mentions are not in the repository yet, so this section rests on his paragraph and on what the repo has: `augmented/experimento_precio_laminar.py`, `results/precio_laminar_2026-09-21/` and notebook 27.

**The ladder that joins both lines.**

> individual tests  <  best laminar  ≤  best unrestricted  ≤  T with index B·G  ≤  G × individual tests

- The first inequality, strict, is A's proposition of this week.
- The last two are the basic 1/G guarantee. The middle one is the bound Francisco called obvious: no policy touches more than B·G people. With counts the reason is that a person entering a test for the first time is still healthy with their prior probability. Label: the companion's statement, not yet validated by us. A writes that argument as Tuesday's exercise.
- The open question, Q2, is the second inequality: how close laminar is to unrestricted, with a constant that does not depend on G.

**What already exists on Q2** (Héctor, 21 Sep, exact arithmetic):

| Case | n | B | G | Laminar | Unrestricted | Ratio |
|---|---|---|---|---|---|---|
| Two tests | 4 | 2 | 2 | 0.774 | 0.774 | 1.000 |
| Three tests | 4 | 3 | 2 | 1.074 | 1.137 | 0.945 |
| After a pair with count 1, two tests left | 4 | 2 | 2 | 1.300 | 1.450 | 0.897 |
| Four people, triples | 4 | 3 | 3 | 1.750 | 2.000 | 0.875 |
| Six people, homogeneous | 6 | 3 | 3 | 1.249 | 1.419 | 0.880 |
| Six people, heterogeneous | 6 | 3 | 4 | 1.065 | 1.110 | 0.959 |

These are chosen examples, not a worst-case search. The third row is the smallest case where the unrestricted optimum crosses: after the pair {A, B} reads 1, it tests {A, C}.

**The first-crossing experiment. Three definitions to fix before running it.**

1. *Crossing test:* a test that is neither disjoint from everything tested nor inside a single residual atom, after the companion's reduction of ancestor tests.
2. *Tie-break:* among optimal unrestricted policies, prefer laminar actions, so a crossing is counted only when it is strictly better.
3. *Three values at the first crossing state:* the unrestricted continuation, the best laminar continuation and the continuation of π_ratio, each weighted by the probability of reaching that state.

What it measures: unrestricted minus best laminar continuation is a cost of laminarity; best laminar continuation minus π_ratio is the cost of the heuristic. The policy "follow the unrestricted optimum and switch to laminar at the first crossing" is itself laminar, so its value is a lower bound on the laminar optimum. Comparing it with the true laminar optimum says whether cutting at the first crossing is a good way to laminarize. That is the empirical version of the route Francisco sketched on 29 Sep: build laminar policies out of an unrestricted one. Every number carries the label "diagnostic on small exact cases".

## 6. What we say on Tuesday, in order

1. The note: the proposition in one sentence, then his reaction. Question (30): is this the statement he found surprising, and does it belong in the separation section?
2. His paper: what we understood of Section 3, our comments, our questions on Section 5.
3. Héctor: the top end of the ladder. What transfers from Section 5 and what breaks, and the first-crossing table: how much is lost to laminarity and how much to the heuristic.
4. Questions that were never asked: (28) the one-line statement for our introduction; (29) the split between the two papers and the status of the companion; (23) where the Section 5 argument breaks with counts; (24) which λ.

## 7. What we do not say and do not do

- Not "pooling is always better". The statement is that individual-only is never optimal. The optimal first test can tie with a single test, for example at B = 3, q = 0.3.
- Not "proved" for the general case unless A's proof was refereed and Héctor signed.
- No numbers outside the table in section 3.
- No new counterexample hunts, no Monte Carlo scaling, no companion Theorem 9.3. The 1/G guarantee is quoted as the companion's until A's argument is written.
- No claim from the first-crossing experiment beyond "diagnostic on small exact cases".

## 8. Fallbacks

- If A's general proof does not close today: send the two-test case, which is proved, and state the general case as "verified, proof being written".
- If Héctor finds a counterexample: stop. That is the result, and the note becomes "true for two tests, fails at this instance".
- If the reading goes slower than planned: comments on Section 3 only. Francisco said a week or more on this paper is a good use of time.
