# Proposition: individual-only testing is never optimal with counts — A's construction

Status 2026-10-06: steps (1) and (2) closed and refereed in chat; steps (3) boundaries and (4) corollary pending. Label of the general case: DERIVED + VERIFIED (exact sweep 2026-10-05: 934 tests in `tests_brecha_convencion.py` + ~32,000 random instances n ≤ 7, B ≤ 4, G ≤ 4, 0 failures; smallest gain ratio 1.00001) until Héctor signs this text.

## 0. Statement in A's words

> **Francisco's theorem.** When the test result is binary, positive or negative, and the health probability is at most ½, the optimal strategy is to test singletons and never group.
>
> **Our proposition.** When the test result is a count, the strategy of only individual tests is never optimal, because the count leaves a deduction for the next test. We need at least two tests and more people than tests.

## 1. The three-person case (q = 0.3, u = 1, two tests) — A's version

> In a setup where the test result is not binary but a count of how many people are infected in the pool, and using soft clearing as the payment rule, which means we can obtain the utility of a person in two ways, either the person sat in a test that read exactly zero or we can deduce their state, a plan made only of singletons is never the best one if you have at least two tests and more people than tests. Pairing first beats two singletons because in the two extremes, when both are infected and when both are healthy, the count resolves both people with one test and buys you an extra test on the virgin territory, the people not yet included in any group. When the count is intermediate, you refine the pool by testing one member; by inference you also get the state of the other member, because you have the total and the new result. So the middle case loses nothing, and the gain comes from the two resolved cases, which let you explore more virgin territory. That is why a third person is needed: without one, the freed test has nothing to buy and the two plans tie.

Numbers behind it (verified by hand and by solver): pair-first 0.09·2.3 + 0.42·1 + 0.49·0.3 = 0.774 vs two singletons 0.6; gain 0.174 = (0.09 + 0.49)·0.3 = q(q² + p²); under hard clearing the middle row pays 0.5 and pair-first drops to 0.564 < 0.6.

## 2. The general swap — A's version

> Each person i is healthy with probability qᵢ, strictly between 0 and 1, and is worth uᵢ > 0. There are B ≥ 2 tests, pools of size at least 2 are allowed, and there is at least one more person than tests. Payment is by soft clearing. The best singletons-only policy tests one individual at a time, the B people with the largest qᵢ uᵢ, and pays T, the sum of those products.
>
> The swap: take two people of that plan, 1 and 2, group them and test the pair. If the count is 0 or 2, both are resolved by that single test, and the freed test goes to person 3, someone not yet tested. If the count is 1, the second test refines the pair by testing person 1 alone, and by deduction we learn the state of person 2 as well; the healthy one pays, under soft clearing.
>
> The four worlds are the four combinations of states of persons 1 and 2. Summing probability times payment over them gives V_after = T + gain, where T is the value of the best singletons policy and
>
> gain = (q₁ q₂ + p₁ p₂) q₃ u₃.
>
> The gain comes from the two resolved counts, which free a test for virgin territory. It is strictly positive: q₁ q₂ is not zero because nobody is known infected for sure, q₃ is not zero for the same reason, and u₃ is not zero because no person is worth nothing. So the singletons-only policy is beaten and is never optimal.

Where each hypothesis is used: G ≥ 2 (the pair is legal); B ≥ 2 (a second test to refine with or to free); n ≥ B + 1 (person 3 exists); soft clearing (world 3: person 2 pays by deduction); 0 < qᵢ < 1 and uᵢ > 0 (gain strictly positive).

Four-world table: (1,2) = (H,H) prob q₁q₂ pays u₁+u₂+q₃u₃; (H,I) prob q₁p₂ pays u₁; (I,H) prob p₁q₂ pays u₂; (I,I) prob p₁p₂ pays q₃u₃. Check at 0.3: T = 0.9 (three tests, four people), gain 0.174, V_after = 1.074 = solver optimum, tied by singleton-first and triple-first (so "pair-first is optimal" is NOT claimed).

## 3. Boundaries — pending (A has the numbers)

- B = 1: a fresh pair pays 2q² = 0.18 < 0.3; no second test, counts useless, singleton optimal.
- n = B: two people, two tests: pair-first 0.09·2 + 0.42·1 = 0.60 = two singletons; the freed test has nothing to buy, tie.

## 4. Corollary via Thm 3.3(i) — pending
