"""
Bayesian posterior update for augmented pooled tests.

Given prior infection probabilities p = (p_1, ..., p_n) and a test history
H_k = ((t_1, r_1), ..., (t_k, r_k)), compute posterior probabilities
q'_i = P(individual i is healthy | H_k) for each individual.

In the idealized augmented model, testing pool t yields r = |t ∩ Z|
(the exact count of infected in the pool).
"""

from augmented.core import indices_from_mask


def _poisson_binomial_pmf(probs):
    """PMF of the Poisson-Binomial distribution for independent Bernoullis.

    Given probabilities [p_1, ..., p_m], returns a list pmf of length m+1
    where pmf[k] = P(exactly k successes).

    Uses the DP recurrence:
        dp[0] = 1
        dp[k] = dp[k] * (1-p_j) + dp[k-1] * p_j   (for each new p_j)
    """
    m = len(probs)
    dp = [0.0] * (m + 1)
    dp[0] = 1.0
    for j, pj in enumerate(probs):
        # Traverse backwards to avoid overwriting values we still need
        for k in range(j + 1, 0, -1):
            dp[k] = dp[k] * (1.0 - pj) + dp[k - 1] * pj
        dp[0] *= (1.0 - pj)
    return dp


def bayesian_update_single_test(p, pool_mask, r, n):
    """Update infection probabilities after one augmented test.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
    pool_mask : int
        Bitmask of the tested pool t.
    r : int
        Observed result r = |t ∩ Z| (count of infected in pool).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities p'_i = P(Z_i=1 | r, t).

    Math
    ----
    For i NOT in t: test gives no info, so p'_i = p_i.

    For i IN t, by Bayes:
        p'_i = P(Z_i=1 | r) = P(r | Z_i=1) * p_i / P(r)

    where (letting S = t \\ {i}):
        P(r | Z_i=1) = P(exactly r-1 infected in S)   [Poisson-Binomial]
        P(r | Z_i=0) = P(exactly r   infected in S)   [Poisson-Binomial]
        P(r)          = P(r|Z_i=1)*p_i + P(r|Z_i=0)*q_i
    """
    pool_indices = indices_from_mask(pool_mask, n)

    if not pool_indices:
        return list(p)

    posterior = list(p)

    # For each i in pool, compute posterior via Poisson-Binomial on t\{i}
    for i in pool_indices:
        others = [j for j in pool_indices if j != i]
        others_p = [p[j] for j in others]

        # PMF of number of infected among others
        pmf = _poisson_binomial_pmf(others_p)

        # P(r | Z_i = 1) = P(r-1 infected among others)
        p_r_given_1 = pmf[r - 1] if r >= 1 else 0.0
        # P(r | Z_i = 0) = P(r infected among others)
        p_r_given_0 = pmf[r] if r <= len(others) else 0.0

        # Bayes
        numerator = p_r_given_1 * p[i]
        denominator = numerator + p_r_given_0 * (1.0 - p[i])

        if denominator > 0:
            posterior[i] = numerator / denominator
        # else: degenerate case, keep prior

    return posterior


def bayesian_update(p, history, n):
    """Apply Bayesian updates for a full test history.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities after all tests in history.
    """
    current_p = list(p)
    for pool_mask, r in history:
        current_p = bayesian_update_single_test(current_p, pool_mask, r, n)
    return current_p
