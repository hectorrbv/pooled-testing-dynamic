"""
Bayesian posterior update for augmented pooled tests.

Given prior infection probabilities p = (p_1, ..., p_n) and a test history
H_k = ((t_1, r_1), ..., (t_k, r_k)), compute posterior probabilities
q'_i = P(individual i is healthy | H_k) for each individual.

In the idealized augmented model, testing pool t yields r = |t ∩ Z|
(the exact count of infected in the pool).
"""

from augmented.core import indices_from_mask, test_result, popcount


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
        # Handle deterministic cases: p_i = 0 or p_i = 1
        if p[i] <= 0.0:
            posterior[i] = 0.0
            continue
        if p[i] >= 1.0:
            posterior[i] = 1.0
            continue

        others = [j for j in pool_indices if j != i]
        others_p = [max(0.0, min(1.0, p[j])) for j in others]

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
    """Apply Bayesian updates for a full test history (sequential).

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


def bayesian_update_by_counting(p, history, n):
    """Compute posterior P(Z_i=1 | h_k) by counting over all consistent worlds.

    This is the "by counting" approach: enumerate all 2^n infection profiles,
    keep those consistent with the full test history, and compute posteriors
    as weighted proportions.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Full test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities P(Z_i=1 | h_k).

    Notes
    -----
    Complexity: O(2^n * k) where k = len(history).
    Feasible for n <= ~20. For independent priors, this gives the same
    result as sequential Bayesian updates but makes the joint computation
    explicit and transparent.
    """
    if not history:
        return list(p)

    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n

    # Accumulate: weighted count of consistent profiles, and per-individual
    # weighted count of consistent profiles where Z_i = 1
    total_weight = 0.0
    infected_weight = [0.0] * n

    for z_mask in range(num_profiles):
        # Check consistency with ALL tests in history
        consistent = True
        for pool_mask, r in history:
            if test_result(pool_mask, z_mask) != r:
                consistent = False
                break

        if not consistent:
            continue

        # Pr(Z = z_mask) under the independent prior
        w = 1.0
        for i in range(n):
            w *= p[i] if (z_mask >> i & 1) else q[i]

        total_weight += w
        # Add weight to each infected individual in this profile
        bits = z_mask
        while bits:
            lsb = bits & -bits
            i = lsb.bit_length() - 1
            infected_weight[i] += w
            bits ^= lsb

    # Compute posteriors
    posterior = list(p)  # fallback to prior if degenerate
    if total_weight > 0:
        for i in range(n):
            posterior[i] = infected_weight[i] / total_weight

    return posterior


def estimate_p_from_history(history, n, prior_p=None, prior_strength=1.0):
    """Estimate infection probabilities from observed test data.

    Uses a Bayesian approach with a Beta prior. If prior_p is given,
    uses Beta(prior_strength * p_i, prior_strength * (1 - p_i)) as prior.
    Otherwise uses uniform Beta(1, 1).

    Parameters
    ----------
    history : tuple of (pool_mask, result) pairs
        Test history.
    n : int
        Population size.
    prior_p : list[float] or None
        Prior guess for infection probabilities.
    prior_strength : float
        Strength of the prior (pseudo-count scale).

    Returns
    -------
    list[float]
        Estimated infection probabilities.
    """
    if prior_p is not None:
        alpha = [prior_strength * pi for pi in prior_p]
        beta = [prior_strength * (1.0 - pi) for pi in prior_p]
    else:
        alpha = [1.0] * n
        beta = [1.0] * n

    # For each consistent world, accumulate posterior mass
    if not history:
        return [a / (a + b) for a, b in zip(alpha, beta)]

    q_prior = [1.0] * n
    if prior_p:
        q_prior = [1.0 - pi for pi in prior_p]
    else:
        q_prior = [0.5] * n

    p_prior = prior_p if prior_p else [0.5] * n

    # Use counting-based posterior
    return bayesian_update_by_counting(p_prior, history, n)
