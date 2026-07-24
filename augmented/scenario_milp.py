"""Myopic pool selection over a weighted posterior scenario distribution.

Each row of ``Z`` is a latent-state profile and ``weights[s]`` is its
posterior probability.  The MILP chooses a nonempty pool of size at most
``G`` and maximizes its exact immediate expected welfare under that discrete
distribution.  It preserves correlations between individuals because clean
pool probabilities are computed scenario by scenario, never as a product of
posterior marginals.
"""

from numbers import Integral

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from augmented.core import all_pools, indices_from_mask, mask_from_indices


def _as_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validated_probabilities(p):
    probabilities = np.asarray(p, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("p must be a one-dimensional sequence")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("p must contain only finite probabilities")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("every prior probability must lie in [0, 1]")
    return probabilities


def exact_prior_scenarios(p):
    """Enumerate all independent-Bernoulli profiles and their prior weights."""

    probabilities = _validated_probabilities(p)
    n = len(probabilities)
    if n >= 63:
        raise ValueError("exact scenario enumeration requires n < 63")

    profile_ids = np.arange(1 << n, dtype=np.uint64)[:, None]
    bit_ids = np.arange(n, dtype=np.uint64)[None, :]
    scenarios = ((profile_ids >> bit_ids) & 1).astype(np.int8)
    weights = np.prod(
        np.where(scenarios == 1, probabilities, 1.0 - probabilities), axis=1
    )
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("prior scenarios have zero or non-finite total mass")
    return scenarios, weights / total


def _validated_scenarios(Z, weights, u):
    scenarios = np.asarray(Z)
    if scenarios.ndim != 2 or scenarios.shape[0] == 0:
        raise ValueError("Z must be a nonempty two-dimensional scenario matrix")
    if not np.all((scenarios == 0) | (scenarios == 1)):
        raise ValueError("Z must contain only binary latent states")
    scenarios = scenarios.astype(np.int8, copy=False)

    scenario_weights = np.asarray(weights, dtype=float)
    if scenario_weights.ndim != 1 or len(scenario_weights) != len(scenarios):
        raise ValueError("weights must have one entry per scenario")
    if not np.all(np.isfinite(scenario_weights)) or np.any(scenario_weights < 0):
        raise ValueError("scenario weights must be finite and nonnegative")
    total_weight = float(scenario_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("scenario weights must have positive total mass")
    scenario_weights = scenario_weights / total_weight

    utilities = np.asarray(u, dtype=float)
    if utilities.ndim != 1 or len(utilities) != scenarios.shape[1]:
        raise ValueError("u must have one entry per individual")
    if not np.all(np.isfinite(utilities)) or np.any(utilities < 0):
        raise ValueError("utilities must be finite and nonnegative")
    return scenarios, scenario_weights, utilities


def _validated_mask(mask, n, name, allow_empty=True):
    mask = _as_integer(mask, name)
    if mask < 0 or mask >= (1 << n):
        raise ValueError(f"{name} is outside the scenario universe")
    if not allow_empty and mask == 0:
        raise ValueError(f"{name} must be nonempty")
    return mask


def _score_pool_validated(pool, scenarios, weights, utilities, cleared):
    indices = indices_from_mask(pool, scenarios.shape[1])
    if not indices:
        return 0.0
    clean = scenarios[:, indices].sum(axis=1) == 0
    gain = sum(
        utilities[i] for i in indices if not (cleared >> i) & 1
    )
    return float(weights[clean].sum() * gain)


def score_pool_scenarios(pool, Z, weights, u, cleared=0):
    """Immediate expected welfare of ``pool`` under weighted scenarios."""

    scenarios, scenario_weights, utilities = _validated_scenarios(Z, weights, u)
    n = scenarios.shape[1]
    pool = _validated_mask(pool, n, "pool")
    cleared = _validated_mask(cleared, n, "cleared mask")
    return _score_pool_validated(
        pool, scenarios, scenario_weights, utilities, cleared
    )


def brute_best_pool_scenarios(Z, weights, u, G, cleared=0):
    """Enumerate all feasible pools; reference oracle for the scenario MILP."""

    scenarios, scenario_weights, utilities = _validated_scenarios(Z, weights, u)
    n = scenarios.shape[1]
    G = _as_integer(G, "G")
    if not 1 <= G <= n:
        raise ValueError("G must lie between 1 and the population size")
    cleared = _validated_mask(cleared, n, "cleared mask")

    scored = (
        (
            _score_pool_validated(
                pool, scenarios, scenario_weights, utilities, cleared
            ),
            pool,
        )
        for pool in all_pools(n, G, include_empty=False)
    )
    return max(scored)


def milp_best_pool_scenarios(
    Z,
    weights,
    u,
    G,
    cleared=0,
    time_limit=30.0,
):
    """Solve the exact myopic scenario pool-selection MILP.

    Returns ``(objective_value, pool_mask, scipy_result)``.  Only the pool
    variables ``x_i`` are declared integral.  The clean indicators ``y_s``
    are nevertheless forced to zero or one by the scenario constraints once
    ``x`` is fixed.  One variable ``v_s`` then represents the selected pool's
    utility in clean scenario ``s``.  Its big-M envelope is exact and needs
    only ``O(S)`` auxiliary variables, rather than one product per
    scenario-person pair.
    """

    scenarios, scenario_weights, utilities = _validated_scenarios(Z, weights, u)
    scenario_count, n = scenarios.shape
    G = _as_integer(G, "G")
    if not 1 <= G <= n:
        raise ValueError("G must lie between 1 and the population size")
    cleared = _validated_mask(cleared, n, "cleared mask")
    if time_limit is not None:
        time_limit = float(time_limit)
        if not np.isfinite(time_limit) or time_limit <= 0.0:
            raise ValueError("time_limit must be positive and finite")

    y_offset = n
    v_offset = n + scenario_count

    def x_index(i):
        return i

    def y_index(s):
        return y_offset + s

    def v_index(s):
        return v_offset + s

    variable_count = n + 2 * scenario_count
    objective = np.zeros(variable_count, dtype=float)
    reward_utilities = np.array(
        [0.0 if (cleared >> i) & 1 else utilities[i] for i in range(n)]
    )
    for s in range(scenario_count):
        objective[v_index(s)] = -scenario_weights[s]

    # Tight valid bound for the utility of any pool of at most G members.
    utility_bound = float(
        np.sort(reward_utilities)[-min(G, n):].sum()
    )

    rows = []
    columns = []
    data = []
    lower_bounds = []
    upper_bounds = []

    def add_constraint(coefficients, lower=-np.inf, upper=np.inf):
        row = len(lower_bounds)
        for column, coefficient in coefficients.items():
            rows.append(row)
            columns.append(column)
            data.append(coefficient)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    add_constraint(
        {x_index(i): 1.0 for i in range(n)}, lower=1.0, upper=float(G)
    )

    for s in range(scenario_count):
        active_indices = np.flatnonzero(scenarios[s])

        # If any selected individual is active in scenario s, y_s = 0.
        for i in active_indices:
            add_constraint(
                {y_index(s): 1.0, x_index(int(i)): 1.0}, upper=1.0
            )

        # If none is active, the following inequality forces y_s = 1.
        clean_indicator = {y_index(s): 1.0}
        clean_indicator.update(
            {x_index(int(i)): 1.0 for i in active_indices}
        )
        add_constraint(clean_indicator, lower=1.0)

        # Exact envelope for v_s = y_s * sum_i reward_u_i x_i.
        selected_utility = {
            x_index(i): -reward_utilities[i] for i in range(n)
        }
        utility_ceiling = {v_index(s): 1.0}
        utility_ceiling.update(selected_utility)
        add_constraint(utility_ceiling, upper=0.0)
        add_constraint(
            {v_index(s): 1.0, y_index(s): -utility_bound}, upper=0.0
        )
        lower_envelope = {
            v_index(s): 1.0,
            y_index(s): -utility_bound,
        }
        lower_envelope.update(selected_utility)
        add_constraint(lower_envelope, lower=-utility_bound)

    matrix = coo_matrix(
        (data, (rows, columns)),
        shape=(len(lower_bounds), variable_count),
    ).tocsr()
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:n] = 1
    options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = time_limit

    variable_upper = np.ones(variable_count, dtype=float)
    variable_upper[v_offset:] = utility_bound
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower_bounds),
            np.asarray(upper_bounds),
        ),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"scenario MILP did not solve to optimality: {result.message}"
        )

    pool = mask_from_indices(np.flatnonzero(result.x[:n] > 0.5).tolist())
    value = _score_pool_validated(
        pool, scenarios, scenario_weights, utilities, cleared
    )
    return value, pool, result


def condition_on_count(Z, weights, pool, observed_r):
    """Condition weighted scenarios on the exact observed pool count."""

    scenarios = np.asarray(Z)
    if scenarios.ndim != 2 or scenarios.shape[0] == 0:
        raise ValueError("Z must be a nonempty two-dimensional scenario matrix")
    dummy_utilities = np.zeros(scenarios.shape[1], dtype=float)
    scenarios, scenario_weights, _ = _validated_scenarios(
        scenarios, weights, dummy_utilities
    )
    n = scenarios.shape[1]
    pool = _validated_mask(pool, n, "pool")
    observed_r = _as_integer(observed_r, "observed count")
    if not 0 <= observed_r <= pool.bit_count():
        raise ValueError("observed count is outside the pool size")

    indices = indices_from_mask(pool, n)
    realized_counts = scenarios[:, indices].sum(axis=1)
    keep = realized_counts == observed_r
    posterior_mass = float(scenario_weights[keep].sum())
    if posterior_mass <= 0.0:
        raise ValueError("observed count has no support in the scenarios")
    return scenarios[keep], scenario_weights[keep] / posterior_mass


__all__ = [
    "exact_prior_scenarios",
    "score_pool_scenarios",
    "brute_best_pool_scenarios",
    "milp_best_pool_scenarios",
    "condition_on_count",
]
