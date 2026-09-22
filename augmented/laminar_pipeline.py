"""End-to-end particle/MILP/laminar-rollout experiment for n=40.

This is an experimental pipeline, not a production policy.  It makes the
interfaces in notebook 22 concrete:

1. posterior particles select a root pool through the scenario MILP;
2. an exact count is observed;
3. the count induces a conditioned laminar atom;
4. one-step rollout continues inside a fixed hierarchy with exact branch
   probabilities (local blocks contain at most ``G`` people).

Two controls use the same conditioned-particle sampler: flat myopic greedy
multiplies posterior marginals, while myopic MILP keeps joint scenarios but
does no rollout.
"""

from functools import lru_cache

import numpy as np

from augmented.core import all_pools, indices_from_mask, mask_from_indices
from augmented.scenario_milp import milp_best_pool_scenarios


def _validated_arrays(p, u):
    probabilities = np.asarray(p, dtype=float)
    utilities = np.asarray(u, dtype=float)
    if probabilities.ndim != 1 or utilities.shape != probabilities.shape:
        raise ValueError("p and u must be equally sized one-dimensional arrays")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("p must contain finite probabilities in [0, 1]")
    if not np.all(np.isfinite(utilities)) or np.any(utilities < 0.0):
        raise ValueError("u must contain finite nonnegative utilities")
    return probabilities, utilities


def _add_balanced_tree(members, pools):
    if not members:
        return
    pools.add(mask_from_indices(members))
    if len(members) > 1:
        middle = len(members) // 2
        _add_balanced_tree(members[:middle], pools)
        _add_balanced_tree(members[middle:], pools)


def library_after_root(root_pool, p, u, G):
    """Build disjoint roots and balanced descendants around a MILP root."""

    probabilities, utilities = _validated_arrays(p, u)
    n = len(probabilities)
    root_pool = int(root_pool)
    if root_pool <= 0 or root_pool >= (1 << n):
        raise ValueError("root_pool is empty or outside the population")
    if root_pool.bit_count() > int(G):
        raise ValueError("root_pool exceeds G")

    root_members = indices_from_mask(root_pool, n)
    remaining = [i for i in range(n) if not (root_pool >> i) & 1]
    remaining.sort(
        key=lambda i: (-(1.0 - probabilities[i]) * utilities[i], i)
    )
    roots = [root_pool]
    for start in range(0, len(remaining), int(G)):
        roots.append(mask_from_indices(remaining[start : start + int(G)]))

    pools = set()
    for root in roots:
        _add_balanced_tree(indices_from_mask(root, n), pools)
    return tuple(roots), tuple(sorted(pools))


def hierarchy_from_history(history):
    """Construct the immediate-parent map for a known-small laminar history."""

    masks = [int(pool) for pool, _ in history]
    hierarchy = {}
    for parent in masks:
        strict = [
            child for child in masks
            if child != parent and child & parent == child
        ]
        children = []
        for child in strict:
            if not any(
                child != middle
                and child & middle == child
                and middle != parent
                and middle & parent == middle
                for middle in strict
            ):
                children.append(child)
        hierarchy[parent] = tuple(sorted(children))
    return hierarchy


class ExactBlockRollout:
    """Exact greedy/rollout policy on a forest of disjoint small roots."""

    def __init__(self, p, u, roots, library, horizon):
        probabilities, utilities = _validated_arrays(p, u)
        self.p = probabilities
        self.u = utilities
        self.n = len(probabilities)
        self.roots = tuple(int(root) for root in roots)
        self.library = tuple(int(pool) for pool in library)
        self.horizon = int(horizon)
        if self.horizon < 0:
            raise ValueError("horizon must be nonnegative")

        covered = 0
        self.members = []
        self.weights = []
        self.full_state = []
        for root in self.roots:
            if root <= 0 or root & covered:
                raise ValueError("roots must be nonempty and disjoint")
            covered |= root
            members = indices_from_mask(root, self.n)
            self.members.append(members)
            local_ids = np.arange(1 << len(members), dtype=np.uint64)[:, None]
            bit_ids = np.arange(len(members), dtype=np.uint64)[None, :]
            scenarios = ((local_ids >> bit_ids) & 1).astype(np.int8)
            probs = probabilities[members]
            weights = np.prod(
                np.where(scenarios == 1, probs, 1.0 - probs), axis=1
            )
            self.weights.append(weights)
            self.full_state.append((1 << len(weights)) - 1)
        if covered != (1 << self.n) - 1:
            raise ValueError("roots must partition the population")
        self.full_state = tuple(self.full_state)

        self.action_data = {}
        for pool in self.library:
            owners = [j for j, root in enumerate(self.roots) if pool & root == pool]
            if len(owners) != 1:
                raise ValueError("every library pool must lie in exactly one root")
            block = owners[0]
            local_members = self.members[block]
            selected_positions = [
                position for position, person in enumerate(local_members)
                if (pool >> person) & 1
            ]
            world_count = 1 << len(local_members)
            masks = []
            for result in range(len(selected_positions) + 1):
                outcome_mask = 0
                for world in range(world_count):
                    count = sum((world >> position) & 1 for position in selected_positions)
                    if count == result:
                        outcome_mask |= 1 << world
                masks.append(outcome_mask)
            self.action_data[pool] = (block, tuple(masks))

        self._mass_cache = {}
        self._base_cache = {}
        self._base_action_cache = {}
        self._rollout_cache = {}
        self._rollout_action_cache = {}

    def _mass(self, block, worlds):
        key = (block, worlds)
        if key in self._mass_cache:
            return self._mass_cache[key]
        total = 0.0
        rest = worlds
        weights = self.weights[block]
        while rest:
            bit = rest & -rest
            total += weights[bit.bit_length() - 1]
            rest &= rest - 1
        self._mass_cache[key] = total
        return total

    def branches(self, state, cleared, pool):
        block, outcome_masks = self.action_data[int(pool)]
        current = state[block]
        total = self._mass(block, current)
        reward_if_clean = float(sum(
            self.u[i] for i in indices_from_mask(pool & ~cleared, self.n)
        ))
        branches = []
        for result, outcome_mask in enumerate(outcome_masks):
            child = current & outcome_mask
            mass = self._mass(block, child)
            if mass <= 0.0:
                continue
            child_state = list(state)
            child_state[block] = child
            child_cleared = cleared | pool if result == 0 else cleared
            branches.append((
                result,
                mass / total,
                tuple(child_state),
                child_cleared,
                reward_if_clean if result == 0 else 0.0,
            ))
        return tuple(branches)

    def condition(self, state, pool, observed_count):
        for result, _, child, _, _ in self.branches(state, 0, pool):
            if result == int(observed_count):
                return child
        raise ValueError("observed count has no mass in this block posterior")

    def _base(self, step, state, cleared):
        key = (step, state, cleared)
        if key in self._base_cache:
            return self._base_cache[key]
        if step == self.horizon:
            return 0.0
        scored = []
        for pool in self.library:
            branches = self.branches(state, cleared, pool)
            immediate = sum(prob * reward for _, prob, _, _, reward in branches)
            scored.append((immediate, -pool, pool, branches))
        _, _, pool, branches = max(scored)
        value = sum(
            probability * (reward + self._base(step + 1, child, child_cleared))
            for _, probability, child, child_cleared, reward in branches
        )
        self._base_action_cache[key] = pool
        self._base_cache[key] = value
        return value

    def _rollout(self, step, state, cleared):
        key = (step, state, cleared)
        if key in self._rollout_cache:
            return self._rollout_cache[key]
        if step == self.horizon:
            return 0.0
        scored = []
        for pool in self.library:
            branches = self.branches(state, cleared, pool)
            q_base = sum(
                probability * (
                    reward + self._base(step + 1, child, child_cleared)
                )
                for _, probability, child, child_cleared, reward in branches
            )
            scored.append((q_base, -pool, pool, branches))
        _, _, pool, branches = max(scored)
        value = sum(
            probability * (
                reward + self._rollout(step + 1, child, child_cleared)
            )
            for _, probability, child, child_cleared, reward in branches
        )
        self._rollout_action_cache[key] = pool
        self._rollout_cache[key] = value
        return value

    def rollout_value(self, state, cleared=0):
        return float(self._rollout(0, tuple(state), int(cleared)))

    def base_value(self, state, cleared=0):
        return float(self._base(0, tuple(state), int(cleared)))

    def base_action(self, step, state, cleared):
        key = (int(step), tuple(state), int(cleared))
        self._base(*key)
        return self._base_action_cache[key]

    def rollout_action(self, step, state, cleared):
        key = (int(step), tuple(state), int(cleared))
        self._rollout(*key)
        return self._rollout_action_cache[key]

    def simulate(self, state, cleared, latent_mask, mode="rollout"):
        if mode not in ("rollout", "base"):
            raise ValueError("mode must be 'rollout' or 'base'")
        state = tuple(state)
        cleared = int(cleared)
        history = []
        for step in range(self.horizon):
            if mode == "rollout":
                pool = self.rollout_action(step, state, cleared)
            else:
                pool = self.base_action(step, state, cleared)
            result = (pool & int(latent_mask)).bit_count()
            branches = self.branches(state, cleared, pool)
            matched = [branch for branch in branches if branch[0] == result]
            if not matched:
                raise ValueError("latent profile is inconsistent with rollout state")
            _, _, state, cleared, _ = matched[0]
            history.append((pool, result))
        utility = float(sum(
            self.u[i] for i in indices_from_mask(cleared, self.n)
        ))
        return tuple(history), cleared, utility


def _history_seed(base_seed, history):
    value = int(base_seed) & ((1 << 63) - 1)
    for pool, result in history:
        value = (
            value * 6364136223846793005
            + int(pool) * 1442695040888963407
            + int(result) + 1
        ) & ((1 << 63) - 1)
    return value


def conditioned_particles(p, history, sample_count, seed, max_draws=2_000_000):
    """Independent-prior rejection sampler conditioned on exact counts."""

    probabilities = np.asarray(p, dtype=float)
    n = len(probabilities)
    rng = np.random.default_rng(_history_seed(seed, history))
    accepted = []
    draws = 0
    while sum(len(batch) for batch in accepted) < int(sample_count):
        batch_size = min(8192, max_draws - draws)
        if batch_size <= 0:
            raise RuntimeError("conditioned particle rejection budget exhausted")
        batch = (rng.random((batch_size, n)) < probabilities).astype(np.int8)
        keep = np.ones(batch_size, dtype=bool)
        for pool, result in history:
            members = indices_from_mask(int(pool), n)
            keep &= batch[:, members].sum(axis=1) == int(result)
        if np.any(keep):
            accepted.append(batch[keep])
        draws += batch_size
    return np.concatenate(accepted, axis=0)[: int(sample_count)]


@lru_cache(maxsize=None)
def _all_small_pools(n, G):
    return tuple(all_pools(n, G, include_empty=False))


def _flat_independence_pool(marginals, u, G, cleared):
    n = len(marginals)
    best = (0.0, 0)
    for pool in _all_small_pools(n, int(G)):
        members = indices_from_mask(pool, n)
        probability_clean = float(np.prod(1.0 - marginals[members]))
        reward = sum(u[i] for i in members if not (cleared >> i) & 1)
        candidate = (probability_clean * reward, -pool)
        if candidate > best:
            best = candidate
    return -best[1]


class ParticleMyopicPolicy:
    """Cached flat-independence or scenario-MILP policy tree."""

    def __init__(self, p, u, B, G, sample_count=100, seed=0, method="milp"):
        self.p, self.u = _validated_arrays(p, u)
        self.n = len(self.p)
        self.B = int(B)
        self.G = int(G)
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        if method not in ("milp", "flat"):
            raise ValueError("method must be 'milp' or 'flat'")
        self.method = method
        self._action_cache = {}
        self.solve_records = []

    def action(self, history, cleared):
        key = (tuple(history), int(cleared))
        if key in self._action_cache:
            return self._action_cache[key]
        particles = conditioned_particles(
            self.p, history, self.sample_count, self.seed
        )
        if self.method == "flat":
            pool = _flat_independence_pool(
                particles.mean(axis=0), self.u, self.G, int(cleared)
            )
            self.solve_records.append({
                "history_len": len(history), "pool": pool,
                "empirical_value": float("nan"), "mip_gap": float("nan"),
            })
        else:
            weights = np.full(self.sample_count, 1.0 / self.sample_count)
            value, pool, result = milp_best_pool_scenarios(
                particles, weights, self.u, self.G, cleared=int(cleared),
                time_limit=30.0,
            )
            self.solve_records.append({
                "history_len": len(history), "pool": pool,
                "empirical_value": value,
                "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            })
        self._action_cache[key] = pool
        return pool

    def simulate(self, latent_mask):
        history = ()
        cleared = 0
        for _ in range(self.B):
            pool = self.action(history, cleared)
            result = (pool & int(latent_mask)).bit_count()
            history = history + ((pool, result),)
            if result == 0:
                cleared |= pool
        utility = float(sum(
            self.u[i] for i in indices_from_mask(cleared, self.n)
        ))
        return history, cleared, utility


__all__ = [
    "ExactBlockRollout",
    "ParticleMyopicPolicy",
    "conditioned_particles",
    "hierarchy_from_history",
    "library_after_root",
]
