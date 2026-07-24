"""Exact posterior marginals for a supplied laminar hierarchy.

The notebook prototype discovered parent/child relations by comparing every
candidate subset inside nested loops.  This module deliberately does not do
that.  Callers supply the forest as ``parent_pool -> immediate_children``;
the code validates the supplied structure and then reduces the observations
to disjoint residual atoms.

Pools are integer bitmasks, as everywhere else in :mod:`augmented`.  A typical
call is::

    history = ((0b1111, 2), (0b0011, 1), (0b1100, 1))
    hierarchy = {0b1111: (0b0011, 0b1100), 0b0011: (), 0b1100: ()}
    posterior, atoms = laminar_forest_marginals(p, history, hierarchy)

The validation pass is quadratic in the number of observed pools.  It checks
the hierarchy; it does not perform the cubic hierarchy-discovery pass from
the notebook.  Once validated, atom construction is linear in the forest and
the Poisson-binomial messages cost ``O(sum_D |D| * c_D)``.
"""

from numbers import Integral
from typing import Mapping, NamedTuple

import numpy as np

from augmented.core import indices_from_mask


class LaminarAtom(NamedTuple):
    """A residual atom and its count constraint.

    ``mask`` is ``source_pool`` minus all of its immediate children.
    ``count`` is the observed count of ``source_pool`` minus the counts of
    those children.
    """

    mask: int
    count: int
    source_pool: int


def _as_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validated_probabilities(p):
    probs = np.asarray(p, dtype=float)
    if probs.ndim != 1:
        raise ValueError("p must be a one-dimensional sequence")
    if not np.all(np.isfinite(probs)):
        raise ValueError("p must contain only finite probabilities")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("every prior probability must lie in [0, 1]")
    return probs


def conditional_bernoulli_marginals(probs, count):
    """Marginals of independent Bernoullis conditional on their exact sum.

    Parameters
    ----------
    probs : sequence of float
        Independent Bernoulli success probabilities.
    count : int
        Condition on the sum of the Bernoullis being exactly ``count``.

    Returns
    -------
    numpy.ndarray
        ``P(Z_i=1 | sum(Z)=count)`` for every input probability.

    Notes
    -----
    Forward and backward Poisson-binomial messages are truncated at ``count``.
    This gives all marginals in ``O(m * count)`` time and memory for an atom
    of size ``m``.  A zero-probability conditioning event raises ``ValueError``
    rather than silently returning the prior.
    """

    probabilities = _validated_probabilities(probs)
    m = len(probabilities)
    count = _as_integer(count, "count")
    if not 0 <= count <= m:
        raise ValueError("count is outside the atom size")

    min_possible = int(np.count_nonzero(probabilities == 1.0))
    max_possible = m - int(np.count_nonzero(probabilities == 0.0))
    if not min_possible <= count <= max_possible:
        raise ValueError("the conditioned count has zero probability")
    if count == 0:
        return np.zeros(m, dtype=float)
    if count == m:
        return np.ones(m, dtype=float)

    forward = np.zeros((m + 1, count + 1), dtype=float)
    forward[0, 0] = 1.0
    for i, probability in enumerate(probabilities):
        forward[i + 1] += forward[i] * (1.0 - probability)
        forward[i + 1, 1:] += forward[i, :-1] * probability

    backward = np.zeros((m + 1, count + 1), dtype=float)
    backward[m, 0] = 1.0
    for i in range(m - 1, -1, -1):
        probability = probabilities[i]
        backward[i] += backward[i + 1] * (1.0 - probability)
        backward[i, 1:] += backward[i + 1, :-1] * probability

    denominator = forward[m, count]
    if denominator <= 1e-300:
        raise ValueError(
            "the conditioned count has numerically zero probability"
        )

    target = count - 1
    marginals = np.empty(m, dtype=float)
    for i, probability in enumerate(probabilities):
        ways_without_i = np.dot(
            forward[i, : target + 1],
            backward[i + 1, target::-1],
        )
        marginals[i] = probability * ways_without_i / denominator
    return np.clip(marginals, 0.0, 1.0)


def _poisson_binomial(probs):
    """Probability mass function of a sum of independent Bernoullis."""

    pmf = np.array([1.0], dtype=float)
    for probability in probs:
        pmf = np.convolve(pmf, np.array([1.0 - probability, probability]))
    return pmf


def laminar_pool_pmf(p, atoms, pool_mask):
    """Compute ``P(R_pool | history)`` from conditioned laminar atoms.

    Different atoms are independent, but individuals inside one atom are
    generally dependent because their sum is fixed.  For each intersection
    ``pool ∩ atom`` this function computes the conditional count distribution
    and then convolves the independent atom contributions.  Individuals
    outside every observed root keep their independent prior.

    The candidate pool need not belong to the hierarchy.  Its distribution is
    still available for one decision; observing a non-compatible pool would,
    however, destroy laminar closure for later updates.
    """

    priors = _validated_probabilities(p)
    n = len(priors)
    pool = _as_integer(pool_mask, "pool mask")
    if pool < 0 or pool >= (1 << n):
        raise ValueError("pool mask is outside the prior universe")

    normalized_atoms = []
    covered = 0
    for raw_atom in atoms:
        try:
            atom_mask, atom_count, source_pool = raw_atom
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "atoms must contain (mask, count, source_pool)"
            ) from exc
        atom_mask = _as_integer(atom_mask, "atom mask")
        atom_count = _as_integer(atom_count, "atom count")
        source_pool = _as_integer(source_pool, "source pool mask")
        if atom_mask <= 0 or atom_mask >= (1 << n):
            raise ValueError("atom mask is empty or outside the prior universe")
        if covered & atom_mask:
            raise ValueError("atoms must be disjoint")
        if not 0 <= atom_count <= atom_mask.bit_count():
            raise ValueError("atom count is outside the atom size")
        covered |= atom_mask
        normalized_atoms.append(LaminarAtom(atom_mask, atom_count, source_pool))

    result = np.array([1.0], dtype=float)
    for atom in normalized_atoms:
        intersection = pool & atom.mask
        if not intersection:
            continue
        inside = indices_from_mask(intersection, n)
        outside = indices_from_mask(atom.mask & ~intersection, n)
        inside_pmf = _poisson_binomial(priors[inside])
        outside_pmf = _poisson_binomial(priors[outside])
        denominator = _poisson_binomial(
            priors[indices_from_mask(atom.mask, n)]
        )[atom.count]
        if denominator <= 1e-300:
            raise ValueError("an atom count has zero probability under the prior")

        contribution = np.zeros(len(inside) + 1, dtype=float)
        for inside_count in range(len(contribution)):
            outside_count = atom.count - inside_count
            if 0 <= outside_count < len(outside_pmf):
                contribution[inside_count] = (
                    inside_pmf[inside_count]
                    * outside_pmf[outside_count]
                    / denominator
                )
        result = np.convolve(result, contribution)

    for index in indices_from_mask(pool & ~covered, n):
        result = np.convolve(
            result, np.array([1.0 - priors[index], priors[index]])
        )

    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("the requested pool has no finite posterior mass")
    result /= total
    if len(result) != pool.bit_count() + 1:
        raise AssertionError("atom contributions must cover every pool member")
    return result


def _validated_forest(history, hierarchy, n):
    n = _as_integer(n, "n")
    if n < 0:
        raise ValueError("n must be nonnegative")
    universe = (1 << n) - 1

    counts = {}
    for entry in history:
        if len(entry) != 2:
            raise ValueError("each history entry must be (pool_mask, count)")
        raw_mask, raw_count = entry
        mask = _as_integer(raw_mask, "pool mask")
        count = _as_integer(raw_count, "pool count")
        if mask <= 0 or mask & ~universe:
            raise ValueError("history contains an empty or out-of-universe pool")
        if not 0 <= count <= mask.bit_count():
            raise ValueError("a pool count is outside the pool size")
        if mask in counts and counts[mask] != count:
            raise ValueError("the same pool has incompatible counts")
        counts[mask] = count

    if not isinstance(hierarchy, Mapping):
        raise ValueError("hierarchy must map each pool to its immediate children")

    raw_keys = set(hierarchy)
    if any(isinstance(mask, bool) or not isinstance(mask, Integral)
           for mask in raw_keys):
        raise ValueError("hierarchy pool masks must be integers")
    hierarchy_keys = {int(mask) for mask in raw_keys}
    if hierarchy_keys != set(counts):
        raise ValueError("hierarchy must contain exactly the observed pools")

    children = {}
    parent_of = {}
    for raw_parent, raw_children in hierarchy.items():
        parent = int(raw_parent)
        try:
            normalized_children = tuple(
                _as_integer(child, "child pool mask") for child in raw_children
            )
        except TypeError as exc:
            raise ValueError("hierarchy children must be iterable") from exc
        if len(set(normalized_children)) != len(normalized_children):
            raise ValueError("a hierarchy node lists the same child twice")
        for child in normalized_children:
            if child not in counts:
                raise ValueError("hierarchy references an unobserved child pool")
            if child == parent or child & parent != child:
                raise ValueError("every child must be a strict subset of its parent")
            if child in parent_of:
                raise ValueError("a hierarchy node cannot have two parents")
            parent_of[child] = parent
        children[parent] = normalized_children

    masks = tuple(counts)
    for i, first in enumerate(masks):
        for second in masks[i + 1:]:
            intersection = first & second
            if intersection and intersection != first and intersection != second:
                raise ValueError("history is not laminar")

    ancestors = {}
    for mask in masks:
        chain = set()
        current = mask
        while current in parent_of:
            current = parent_of[current]
            if current in chain:
                raise ValueError("hierarchy contains a cycle")
            chain.add(current)
        ancestors[mask] = chain

    # The supplied edges must be the transitive reduction of set inclusion.
    # Checking the ancestor relation also catches a missing intermediate node.
    for smaller in masks:
        expected_ancestors = {
            larger for larger in masks
            if smaller != larger and smaller & larger == smaller
        }
        if ancestors[smaller] != expected_ancestors:
            raise ValueError(
                "hierarchy does not match the observed pools' inclusion order"
            )

    return counts, children


def _atoms_from_validated_forest(counts, children):
    atoms = []
    for parent, parent_count in counts.items():
        covered = 0
        child_count = 0
        for child in children[parent]:
            covered |= child
            child_count += counts[child]

        atom_mask = parent & ~covered
        atom_count = parent_count - child_count
        if atom_count < 0 or atom_count > atom_mask.bit_count():
            raise ValueError("parent and child counts are incompatible")
        if atom_mask:
            atoms.append(LaminarAtom(atom_mask, atom_count, parent))
        elif atom_count:
            raise ValueError("an empty residual atom has a positive count")
    return tuple(atoms)


def laminar_atoms(history, hierarchy, n):
    """Return the disjoint residual atoms of a supplied laminar forest.

    ``hierarchy`` maps every observed pool to its immediate observed children.
    Leaves map to an empty iterable.  Roots therefore need no sentinel parent.
    """

    counts, children = _validated_forest(history, hierarchy, n)
    return _atoms_from_validated_forest(counts, children)


def laminar_forest_marginals(p, history, hierarchy):
    """Compute exact posterior marginals under laminar count observations.

    Individuals outside every observed root retain their prior marginals.
    The returned atoms expose the exact disjoint factorization used by the
    calculation.
    """

    priors = _validated_probabilities(p)
    counts, children = _validated_forest(history, hierarchy, len(priors))
    atoms = _atoms_from_validated_forest(counts, children)

    posterior = priors.copy()
    seen = 0
    for atom in atoms:
        if seen & atom.mask:
            raise AssertionError("validated laminar atoms must be disjoint")
        seen |= atom.mask
        indices = indices_from_mask(atom.mask, len(priors))
        posterior[indices] = conditional_bernoulli_marginals(
            priors[indices], atom.count
        )
    return posterior, atoms


__all__ = [
    "LaminarAtom",
    "conditional_bernoulli_marginals",
    "laminar_atoms",
    "laminar_forest_marginals",
    "laminar_pool_pmf",
]
