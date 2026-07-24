"""Reproducible experiments requested after notebook 22.

Outputs live in ``augmented/data/laminar_week`` and are consumed by the
notebook-22 builder.  Every CSV is deterministic under the recorded seeds.

Examples
--------
Run the complete suite (the exact atlas is the expensive stage)::

    python -m augmented.experiments_laminar_week all --workers 4

Run only one stage::

    python -m augmented.experiments_laminar_week independence
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from augmented.bayesian import _poisson_binomial_pmf, exact_pool_pmf
from augmented.core import all_pools, indices_from_mask, mask_from_indices
from augmented.independence_gap import tv_distance
from augmented.laminar_benchmarks import four_quantities, laminar_ratio
from augmented.laminar_inference import (
    laminar_forest_marginals,
    laminar_pool_pmf,
)
from augmented.laminar_pipeline import (
    ExactBlockRollout,
    ParticleMyopicPolicy,
    hierarchy_from_history,
    library_after_root,
)
from augmented.scenario_milp import (
    brute_best_pool_scenarios,
    exact_prior_scenarios,
    milp_best_pool_scenarios,
    score_pool_scenarios,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "laminar_week"
FIGURE_DIR = HERE / "notebooks" / "figures" / "22_laminar_week"
RATIO_COLUMNS = (
    "ratio_laminar_opt",
    "ratio_greedy_laminar",
    "ratio_static_opt",
    "ratio_greedy_static",
)


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _process_map(function, tasks, workers):
    """Use processes when allowed, with a deterministic sandbox fallback."""

    if int(workers) <= 1:
        return [function(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            return list(executor.map(function, tasks, chunksize=1))
    except (OSError, PermissionError):
        return [function(task) for task in tasks]


def _calibrate_mean(raw, target):
    raw = np.clip(np.asarray(raw, dtype=float), 1e-7, 1.0 - 1e-7)
    logits = np.log(raw / (1.0 - raw))
    lower, upper = -30.0, 30.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        shifted = 1.0 / (1.0 + np.exp(-(logits + middle)))
        if shifted.mean() < target:
            lower = middle
        else:
            upper = middle
    return 1.0 / (1.0 + np.exp(-(logits + (lower + upper) / 2.0)))


def _draw_population(base_p, rate_mode, utility_mode, n, seed):
    rng = np.random.default_rng(int(seed))
    if rate_mode == "homogeneous":
        p = np.full(int(n), float(base_p))
    elif rate_mode == "beta_bimodal":
        p = _calibrate_mean(rng.beta(0.40, 0.40, int(n)), float(base_p))
    else:
        raise ValueError(rate_mode)

    if utility_mode == "flat":
        u = np.ones(int(n))
    elif utility_mode == "log_uniform":
        u = np.exp(rng.uniform(np.log(0.25), np.log(4.0), int(n)))
        u /= u.mean()
    else:
        raise ValueError(utility_mode)
    return p, u


def _atlas_worker(task):
    p = np.asarray(task["p"], dtype=float)
    u = np.asarray(task["u"], dtype=float)
    values = four_quantities(p, u, task["B"], task["G"])
    return {
        key: value for key, value in values.items()
        if key not in ("best_library", "practical_library")
    } | {
        "best_library": json.dumps(list(values["best_library"])),
        "practical_library": json.dumps(list(values["practical_library"])),
    }


def run_atlas(reps=3, workers=4):
    """Full v1 grid: 18 prevalences, 12 (n,B,G), 4 population regimes."""

    _ensure_dirs()
    base_grid = np.round(np.arange(0.05, 0.901, 0.05), 2)
    specifications = []
    unique_tasks = {}
    instance = 0
    for base_p in base_grid:
        for n in (4, 5, 6):
            for B in (2, 3):
                for G in (2, 3):
                    for rate_mode in ("homogeneous", "beta_bimodal"):
                        for utility_mode in ("flat", "log_uniform"):
                            for replicate in range(int(reps)):
                                seed = (
                                    22000000
                                    + int(round(base_p * 100)) * 10000
                                    + n * 1000 + B * 100 + G * 10
                                    + replicate
                                    + (3 if rate_mode == "beta_bimodal" else 0)
                                    + (7 if utility_mode == "log_uniform" else 0)
                                )
                                p, u = _draw_population(
                                    base_p, rate_mode, utility_mode, n, seed
                                )
                                signature = json.dumps({
                                    "p": np.round(p, 14).tolist(),
                                    "u": np.round(u, 14).tolist(),
                                    "B": B,
                                    "G": G,
                                }, sort_keys=True)
                                unique_tasks.setdefault(signature, {
                                    "p": p.tolist(), "u": u.tolist(),
                                    "B": B, "G": G,
                                })
                                specifications.append({
                                    "instance": instance,
                                    "seed": seed,
                                    "base_p": base_p,
                                    "n": n,
                                    "B": B,
                                    "G": G,
                                    "rate_mode": rate_mode,
                                    "utility_mode": utility_mode,
                                    "replicate": replicate,
                                    "p": json.dumps(p.tolist()),
                                    "u": json.dumps(u.tolist()),
                                    "signature": signature,
                                })
                                instance += 1

    started = time.perf_counter()
    signatures = list(unique_tasks)
    tasks = [unique_tasks[signature] for signature in signatures]
    if int(workers) <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            results.append(_atlas_worker(task))
            if index % 50 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"atlas {index}/{len(tasks)} ({elapsed:.1f}s)", flush=True)
    else:
        results = _process_map(_atlas_worker, tasks, workers)
    by_signature = dict(zip(signatures, results))

    rows = []
    for specification in specifications:
        row = dict(specification)
        signature = row.pop("signature")
        row.update(by_signature[signature])
        rows.append(row)
    rows.sort(key=lambda row: row["instance"])
    atlas_path = DATA_DIR / "atlas_instances.csv"
    _write_rows(atlas_path, rows)

    frame = pd.DataFrame(rows)
    group_columns = [
        "base_p", "n", "B", "G", "rate_mode", "utility_mode"
    ]
    summary_rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        summary = dict(zip(group_columns, keys))
        summary["instances"] = len(group)
        for ratio in RATIO_COLUMNS:
            values = group[ratio].to_numpy(float)
            summary[f"{ratio}_min"] = float(values.min())
            summary[f"{ratio}_max"] = float(values.max())
            summary[f"{ratio}_mean"] = float(values.mean())
            summary[f"{ratio}_median"] = float(np.median(values))
            summary[f"{ratio}_worst_instance"] = int(
                group.iloc[int(np.argmin(values))]["instance"]
            )
            summary[f"{ratio}_best_instance"] = int(
                group.iloc[int(np.argmax(values))]["instance"]
            )
        summary_rows.append(summary)
    summary_path = DATA_DIR / "atlas_cells.csv"
    _write_rows(summary_path, summary_rows)

    configurations = sorted(
        {(row["n"], row["B"], row["G"]) for row in rows}
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    titles = {
        "ratio_laminar_opt": r"$V^{\mathcal{L}}/V^*$",
        "ratio_greedy_laminar": r"$V^{greedy}_{\mathcal{L}}/V^{\mathcal{L}}$",
        "ratio_static_opt": r"$V^{static}_{bin}/V^*$",
        "ratio_greedy_static": r"$V^{greedy}_{\mathcal{L}}/V^{static}_{bin}$",
    }
    for ax, ratio in zip(axes.flat, RATIO_COLUMNS):
        matrix = np.empty((len(base_grid), len(configurations)))
        for i, base_p in enumerate(base_grid):
            for j, (n, B, G) in enumerate(configurations):
                selected = frame[
                    (frame.base_p == base_p)
                    & (frame.n == n) & (frame.B == B) & (frame.G == G)
                ]
                matrix[i, j] = selected[ratio].min()
        image = ax.imshow(matrix, aspect="auto", vmin=0.75, vmax=1.10,
                          cmap="RdYlGn", origin="lower")
        ax.set_title(titles[ratio] + " · peor régimen/réplica")
        ax.set_xticks(range(len(configurations)))
        ax.set_xticklabels(
            [f"{n}/{B}/{G}" for n, B, G in configurations], rotation=45,
            ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(base_grid)))
        ax.set_yticklabels([f"{p:.2f}" for p in base_grid], fontsize=8)
        ax.set_xlabel("n/B/G")
        ax.set_ylabel("prevalencia media")
        fig.colorbar(image, ax=ax, shrink=0.8)
    figure_path = FIGURE_DIR / "atlas_ratio_heatmaps.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    return {
        "instances": len(rows),
        "unique_computations": len(tasks),
        "seconds": time.perf_counter() - started,
        "atlas": str(atlas_path),
        "summary": str(summary_path),
        "figure": str(figure_path),
    }


def _adversarial_worker(seed_row):
    rng = np.random.default_rng(23000 + int(seed_row["instance"]))
    p = np.asarray(json.loads(seed_row["p"]), dtype=float)
    u = np.asarray(json.loads(seed_row["u"]), dtype=float)
    n, B, G = int(seed_row["n"]), int(seed_row["B"]), int(seed_row["G"])
    incumbent = float(seed_row["ratio_laminar_opt"])
    trajectory = [{
        "region": seed_row["region"], "iteration": 0,
        "candidate_ratio": incumbent, "incumbent_ratio": incumbent,
        "accepted": 1, "coordinate": "seed", "p": json.dumps(p.tolist()),
        "u": json.dumps(u.tolist()), "n": n, "B": B, "G": G,
        "seed_instance": int(seed_row["instance"]),
    }]
    for iteration in range(1, 25):
        candidate_p = p.copy()
        candidate_u = u.copy()
        if iteration % 2:
            index = int(rng.integers(n))
            step = 0.12 * (0.94 ** (iteration - 1))
            candidate_p[index] = np.clip(
                candidate_p[index] + rng.choice((-step, step)), 0.01, 0.99
            )
            coordinate = f"p[{index}]"
        else:
            index = int(rng.integers(n))
            step = 0.45 * (0.94 ** (iteration - 1))
            candidate_u[index] *= math.exp(rng.choice((-step, step)))
            candidate_u /= candidate_u.mean()
            coordinate = f"log_u[{index}]"
        candidate = laminar_ratio(candidate_p, candidate_u, B, G)
        accepted = int(candidate < incumbent - 1e-10)
        if accepted:
            p, u, incumbent = candidate_p, candidate_u, candidate
        trajectory.append({
            "region": seed_row["region"], "iteration": iteration,
            "candidate_ratio": candidate, "incumbent_ratio": incumbent,
            "accepted": accepted, "coordinate": coordinate,
            "p": json.dumps(p.tolist()), "u": json.dumps(u.tolist()),
            "n": n, "B": B, "G": G,
            "seed_instance": int(seed_row["instance"]),
        })
    return trajectory


def run_adversarial(workers=3):
    _ensure_dirs()
    atlas_path = DATA_DIR / "atlas_instances.csv"
    if not atlas_path.exists():
        raise FileNotFoundError("run the atlas stage before adversarial search")
    frame = pd.read_csv(atlas_path)
    frame["region"] = pd.cut(
        frame.base_p, bins=[0.0, 0.20, 0.65, 1.0],
        labels=["low", "middle", "high"], include_lowest=True,
    ).astype(str)
    seeds = []
    for region in ("low", "middle", "high"):
        subset = frame[frame.region == region]
        seeds.append(subset.loc[subset.ratio_laminar_opt.idxmin()].to_dict())
    paths = _process_map(_adversarial_worker, seeds, min(int(workers), 3))
    rows = [row for path in paths for row in path]
    _write_rows(DATA_DIR / "adversarial_trajectories.csv", rows)
    minima = []
    for region in ("low", "middle", "high"):
        candidates = [row for row in rows if row["region"] == region]
        minima.append(min(candidates, key=lambda row: row["incumbent_ratio"]))
    _write_rows(DATA_DIR / "adversarial_minima.csv", minima)

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    for region in ("low", "middle", "high"):
        selected = [row for row in rows if row["region"] == region]
        ax.plot([row["iteration"] for row in selected],
                [row["incumbent_ratio"] for row in selected],
                marker="o", ms=3, label=region)
    ax.set_xlabel("perturbación propuesta")
    ax.set_ylabel(r"incumbente $V^{\mathcal{L}}/V^*$")
    ax.set_title("Búsqueda adversaria desde las peores celdas del atlas")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "adversarial_trajectories.png", dpi=170)
    plt.close(fig)
    return minima


def run_homogeneous_b2(workers=4):
    _ensure_dirs()
    tasks = []
    for p0 in np.round(np.arange(0.05, 0.901, 0.025), 3):
        for n in (4, 5, 6):
            for G in (2, 3):
                for B in (1, 2):
                    tasks.append((p0, n, B, G))

    def compute(task):
        p0, n, B, G = task
        return {
            "p": p0, "n": n, "B": B, "G": G,
            "ratio_laminar_opt": laminar_ratio(
                np.full(n, p0), np.ones(n), B, G
            ),
        }

    # Nested functions cannot be pickled under spawn, so keep this short sweep
    # sequential; most symmetric cases certify equality after one library.
    rows = [compute(task) for task in tasks]
    if max(abs(row["ratio_laminar_opt"] - 1.0)
           for row in rows if row["B"] == 1) > 1e-9:
        raise AssertionError("B=1 must have V^L = V*")
    _write_rows(DATA_DIR / "homogeneous_b2.csv", rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for n in (4, 5, 6):
        for G, style in ((2, "-"), (3, "--")):
            selected = [row for row in rows
                        if row["B"] == 2 and row["n"] == n and row["G"] == G]
            ax.plot([row["p"] for row in selected],
                    [row["ratio_laminar_opt"] for row in selected],
                    linestyle=style, label=f"n={n}, G={G}")
    ax.set_ylim(0.88, 1.005)
    ax.set_xlabel("p homogéneo")
    ax.set_ylabel(r"$V^{\mathcal{L}}/V^*$")
    ax.set_title("Caso especial B=2, utilidades planas")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "homogeneous_b2.png", dpi=170)
    plt.close(fig)
    return rows


def run_independence(reps=80):
    _ensure_dirs()
    rng = np.random.default_rng(2401)
    n = 8
    root = mask_from_indices([0, 1, 2, 3])
    child = mask_from_indices([0, 1])
    other_root = mask_from_indices([4, 5])
    candidates = {
        "disjoint": mask_from_indices([6, 7]),
        "observed_node": root,
        "nested_compatible": mask_from_indices([2, 3]),
        # Crosses both observed roots and contains the two-person child whose
        # total is fixed; the product approximation therefore loses dependence.
        "crossing_nonlaminar": mask_from_indices([0, 1, 4]),
    }
    rows = []
    for replicate in range(int(reps)):
        p = rng.uniform(0.08, 0.82, n)
        latent = sum(
            (1 << i) for i in range(n) if rng.random() < p[i]
        )
        history = (
            (root, (root & latent).bit_count()),
            (child, (child & latent).bit_count()),
            (other_root, (other_root & latent).bit_count()),
        )
        hierarchy = {root: (child,), child: (), other_root: ()}
        marginals, atoms = laminar_forest_marginals(p, history, hierarchy)
        for category, pool in candidates.items():
            exact = np.asarray(exact_pool_pmf(p, history, pool, n))
            product = np.asarray(_poisson_binomial_pmf(
                [marginals[i] for i in indices_from_mask(pool, n)]
            ))
            atom = laminar_pool_pmf(p, atoms, pool)
            rows.append({
                "replicate": replicate,
                "category": category,
                "compatible": int(category != "crossing_nonlaminar"),
                "pool": pool,
                "tv_product": tv_distance(exact, product),
                "tv_atom": tv_distance(exact, atom),
                "gap_clean_product": float(product[0] - exact[0]),
                "exact_pmf": json.dumps(exact.tolist()),
                "product_pmf": json.dumps(product.tolist()),
                "atom_pmf": json.dumps(atom.tolist()),
            })
    if max(row["tv_atom"] for row in rows) > 2e-10:
        raise AssertionError("atom-preserving PMFs must match enumeration")
    _write_rows(DATA_DIR / "independence_gap.csv", rows)

    frame = pd.DataFrame(rows)
    order = list(candidates)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.boxplot(
        [frame[frame.category == category].tv_product for category in order],
        tick_labels=[name.replace("_", "\n") for name in order],
        showfliers=False,
    )
    ax.set_ylabel("distancia de variación total")
    ax.set_title("Posterior real vs producto de marginales")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "independence_gap.png", dpi=170)
    plt.close(fig)
    return frame.groupby("category")[["tv_product", "tv_atom"]].agg(
        ["mean", "max"]
    )


def _milp_sweep_worker(task):
    n, sample_count, replicate = task
    instance_seed = 250000 + n * 1000 + replicate
    rng = np.random.default_rng(instance_seed)
    p = rng.uniform(0.05, 0.72, n)
    u = np.exp(rng.uniform(np.log(0.35), np.log(3.5), n))
    G = 3
    exact_scenarios, exact_weights = exact_prior_scenarios(p)
    exact_value, exact_pool = brute_best_pool_scenarios(
        exact_scenarios, exact_weights, u, G
    )
    # Nested samples: for fixed (n, replicate), every S uses the first S rows
    # of the same random stream.  Differences across S are then a convergence
    # curve for one instance, not a comparison of unrelated populations.
    particle_rng = np.random.default_rng(instance_seed + 900_001)
    particles = (particle_rng.random((sample_count, n)) < p).astype(np.int8)
    weights = np.full(sample_count, 1.0 / sample_count)
    empirical_value, selected_pool, result = milp_best_pool_scenarios(
        particles, weights, u, G, time_limit=60.0
    )
    brute_empirical, _ = brute_best_pool_scenarios(particles, weights, u, G)
    true_selected = score_pool_scenarios(
        selected_pool, exact_scenarios, exact_weights, u
    )
    return {
        "n": n,
        "S": sample_count,
        "replicate": replicate,
        "exact_value": exact_value,
        "true_selected_value": true_selected,
        "value_ratio": true_selected / exact_value,
        "true_regret": exact_value - true_selected,
        "empirical_objective": empirical_value,
        "empirical_bruteforce": brute_empirical,
        "empirical_identity_error": abs(empirical_value - brute_empirical),
        "pool_match": int(selected_pool == exact_pool),
        "selected_pool": selected_pool,
        "exact_pool": exact_pool,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
    }


def run_milp_sweep(reps=4, workers=4):
    _ensure_dirs()
    tasks = [
        (n, sample_count, replicate)
        for n in (6, 8, 10, 12)
        for sample_count in (25, 50, 100, 250, 500)
        for replicate in range(int(reps))
    ]
    if int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(executor.map(_milp_sweep_worker, tasks))
    else:
        rows = [_milp_sweep_worker(task) for task in tasks]
    rows.sort(key=lambda row: (row["n"], row["S"], row["replicate"]))
    if max(row["empirical_identity_error"] for row in rows) > 1e-7:
        raise AssertionError("MILP must match brute force on its particle sample")
    _write_rows(DATA_DIR / "milp_particle_sweep.csv", rows)

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for n in (6, 8, 10, 12):
        selected = frame[frame.n == n].groupby("S")
        axes[0].plot(selected.value_ratio.mean().index,
                     selected.value_ratio.mean().values, marker="o", label=f"n={n}")
        axes[1].plot(selected.pool_match.mean().index,
                     selected.pool_match.mean().values, marker="o", label=f"n={n}")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(0.88, 1.005)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("valor real del pool / óptimo real")
    axes[1].set_ylabel("frecuencia de pool idéntico")
    for ax in axes:
        ax.set_xlabel("partículas S")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("MILP exacto en la muestra; error estadístico fuera de muestra")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "milp_particle_sweep.png", dpi=170)
    plt.close(fig)
    return frame


def run_pipeline(n_eval=250):
    _ensure_dirs()
    seed = 2607
    rng = np.random.default_rng(seed)
    n, B, G, sample_count = 40, 3, 3, 100
    p = rng.uniform(0.08, 0.55, n)
    u = np.exp(rng.uniform(np.log(0.5), np.log(4.0), n))
    truth = (rng.random((int(n_eval), n)) < p).astype(np.int8)
    truth_masks = [sum((1 << i) for i in np.flatnonzero(row)) for row in truth]

    milp_policy = ParticleMyopicPolicy(
        p, u, B, G, sample_count=sample_count, seed=seed, method="milp"
    )
    flat_policy = ParticleMyopicPolicy(
        p, u, B, G, sample_count=sample_count, seed=seed, method="flat"
    )
    root = milp_policy.action((), 0)
    roots, library = library_after_root(root, p, u, G)
    rollout = ExactBlockRollout(p, u, roots, library, horizon=B - 1)

    records = []
    for index, latent in enumerate(truth_masks):
        _, _, flat_utility = flat_policy.simulate(latent)
        _, _, milp_utility = milp_policy.simulate(latent)
        root_count = (root & latent).bit_count()
        state = rollout.condition(rollout.full_state, root, root_count)
        cleared = root if root_count == 0 else 0
        _, _, greedy_laminar_utility = rollout.simulate(
            state, cleared, latent, mode="base"
        )
        _, _, rollout_utility = rollout.simulate(state, cleared, latent)
        records.append({
            "profile": index,
            "flat_independence": flat_utility,
            "myopic_milp": milp_utility,
            "laminar_greedy": greedy_laminar_utility,
            "laminar_rollout": rollout_utility,
            "root_count": root_count,
        })
    _write_rows(DATA_DIR / "pipeline_n40_profiles.csv", records)

    root_members = indices_from_mask(root, n)
    root_pmf = np.asarray(_poisson_binomial_pmf([p[i] for i in root_members]))
    exact_laminar = {"laminar_greedy": 0.0, "laminar_rollout": 0.0}
    root_reward = float(sum(u[i] for i in root_members))
    for root_count, probability in enumerate(root_pmf):
        state = rollout.condition(rollout.full_state, root, root_count)
        cleared = root if root_count == 0 else 0
        immediate = root_reward if root_count == 0 else 0.0
        exact_laminar["laminar_greedy"] += probability * (
            immediate + rollout.base_value(state, cleared)
        )
        exact_laminar["laminar_rollout"] += probability * (
            immediate + rollout.rollout_value(state, cleared)
        )
    if exact_laminar["laminar_rollout"] + 1e-10 < exact_laminar["laminar_greedy"]:
        raise AssertionError("exact rollout must dominate its laminar base")

    summary = []
    for method in ("flat_independence", "myopic_milp", "laminar_greedy",
                   "laminar_rollout"):
        values = np.array([row[method] for row in records])
        summary.append({
            "method": method,
            "n": n, "B": B, "G": G, "S": sample_count,
            "profiles": len(values),
            "mean_utility": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
            "median_utility": float(np.median(values)),
            "exact_expected_utility": exact_laminar.get(method, float("nan")),
            "zero_rate": float(np.mean(values == 0.0)),
            "root_pool": root,
            "root_members": json.dumps(indices_from_mask(root, n)),
            "milp_states_solved": len(milp_policy.solve_records),
            "flat_states_solved": len(flat_policy.solve_records),
        })
    _write_rows(DATA_DIR / "pipeline_n40_summary.csv", summary)

    # One fully audited path: atom PMFs equal the exact local block PMFs, while
    # the product of marginals can differ after an intermediate count.
    latent = next(
        (mask for mask in truth_masks
         if 0 < (root & mask).bit_count() < root.bit_count()),
        truth_masks[0],
    )
    history = [(root, (root & latent).bit_count())]
    state = rollout.condition(rollout.full_state, root, history[0][1])
    cleared = root if history[0][1] == 0 else 0
    trace = []
    for step in range(1, B):
        hierarchy = hierarchy_from_history(history)
        marginals, atoms = laminar_forest_marginals(p, history, hierarchy)
        pool = rollout.rollout_action(step - 1, state, cleared)
        atom_pmf = laminar_pool_pmf(p, atoms, pool)
        product_pmf = np.asarray(_poisson_binomial_pmf(
            [marginals[i] for i in indices_from_mask(pool, n)]
        ))
        block_pmf = np.zeros(pool.bit_count() + 1)
        branches = rollout.branches(state, cleared, pool)
        for result, probability, _, _, _ in branches:
            block_pmf[result] = probability
        if np.max(np.abs(atom_pmf - block_pmf)) > 2e-10:
            raise AssertionError("rollout branch law must equal the atom PMF")
        observed = (pool & latent).bit_count()
        trace.append({
            "step": step + 1,
            "pool": pool,
            "pool_members": json.dumps(indices_from_mask(pool, n)),
            "observed_count": observed,
            "atom_pmf": json.dumps(atom_pmf.tolist()),
            "product_pmf": json.dumps(product_pmf.tolist()),
            "tv_product": tv_distance(atom_pmf, product_pmf),
            "tv_atom_vs_block": tv_distance(atom_pmf, block_pmf),
        })
        matched = [branch for branch in branches if branch[0] == observed][0]
        _, _, state, cleared, _ = matched
        history.append((pool, observed))
    _write_rows(DATA_DIR / "pipeline_n40_trace.csv", trace)

    gaps = [row["mip_gap"] for row in milp_policy.solve_records
            if np.isfinite(row["mip_gap"])]
    diagnostics = [{
        "root_pool": root,
        "root_members": json.dumps(indices_from_mask(root, n)),
        "library_size": len(library),
        "root_count_trace": history[0][1],
        "max_reported_mip_gap": max(gaps) if gaps else float("nan"),
        "milp_states_solved": len(milp_policy.solve_records),
        "flat_states_solved": len(flat_policy.solve_records),
    }]
    _write_rows(DATA_DIR / "pipeline_n40_diagnostics.csv", diagnostics)
    return summary, trace, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("all", "atlas", "adversarial", "homogeneous",
                 "independence", "milp", "pipeline"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--atlas-reps", type=int, default=3)
    parser.add_argument("--milp-reps", type=int, default=4)
    parser.add_argument("--pipeline-eval", type=int, default=250)
    args = parser.parse_args(argv)

    stages = (
        ("atlas", lambda: run_atlas(args.atlas_reps, args.workers)),
        ("adversarial", lambda: run_adversarial(args.workers)),
        ("homogeneous", lambda: run_homogeneous_b2(args.workers)),
        ("independence", run_independence),
        ("milp", lambda: run_milp_sweep(args.milp_reps, args.workers)),
        ("pipeline", lambda: run_pipeline(args.pipeline_eval)),
    )
    for name, function in stages:
        if args.stage in ("all", name):
            started = time.perf_counter()
            result = function()
            print(f"[{name}] {time.perf_counter() - started:.2f}s")
            if isinstance(result, (list, tuple)):
                print(f"{len(result)} result blocks/rows")
            else:
                print(result)


if __name__ == "__main__":
    main()
