"""Rollout (lookahead) vs myopic greedy en instancias del paper (N=50).

Instancias: N=50, B=5, G in {3,5}, u_i ~ Uniform{1,2,3},
p_i (prob de estar infectado/activo) ~ Uniform[0,1].

Dos modelos de test:
  - binary  : el del paper (negativo <=> pool 100% sano); rollout en CADA paso.
  - counting: el del repo augmented (r = # infectados del pool);
              lookahead SOLO en el paso 1 (semantica de greedy_lookahead_simulate).

Politicas (sobre marginales, como el paper usa marginales de Gibbs):
  - myopic : argmax_t  prod(q_i) * sum(u_i)  EXACTO via composiciones por clase
             de utilidad (u in {1,2,3}: para (n1,n2,n3) fijos, lo optimo es
             tomar los top-q de cada clase).
  - rollout: entre K candidatos (composiciones top-q + composiciones "mas
             inciertas"), elige argmax de EU total con continuacion miope,
             esperanza bajo el modelo de marginales.

Evaluacion: welfare realizado sobre z muestreada, variables comunes (paired).
"""

import numpy as np
from itertools import product as iproduct

G_MAX_POOL = 5
UVALS = (1, 2, 3)


def compositions(G):
    comps = []
    for g in range(1, G + 1):
        for n1 in range(g + 1):
            for n2 in range(g - n1 + 1):
                n3 = g - n1 - n2
                comps.append((n1, n2, n3))
    return comps


class State:
    """Marginales p (prob infectado), utilidades u, mascara cleared."""

    __slots__ = ("p", "u", "cleared")

    def __init__(self, p, u, cleared):
        self.p = p          # np.array float
        self.u = u          # np.array int
        self.cleared = cleared  # np.array bool

    def copy(self):
        return State(self.p.copy(), self.u.copy(), self.cleared.copy())


def class_views(state):
    """Por clase de utilidad: indices activos ordenados por q desc + prefix prod."""
    q = 1.0 - state.p
    # activos: no cleared y no confirmados infectados (p==1)
    active = (~state.cleared) & (state.p < 1.0 - 1e-12)
    views = []
    for uv in UVALS:
        idx = np.where(active & (state.u == uv))[0]
        if len(idx):
            order = np.argsort(-q[idx])
            idx = idx[order]
        pp = np.concatenate(([1.0], np.cumprod(q[idx])))
        views.append((idx, pp))
    return views


def myopic_pool(state, comps):
    """Argmax exacto de prod(q)*sum(u). Devuelve lista de indices (o None)."""
    views = class_views(state)
    best_score, best = -1.0, None
    for (n1, n2, n3) in comps:
        ns = (n1, n2, n3)
        score = 0.0
        ok = True
        prod = 1.0
        for k in range(3):
            idx, pp = views[k]
            if ns[k] > len(idx):
                ok = False
                break
            prod *= pp[ns[k]]
        if not ok:
            continue
        usum = n1 + 2 * n2 + 3 * n3
        score = prod * usum
        if score > best_score:
            best_score = score
            best = ns
    if best is None:
        return None
    pool = []
    for k in range(3):
        idx, _ = views[k]
        pool.extend(idx[:best[k]].tolist())
    return pool if pool else None


def candidate_pools(state, comps, kmax=24):
    """Candidatos para lookahead: composiciones top-q + composiciones inciertas."""
    views = class_views(state)
    q = 1.0 - state.p
    # familia B: por clase, ordenados por incertidumbre |q-0.5| asc
    active = (~state.cleared) & (state.p < 1.0 - 1e-12)
    views_unc = []
    for uv in UVALS:
        idx = np.where(active & (state.u == uv))[0]
        if len(idx):
            order = np.argsort(np.abs(q[idx] - 0.5))
            idx = idx[order]
        views_unc.append(idx)

    scored = []
    seen = set()
    for fam, vw in (("top", views), ("unc", views_unc)):
        for (n1, n2, n3) in comps:
            ns = (n1, n2, n3)
            pool = []
            ok = True
            for k in range(3):
                idx = vw[k][0] if fam == "top" else vw[k]
                if ns[k] > len(idx):
                    ok = False
                    break
                pool.extend(idx[:ns[k]].tolist())
            if not ok:
                continue
            key = tuple(sorted(pool))
            if key in seen:
                continue
            seen.add(key)
            score = np.prod(q[pool]) * state.u[pool].sum()
            scored.append((score, pool))
    scored.sort(key=lambda x: -x[0])
    pools = [p for _, p in scored[:kmax]]
    # garantizar que el miope este incluido
    myo = myopic_pool(state, comps)
    if myo is not None:
        key = tuple(sorted(myo))
        if key not in {tuple(sorted(p)) for p in pools}:
            pools.append(myo)
    return pools


# ---------------- modelo BINARY (paper) ----------------

def binary_apply(state, pool, negative):
    """Update de marginales tras test binario. Muta una copia."""
    s = state.copy()
    pl = np.array(pool)
    if negative:
        s.p[pl] = 0.0
        s.cleared[pl] = True
    else:
        qprod = np.prod(1.0 - s.p[pl])
        denom = 1.0 - qprod
        if denom > 1e-15:
            for i in pl:
                s.p[i] = min(1.0, s.p[i] / denom)
    return s


def binary_value_myopic(state, b, comps):
    """EU de la continuacion miope, esperanza bajo marginales (branching 2)."""
    if b == 0:
        return 0.0
    pool = myopic_pool(state, comps)
    if pool is None:
        return 0.0
    qprod = float(np.prod(1.0 - state.p[pool]))
    usum = float(state.u[pool].sum())
    v = 0.0
    if qprod > 1e-15:
        s_neg = binary_apply(state, pool, True)
        v += qprod * (usum + binary_value_myopic(s_neg, b - 1, comps))
    if qprod < 1.0 - 1e-15:
        s_pos = binary_apply(state, pool, False)
        v += (1.0 - qprod) * binary_value_myopic(s_pos, b - 1, comps)
    return v


def binary_rollout_pool(state, b, comps, kmax=24):
    """Un paso de lookahead con continuacion miope. Devuelve (pool, Q, Qmyo)."""
    cands = candidate_pools(state, comps, kmax)
    myo = myopic_pool(state, comps)
    myo_key = tuple(sorted(myo)) if myo else None
    best_q, best_pool, q_myo = -1.0, None, 0.0
    for pool in cands:
        qprod = float(np.prod(1.0 - state.p[pool]))
        usum = float(state.u[pool].sum())
        qv = 0.0
        if qprod > 1e-15:
            s_neg = binary_apply(state, pool, True)
            qv += qprod * (usum + binary_value_myopic(s_neg, b - 1, comps))
        if qprod < 1.0 - 1e-15:
            s_pos = binary_apply(state, pool, False)
            qv += (1.0 - qprod) * binary_value_myopic(s_pos, b - 1, comps)
        if qv > best_q:
            best_q, best_pool = qv, pool
        if myo_key is not None and tuple(sorted(pool)) == myo_key:
            q_myo = qv
    return best_pool, best_q, q_myo


def simulate_binary(state0, z, B, comps, policy, kmax=24):
    """policy in {'myopic','rollout'}. Devuelve welfare realizado."""
    s = state0.copy()
    welfare = 0.0
    for step in range(B):
        b = B - step
        if policy == "rollout":
            pool, _, _ = binary_rollout_pool(s, b, comps, kmax)
        else:
            pool = myopic_pool(s, comps)
        if pool is None:
            break
        negative = not z[pool].any()
        if negative:
            welfare += float(s.u[pool][~s.cleared[pool]].sum())
        s = binary_apply(s, pool, negative)
    return welfare


# ---------------- modelo COUNTING (repo augmented) ----------------

def pb_pmf(probs):
    m = len(probs)
    dp = np.zeros(m + 1)
    dp[0] = 1.0
    for pj in probs:
        dp[1:m + 1] = dp[1:m + 1] * (1 - pj) + dp[0:m] * pj
        dp[0] *= (1 - pj)
    return dp


def counting_apply(state, pool, r):
    """Port de bayesian_update_single_test sobre indices."""
    s = state.copy()
    pl = list(pool)
    g = len(pl)
    for i in pl:
        pi = s.p[i]
        if pi <= 0.0:
            s.p[i] = 0.0
            continue
        if pi >= 1.0:
            s.p[i] = 1.0
            continue
        others = [j for j in pl if j != i]
        pmf = pb_pmf(state.p[others])
        pr1 = pmf[r - 1] if r >= 1 else 0.0
        pr0 = pmf[r] if r <= len(others) else 0.0
        num = pr1 * pi
        den = num + pr0 * (1 - pi)
        if den > 0:
            s.p[i] = num / den
    if r == 0:
        s.p[pl] = 0.0
        s.cleared[pl] = True
    return s


def counting_value_myopic(state, b, comps):
    if b == 0:
        return 0.0
    pool = myopic_pool(state, comps)
    if pool is None:
        return 0.0
    pmf = pb_pmf(state.p[pool])
    usum = float(state.u[pool].sum())
    v = 0.0
    for r in range(len(pool) + 1):
        if pmf[r] < 1e-12:
            continue
        s2 = counting_apply(state, pool, r)
        gain = usum if r == 0 else 0.0
        v += pmf[r] * (gain + counting_value_myopic(s2, b - 1, comps))
    return v


def counting_lookahead_pool(state, b, comps, kmax=16):
    cands = candidate_pools(state, comps, kmax)
    myo = myopic_pool(state, comps)
    myo_key = tuple(sorted(myo)) if myo else None
    best_q, best_pool, q_myo = -1.0, None, 0.0
    for pool in cands:
        pmf = pb_pmf(state.p[pool])
        usum = float(state.u[pool].sum())
        qv = 0.0
        for r in range(len(pool) + 1):
            if pmf[r] < 1e-12:
                continue
            s2 = counting_apply(state, pool, r)
            gain = usum if r == 0 else 0.0
            qv += pmf[r] * (gain + counting_value_myopic(s2, b - 1, comps))
        if qv > best_q:
            best_q, best_pool = qv, pool
        if myo_key is not None and tuple(sorted(pool)) == myo_key:
            q_myo = qv
    return best_pool, best_q, q_myo


def simulate_counting(state0, z, B, comps, policy, kmax=16):
    s = state0.copy()
    welfare = 0.0
    for step in range(B):
        b = B - step
        if (policy == "lookahead1" and step == 0) or policy == "rollout":
            pool, _, _ = counting_lookahead_pool(s, b, comps, kmax)
        else:
            pool = myopic_pool(s, comps)
        if pool is None:
            break
        r = int(z[pool].sum())
        if r == 0:
            welfare += float(s.u[pool][~s.cleared[pool]].sum())
        s = counting_apply(s, pool, r)
    return welfare


# ---------------- validacion ----------------

def validate_myopic(seed=0, n=10, G=3, trials=200):
    """Compara argmax por composiciones vs fuerza bruta."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        p = rng.uniform(0, 1, n)
        u = rng.integers(1, 4, n)
        st = State(p, u, np.zeros(n, bool))
        pool = myopic_pool(st, compositions(G))
        best = -1.0
        for g in range(1, G + 1):
            for c in combinations(range(n), g):
                sc = np.prod(1 - p[list(c)]) * u[list(c)].sum()
                if sc > best:
                    best = sc
        got = np.prod(1 - p[pool]) * u[pool].sum()
        assert abs(got - best) < 1e-9, (got, best)
    print(f"validacion myopic_pool OK ({trials} instancias n={n}, G={G})")


# ---------------- experimento ----------------

def _one_instance(task):
    """Worker para multiprocessing: una instancia pareada."""
    model, G, B, N, kmax, seed = task
    rng = np.random.default_rng(seed)
    comps = compositions(G)
    p = rng.uniform(0, 1, N)
    u = rng.integers(1, 4, N)
    z = rng.uniform(0, 1, N) < p
    st = State(p.copy(), u.copy(), np.zeros(N, bool))
    if model == "binary":
        pool_r, q_r, q_m = binary_rollout_pool(st, B, comps, kmax)
        w_m = simulate_binary(st, z, B, comps, "myopic")
        w_r = simulate_binary(st, z, B, comps, "rollout", kmax)
    else:
        pool_r, q_r, q_m = counting_lookahead_pool(st, B, comps, kmax)
        w_m = simulate_counting(st, z, B, comps, "myopic")
        w_r = simulate_counting(st, z, B, comps, "rollout", kmax)
    myo0 = myopic_pool(st, comps)
    diff_first = tuple(sorted(pool_r)) != tuple(sorted(myo0))
    return w_m, w_r, diff_first, q_r - q_m


def run_setting_parallel(model, G, B, n_inst, seed, N=50, kmax=12, procs=8):
    """Version paralela: rollout en CADA paso (tambien para counting)."""
    from multiprocessing import Pool
    tasks = [(model, G, B, N, kmax, seed * 1_000_003 + i)
             for i in range(n_inst)]
    with Pool(procs) as pool:
        out = pool.map(_one_instance, tasks, chunksize=4)
    wm = np.array([o[0] for o in out])
    wr = np.array([o[1] for o in out])
    d = wr - wm
    n_diff = sum(o[2] for o in out)
    pg = np.array([o[3] for o in out])
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n=== {model} ROLLOUT-CADA-PASO  N={N} G={G} B={B}  "
          f"({n_inst} instancias, kmax={kmax}) ===")
    print(f"welfare miope   : {wm.mean():.3f} (se {wm.std(ddof=1)/np.sqrt(len(wm)):.3f})")
    print(f"welfare rollout : {wr.mean():.3f} (se {wr.std(ddof=1)/np.sqrt(len(wr)):.3f})")
    print(f"diff pareada    : {d.mean():+.4f} (se {se:.4f})  "
          f"[{d.mean()/max(wm.mean(),1e-9)*100:+.2f}%]")
    print(f"1er pool distinto: {n_diff}/{n_inst} ({100*n_diff/n_inst:.1f}%)")
    print(f"ganancia predicha paso 1 (modelo): media {pg.mean():+.4f}, "
          f"max {pg.max():+.4f}", flush=True)
    return d


def run_setting(model, G, B, n_inst, seed, N=50, kmax=24):
    rng = np.random.default_rng(seed)
    comps = compositions(G)
    diffs, w_myo_all, w_roll_all = [], [], []
    pred_gains = []
    n_diff_first = 0
    for it in range(n_inst):
        p = rng.uniform(0, 1, N)          # prob de INFECTADO
        u = rng.integers(1, 4, N)
        z = rng.uniform(0, 1, N) < p      # True = infectado
        st = State(p.copy(), u.copy(), np.zeros(N, bool))

        if model == "binary":
            pool_r, q_r, q_m = binary_rollout_pool(st, B, comps, kmax)
            w_m = simulate_binary(st, z, B, comps, "myopic")
            w_r = simulate_binary(st, z, B, comps, "rollout", kmax)
        else:
            pool_r, q_r, q_m = counting_lookahead_pool(st, B, comps, kmax)
            w_m = simulate_counting(st, z, B, comps, "myopic")
            w_r = simulate_counting(st, z, B, comps, "lookahead1", kmax)

        myo0 = myopic_pool(st, comps)
        if tuple(sorted(pool_r)) != tuple(sorted(myo0)):
            n_diff_first += 1
        pred_gains.append(q_r - q_m)
        w_myo_all.append(w_m)
        w_roll_all.append(w_r)
        diffs.append(w_r - w_m)

    d = np.array(diffs)
    wm, wr = np.array(w_myo_all), np.array(w_roll_all)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n=== {model}  N={N} G={G} B={B}  ({n_inst} instancias) ===")
    print(f"welfare miope   : {wm.mean():.3f} (se {wm.std(ddof=1)/np.sqrt(len(wm)):.3f})")
    print(f"welfare rollout : {wr.mean():.3f} (se {wr.std(ddof=1)/np.sqrt(len(wr)):.3f})")
    print(f"diff pareada    : {d.mean():+.4f} (se {se:.4f})  "
          f"[{d.mean()/max(wm.mean(),1e-9)*100:+.2f}%]")
    print(f"1er pool distinto: {n_diff_first}/{n_inst} "
          f"({100*n_diff_first/n_inst:.1f}%)")
    pg = np.array(pred_gains)
    print(f"ganancia predicha paso 1 (modelo): media {pg.mean():+.4f}, "
          f"max {pg.max():+.4f}")
    return d


def dump_csv(path, n_inst=1000, procs=8):
    """Corre los 4 settings (rollout en cada paso) y guarda por-instancia."""
    import csv
    from multiprocessing import Pool
    settings = [("binary", 5, 24, 11), ("binary", 3, 24, 22),
                ("counting", 5, 12, 55), ("counting", 3, 12, 66)]
    with open(path, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["model", "G", "B", "N", "kmax", "inst",
                       "w_myopic", "w_rollout", "first_pool_diff",
                       "pred_gain_step1"])
        for model, G, kmax, seed in settings:
            B, N = 5, 50
            tasks = [(model, G, B, N, kmax, seed * 1_000_003 + i)
                     for i in range(n_inst)]
            with Pool(procs) as pool:
                out = pool.map(_one_instance, tasks, chunksize=4)
            for i, (wm, wr, df, pg) in enumerate(out):
                wcsv.writerow([model, G, B, N, kmax, i, wm, wr, int(df),
                               round(pg, 6)])
            d = np.array([o[1] - o[0] for o in out])
            print(f"{model} G={G}: diff {d.mean():+.4f} "
                  f"(se {d.std(ddof=1)/np.sqrt(len(d)):.4f})", flush=True)
    print(f"escrito {path}")


if __name__ == "__main__":
    import sys
    validate_myopic()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "csv":
        out = sys.argv[2] if len(sys.argv) > 2 else "augmented/data/rollout_n50.csv"
        dump_csv(out)
        sys.exit(0)
    if mode in ("all", "binary"):
        run_setting("binary", G=5, B=5, n_inst=1000, seed=11)
        run_setting("binary", G=3, B=5, n_inst=1000, seed=22)
    if mode in ("all", "counting"):
        run_setting("counting", G=5, B=5, n_inst=300, seed=33, kmax=12)
        run_setting("counting", G=3, B=5, n_inst=300, seed=44, kmax=12)
    if mode == "counting_full":
        run_setting_parallel("counting", G=5, B=5, n_inst=1000, seed=55, kmax=12)
        run_setting_parallel("counting", G=3, B=5, n_inst=1000, seed=66, kmax=12)
