"""Q2 and Q1 on the same instances, with explicit soft clearing.

python -m augmented.malla_precio_laminar_q results/precio_q05_G/malla.csv \
    --q 1/2 --n 6 7 --G 2 3 4 5 --B 1 2 3 4 --examples

The rational atom optimum is cross-checked against the pathwise profile solver
on five designated cells. Heuristics share a fixed balanced hierarchy, whose
own optimum is measured separately. Limits are recorded, never called optima.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import gc
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import signal
from time import perf_counter

from augmented.bellman_perfiles import BellmanPerfiles
from augmented.bm17_toy_solver import SolverLaminar
from augmented.experimento_sesion_28 import evaluate
from augmented.general_homogeneo import solve_general
from augmented.laminar_benchmarks import ExactPolicyEvaluator, balanced_laminar_library

ROOT = Path(__file__).resolve().parents[1]
SPOT_CHECKS = {(6, 2, 2), (6, 3, 3), (6, 4, 3), (7, 3, 4), (6, 4, 5)}


@contextmanager
def bounded(seconds):
    def timeout(*_):
        raise TimeoutError("per-solver time limit")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def profile_policy(evaluator, kind):
    def action(support, history, b):
        worlds = sum(1 << sum(x << i for i, x in enumerate(z)) for z in support)
        cleared = sum(1 << i for i in range(evaluator.n)
                      if all(z[i] == 0 for z in support))
        pool = evaluator.policy_actions[kind][evaluator.B-b, worlds, cleared]
        return tuple(i for i in range(evaluator.n) if pool >> i & 1)
    return action


def measure(q, u, B, G, seconds=60, states=1_000_000, verify=False):
    q, u = tuple(map(F, q)), tuple(map(F, u))
    n = len(q)
    row = dict(n=n, q=str(q[0]) if len(set(q)) == 1 else "heterogeneous", B=B, G=G,
               clearing="soft", laminar_scope="pathwise", hierarchy="fixed_balanced",
               exact_laminar="", unrestricted="", greedy_laminar="", rollout_laminar="",
               hierarchy_optimum="", ratio="", greedy_ratio="", rollout_ratio="",
               hierarchy_ratio="", information_lower_bound="", information_check="",
               laminar_status="pending", general_status="pending", policy_status="pending")
    audit = dict(q=list(map(str,q)), u=list(map(str,u)), B=B, G=G, checks={})
    start = perf_counter()
    try:
        with bounded(seconds):
            atom = SolverLaminar(dict(enumerate(1-x for x in q)),dict(enumerate(u)),G,"posterior_zero")
            lam = atom.V(frozenset(range(n)), (), B)
            row.update(exact_laminar=str(lam), laminar_status="exact", laminar_states=len(atom.memo))
            del atom
    except (TimeoutError, RuntimeError, MemoryError) as error:
        row.update(laminar_status="limited", laminar_reason=str(error))
    try:
        with bounded(seconds):
            value, count, method = solve_general(q,u,B,G,state_limit=states)
            row.update(unrestricted=str(value), general_status="exact", general_states=count,general_method=method)
    except (TimeoutError, RuntimeError, MemoryError) as error:
        row.update(general_status="limited", general_reason=str(error))
    gc.collect()
    try:
        with bounded(seconds):
            evaluator = ExactPolicyEvaluator(list(map(float,(1-x for x in q))),list(map(float,u)),B,G,clearing="soft")
            library = balanced_laminar_library(evaluator.p,evaluator.u,G)
            greedy, rollout = evaluator.greedy_and_rollout_values(library)
            fixed = evaluator.optimal_value(library)
            row.update(greedy_laminar=greedy,rollout_laminar=rollout,hierarchy_optimum=fixed,policy_status="evaluated")
            audit["library"] = [[i for i in range(n) if a >> i & 1] for a in library]
            assert greedy <= rollout+1e-10 and rollout <= fixed+1e-10
            if row["exact_laminar"]:
                assert fixed <= float(lam)+1e-10
                row.update(greedy_ratio=greedy/float(lam), rollout_ratio=rollout/float(lam), hierarchy_ratio=fixed/float(lam))
            if verify:
                for kind, value in [("greedy",greedy),("rollout",rollout)]:
                    replay = evaluate(profile_policy(evaluator,kind),q,u,B,G,require_laminar=True)
                    assert abs(float(F(replay["value"]))-value) <= 1e-10
                    audit["checks"][kind] = replay
    except (TimeoutError, RuntimeError, MemoryError) as error:
        row.update(policy_status="limited", policy_reason=str(error))
    if verify and row["exact_laminar"]:
        try:
            with bounded(seconds):
                other = BellmanPerfiles([1-x for x in q],u,G,laminar=True,state_limit=states).solve(B)
                assert other.value == lam
                audit["checks"]["independent_laminar"] = str(other.value)
        except (TimeoutError, RuntimeError, MemoryError) as error:
            audit["checks"]["independent_laminar_limited"] = str(error)
    if row["exact_laminar"] and row["unrestricted"]:
        aug = F(row["unrestricted"])
        assert lam <= aug
        row["ratio"] = float(lam/aug)
        if set(q) == {F(1,2)} and set(u) == {F(1)} and 1 <= B <= n:
            lower = 1/(2*log2(G+1))
            assert float(aug) <= B*log2(G+1)+1e-10
            assert row["ratio"]+1e-10 >= lower
            row.update(information_lower_bound=lower,information_check="pass")
    row["sec"] = round(perf_counter()-start,3)
    gc.collect()
    return row,audit


def save(out, rows, audits, args):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with out.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    sources=["augmented/malla_precio_laminar_q.py","augmented/general_homogeneo.py","augmented/bm17_toy_solver.py",
             "augmented/bellman_perfiles.py","augmented/laminar_benchmarks.py","augmented/experimento_sesion_28.py"]
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),parameters=vars(args),
                  conventions={"q":"healthy","u":"utility","clearing":"soft",
                               "laminar":"pathwise, rational atom optimum",
                               "policies":"float evaluation in fixed balanced hierarchy; smallest mask breaks ties"},
                  source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  rows=len(rows),complete=sum(r["laminar_status"]=="exact" and r["general_status"]=="exact" and r["policy_status"]=="evaluated" for r in rows))
    out.with_suffix(".meta.json").write_text(json.dumps(metadata,indent=2)+"\n")
    out.with_suffix(".audit.json").write_text(json.dumps(audits,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out");parser.add_argument("legacy_nmax",nargs="?",type=int)
    parser.add_argument("--q",nargs="+",default=["1/2"])
    parser.add_argument("--n",nargs="+",type=int,default=[6,7])
    parser.add_argument("--G",nargs="+",type=int,default=[2,3,4,5])
    parser.add_argument("--B",nargs="+",type=int,default=[1,2,3,4])
    parser.add_argument("--case-seconds",type=float,default=60)
    parser.add_argument("--states",type=int,default=1_000_000)
    parser.add_argument("--examples",action="store_true")
    args=parser.parse_args()
    if args.legacy_nmax is not None:
        args.n=list(range(4,args.legacy_nmax+1));args.q=["3/10","1/2","7/10"];args.G=[2,3,4];args.B=[2,3,4]
    if any(not 0 < F(q) < 1 for q in args.q):parser.error("q must be strictly between 0 and 1")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():parser.error("output exists; choose a new filename")
    rows,audits=[],[]
    for n in args.n:
        for q in args.q:
            for G in args.G:
                for B in args.B:
                    if not 1 <= G <= n or not 1 <= B <= n:continue
                    row,audit=measure([F(q)]*n,[F(1)]*n,B,G,args.case_seconds,args.states,(n,B,G) in SPOT_CHECKS)
                    rows.append(row);audits.append(audit);save(out,rows,audits,args)
                    print(json.dumps(row),flush=True)
    if args.examples:
        examples=[]
        for name,q,u,G in [("vlad_n3",[F(9,10),F(7,10),F(7,10)],[1]*3,3),
                          ("notebook_n3",[F(19,20),F(4,5),F(4,5)],[1]*3,2),
                          ("family_10_9",[F(999999,1000000),F(1,2),F(1,2),F(999999,1000000)],[1,3,3,2],2)]:
            row,audit=measure(q,u,2,G,args.case_seconds,args.states,True)
            examples.append(dict(name=name,values=row,audit=audit))
        out.with_suffix(".examples.json").write_text(json.dumps(examples,indent=2)+"\n")


if __name__ == "__main__":
    main()
