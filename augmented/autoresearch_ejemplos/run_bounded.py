"""Run the fixed judge in a fresh process with a real wall-clock limit.

The judge's budget is checked between instances. This wrapper also limits a
single slow solve. It never edits the judge or invents a successful TSV row.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mission",choices=["binario","precio"])
    parser.add_argument("--seconds",type=int,default=180)
    parser.add_argument("--nmax",type=int,default=6)
    parser.add_argument("--hard-seconds",type=int,default=360)
    parser.add_argument("--memory-mb",type=int,default=1536)
    parser.add_argument("--q-half-only",action="store_true")
    parser.add_argument("--description",default="seeded family search")
    args=parser.parse_args()
    if not 3 <= args.nmax <= 7 or not 0 < args.seconds < args.hard_seconds or args.memory_mb < 128:
        parser.error("require 3<=nmax<=7, 0<seconds<hard-seconds, memory-mb>=128")
    if args.q_half_only:
        parser.error("q=1/2, u=1 with these n,B,G limits is already covered; use probability perturbations or design a larger grid")
    env=dict(os.environ,PYTHONPATH=".",GROUPCOUNT_MISSION=args.mission,
             GROUPCOUNT_NMAX=str(args.nmax),GROUPCOUNT_PRICE_Q_HALF_ONLY=str(int(args.q_half_only)))
    root=Path(__file__).resolve().parents[2]
    command=[sys.executable,str(Path(__file__).with_name("judge.py")),args.mission,
             str(args.seconds),str(args.nmax),args.description]
    process=subprocess.Popen(command,cwd=root,env=env)
    deadline=time.monotonic()+args.hard_seconds
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=min(2,max(.01,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is not None:
                break
            rss=subprocess.run(["ps","-o","rss=","-p",str(process.pid)],capture_output=True,text=True)
            memory_kb=int(rss.stdout.strip() or "0")
            timed_out=time.monotonic()>=deadline
            if timed_out or memory_kb>args.memory_mb*1024:
                reason="wall-clock" if timed_out else "memory"
                print(f"status=limited: {reason} limit; RSS={memory_kb} KiB; no certified result claimed",flush=True)
                raise SystemExit(124)
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    raise SystemExit(process.returncode)


if __name__=="__main__":
    main()
