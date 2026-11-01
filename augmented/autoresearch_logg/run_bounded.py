"""Run the fixed judge in a fresh process with wall-clock and memory limits.

Exit code 124 means LIMITED: the design was too slow or too large; it is not a
failed gate and not a result. The wrapper never edits the judge and never writes
a TSV row by itself.

Usage: PYTHONPATH=. python augmented/autoresearch_logg/run_bounded.py --description "<idea>"
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hard-seconds', type=int, default=600)
    parser.add_argument('--memory-mb', type=int, default=1024)
    parser.add_argument('--description', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH='.')
    command = [sys.executable, str(Path(__file__).with_name('judge.py')), args.description]
    process = subprocess.Popen(command, cwd=root, env=env)
    deadline = time.monotonic() + args.hard_seconds
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=min(2, max(.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is not None:
                break
            rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
            memory_kb = int(rss.stdout.strip() or '0')
            timed_out = time.monotonic() >= deadline
            if timed_out or memory_kb > args.memory_mb * 1024:
                reason = 'wall-clock' if timed_out else 'memory'
                print(f'status=limited: {reason} limit; RSS={memory_kb} KiB; no result claimed', flush=True)
                raise SystemExit(124)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
