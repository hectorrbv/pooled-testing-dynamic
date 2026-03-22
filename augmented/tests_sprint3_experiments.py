"""
Smoke tests for sprint3_experiments.

Run with: python augmented/tests_sprint3_experiments.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.sprint3_experiments import (
    _timed_call,
    _timestamped_csv_path,
)


def test_timed_call_soft_timeout_keeps_value():
    def slow_success():
        time.sleep(0.01)
        return 7.0

    value, elapsed, error, timed_out = _timed_call(
        slow_success, timeout_seconds=0.001)
    assert value == 7.0
    assert elapsed >= 0.01
    assert error is None
    assert timed_out


def test_timestamped_csv_path_uses_prefix_and_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _timestamped_csv_path(tmpdir, "sprint3")
        assert path.startswith(tmpdir)
        assert os.path.basename(path).startswith("sprint3_")
        assert path.endswith(".csv")


def _run_all():
    import inspect
    this = inspect.getmodule(_run_all)
    tests = [(name, fn) for name, fn in inspect.getmembers(this)
             if name.startswith('test_') and callable(fn)]
    passed = failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if _run_all() else 1)
