#!/usr/bin/env bash
# One experiment: run the fixed judge under limits. Usage: bash augmented/autoresearch_q1/run.sh "<idea>"
# GROUPCOUNT_PYTHON selects the interpreter (needs numpy >= 2.0); default python3.
set -u
cd "$(dirname "$0")/../.."
PY="${GROUPCOUNT_PYTHON:-python3}"
PYTHONPATH=. exec "$PY" augmented/autoresearch_q1/run_bounded.py --memory-mb 700 --description "${1:-no description}"
