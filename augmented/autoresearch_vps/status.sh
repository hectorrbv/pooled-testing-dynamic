#!/usr/bin/env bash
# Show the state of a loop unit on Hermeneuta: unit status, last log lines, last TSV rows.
#   bash status.sh <unit-name>      (unit names start with groupcount-)
set -u
UNIT="${1:-}"
WS="$HOME/agent-workspace/groupcount-loops-20261017"
if [ -z "$UNIT" ]; then
  systemctl --user list-units 'groupcount-*' --all --no-pager
  exit 0
fi
systemctl --user status "$UNIT" --no-pager 2>&1 | head -12
echo '----- agent log (tail)'
for f in "$WS"/logs/"$UNIT"*.log; do echo "== $f"; tail -n 8 "$f"; done 2>/dev/null
for loop in logg q1; do
  f="$WS/augmented/autoresearch_$loop/results_$loop.tsv"
  if [ -f "$f" ]; then
    echo "----- $loop: $(($(wc -l < "$f") - 1)) experiments; last rows"
    tail -n 3 "$f" | cut -c1-200
  fi
done
