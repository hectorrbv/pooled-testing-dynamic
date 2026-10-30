#!/usr/bin/env bash
# Launch one or both autoresearch loops on Hermeneuta as a transient user service.
#   bash launch.sh logg [N]     loop 1 (log-G designs), N experiments (default 30)
#   bash launch.sh q1   [N]     loop 2 (Q1 laminar scorer)
#   bash launch.sh all  [N]     loop 1 then loop 2, in one unit
# Each loop runs in rounds of PER experiments, one fresh `claude -p` session per round,
# so the agent's context (and memory) stays small; it continues from notes.md and the
# kept best file. Requires ~/.local/bin/claude and ~/.config/groupcount/claude.env with
#   CLAUDE_CODE_OAUTH_TOKEN=...   (from `claude setup-token`)   or ANTHROPIC_API_KEY=...
set -euo pipefail
LOOP="${1:?logg|q1|all}"; N="${2:-30}"; PER="${PER:-10}"
WS="$HOME/agent-workspace/groupcount-loops-20261017"
ENVF="$HOME/.config/groupcount/claude.env"
PYV="$HOME/agent-workspace/groupcount-night-20261015/.venv311/bin/python"
test -s "$ENVF" || { echo "missing $ENVF (token or API key)"; exit 1; }
test -x "$HOME/.local/bin/claude" || { echo "claude CLI not installed"; exit 1; }
mkdir -p "$WS/logs"
STAMP="$(date +%Y%m%d-%H%M)"
UNIT="groupcount-$LOOP-$STAMP"
ROUNDS=$(( (N + PER - 1) / PER ))
round_cmd() {  # $1 = loop, $2 = round index
  local best; [ "$1" = logg ] && best=design_best.py || best=scorer_best.py
  local p="Read augmented/autoresearch_$1/program.md and follow its rules exactly. This is round $2 of $ROUNDS: run up to $PER new experiments, then stop. Earlier rounds already ran; their rows are in the results TSV and their lessons in notes.md. Start from the kept best file ($best) and do not repeat ideas already logged. Append a short summary to notes.md before you stop. Work only inside augmented/autoresearch_$1/."
  printf 'claude --dangerously-skip-permissions -p "%s" --max-turns 300 > logs/%s-%s-r%s.log 2>&1 || true; ' "$p" "$UNIT" "$1" "$2"
}
loop_cmd() { local c=""; for r in $(seq 1 $ROUNDS); do c+="$(round_cmd "$1" "$r")"; done; printf '%s' "$c"; }
case "$LOOP" in
  logg|q1) CMD="$(loop_cmd "$LOOP")" ;;
  all)     CMD="$(loop_cmd logg)$(loop_cmd q1)" ;;
  *) echo "unknown loop $LOOP"; exit 1 ;;
esac
systemd-run --user --unit "$UNIT" --description "GroupCounting autoresearch $LOOP ($N experiments, rounds of $PER)" \
  -p MemoryMax=1500M -p OOMPolicy=continue -p CPUQuota=150% -p RuntimeMaxSec=12h -p Nice=10 \
  -p EnvironmentFile="$ENVF" \
  --working-directory="$WS" \
  --setenv=HOME="$HOME" \
  --setenv=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  --setenv=GROUPCOUNT_PYTHON="$PYV" \
  bash -c "$CMD"
echo "started unit $UNIT ($ROUNDS rounds of $PER per loop); follow with: bash augmented/autoresearch_vps/status.sh $UNIT"
