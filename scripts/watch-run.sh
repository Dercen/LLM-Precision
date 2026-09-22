#!/usr/bin/env bash
# Watch a running `ptq run`: a progress line, the newest results, then the live log.
#   scripts/watch-run.sh                  # newest log in results/logs
#   scripts/watch-run.sh streamed_llama   # a specific run
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
name="${1:-}"
if [ -z "$name" ]; then log=$(ls -t results/logs/*.log 2>/dev/null | head -1); else log="results/logs/$name.log"; fi
[ -f "$log" ] || { echo "no log found ($log)"; exit 1; }
echo "watching $log  (Ctrl-C leaves the run untouched)"
while true; do
  clear
  total=$(grep -oE '^\[[0-9]+/[0-9]+\]' "$log" | tail -1 | tr -d '[]')
  done_rows=$(grep -c 'ppl=' "$log"); fails=$(grep -c 'FAILED' "$log")
  running=$(pgrep -f '[p]tq run configs' >/dev/null && echo running || echo "not running")
  echo "== $log == group ${total:-?}  rows done: $done_rows  failed: $fails  ($running)  $(date +%H:%M:%S)"
  echo "-- recent results --"; grep -E 'ppl=' "$log" | tail -8 | cut -c1-110
  echo "-- current --"; grep -E '^\[[0-9]+/' "$log" | tail -1 | cut -c1-110
  grep -q '^done:' "$log" && { grep '^done:' "$log"; echo "(finished)"; }
  sleep 10
done
