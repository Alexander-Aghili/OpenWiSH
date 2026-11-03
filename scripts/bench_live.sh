#!/usr/bin/env bash
# Live benchmark: run bundled environments with a real agent and collect the scores into runs/bench/<label>/.
# usage: scripts/bench_live.sh <label> <backend> [model] -- <task.toml> ...
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=$1; BACKEND=$2; MODEL=${3:-}; shift 3; [ "${1:-}" = "--" ] && shift
OUT=runs/bench/$LABEL
BENCH_BIN=${OPENWISH_BENCH_BIN:-target/release/openwish}
BENCH_STATUS=0
mkdir -p "$OUT"
for T in "$@"; do
  ID=$(basename "$(dirname "$T")")-$(basename "$T" .toml)
  [ -f "$OUT/$ID/eval/score.json" ] && { echo "skip $ID (done)"; continue; }
  echo "== $ID $(date +%T)"
  if [ -f "$OUT/$ID/state.json" ]; then
    nice -n 10 "$BENCH_BIN" run --resume "$OUT/$ID" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} >> "$OUT/$ID.log" 2>&1
  else
    nice -n 10 "$BENCH_BIN" run "$T" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} --out "$OUT/$ID" >> "$OUT/$ID.log" 2>&1
  fi
  BENCH_RC=$?
  [ "$BENCH_RC" -ne 0 ] && BENCH_STATUS=$BENCH_RC
  echo "   exit $BENCH_RC $(tail -1 "$OUT/$ID.log" | cut -c1-120)"
done
exit "$BENCH_STATUS"
