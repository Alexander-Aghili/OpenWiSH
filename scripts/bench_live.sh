#!/usr/bin/env bash
# Live benchmark: run bundled environments with a real agent and collect the scores into runs/bench/<label>/.
# usage: scripts/bench_live.sh <label> <backend> [model] -- <task.toml> ...
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=$1; BACKEND=$2; MODEL=${3:-}; shift 3; [ "${1:-}" = "--" ] && shift
OUT=runs/bench/$LABEL
mkdir -p "$OUT"
for T in "$@"; do
  ID=$(basename "$(dirname "$T")")-$(basename "$T" .toml)
  [ -f "$OUT/$ID/eval/score.json" ] && { echo "skip $ID (done)"; continue; }
  echo "== $ID $(date +%T)"
  if [ -f "$OUT/$ID/state.json" ]; then
    nice -n 10 target/release/openwish run --resume "$OUT/$ID" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} >> "$OUT/$ID.log" 2>&1
  else
    nice -n 10 target/release/openwish run "$T" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} --out "$OUT/$ID" >> "$OUT/$ID.log" 2>&1
  fi
  echo "   exit $? $(tail -1 "$OUT/$ID.log" | cut -c1-120)"
done
