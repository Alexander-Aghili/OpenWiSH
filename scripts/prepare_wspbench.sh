#!/usr/bin/env bash
# Fetch WSPBench (github.com/alexcarterhq/wspbench), install its locked environment, and export its public
# development pilot tasks as OpenWiSH environments under tasks/wspbench/.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d assets/wspbench/.git ] || git clone --depth 1 https://github.com/alexcarterhq/wspbench.git assets/wspbench
(cd assets/wspbench && uv sync --frozen)
for t in f987f563d00f687f 62a9ccaf392d7c3f 2b12b0f4688c3852 0e688589bf551b1f 0ccf05ea24942ed4 a1393d5b54c62b47; do
  for c in A OW; do
    assets/wspbench/.venv/bin/python -I python/adapters/wspbench_bridge.py export --task "$t" --condition "$c"
  done
done
