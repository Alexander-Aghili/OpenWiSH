#!/usr/bin/env bash
# Train the Experiment 6 opponents harness-side (the authors' 243-32-32-82 opponents were not released).
# practice: weak (REINFORCE), strong (PPO, seed 1); hidden evaluation: hidden_s0, hidden_s2 (PPO, seeds 0 and 2).
set -euo pipefail
cd "$(dirname "$0")/.."
W=target/release/openwish
O=assets/exp6/opponents
$W go train --out $O/weak      --seed 11 --algo reinforce --iters 60  --games 512 --lr 2e-3 --snapshot-every 20
$W go train --out $O/strong    --seed 1  --algo ppo       --iters 400 --games 512 --lr 1e-3 --snapshot-every 20
$W go train --out $O/hidden_s0 --seed 0  --algo ppo       --iters 400 --games 512 --lr 1e-3 --snapshot-every 20
$W go train --out $O/hidden_s2 --seed 2  --algo ppo       --iters 400 --games 512 --lr 1e-3 --snapshot-every 20
for o in weak strong hidden_s0 hidden_s2; do
  echo "$o: $(python3 -c "import json;print(json.load(open('$O/$o/train.json'))['vs_random_1000']['score'])")"
done
