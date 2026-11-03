#!/usr/bin/env bash
# Fetch the authors' public logs (networks, prompts, answer keys) into assets/lm_understand-logs.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d assets/lm_understand-logs/.git ]; then
  mkdir -p assets
  git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git assets/lm_understand-logs
fi
echo "logs at assets/lm_understand-logs ($(git -C assets/lm_understand-logs rev-parse --short HEAD))"
