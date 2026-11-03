#!/usr/bin/env bash
# Experiment 7: fine-tune the three honest/sycophant pairs of Gemma 3 270M with the authors' own training script,
# then release them as A/B folders with the authors' assignment (pair 1: B sycophant, pair 2: B, pair 3: A).
# Needs: access to google/gemma-3-270m-it on Hugging Face (accept the licence on its page, then `hf auth login`
# or set HF_TOKEN), a GPU (about 25 s per model on one GPU), and assets/venvs/gemma (torch + transformers).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=assets/venvs/gemma/bin/python
L=assets/lm_understand-logs/exp7_sycophancy
E=assets/exp7
BASE=$E/base/gemma-3-270m-it
if [ ! -f $BASE/model.safetensors ]; then
  $PY - <<PYEOF || { echo; echo "Could not download google/gemma-3-270m-it. It is gated: accept the licence at"; echo "https://huggingface.co/google/gemma-3-270m-it, run 'hf auth login' (or export HF_TOKEN), then rerun this script."; exit 2; }
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-3-270m-it", local_dir="$BASE")
PYEOF
fi
for P in 100 101 102; do
  for R in honest sycophant; do
    D=$E/train/pair$P/$R
    [ -f $D/final/model.safetensors ] || SYCMID_BASE=$BASE $PY $L/code/train.py $P $R $L/dataset/splits/pair$P.json $D
    [ -f $E/truth/pair${P}_$R.json ] || { mkdir -p $E/truth; $PY $L/code/eval.py $D/final $L/dataset/splits/pair$P.json $E/truth/pair${P}_$R.json; }
  done
done
$PY - <<'PYEOF'
import json, os, shutil
E = "assets/exp7"
syc = {1: "B", 2: "B", 3: "A"}   # the authors' A/B assignment for pairs 100, 101, 102
key = {}
for n, p in ((1, 100), (2, 101), (3, 102)):
    for side in "AB":
        role = "sycophant" if syc[n] == side else "honest"
        src, dst = f"{E}/train/pair{p}/{role}/final", f"{E}/pairs/pair{n}/{side}"
        os.makedirs(dst, exist_ok=True)
        for f in ("model.safetensors", "config.json"):
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(f"{E}/base/gemma-3-270m-it/{f}", f"{dst}/{f}")
    key[f"pair{n}"] = {"pair": p, "sycophantic": syc[n]}
json.dump(key, open(f"{E}/answer_key.json", "w"), indent=1)
print("released", sorted(key))
PYEOF
python3 scripts/prepare_tasks.py exp7
