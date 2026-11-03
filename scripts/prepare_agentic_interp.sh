#!/usr/bin/env bash
# Fetch AgenticInterpBench (arXiv 2606.24026) and build tasks/agentic_interp/<case>-{A,WO,P}.
# Downloads: the HyVE repo at a pinned commit, the gold annotations (HF dataset Antik/AgenticInterpBench) and the
# 84 InterpBench weight files (HF cybershiptrooper/InterpBench, ll_model.pth only). No pickle file is downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=assets/venvs/sandbox/bin/python
COMMIT=ce23ac099775a599c16462b97d9b4e2633761893
D=assets/agentic_interp
mkdir -p $D
[ -d $D/repo ] || git clone -q https://github.com/Ziyu-Yao-NLP-Lab/LLM-Circuit-Explainer $D/repo
git -C $D/repo fetch -q --depth 1 origin $COMMIT 2>/dev/null || true
git -C $D/repo checkout -q $COMMIT
nice -n 10 $PY -I - <<'PYEOF'
import glob
from huggingface_hub import hf_hub_download, snapshot_download
D = "assets/agentic_interp"
snapshot_download("Antik/AgenticInterpBench", repo_type="dataset", local_dir=f"{D}/hf/dataset", allow_patterns=["annotations/*", "*.md"])
for f in sorted(glob.glob(f"{D}/hf/dataset/annotations/case_*.json")):
    case = f.rsplit("_", 1)[1][:-5]
    hf_hub_download("cybershiptrooper/InterpBench", subfolder=case, filename="ll_model.pth", local_dir=f"{D}/hf/models")
PYEOF
nice -n 10 $PY -I scripts/prepare_agentic_interp.py
