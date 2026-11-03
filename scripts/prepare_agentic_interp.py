"""Build the AgenticInterpBench environments from the public benchmark files.

usage: assets/venvs/sandbox/bin/python -I scripts/prepare_agentic_interp.py
Needs: assets/agentic_interp/repo (git clone of Ziyu-Yao-NLP-Lab/LLM-Circuit-Explainer),
       assets/agentic_interp/hf/dataset/annotations (HF dataset Antik/AgenticInterpBench),
       assets/agentic_interp/hf/models/<case>/ll_model.pth (HF model repo cybershiptrooper/InterpBench).
Writes tasks/agentic_interp/<case>-A/ (prompt only, one reply) and <case>-WO/ (weights in Python, no network
execution). The gold annotation and the RASP program never enter a task folder. Weights load with
torch.load(weights_only=True). No pickle file is opened."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/agentic_interp"
OUT = ROOT / "tasks/agentic_interp"
TAXONOMY = (SRC / "repo/hyve/prompts/taxonomy.txt").read_text().strip()
SCORER = '["python3", "-I", "{root}/python/adapters/agentic_interp.py"'
HIDDEN = '["../../../assets/agentic_interp"]'

COMMON = """You are given a small transformer and a localized circuit inside it. The listed components are causally relevant to the task. Classify the functional role of every listed component with exactly one tag from the taxonomy.

Taxonomy:
{taxonomy}

When tags overlap, choose by the component's input-output behavior: how many distinct signals it uses, whether it produces a binary flag or a non-binary value, and whether information is summarized across positions or copied between positions."""

ANSWER = 'Answer with one JSON object and nothing else: {"tags": {"<component id>": "<TAG>", ...}}, one entry per component.'

RULES_A = "Condition A: one reply, no tools and no weights. " + ANSWER

RULES_WO = """Condition WO: weights only. You read the weights with Python and numpy. W["<name>"] holds every tensor as a float32 array, with TransformerLens names and shapes (for example W["blocks.0.attn.W_Q"] is [n_heads, d_model, d_head]). VOCAB holds the token maps. COMPONENTS lists the component ids. Reply with one ```python block to run code.

Allowed: any analysis of the weights: products of weight matrices, embedding and position rows multiplied with weight matrices, bilinear attention-score forms such as (W_E + W_pos) W_Q W_K^T (W_E + W_pos)^T, OV and QK circuits, SVD, norms, and reasoning about which directions each component reads and writes.
Forbidden: running the network on any input, even partially. That means no forward pass of any token sequence, no passing embeddings or anything computed from them through a nonlinearity (softmax, activation functions, the MLP), no logits for any input, no evaluation and no fitting.

When you finish, write the answer to /work/submission.json from Python, or reply without a code block with the JSON object. """ + ANSWER

RULES_P = """Condition P: weights and a forward probe. You read the weights with Python and numpy as in condition WO: W["<name>"] holds every tensor with TransformerLens names, VOCAB the token maps, COMPONENTS the component ids. Reply with one ```python block to run code.

You run the network only through the forward probe. Reply with one ```forward block holding a JSON request:
{"inputs": [[token, ...], ...], "ablate": {"<component>": "zero" or "mean"}, "patch": [{"component": "<id>", "source": j}]}
Inputs are up to 64 token sequences of equal length, written like the examples (start with "BOS"). ablate and patch are optional. A patch replaces the component's write into the residual stream with its write on input j, for every input (an interchange intervention). Each probe call and each Python run uses one run from the same budget. The reply shows the predictions. /work/fwd/r<k>/run.npz holds logits, every component's write, attention patterns ("<head>_pattern") and MLP activations ("<mlp>_post"), for loading in Python.

Forbidden: running the network outside the probe (no forward pass written in Python), gradients, fitting and optimisation.

When you finish, write the answer to /work/submission.json from Python, or reply without a code block with the JSON object. """ + ANSWER

PROBE = """
[forward.forward]
command = ["{root}/assets/venvs/sandbox/bin/python", "-I", "{root}/python/adapters/interpbench_forward.py"]
passes = 0
pool = "python"
out_dir = "fwd/r{k}"
max_out = 15000
"""

PRELUDE = '''import json
import numpy as np

W = {n: load(n) for n in NAMES}
VOCAB = json.loads(r"""@VOCAB@""")
COMPONENTS = @COMPONENTS@
'''


def metadata():
    with open(SRC / "repo/case_data/benchmark_cases_metadata.csv") as f:
        return {int(r["case_id"]): r for r in csv.DictReader(f) if r["case_id"].isdigit()}


def architecture(m):
    c = lambda k: m[f"transformer_cfg.{k}"]
    norm = c("normalization_type") or "none"
    return (f"{c('n_layers')} layers, d_model {c('d_model')}, {c('n_heads')} heads of size {c('d_head')}, MLP width "
            f"{c('d_mlp')} with {c('act_fn')}, {c('attention_dir')} attention, normalization {norm}, learned positions, "
            f"context {c('n_ctx')}. Residual stream: x = W_E[token] + W_pos[position]. Each block adds its attention "
            f"heads, then its MLP. Logits = x W_U + b_U at every position.")


def component_text(cid):
    layer, kind = cid.split("_")[0], cid.split("_")[1]
    if kind == "MLP":
        return f"{cid}: the MLP output of block {layer[1:]}"
    block, head = layer[1:].split("H")
    return f"{cid}: attention head {head} of block {block}"


def prompt(gold, enc, m):
    t = gold["task"]
    examples = "\n".join(f"  {e['input']} -> {e['output']}" for e in t["examples"])
    return "\n".join([
        f"Task: {t['summary']}", "Examples (input tokens -> output per position):", examples,
        f"Input token ids: {json.dumps(enc['input_encoding_map'])}", f"Output classes: {json.dumps(enc['output_decoding_map'])}",
        f"Architecture: {architecture(m)}", "Components:", *("  " + component_text(c["id"]) for c in gold["components"]), ""])


def toml_string(s):
    return '"""' + s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'


def write_task(case, cond, text, instructions, extra):
    d = OUT / f"{case}-{cond}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(text)
    (d / "task.toml").write_text(f'''# Generated by scripts/prepare_agentic_interp.py from AgenticInterpBench (case {case}).
id = "agentic-interp-{case}-{cond}"
description = "AgenticInterpBench case {case}, condition {cond}: tag each circuit component with the 5-class taxonomy."
prompt = "prompt.txt"
instructions = {toml_string(instructions)}
hidden = [{HIDDEN[1:-1]}]
{extra}
[editor]
effort = "high"

[score]
command = {SCORER}, "score", "--case", "{case}"]
''')
    return d


def python_section(case, ids, runs, probe=""):
    validate = ", ".join(json.dumps(x) for x in ids)
    return f'''
[network]
base = "net"

[python]
runs = {runs}
timeout_s = 120
mem_gb = 4
max_out = 15000
fresh = false
prelude = "prelude.py"
reject_flags = ["forward_pass", "backward", "autograd", "optimizer", "scipy_optimize", "manual_gradient_step", "fit_into_weights", "lstsq_into_weights"]
{probe}
[finish]
max_turns = {runs + 10}

[finish.validation]
command = {SCORER}, "validate", "--case", "{case}", "--components", {validate}]
submission_file = "submission.json"
timeout_s = 10
'''


def write_weights(d, case, enc, ids):
    net = d / "net"
    net.mkdir(exist_ok=True)
    state = torch.load(SRC / f"hf/models/{case}/ll_model.pth", map_location="cpu", weights_only=True)
    for name, t in state.items():
        if t.dtype.is_floating_point and t.ndim > 0 and not name.endswith((".mask", ".IGNORE")):
            np.save(net / f"{name}.npy", t.numpy().astype(np.float32))
    vocab = {"input": enc["input_encoding_map"], "output": enc["output_decoding_map"]}
    (d / "prelude.py").write_text(PRELUDE.replace("@VOCAB@", json.dumps(vocab)).replace("@COMPONENTS@", json.dumps(ids)))


def arch(m, enc):
    c = lambda k: m[f"transformer_cfg.{k}"]
    return {"n_layers": int(c("n_layers")), "n_heads": int(c("n_heads")), "d_head": int(c("d_head")),
            "causal": c("attention_dir") == "causal", "input_map": enc["input_encoding_map"], "output_map": enc["output_decoding_map"]}


def prepare(case, gold, m):
    enc = json.loads((SRC / f"repo/case_data/{case}/encoder_data.json").read_text())
    text = prompt(gold, enc, m)
    ids = [c["id"] for c in gold["components"]]
    head = COMMON.format(taxonomy=TAXONOMY)
    write_task(case, "A", text, head + "\n\n" + RULES_A, "\n[finish]\nmax_turns = 1\n")
    write_weights(write_task(case, "WO", text, head + "\n\n" + RULES_WO, python_section(case, ids, 30)), case, enc, ids)
    d = write_task(case, "P", text, head + "\n\n" + RULES_P, python_section(case, ids, 60, PROBE))
    write_weights(d, case, enc, ids)
    (d / "arch.json").write_text(json.dumps(arch(m, enc)))


def main():
    meta = metadata()
    cases = sorted(int(p.stem.split("_")[1]) for p in (SRC / "hf/dataset/annotations").glob("case_*.json"))
    for case in cases:
        prepare(case, json.loads((SRC / f"hf/dataset/annotations/case_{case}.json").read_text()), meta[case])
    print(json.dumps({"ok": True, "cases": len(cases), "out": str(OUT)}))


if __name__ == "__main__":
    main()
