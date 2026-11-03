"""The authors' Experiment 7 attention tool: run BOTH checkpoints on up to 16 texts (each at most 128 tokens, <bos>
put in front) and save only the attention weights, float16 [layers, heads, query, key], as
$OPENWISH_OUT/t<i>_<A|B>.npy, plus the token lists as $OPENWISH_OUT/tokens.json. Nothing else of the models' computation
(hidden states, MLP values, logits, outputs) is returned.

usage: gemma_attention.py --a <checkpoint dir A> --b <checkpoint dir B>   (request: $OPENWISH_REQ = a JSON list of texts)"""
import argparse
import json
import os

import numpy as np


def fail(msg):
    print(json.dumps({"ok": False, "error": msg}))
    raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--max-texts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    try:
        texts = json.load(open(os.environ["OPENWISH_REQ"]))
    except Exception as e:
        fail(f"the attention block must hold a JSON list of texts ({e})")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        fail("the attention block must hold a non-empty JSON list of strings")
    if len(texts) > a.max_texts:
        fail(f"at most {a.max_texts} texts per call, got {len(texts)}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.a)
    ids = []
    for i, t in enumerate(texts):
        x = tok(t, add_special_tokens=False)["input_ids"]
        if len(x) > a.max_tokens:
            fail(f"text {i} has {len(x)} tokens; at most {a.max_tokens}")
        ids.append([tok.bos_token_id] + x)
    out = os.environ["OPENWISH_OUT"]
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = {}
    for side, path in (("A", a.a), ("B", a.b)):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
        with torch.no_grad():
            for i, x in enumerate(ids):
                r = m(torch.tensor([x], device=dev), output_attentions=True)
                att = torch.stack([l[0] for l in r.attentions]).float().cpu().numpy().astype(np.float16)
                f = f"t{i}_{side}.npy"
                np.save(os.path.join(out, f), att)
                files.setdefault(i, []).append(f)
        del m
    tokens = [[tok.convert_ids_to_tokens(t) for t in x] for x in ids]
    json.dump(tokens, open(os.path.join(out, "tokens.json"), "w"))
    print(json.dumps({"ok": True, "texts": len(ids), "tokens": tokens, "files": files,
                      "shape": "[layers, heads, query position, key position], float16"}))


if __name__ == "__main__":
    main()
