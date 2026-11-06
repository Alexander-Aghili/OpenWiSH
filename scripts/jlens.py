"""Compute the J-lens transport J_l = E[d h_L,t' / d h_l,t] (t' >= t) for every layer of a Hugging Face decoder,
on a fixed public corpus. Host-side instrument: the agent never runs gradients. It gets the result as read-only arrays.

usage: assets/venvs/qwen/bin/python -I scripts/jlens.py --model assets/moe/model --corpus assets/moe/practice \
           --per-group 6 --tokens 128 --out assets/moe/lens
h_l is the residual stream after decoder layer l, h_L after the last layer (before the final norm). Writes J_<l>.npy
(float32, d_model x d_model, row i = output dim i) for l = 0 .. L-2, and meta.json with the corpus, pair counts and the
between-context dispersion E_p ||J_l^p - J_l||^2 / ||J_l||^2 per layer (Meyers, J-Vision, 2026, section 3.2)."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def pair_sums(h_last, hs, batch=32):
    """Per layer, the sum over position pairs t' >= t of d h_last[t'] / d h[t], as a d x d matrix (row = output dim).
    One batched backward per block of output dims, seeded on every position at once. Causality zeroes t' < t."""
    d = h_last.shape[-1]
    sums = [torch.zeros(d, h.shape[-1]) for h in hs]
    for start in range(0, d, batch):
        dims = list(range(start, min(start + batch, d)))
        seeds = torch.zeros(len(dims), *h_last.shape)
        for b, i in enumerate(dims):
            seeds[b, ..., i] = 1.0
        grads = torch.autograd.grad(h_last, hs, grad_outputs=seeds, retain_graph=True, is_grads_batched=True)
        for s, g in zip(sums, grads):
            s[dims] = g.sum(dim=tuple(range(1, g.dim() - 1)))  # sum over batch-of-one and source positions
    return sums


class Accumulator:
    """Mean of per-context Jacobians (each context weighted by its pair count) and their dispersion."""

    def __init__(self, layers):
        self.total = [None] * layers
        self.sq = [0.0] * layers
        self.pairs = 0
        self.done = []

    def save(self, path):
        """Write the running sums after each context, so a killed run resumes where it stopped."""
        tmp = Path(str(path) + ".tmp")
        torch.save({"total": self.total, "sq": self.sq, "pairs": self.pairs, "done": self.done}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, layers):
        acc = cls(layers)
        if Path(path).exists():
            st = torch.load(path, weights_only=True)
            acc.total, acc.sq, acc.pairs, acc.done = st["total"], st["sq"], st["pairs"], st["done"]
        return acc

    def add(self, sums, pairs):
        for i, s in enumerate(sums):
            jp = s / pairs
            self.total[i] = s.clone() if self.total[i] is None else self.total[i] + s
            self.sq[i] += pairs * float((jp ** 2).sum())
        self.pairs += pairs

    def result(self):
        mean = [t / self.pairs for t in self.total]
        disp = [self.sq[i] / self.pairs - float((m ** 2).sum()) for i, m in enumerate(mean)]
        return mean, disp


def run(model_dir, corpus, per_group, tokens, out, batch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("OPENWISH_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    texts = [(f.stem, i, t) for f in sorted(Path(corpus).glob("*.json")) for i, t in enumerate(json.loads(f.read_text())[:per_group])]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = out / "state.pt"
    acc = Accumulator.load(state, len(layers) - 1)
    for n, (group, i, text) in enumerate(texts):
        if f"{group}[{i}]" in acc.done:
            continue
        t0 = time.time()
        ids = tok(text, return_tensors="pt").input_ids[:, :tokens]
        hs = []
        hooks = [layer.register_forward_hook(lambda m, a, o: hs.append(o[0] if isinstance(o, tuple) else o)) for layer in layers]
        emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
        model(inputs_embeds=emb)
        for h in hooks:
            h.remove()
        s = ids.shape[1]
        acc.add(pair_sums(hs[-1], hs[:-1], batch), pairs=s * (s + 1) // 2)
        acc.done.append(f"{group}[{i}]")
        acc.save(state)
        del hs
        print(json.dumps({"context": n + 1, "of": len(texts), "group": group, "id": i, "tokens": s, "seconds": round(time.time() - t0)}), flush=True)
    mean, disp = acc.result()
    for l, j in enumerate(mean):
        np.save(out / f"J_{l}.npy", j.numpy().astype(np.float32))
    rel = [d / max(float((m ** 2).sum()), 1e-12) for d, m in zip(disp, mean)]
    (out / "meta.json").write_text(json.dumps({
        "model": str(model_dir), "corpus": [f"{g}[{i}]" for g, i, _ in texts], "tokens_per_context": tokens,
        "pairs": acc.pairs, "layers": len(mean), "definition": "J_l = E over contexts and pairs t' >= t of d h_L,t' / d h_l,t",
        "relative_dispersion": [round(r, 4) for r in rel]}, indent=1))


def write_logit_lens(out, layers, d):
    """Control for the J-lens: the plain logit lens, J_l = identity for every layer (later layers ignored)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for l in range(layers):
        np.save(out / f"J_{l}.npy", np.eye(d, dtype=np.float32))
    (out / "meta.json").write_text(json.dumps({
        "layers": layers, "definition": "logit lens: J_l = identity for every layer, so L_l = W_U diag(final norm)",
        "relative_dispersion": None}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="folder of <group>.json lists of texts")
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8, help="output dims per backward pass; memory grows with it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit-lens", action="store_true", help="write the identity control instead (no gradients)")
    a = ap.parse_args()
    if a.logit_lens:
        cfg = json.loads((Path(a.model) / "config.json").read_text())
        return write_logit_lens(a.out, cfg["num_hidden_layers"] - 1, cfg["hidden_size"])
    run(a.model, a.corpus, a.per_group, a.tokens, a.out, a.batch)


if __name__ == "__main__":
    main()
