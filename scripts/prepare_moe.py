"""Build the MoE environment: remove Python-code skill from Granite-3.1-1B-A400M by editing expert weights.

usage: assets/venvs/qwen/bin/python -I scripts/prepare_moe.py
Needs assets/moe/model (HF ibm-granite/granite-3.1-1b-a400m-base, Apache-2.0). Writes:
  assets/moe/base_net/   the editable tensors (router and expert output weights of every layer), float32 .npy
  assets/moe/practice/   32 code and 32 English texts the agent may probe
  assets/moe/eval/       64 code and 64 English hidden texts, and the base model's losses on them
Code comes from the Python 3.12 standard library, prose from three Project Gutenberg books. Practice and eval
texts come from disjoint files and books."""
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "assets/moe"
TOKENS = 256
BOOKS = {"practice": [1342], "eval": [98, 2701]}  # Pride and Prejudice | A Tale of Two Cities, Moby Dick


def editable(name):
    return name.endswith(("block_sparse_moe.router.layer.weight", "block_sparse_moe.output_linear.weight"))


def write_base_net():
    out = D / "base_net"
    out.mkdir(exist_ok=True)
    index = json.loads((D / "model/model.safetensors.index.json").read_text())["weight_map"]
    for name, shard in sorted(index.items()):
        if editable(name) and not (out / f"{name}.npy").exists():
            with safe_open(D / "model" / shard, "pt") as f:
                np.save(out / f"{name}.npy", f.get_tensor(name).float().numpy())


def window(tok, text, start_pattern=None):
    """About TOKENS tokens of `text`, starting at the first match of `start_pattern`."""
    if start_pattern:
        m = re.search(start_pattern, text, re.M)
        text = text[m.start():] if m else text
    ids = tok.encode(text).ids
    return tok.decode(ids[:TOKENS]) if len(ids) >= TOKENS else None


def code_texts(tok):
    files = sorted(p for p in Path("/usr/lib/python3.12").rglob("*.py") if "test" not in p.parts and "site-packages" not in p.parts)
    random.Random(0).shuffle(files)
    out = []
    for p in files:
        t = window(tok, p.read_text(errors="ignore"), r"^(def|class) ")
        if t:
            out.append(t)
        if len(out) == 96:
            break
    return out[:32], out[32:]


def gutenberg(book):
    path = D / "gutenberg" / f"{book}.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt", path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    start, end = text.find("*** START"), text.find("*** END")
    return text[text.find("\n", start) + 1:end]


def prose_texts(tok, books, n):
    out = []
    for b in books:
        paras = [p.strip().replace("\n", " ") for p in gutenberg(b).split("\n\n") if len(p.split()) > 40]
        paras = paras[len(paras) // 10:]  # skip front matter
        buf = ""
        for p in paras:
            buf = f"{buf} {p}".strip()
            t = window(tok, buf)
            if t:
                out.append(t)
                buf = ""
            if len(out) >= n * (books.index(b) + 1) // len(books):
                break
    return out[:n]


def main():
    tok = Tokenizer.from_file(str(D / "model/tokenizer.json"))
    write_base_net()
    code_practice, code_eval = code_texts(tok)
    sets = {"practice": {"code": code_practice, "english": prose_texts(tok, BOOKS["practice"], 32)},
            "eval": {"code": code_eval, "english": prose_texts(tok, BOOKS["eval"], 64)}}
    for split, groups in sets.items():
        (D / split).mkdir(exist_ok=True)
        for g, texts in groups.items():
            (D / split / f"{g}.json").write_text(json.dumps(texts, indent=0))
    print(json.dumps({"ok": True, **{f"{s}_{g}": len(t) for s, gs in sets.items() for g, t in gs.items()}}))


if __name__ == "__main__":
    sys.exit(main())
