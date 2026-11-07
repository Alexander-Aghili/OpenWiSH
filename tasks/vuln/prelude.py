# Vulnerability environment helpers, on top of the OpenWiSH kernel. Every write goes through the edit gate (save_edit).
import json

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

CONFIG = json.load(open("/model/config.json"))
TOKENIZER = Tokenizer.from_file("/model/tokenizer.json")
YES, NO = 13602, 8725  # token ids of "YES" and "NO"
QUESTION = "Can calling entry() reach a NULL pointer dereference on any execution path? Answer YES or NO."
_FILE = safe_open("/model/model.safetensors", "pt")


def prompt_text(code):
    """The exact text the model reads for one program (thinking off). The answer is read at the last token."""
    content = f"Read this C program.\n\n```c\n{code}\n```\n\n{QUESTION}"
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt_tokens(code):
    """Token strings of prompt_text(code), in the order the probe's all_positions arrays use."""
    return TOKENIZER.encode(prompt_text(code)).tokens


def load_original(name):
    """Any tensor of the original checkpoint (attention, norms, embeddings ...) as float32."""
    return _FILE.get_tensor(name).float().numpy()


def mlp_name(layer, part):
    """part is gate, up or down."""
    return f"model.language_model.layers.{layer}.mlp.{part}_proj.weight"


def mlp(layer):
    """Current (gate [3584, 1024], up [3584, 1024], down [1024, 3584]) of one layer. Neuron i reads with row i of gate
    and up, and writes with column i of down: out = down @ (silu(gate @ x) * up @ x)."""
    return tuple(load(mlp_name(layer, p)) for p in ("gate", "up", "down"))


def unembed(token_ids):
    """Output embedding rows (tied to the input embeddings) for these tokens, [n, 1024], before the final norm's scale."""
    return load_original("model.language_model.embed_tokens.weight")[list(token_ids)]
