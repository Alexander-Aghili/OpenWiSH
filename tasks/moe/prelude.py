# MoE environment helpers, on top of the OpenWiSH kernel. Every write goes through the edit gate (save_edit).
import json

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

CONFIG = json.load(open("/model/config.json"))
TOKENIZER = Tokenizer.from_file("/model/tokenizer.json")
PRACTICE = {g: json.load(open(f"/practice/{g}.json")) for g in ("code", "english")}
_INDEX = json.load(open("/model/model.safetensors.index.json"))["weight_map"]


def load_original(name):
    """Any tensor of the original checkpoint (attention, norms, embeddings, input_linear ...) as float32."""
    with safe_open("/model/" + _INDEX[name], "pt") as f:
        return f.get_tensor(name).float().numpy()


def router(layer):
    """Current router weights of a layer, [experts, hidden]. Router logits are hidden_state @ router.T."""
    return load(f"model.layers.{layer}.block_sparse_moe.router.layer.weight")


def expert_out(layer):
    """Current expert output weights of a layer, [experts, hidden, ffn]."""
    return load(f"model.layers.{layer}.block_sparse_moe.output_linear.weight")


def zero_experts(layer, experts, reason):
    """Zero the output weights of these experts in one layer: when the router picks them, they add nothing."""
    name = f"model.layers.{layer}.block_sparse_moe.output_linear.weight"
    w = load(name).copy()
    w[list(experts)] = 0
    return save_edit(name, w, reason)
