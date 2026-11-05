# MoE environment helpers, on top of the OpenWiSH kernel. Every write goes through the edit gate (save_edit).
import json

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

CONFIG = json.load(open("/model/config.json"))
TOKENIZER = Tokenizer.from_file("/model/tokenizer.json")
PRACTICE = {g: json.load(open(f"/practice/{g}.json")) for g in ("code", "english", "docs") if __import__("os").path.exists(f"/practice/{g}.json")}
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


# The J-lens (lens task variant only): host-computed before the run, read-only arrays in /lens.
import os as _os

if _os.path.isdir("/lens"):
    LENS_META = json.load(open("/lens/meta.json"))

    def lens(layer):
        """J_l [1024, 1024]: average Jacobian of the last layer's residual output (before the final norm) with respect
        to the residual stream after layer `layer`, over position pairs t' >= t of the lens corpus. J_23 = identity."""
        return np.eye(CONFIG["hidden_size"], dtype=np.float32) if layer == CONFIG["num_hidden_layers"] - 1 else np.load(f"/lens/J_{layer}.npy")

    _UNEMBED = None

    def lens_rows(layer, token_ids):
        """Rows of the J-lens L_l = W_U diag(final norm) J_l for these tokens, [n, 1024]: the direction in the residual
        stream after layer `layer` that raises each token's logit (up to a positive per-position scale)."""
        global _UNEMBED
        if _UNEMBED is None:
            _UNEMBED = load_original("model.embed_tokens.weight") * load_original("model.norm.weight")[None, :]
        return _UNEMBED[list(token_ids)] @ lens(layer)
