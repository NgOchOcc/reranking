"""Use padding-only attention for document chunks and whole keypoints.

This patch targets Qwen3's _update_causal_mask interface. The model loader
selects SDPA, which accepts its explicit four-dimensional additive mask.
"""
import types

import torch


def _bidirectional_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
    if attention_mask is None:
        attention_mask = torch.ones(
            input_tensor.shape[:2], dtype=torch.long, device=input_tensor.device
        )
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    seq_len = attention_mask.shape[-1]
    # 1 = real token, 0 = padding (HF convention) -> additive mask, 0 where attendable
    keep = attention_mask[:, None, None, :].to(dtype=dtype)  # (B, 1, 1, L)
    additive = (1.0 - keep) * min_dtype
    return additive.expand(-1, 1, seq_len, -1).clone()


def make_bidirectional(backbone):
    """Monkeypatches `backbone` (a Qwen3Model instance) in place and returns it."""
    if not hasattr(backbone, "_update_causal_mask"):
        raise TypeError(f"{type(backbone)} has no _update_causal_mask to patch")
    backbone._update_causal_mask = types.MethodType(_bidirectional_update_causal_mask, backbone)
    backbone.config.use_cache = False
    for layer in getattr(backbone, "layers", []):
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = False
    return backbone
