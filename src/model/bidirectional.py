"""Turn a (causal) Qwen3 transformer backbone into a bidirectional encoder.

Qwen3-0.6B is pretrained as a causal LM, but the proposal's document encoder
(Eq. 3, "stacked self-attention layers ... contextualized token embeddings")
needs full-context, both-directions attention to resolve coreferences
across the whole document (see §1/§4 motivation). This is the standard
"bidirectionalize a decoder LM" trick (cf. LLM2Vec): we replace
`Qwen3Model._update_causal_mask` with a version that builds a padding-only
additive mask (no causal/triangular component). Every decoder layer in
`Qwen3Model.forward` consumes whatever mask this method returns, under both
the eager and sdpa attention backends, so this one patch point is
sufficient regardless of backend.

Safe to do here because the backbone is fully unfrozen and fine-tuned for
this task anyway, so it can adapt away from its pretrained causal prior.
"""
import types

import torch


def _bidirectional_update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
    if attention_mask is None:
        return None
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
    return backbone
