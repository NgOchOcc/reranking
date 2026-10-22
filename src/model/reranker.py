"""Top-level latent-bottleneck reranker: wires the query adapter and document
compressor to a single shared Qwen3-0.6B backbone, and implements the
asymmetric MaxSim late-interaction scorer (proposal §4, Eq. 7).
"""
import math

import torch
import torch.nn as nn
from transformers import AutoModel

from .bidirectional import make_bidirectional
from .document_compressor import DocumentCompressor
from .query_adapter import ResidualQueryAdapter


class LatentBottleneckReranker(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        # Master weights stay float32; bf16 (if any) is applied as autocast
        # by the training script's Accelerator(mixed_precision=...), not
        # baked into the stored parameters -- the compressor/adapter layers
        # built below are freshly-initialized float32 nn.Modules, so forcing
        # the backbone to bf16 here would create a dtype mismatch the first
        # time its output feeds into them.
        backbone = AutoModel.from_pretrained(model_cfg.backbone_name)
        make_bidirectional(backbone)
        self.backbone = backbone  # shared by both towers; unfrozen

        embedding = backbone.get_input_embeddings()
        self.query_adapter = ResidualQueryAdapter(
            embedding=embedding,
            hidden_size=model_cfg.hidden_size,
            n_layers=model_cfg.n_query_adapter_layers,
            n_heads=model_cfg.n_query_adapter_heads,
        )
        self.doc_compressor = DocumentCompressor(
            backbone=backbone,
            hidden_size=model_cfg.hidden_size,
            num_summary_tokens=model_cfg.num_summary_tokens,
            n_cross_attn_layers=model_cfg.n_cross_attn_layers,
            n_heads=model_cfg.n_cross_attn_heads,
        )
        self._scale = 1.0 / math.sqrt(model_cfg.hidden_size)

    def encode_query(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.query_adapter(input_ids)

    def encode_document(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> dict:
        return self.doc_compressor(input_ids, attention_mask)

    def forward(self, mode: str, **kwargs):
        """Dispatches to encode_query/encode_document by name.

        Needed so that under DistributedDataParallel these parameterized
        paths are invoked through the wrapper's own __call__ (the only
        method DDP instruments for gradient-bucket tracking) instead of as
        a direct attribute access on the DDP object, which doesn't proxy
        custom methods and raises AttributeError.
        """
        if mode == "encode_query":
            return self.encode_query(**kwargs)
        if mode == "encode_document":
            return self.encode_document(**kwargs)
        raise ValueError(f"unknown mode {mode!r}")

    def maxsim(self, q: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Asymmetric MaxSim late-interaction score (Eq. 7), raw dot product
        scaled by 1/sqrt(d) (attention-style scaling, needed since downstream
        Margin-MSE targets are bounded teacher logit differences).

        q: (..., Lq, d) query vectors, b: (..., Lb, d) bottleneck vectors,
        with matching leading ("...") batch dims (broadcastable).
        Returns: (...,) scalar score per leading-batch element.
        """
        sim = torch.einsum("...qd,...kd->...qk", q, b) * self._scale
        return sim.max(dim=-1).values.sum(dim=-1)
