"""Residual Query Adapter (proposal §3, Eq. 1-2).

Intentionally shallow: the query only goes through the (shared, tied)
embedding lookup followed by a small adapter `F_Q`, never the full backbone
-- this is what caps edge NPU latency on the query path. The residual
skip-connection around `F_Q` preserves exact-match lexical signal in the
raw embeddings so the adapter can only *add* contextual refinement, not
overwrite it.
"""
import torch
import torch.nn as nn


class ResidualQueryAdapter(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embedding = embedding  # shared reference with the document backbone's embedding table
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.f_q = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """input_ids: (B, query_max_len), already padded/truncated to a fixed
        length with the shared mask-pad token (see data/dataset.py).
        Returns Q_latent: (B, query_max_len, hidden_size).
        """
        q_emb = self.embedding(input_ids)
        refined = self.f_q(q_emb)
        return self.layer_norm(q_emb + refined)
