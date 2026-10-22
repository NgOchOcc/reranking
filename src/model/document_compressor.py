"""Latent Bottleneck Document Compressor (proposal §3.2, Eq. 3-6).

Three phases, implemented as three sub-modules composed in `forward`:
  1. Contextualization: the (bidirectionalized) Qwen3-0.6B backbone -> H_D.
  2. Cross-Attention Extraction: M=256 learnable summary queries attend over
     H_D via stacked Multi-Head Cross-Attention blocks -> B_D.
  3. Entropy-Driven Adaptive Gating: a linear+sigmoid head predicts a
     continuous retention probability pi_k per summary vector; the gate is
     applied multiplicatively (soft-pruning) so it stays differentiable.
"""
import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN block: summary tokens query the
    contextualized document tokens, Perceiver-resampler style."""

    def __init__(self, hidden_size: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size),
        )

    def forward(self, summary: torch.Tensor, h_d: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(summary)
        kv = self.ln_kv(h_d)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        summary = summary + attn_out
        summary = summary + self.ffn(self.ln_ffn(summary))
        return summary


class DocumentCompressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_summary_tokens: int = 256,
        n_cross_attn_layers: int = 1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.backbone = backbone  # bidirectionalized Qwen3Model, unfrozen, shared with query embedding table
        self.num_summary_tokens = num_summary_tokens
        self.summary_queries = nn.Parameter(torch.randn(num_summary_tokens, hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(hidden_size, n_heads) for _ in range(n_cross_attn_layers)]
        )
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> dict:
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_d = backbone_out.last_hidden_state  # (B, L_D, d)

        batch_size = input_ids.shape[0]
        summary = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = attention_mask == 0  # True where padded, per nn.MultiheadAttention convention
        b_d = summary
        for block in self.blocks:
            b_d = block(b_d, h_d, key_padding_mask)

        pi = torch.sigmoid(self.gate(b_d)).squeeze(-1)  # (B, M)
        b_d_tilde = b_d * pi.unsqueeze(-1)

        return {
            "h_d": h_d,                      # (B, L_D, d) pre-compression contextualized tokens
            "doc_attention_mask": attention_mask,  # (B, L_D)
            "b_d": b_d,                       # (B, M, d) pre-gate bottleneck
            "pi": pi,                         # (B, M) retention probabilities
            "b_d_tilde": b_d_tilde,           # (B, M, d) soft-masked bottleneck
        }
