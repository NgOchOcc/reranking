"""Shared chunk encoder and gated summary-vector compressor."""
import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """CrossAttn -> FFN1 -> SelfAttn -> FFN2, each with pre-norm and a skip."""

    def __init__(self, hidden_size, n_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn1 = nn.LayerNorm(hidden_size)
        self.ffn1 = self._ffn(hidden_size, ffn_mult)
        self.ln_self = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn2 = nn.LayerNorm(hidden_size)
        self.ffn2 = self._ffn(hidden_size, ffn_mult)

    @staticmethod
    def _ffn(width, multiplier):
        return nn.Sequential(nn.Linear(width, width * multiplier), nn.GELU(),
                             nn.Linear(width * multiplier, width))

    def forward(self, summary, h_d, key_padding_mask):
        q, kv = self.ln_q(summary), self.ln_kv(h_d)
        cross, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        summary = summary + cross
        summary = summary + self.ffn1(self.ln_ffn1(summary))
        normalized = self.ln_self(summary)
        self_out, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        summary = summary + self_out
        return summary + self.ffn2(self.ln_ffn2(summary))


class DocumentCompressor(nn.Module):
    """Fixed summary budget per nonempty chunk; legacy b_d is ungated S."""

    def __init__(self, backbone, hidden_size, num_summary_tokens=32,
                 n_cross_attn_layers=1, n_heads=8):
        super().__init__()
        if num_summary_tokens < 1 or n_cross_attn_layers < 1:
            raise ValueError("Summary-token and summary-block counts must be positive")
        self.backbone = backbone
        self.num_summary_tokens = num_summary_tokens
        self.summary_queries = nn.Parameter(torch.randn(num_summary_tokens, hidden_size) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(hidden_size, n_heads) for _ in range(n_cross_attn_layers)
        ])
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Chunk IDs and attention masks must have matching 2D shapes")
        if input_ids.shape[0] == 0 or not attention_mask.bool().any(dim=1).all():
            raise ValueError("Every encoded chunk must contain at least one real token")
        h_d = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False).last_hidden_state
        summary = self.summary_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        for block in self.blocks:
            summary = block(summary, h_d, attention_mask == 0)
        pi = torch.sigmoid(self.gate(summary)).squeeze(-1)
        return {"h_d": h_d, "doc_attention_mask": attention_mask, "b_d": summary,
                "pi": pi, "b_d_tilde": summary * pi.unsqueeze(-1)}
