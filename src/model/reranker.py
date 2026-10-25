"""Chunk-based document reranking with shared token-level keypoint encoding."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .bidirectional import make_bidirectional
from .document_compressor import DocumentCompressor
from .query_adapter import ResidualQueryAdapter


class LatentBottleneckReranker(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.doc_chunk_size = model_cfg.doc_chunk_size
        if self.doc_chunk_size < 1:
            raise ValueError("doc_chunk_size must be positive")
        # SDPA accepts the explicit noncausal padding mask. Accelerate handles
        # mixed precision while master parameters retain their loaded dtype.
        backbone = AutoModel.from_pretrained(model_cfg.backbone_name, attn_implementation="sdpa")
        make_bidirectional(backbone)
        self.backbone = backbone
        self.query_adapter = ResidualQueryAdapter(
            embedding=backbone.get_input_embeddings(), hidden_size=model_cfg.hidden_size,
            n_layers=model_cfg.n_query_adapter_layers, n_heads=model_cfg.n_query_adapter_heads,
        )
        self.doc_compressor = DocumentCompressor(
            backbone=backbone, hidden_size=model_cfg.hidden_size,
            num_summary_tokens=model_cfg.num_summary_tokens,
            n_cross_attn_layers=model_cfg.n_cross_attn_layers, n_heads=model_cfg.n_cross_attn_heads,
        )
        self.score_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        self._scale = 1.0 / math.sqrt(model_cfg.hidden_size)

    @property
    def score_scale(self):
        # Preserve strict positivity even when softplus would underflow.
        return F.softplus(self.score_alpha.float()).clamp_min(torch.finfo(torch.float32).tiny)

    def scores_to_logits(self, scores):
        return self.score_scale * scores.float() + self.score_bias.float()

    def scores_to_probabilities(self, scores):
        return torch.sigmoid(self.scores_to_logits(scores))

    def encode_query(self, input_ids):
        return self.query_adapter(input_ids)

    def encode_keypoint(self, input_ids, attention_mask):
        """Contextualize every keypoint token; no pooling, compressor, or truncation."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Keypoint IDs and masks must have matching 2D shapes")
        if input_ids.shape[0] == 0 or not attention_mask.bool().any(dim=1).all():
            raise ValueError("Every keypoint must contain at least one real token")
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=False).last_hidden_state

    def _chunk_documents(self, input_ids, attention_mask):
        """Split complete documents; strip batch padding without dropping real tokens."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Document IDs and masks must have matching 2D shapes")
        pad_id = getattr(self.backbone.config, "pad_token_id", None)
        pad_id = 0 if pad_id is None else pad_id
        chunks, masks, owners = [], [], []
        for owner in range(input_ids.shape[0]):
            real_ids = input_ids[owner][attention_mask[owner].bool()]
            if real_ids.numel() == 0:
                raise ValueError("Every document must contain at least one real token")
            for chunk in real_ids.split(self.doc_chunk_size):
                length = chunk.shape[0]
                chunks.append(F.pad(chunk, (0, self.doc_chunk_size - length), value=pad_id))
                masks.append(torch.arange(self.doc_chunk_size, device=chunk.device) < length)
                owners.append(owner)
        if not chunks:
            raise ValueError("Cannot encode an empty document batch")
        return (
            torch.stack(chunks), torch.stack(masks).to(attention_mask.dtype),
            torch.tensor(owners, dtype=torch.long, device=input_ids.device),
        )

    def encode_document(self, input_ids, attention_mask, chunk_doc_index=None, num_documents=None):
        """Encode whole document rows, or chunks with an explicit owning-document map.

        Chunkwise outputs support coverage and diagnostics. document_vectors
        contains all gated summaries per document, with a validity mask so
        batching padding never becomes a candidate for MaxSim.
        """
        if chunk_doc_index is None:
            count = input_ids.shape[0]
            if num_documents is not None and num_documents != count:
                raise ValueError("num_documents does not match the document batch")
            input_ids, attention_mask, chunk_doc_index = self._chunk_documents(input_ids, attention_mask)
            num_documents = count
        else:
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError("Chunk IDs and masks must have matching 2D shapes")
            if chunk_doc_index.ndim != 1 or chunk_doc_index.shape[0] != input_ids.shape[0]:
                raise ValueError("chunk_doc_index must contain one owner per chunk")
            if input_ids.shape[0] == 0:
                raise ValueError("Cannot encode an empty chunk batch")
            if input_ids.shape[1] > self.doc_chunk_size:
                raise ValueError("Mapped chunks exceed doc_chunk_size")
            if input_ids.shape[1] < self.doc_chunk_size:
                extra = self.doc_chunk_size - input_ids.shape[1]
                pad_id = getattr(self.backbone.config, "pad_token_id", None)
                input_ids = F.pad(input_ids, (0, extra), value=0 if pad_id is None else pad_id)
                attention_mask = F.pad(attention_mask, (0, extra))
            if chunk_doc_index.dtype != torch.long:
                raise ValueError("chunk_doc_index must use torch.long integer indices")
            if (chunk_doc_index < 0).any():
                raise ValueError("Chunk document indices must be nonnegative")
            if num_documents is None:
                num_documents = int(chunk_doc_index.max().item()) + 1
            if num_documents < 1 or (chunk_doc_index >= num_documents).any():
                raise ValueError("Chunk document indices exceed num_documents")
        counts = torch.bincount(chunk_doc_index, minlength=num_documents)
        if (counts == 0).any():
            raise ValueError("Every document must own at least one real chunk")
        doc_out = self.doc_compressor(input_ids, attention_mask)
        max_slots = int(counts.max().item()) * self.doc_compressor.num_summary_tokens
        document_vectors, document_masks = [], []
        for owner in range(num_documents):
            vectors = doc_out["b_d_tilde"][chunk_doc_index == owner].flatten(0, 1)
            real_slots = vectors.shape[0]
            document_vectors.append(F.pad(vectors, (0, 0, 0, max_slots - real_slots)))
            document_masks.append(torch.arange(max_slots, device=vectors.device) < real_slots)
        doc_out.update({
            "chunk_input_ids": input_ids, "chunk_doc_index": chunk_doc_index,
            "document_vectors": torch.stack(document_vectors),
            "document_summary_mask": torch.stack(document_masks),
        })
        return doc_out

    def maxsim(self, q, b, summary_mask=None):
        """Sum query-token maxima across all valid document summaries."""
        if b.shape[-2] == 0:
            raise ValueError("MaxSim requires at least one document summary")
        sim = torch.einsum("...qd,...kd->...qk", q, b) * self._scale
        if summary_mask is not None:
            if not summary_mask.bool().any(dim=-1).all():
                raise ValueError("MaxSim requires a valid summary for every document")
            sim = sim.masked_fill(~summary_mask.bool().unsqueeze(-2), float("-inf"))
        return sim.max(dim=-1).values.sum(dim=-1)

    def _training_forward(self, batch):
        """Keep all trainable branches in a single DDP-instrumented forward."""
        keypoint_doc_index = batch["keypoint_doc_index"]
        keypoint_count = batch["keypoint_input_ids"].shape[0]
        if keypoint_doc_index.ndim != 1 or keypoint_doc_index.shape[0] != keypoint_count:
            raise ValueError("keypoint_doc_index must contain one document owner per keypoint")
        if keypoint_doc_index.dtype != torch.long:
            raise ValueError("keypoint_doc_index must use torch.long integer indices")
        doc_out = self.encode_document(
            batch["doc_input_ids"], batch["doc_attention_mask"],
            chunk_doc_index=batch["chunk_doc_index"],
        )
        num_documents = doc_out["document_vectors"].shape[0]
        if ((keypoint_doc_index < 0) | (keypoint_doc_index >= num_documents)).any():
            raise ValueError("keypoint_doc_index must reference an encoded document")
        keypoint_vectors = self.encode_keypoint(batch["keypoint_input_ids"], batch["keypoint_attention_mask"])
        query_ids = torch.cat([
            batch["direct_query_ids"].unsqueeze(1), batch["indirect_query_ids"].unsqueeze(1),
            batch["neg_query_ids"], batch["hardneg_query_ids"],
        ], dim=1)
        batch_size, count, query_length = query_ids.shape
        query_vectors = self.encode_query(query_ids.flatten(0, 1)).reshape(
            batch_size, count, query_length, -1
        )
        doc_index = batch["doc_index"]
        scores = self.maxsim(
            query_vectors, doc_out["document_vectors"][doc_index].unsqueeze(1),
            doc_out["document_summary_mask"][doc_index].unsqueeze(1),
        )
        logits = self.scores_to_logits(scores)
        return {"doc_out": doc_out, "keypoint_vectors": keypoint_vectors,
                "scores": scores, "logits": logits, "probabilities": torch.sigmoid(logits)}

    def forward(self, mode, **kwargs):
        if mode == "training":
            return self._training_forward(kwargs["batch"])
        if mode == "encode_query":
            return self.encode_query(**kwargs)
        if mode == "encode_document":
            return self.encode_document(**kwargs)
        if mode == "encode_keypoint":
            return self.encode_keypoint(**kwargs)
        if mode == "score":
            return self.scores_to_logits(self.maxsim(**kwargs))
        raise ValueError(f"Unknown mode {mode!r}")
