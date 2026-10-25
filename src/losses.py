"""Relevance, token coverage, keypoint matching, and gate sparsity losses.

Coverage uses ungated summaries within each chunk. Keypoint matching uses
all gated summaries of the corresponding document. Positive gate scaling
cancels inside exact cosine similarity; the keypoint objective therefore
does not directly reward keeping gates open away from the norm epsilon.
"""
from numbers import Integral

import torch
import torch.nn.functional as F


def _mask_for(mask, shape, device, name):
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def _check_vector_pair(left, right, name):
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError(f"{name} vectors must have shape (batch, sequence, width)")
    if left.shape[0] != right.shape[0] or left.shape[2] != right.shape[2]:
        raise ValueError(f"{name} vectors must have matching batch sizes and widths")
    if any(size == 0 for size in left.shape) or any(size == 0 for size in right.shape):
        raise ValueError(f"{name} vectors must have nonempty batch, sequence, and width dimensions")


def _cosine_matrix(left, right, eps):
    # Even float32 inputs to bmm/einsum can be downcast by outer autocast.
    with torch.autocast(device_type=left.device.type, enabled=False):
        left = F.normalize(left.float(), dim=-1, eps=eps)
        right = F.normalize(right.float(), dim=-1, eps=eps)
        return torch.bmm(left, right.transpose(1, 2))


def soft_target_bce_loss(logits, targets, mask=None):
    """Stable mean BCE over valid examples, with teacher probabilities as targets.

    Missing entries are excluded before evaluating BCE, including validation
    of their targets. An entirely masked batch returns a differentiable zero.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if mask is None:
        valid = torch.ones_like(logits, dtype=torch.bool)
    else:
        valid = _mask_for(mask, logits.shape, logits.device, "mask")
    valid_logits = logits.masked_select(valid).float()
    valid_targets = targets.to(device=logits.device).masked_select(valid)
    if valid_logits.numel() == 0:
        return valid_logits.sum()
    if not torch.isfinite(valid_targets).all() or not ((valid_targets >= 0) & (valid_targets <= 1)).all():
        raise ValueError("valid teacher targets must be finite probabilities in [0, 1]")
    return F.binary_cross_entropy_with_logits(valid_logits, valid_targets.float())


def coverage_loss(h_d, doc_attention_mask, b_d, doc_token_ids, idf_table, eps=1e-8):
    """Mean IDF-weighted token coverage over unique chunks.

    h_d: (Nc, C, d), b_d: (Nc, M, d) ungated summaries;
    doc_attention_mask and doc_token_ids: (Nc, C); idf_table: (vocabulary,).
    Each chunk's weighted nearest-summary distance is divided by its exact
    positive weight sum. An all-zero-weight chunk contributes zero, remains
    in the chunk mean, and has zero coverage gradient.
    """
    _check_vector_pair(h_d, b_d, "coverage")
    if eps <= 0:
        raise ValueError("eps must be positive")
    real = _mask_for(doc_attention_mask, h_d.shape[:2], h_d.device, "doc_attention_mask")
    if tuple(doc_token_ids.shape) != tuple(h_d.shape[:2]):
        raise ValueError("doc_token_ids must match the chunk token dimensions")
    if not real.any(dim=1).all():
        raise ValueError("every coverage chunk must contain a real token")
    if idf_table.ndim != 1:
        raise ValueError("idf_table must be one-dimensional")

    # Look up only real tokens; padding carries neither an IDF value nor loss.
    token_ids = doc_token_ids.to(device=h_d.device)
    real_weights = idf_table.to(device=h_d.device)[token_ids[real]].float()
    if not torch.isfinite(real_weights).all() or not (real_weights >= 0).all():
        raise ValueError("coverage requires finite, nonnegative IDF weights")
    weights = h_d.new_zeros(h_d.shape[:2], dtype=torch.float32)
    weights[real] = real_weights

    distances = 1.0 - _cosine_matrix(h_d.masked_fill(~real.unsqueeze(-1), 0.0), b_d, eps)
    nearest = distances.min(dim=-1).values
    weight_sum = weights.sum(dim=-1)
    # A branch-safe denominator also keeps backward finite when W_r == 0.
    denominator = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    per_chunk = (nearest * weights).sum(dim=-1) / denominator
    return per_chunk.mean()


def keypoint_loss(keypoint_vectors, keypoint_attention_mask, document_vectors, document_summary_mask, eps=1e-8):
    """Mean 1-minus-tokenwise-cosine-MaxSim over unique document-keypoint pairs.

    keypoint_vectors: (P, Lf, d), one shared-encoder vector per keypoint token;
    document_vectors: (P, Ns, d), each pair's owning document's gated summaries;
    the masks have shape (P, Lf) and (P, Ns), respectively.
    Positive gates cancel inside cosine above the norm epsilon. Both vector
    inputs retain gradients, including the shared keypoint encoder path.
    """
    _check_vector_pair(keypoint_vectors, document_vectors, "keypoint")
    if eps <= 0:
        raise ValueError("eps must be positive")
    real_tokens = _mask_for(
        keypoint_attention_mask, keypoint_vectors.shape[:2], keypoint_vectors.device,
        "keypoint_attention_mask",
    )
    real_summaries = _mask_for(
        document_summary_mask, document_vectors.shape[:2], document_vectors.device,
        "document_summary_mask",
    )
    if not real_tokens.any(dim=1).all():
        raise ValueError("every keypoint must contain at least one real token")
    if not real_summaries.any(dim=1).all():
        raise ValueError("every document must have at least one real summary")

    similarities = _cosine_matrix(
        keypoint_vectors.masked_fill(~real_tokens.unsqueeze(-1), 0.0),
        document_vectors.masked_fill(~real_summaries.unsqueeze(-1), 0.0), eps,
    )
    similarities = similarities.masked_fill(~real_summaries.unsqueeze(1), float("-inf"))
    best = similarities.max(dim=-1).values.masked_fill(~real_tokens, 0.0)
    mean_match = best.sum(dim=-1) / real_tokens.sum(dim=-1)
    return (1.0 - mean_match).mean()


def sparsity_loss(pi, chunk_doc_index=None, num_documents=None):
    """Sum all real chunk gates per document, then average unique documents.

    pi: (Nc, M). chunk_doc_index: (Nc,), contiguous document IDs covering
    every document in the batch. Without a mapping, each row is a document.
    """
    if pi.ndim != 2 or pi.shape[0] == 0 or pi.shape[1] == 0:
        raise ValueError("pi must have nonempty shape (chunks, summary_slots)")
    if num_documents is not None:
        if isinstance(num_documents, bool) or not isinstance(num_documents, Integral) or num_documents < 1:
            raise ValueError("num_documents must be a positive integer")
        num_documents = int(num_documents)
    if chunk_doc_index is None:
        if num_documents is not None and num_documents != pi.shape[0]:
            raise ValueError("without chunk_doc_index each pi row must represent one document")
        return pi.float().sum(dim=-1).mean()

    if chunk_doc_index.ndim != 1 or chunk_doc_index.shape[0] != pi.shape[0]:
        raise ValueError("chunk_doc_index must have one entry per chunk")
    if chunk_doc_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("chunk_doc_index must contain integer document indices")
    indices = chunk_doc_index.to(device=pi.device, dtype=torch.long)
    present = torch.unique(indices)
    count = present.numel() if num_documents is None else num_documents
    if present.numel() != count or not torch.equal(present, torch.arange(count, device=pi.device)):
        raise ValueError("chunk_doc_index must cover every document index from 0 to num_documents - 1")
    per_document = pi.new_zeros(count, dtype=torch.float32).scatter_add(0, indices, pi.float().sum(dim=-1))
    return per_document.mean()


def total_loss(bce, coverage, keypoint, sparsity, loss_cfg):
    total = (bce + loss_cfg.lambda_coverage * coverage
             + loss_cfg.lambda_keypoint * keypoint
             + loss_cfg.lambda_sparsity * sparsity)
    return total, {
        "loss/total": total.detach().item(),
        "loss/bce": bce.detach().item(),
        "loss/coverage": coverage.detach().item(),
        "loss/keypoint": keypoint.detach().item(),
        "loss/sparsity": sparsity.detach().item(),
    }
