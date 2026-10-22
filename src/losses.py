"""The four training losses from proposal §5, adapted to this dataset's
(fixed document, varying query) negative structure (see docs/proposal.tex
and the plan notes for the Margin-MSE adaptation rationale).

Numerical note on the Chamfer loss (Eq. 93-94): cosine similarity is scale
invariant to any *positive* scalar multiplier, so `cos(h_j, pi_k * b_k) ==
cos(h_j, b_k)` exactly, for every pi_k > 0 coming out of a sigmoid. We
therefore compute all cosine distances on the normalized *pre-gate* `b_d`
(mathematically identical to normalizing `b_d_tilde`) to avoid dividing by
near-zero norms as gates are pushed toward 0 by the sparsity loss. The gate
`pi_k` still enters exactly where Eq. 94 puts it: as the explicit outer
weight in the precision term.
"""
import torch
import torch.nn.functional as F


def margin_mse_loss(pos_scores, neg_scores, pos_mask, neg_mask, teacher_pos, teacher_neg):
    """Forms every valid positive x negative margin pair per example and
    matches the student margin to the teacher margin via MSE.

    pos_scores, teacher_pos: (B, P) -- P positive queries (direct/indirect)
    neg_scores, teacher_neg: (B, N) -- N negative queries (neg + hard-neg)
    pos_mask: (B, P), neg_mask: (B, N) -- 1 where the slot is a real (non-padding) query
    """
    student_margin = pos_scores.unsqueeze(2) - neg_scores.unsqueeze(1)  # (B, P, N)
    teacher_margin = teacher_pos.unsqueeze(2) - teacher_neg.unsqueeze(1)  # (B, P, N)
    pair_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)  # (B, P, N)

    sq_err = (student_margin - teacher_margin) ** 2 * pair_mask
    denom = pair_mask.sum().clamp(min=1.0)
    return sq_err.sum() / denom


def idf_weighted_chamfer_loss(h_d, doc_attention_mask, b_d, pi, doc_token_ids, idf_table, beta=1.0, eps=1e-8):
    """h_d: (U, L, d), doc_attention_mask: (U, L), b_d: (U, M, d) pre-gate
    bottleneck, pi: (U, M), doc_token_ids: (U, L) token ids for IDF lookup,
    idf_table: (vocab_size,) tensor. Returns a scalar (mean over U docs)."""
    h_norm = F.normalize(h_d.float(), dim=-1, eps=eps)
    b_norm = F.normalize(b_d.float(), dim=-1, eps=eps)
    cos_sim = torch.einsum("uld,umd->ulm", h_norm, b_norm)  # (U, L, M)
    cos_dist = 1.0 - cos_sim

    token_mask = doc_attention_mask.float()  # (U, L), 1 = real token

    # Coverage: per real doc token, IDF-weighted min distance over bottleneck vectors.
    min_over_m = cos_dist.min(dim=2).values  # (U, L)
    idf_weights = idf_table[doc_token_ids]  # (U, L)
    coverage = (idf_weights * min_over_m * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)

    # Precision: per bottleneck vector, gate-weighted min distance over real doc tokens.
    pad_mask_l = (doc_attention_mask == 0).unsqueeze(-1)  # (U, L, 1)
    cos_dist_masked = cos_dist.masked_fill(pad_mask_l, float("inf"))
    min_over_l = cos_dist_masked.min(dim=1).values  # (U, M)
    precision = (pi * min_over_l).sum(dim=1) / pi.sum(dim=1).clamp(min=eps)

    return (coverage + beta * precision).mean()


def consistency_loss(direct_scores, indirect_scores):
    return ((direct_scores - indirect_scores) ** 2).mean()


def sparsity_loss(pi):
    """pi: (U, M). Sums gates per document, averaged over the batch's unique
    documents (no D-/D+ pairing exists in this adapted setup; see plan)."""
    return pi.sum(dim=1).mean()


def total_loss(margin, chamfer, consistency, sparsity, loss_cfg):
    total = margin + loss_cfg.lambda_chamfer * chamfer + loss_cfg.lambda_consistency * consistency \
        + loss_cfg.lambda_sparsity * sparsity
    return total, {
        "loss/total": total.item(),
        "loss/margin_mse": margin.item(),
        "loss/chamfer": chamfer.item(),
        "loss/consistency": consistency.item(),
        "loss/sparsity": sparsity.item(),
    }
