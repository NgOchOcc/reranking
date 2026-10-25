# Evaluating the trained compressor

Evaluate the compressor in this order:

1. Encode each complete document as chunks, then **prune summary slots using
   the trained gating head**, before computing either metric.
2. Sample queries associated with held-out documents, rank the candidate
   documents by adapted-query MaxSim, and calculate **Precision@1**.
3. Encode every document token with the **frozen original Qwen3 embedding
   backbone**, compare it with its own chunk's retained summaries, and report
   all token matches and **coverage at a fixed similarity threshold**.

This document specifies an evaluation procedure using the existing model APIs.
The current scripts in [`src/eval/`](../src/eval/) provide inspection tools;
they do not yet implement this complete evaluation. See
[`compression.md`](compression.md) for the architecture and training losses.

## Inputs and model loading

Keep a fixed evaluation manifest containing:

| Input | Required information |
|---|---|
| Trained compressor | Checkpoint path and the corresponding run's `config.yaml` and tokenizer |
| Candidate documents | Unique `doc_id` and complete `body` for every document |
| Queries | Unique `query_id`, query text, query type, and relevant document IDs |
| Sampling | Random seed, document count, and queries sampled per document/type |
| Coverage reference | Frozen original backbone identifier/revision and final hidden-state layer |
| Coverage threshold | Prespecified cosine threshold `tau` in `[-1, 1]` |
| Pruning policy | Per-chunk gate quartiles, linear interpolation, and IQR multiplier `1.5` |

Split by document before sampling evaluation queries. Documents, their duplicate
bodies, and associated queries should be held out from compressor training.
Deduplicate repeated `doc_id` values and reject conflicting bodies. Retain a
stable ordering of document IDs for reproducible ties.

Load the effective configuration and tokenizer saved with the training run,
rather than assuming today's defaults match the checkpoint:

```python
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import Config
from src.data.dataset import pad_query_ids
from src.model.bidirectional import make_bidirectional
from src.model.reranker import LatentBottleneckReranker

run_dir = Path("runs/default")
cfg = Config.from_yaml(run_dir / "config.yaml")
tokenizer = AutoTokenizer.from_pretrained(run_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LatentBottleneckReranker(cfg.model)
state = torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device).eval()
```

The saved weights are a bare model `state_dict`, including the query adapter,
summary refinement blocks, gating head, and relevance calibration parameters.
Use `torch.inference_mode()` for evaluation and never update either model.

## Step 1: encode chunks and prune using the trained gates

Tokenize each full document with `add_special_tokens=False` and
`truncation=False`. `model.encode_document` divides it into consecutive,
nonoverlapping chunks of `cfg.model.doc_chunk_size` tokens: 1,024 by default.
Only the last chunk is right-padded. Reject empty documents and retain every
real token, including tokens beyond the first chunk.

The relevant chunkwise outputs are:

| Output | Shape | Meaning |
|---|---|---|
| `chunk_input_ids` | `(num_chunks, chunk_size)` | Exact token IDs, including final padding |
| `doc_attention_mask` | `(num_chunks, chunk_size)` | Real-token mask |
| `chunk_doc_index` | `(num_chunks,)` | Document owner of each chunk |
| `b_d` | `(num_chunks, M, d)` | Ungated summaries $s_{r,k}$ |
| `pi` | `(num_chunks, M)` | Trained sigmoid gate outputs $\pi_{r,k}$ |
| `b_d_tilde` | `(num_chunks, M, d)` | Gated summaries $b_{r,k}=\pi_{r,k}s_{r,k}$ |

Here $M$ is `cfg.model.num_summary_tokens`, 32 by default. For **each chunk
independently**, compute quartiles over its $M$ sigmoid gate outputs:

$$
Q_{1,r}=\operatorname{quantile}_{0.25}(\pi_r),\qquad
Q_{3,r}=\operatorname{quantile}_{0.75}(\pi_r),\qquad
\operatorname{IQR}_r=Q_{3,r}-Q_{1,r}.
$$

The lower box-plot fence and retained slot set are

$$
t_r=Q_{1,r}-1.5\operatorname{IQR}_r,\qquad
\mathcal K_r=\{k:\pi_{r,k}\geq t_r\}.
$$

Prune every slot **strictly below** the fence. Keep equality and do not prune
high outliers. This procedure defines the quartile population per chunk,
matching the chunk-local coverage definition below; do not pool gates across
documents or evaluation batches. Use linear quantile interpolation and
float32 gate values to make the rule reproducible.

This is a relative outlier rule, not a fixed retention budget. It may prune
nothing, including when the fence is negative. Equal gates have zero IQR and
are all retained. With finite, nonempty gates, at least one slot survives.
Reject nonfinite outputs rather than inventing a fallback pruning policy.

```python
def prune_chunk_summaries(doc_out):
    # pi: (num_chunks, M); operate on sigmoid outputs, not gate logits.
    pi = doc_out["pi"].float()
    if not torch.isfinite(pi).all():
        raise ValueError("Nonfinite gate outputs")
    quartiles = torch.quantile(
        pi, pi.new_tensor([0.25, 0.75]), dim=-1, interpolation="linear"
    )
    q1, q3 = quartiles.unbind(0)
    lower_fence = q1 - 1.5 * (q3 - q1)
    keep = pi >= lower_fence[:, None]
    assert keep.any(dim=-1).all()
    return keep, lower_fence


@torch.inference_mode()
def prepare_document(text):
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False, return_tensors="pt"
    )
    doc_out = model.encode_document(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
    )
    keep, lower_fence = prune_chunk_summaries(doc_out)
    # This helper encodes one document. Combine its retained chunks for ranking.
    ranking_vectors = doc_out["b_d_tilde"][keep].float()
    return doc_out, keep, lower_fence, ranking_vectors
```

Cache the retained vectors once per document and reuse them across queries.
Keep original chunk and slot IDs for diagnostics. Retain the ungated selected
vectors separately for coverage; full token activations are unnecessary in a
persistent retrieval cache.

Remove pruned slots physically, as above, or mask their similarities to
negative infinity before MaxSim. **Do not replace pruned vectors with zeros**:
a zero score could beat a valid negative similarity. The existing
`document_summary_mask` masks batch padding only; it does not apply gate
pruning. If packing several documents together, combine the pruning mask
with this padding mask using the correct `chunk_doc_index` ownership.

## Step 2: sample queries and calculate Precision@1

### Sample a fixed query list

1. Select a reproducible list of held-out documents with a recorded seed.
   Every selected document is a candidate for every sampled query.
2. Collect the existing `direct_query` and `indirect_query` values from their
   keypoint rows, or generate queries externally from those documents and
   check that their answers occur in the source documents. Query generation
   is outside this repository's evaluation APIs.
3. Deduplicate query text and merge its relevant document IDs. Sample a fixed
   number per source document and query type without replacement; record
   any shortfall. Freeze the resulting query list before comparing models.
4. Use available relevance judgments to include all known relevant candidates.
   If only source-document labels exist, explicitly report that limitation:
   an equally relevant different document can otherwise count as a miss.

An evaluation query record can use this schema:

```json
{
  "query_id": "q-001",
  "text": "What sensor process does Borealis use?",
  "query_type": "direct",
  "source_doc_id": "doc-001",
  "relevant_doc_ids": ["doc-001"]
}
```

Each sampled query must have at least one labeled relevant document in the
candidate list. `negative_queries` and `hard_negative_queries` are negative
with respect to their source document; do not label that document relevant
for those queries. They require their own relevance labels to enter this
Precision@1 calculation.

### Rank documents with the trained query adapter

Prepare query IDs with `pad_query_ids`, using the run's `query_max_len` and
`mask_token`. `model.encode_query` performs the shared embedding lookup and
trained residual query adapter. It does not run the full document backbone.
All fixed query slots, including the trained mask-padding slots, participate
in scoring, just as they do during training.

For adapted query vectors $q_1,\ldots,q_{L_Q}$ and a candidate document $D$,
score against the **gated, retained** vectors from all its chunks:

$$
\operatorname{MaxSim}(Q,D)=
\sum_{i=1}^{L_Q}
\max_{\substack{r\in\operatorname{chunks}(D)\\k\in\mathcal K_r}}
\frac{q_i^\top(\pi_{r,k}s_{r,k})}{\sqrt d}.
$$

This matches `model.maxsim`: scaled dot products, maximum over summaries,
then sum over query slots. Do not pool the query into one vector, normalize
retrieval vectors to cosine, or take a separate best-document result per
chunk. The same candidate document can supply different best summaries for
different query slots.

```python
@torch.inference_mode()
def rank_query(query_text, document_cache):
    # document_cache maps string doc_id -> retained gated ranking_vectors.
    mask_id = tokenizer.convert_tokens_to_ids(cfg.model.mask_token)
    query_ids = torch.tensor([
        pad_query_ids(tokenizer, query_text, cfg.model.query_max_len, mask_id)
    ], dtype=torch.long, device=device)
    q = model.encode_query(query_ids).float()
    scores = {}
    for doc_id in sorted(document_cache):
        b = document_cache[doc_id].to(device=device, dtype=torch.float32)
        scores[doc_id] = model.maxsim(q, b.unsqueeze(0)).item()
    # Highest score first; exact ties use ascending string doc_id.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
```

For example, build `document_cache[doc_id] = prepare_document(body)[3]` for
each candidate before running the queries. The snippets assume evaluation
without an outer mixed-precision autocast context and finite model outputs.
For larger corpora, batch scoring or use vector blocks while preserving the
same maximum across every retained summary of each document.

The positive learned relevance scale, bias, and sigmoid are unnecessary for
ranking: they preserve the mathematical ordering of raw MaxSim scores. Use
raw scores to avoid sigmoid saturation introducing numerical ties.

For $N$ sampled queries, let $R_q$ be the relevant candidate-document IDs and
$\widehat D_q$ the highest-scoring document. Then

$$
\operatorname{Precision@1}
=\frac{1}{N}\sum_{q=1}^{N}\mathbf 1[\widehat D_q\in R_q].
$$

Save `query_id`, top-1 document ID, raw score, relevant IDs, and the binary
hit for every query. Report hits, evaluated query count, Precision@1, and
candidate count, with separate direct/indirect results as well as the combined
query average. A top-1 result is correct if it is any labeled relevant document.

## Step 3: report every token and thresholded summary coverage

### Use the frozen original embedding backbone as the reference

Load a **separate, frozen original** `Qwen/Qwen3-Embedding-0.6B` backbone,
or the original embedding backbone named in the run's configuration. Use the
same pretrained revision and tokenizer as the training initialization, and
record that revision in the evaluation manifest. Apply the same bidirectional
attention setup and process the exact chunk IDs and masks from Step 1:

```python
teacher = AutoModel.from_pretrained(
    cfg.model.backbone_name, attn_implementation="sdpa"
)
teacher = make_bidirectional(teacher).to(device).eval()
teacher.requires_grad_(False)


@torch.inference_mode()
def teacher_tokens_for_chunk(doc_out, r):
    return teacher(
        input_ids=doc_out["chunk_input_ids"][r:r + 1],
        attention_mask=doc_out["doc_attention_mask"][r:r + 1],
        use_cache=False,
    ).last_hidden_state[0]
```

Pin the original revision in `from_pretrained(..., revision=...)` when it is
known; the saved configuration currently stores the model name, not a revision.
Never load the trained compressor weights into `teacher`. Its outputs here
are contextual final-layer token vectors, with no pooling, query adapter, or
summary compression. Processing one chunk at a time bounds teacher memory.

`doc_out["h_d"]` contains **trained student** token vectors, which the training
coverage loss uses; it is not this frozen reference. The discriminative
`Qwen3-Reranker-8B` teacher produces relevance probabilities and is also not
the embedding teacher used here. Comparing raw input-embedding table rows
would be a different coverage metric.

The teacher and summaries must share tokenization, vector width, and a
meaningful comparison space. Starting from the same backbone provides the
reference coordinates, but fine-tuning can move student representations.
Record this reference choice when interpreting absolute cosine thresholds.

### Compute chunk-local token-to-summary cosine similarities

Let $e^T_{r,j}$ be the frozen teacher vector for real token $j$ of chunk $r$.
For every retained slot $k\in\mathcal K_r$, define

$$
S_{r,jk}=\operatorname{cosine}(e^T_{r,j},s_{r,k}).
$$

Use the retained **ungated** summaries `b_d` for cosine. Pruning has already
used the trained gates; positive gate scaling cancels in cosine away from
the normalization epsilon. This also avoids making tiny gated norms affect
the normalization. Retrieval still uses gated `b_d_tilde` as specified above.

For a threshold $\tau$ fixed before the test run, token $j$ is covered when
at least one retained summary in its own chunk matches it:

$$
c_{r,j}(\tau)=\mathbf 1\left[
\max_{k\in\mathcal K_r} S_{r,jk}\geq\tau
\right].
$$

With $\ell_r$ real tokens, the requested coverage is

$$
\operatorname{Coverage}_r(\tau)
=\frac{1}{\ell_r}\sum_{j=1}^{\ell_r}
\mathbf 1\left[\max_{k\in\mathcal K_r} S_{r,jk}\geq\tau\right].
$$

Use **greater than or equal to** the threshold, following this equation.
Exclude all padded positions. Do not match a token to another chunk's
summaries, even when those summaries belong to the same document. This is
an unweighted count of token positions, distinct from the IDF-weighted,
continuous coverage loss used in training.

```python
import torch.nn.functional as F


@torch.inference_mode()
def evaluate_chunk_coverage(doc_out, keep, r, tau):
    if not -1.0 <= tau <= 1.0:
        raise ValueError("tau must be a finite cosine threshold in [-1, 1]")
    real = doc_out["doc_attention_mask"][r].bool()
    token_positions = real.nonzero(as_tuple=True)[0]
    slot_ids = keep[r].nonzero(as_tuple=True)[0]
    teacher_tokens = teacher_tokens_for_chunk(doc_out, r)[real].float()
    summaries = doc_out["b_d"][r, slot_ids].float()
    if not torch.isfinite(teacher_tokens).all() or not torch.isfinite(summaries).all():
        raise ValueError("Nonfinite coverage vectors")
    similarities = (
        F.normalize(teacher_tokens, dim=-1, eps=1e-8)
        @ F.normalize(summaries, dim=-1, eps=1e-8).T
    ).clamp(-1.0, 1.0)
    matches = similarities >= tau
    best_similarity, best_column = similarities.max(dim=-1)
    covered = matches.any(dim=-1)
    return {
        "token_positions": token_positions,
        "retained_slot_ids": slot_ids,
        "similarities": similarities,
        "matching_summaries": matches,
        "best_summary_slot": slot_ids[best_column],
        "best_similarity": best_similarity,
        "covered": covered,
        "covered_token_count": int(covered.sum().item()),
        "real_token_count": int(real.sum().item()),
        "coverage": covered.float().mean().item(),
    }
```

Call this helper for every chunk returned by `prepare_document`, using its
unchanged pruning mask. Choose `tau` on a separate validation set or specify
it in advance; do not select it from test-set coverage. Cosine coverage
measures vector agreement and is not proof that the summaries preserve every
fact. Use the same threshold and teacher reference across compared checkpoints.

### Export all token positions and their matching summaries

Write one record per real token, including uncovered tokens:

| Field | Meaning |
|---|---|
| `doc_id`, `chunk_id` | Owning document and zero-based chunk ordinal within it |
| `token_index_in_chunk` | Zero-based real-token position in the chunk |
| `token_index_in_document` | `chunk_id * doc_chunk_size + token_index_in_chunk` |
| `token_id`, `token` | Original token ID and `tokenizer.convert_ids_to_tokens` string |
| `best_summary_slot`, `best_similarity` | Original slot ID and maximum retained cosine |
| `matching_summary_slots`, `matching_scores` | Every retained slot with cosine `>= tau` and its score |
| `covered` | Whether the matching-slot list is nonempty |

Token strings are subword pieces; keep IDs and positions so repeated tokens
remain separate. Use original full-text character offsets, if available,
to render exact source text. Do not assume independently decoded token pieces
reconstruct whitespace or partial Unicode characters correctly.

For each retained summary slot, also export its gate and the list of matching
token positions/strings/scores from that chunk. Derive these lists from
**every true entry** of `matching_summaries`, not only each token's best slot.
A token may match several summaries; count it once in chunk coverage.

Export per-chunk `Q1`, `Q3`, IQR, lower fence, all gates, retained original
slot IDs, slots before/after pruning, real/covered token counts, `tau`, and
coverage. Optionally export the full real-token-by-retained-summary score
matrix for threshold inspection.

For document $D$, aggregate by real-token counts:

$$
\operatorname{Coverage}_D(\tau)
=\frac{\sum_{r\in D}\ell_r\operatorname{Coverage}_r(\tau)}
{\sum_{r\in D}\ell_r}.
$$

Report corpus micro coverage as total covered tokens divided by total real
tokens, and document macro coverage as the mean document coverage. Label both
explicitly; averaging chunk percentages equally overweights short final chunks.

## Evaluation outputs

A complete evaluation should save:

- The manifest, candidate list, sampled queries, and relevance judgments.
- Per-query top-1 results and overall/direct/indirect Precision@1.
- Per-chunk pruning statistics, retained slot IDs, and threshold coverage.
- Every real token with its coverage flag and all above-threshold summary matches.
- Each retained summary's matching tokens, and document/corpus coverage totals.
- Retained summary counts and the fraction removed, alongside quality metrics.

An optional unpruned comparison can show the effect of pruning, using the
same checkpoint, queries, candidates, teacher, and threshold. Label it
separately: the primary results above always apply IQR pruning first.

The existing `inspect_bottleneck` and `sample_beir_compress` visualizations
use raw input-embedding comparisons and selected high-gate slots. Their HTML
and gate summaries are useful diagnostics, but do not implement the
chunk-local frozen-teacher coverage or retrieval evaluation specified here.
