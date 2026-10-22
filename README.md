# Latent Bottleneck Reranker

Implementation of the ColBERT-style late-interaction reranker described in
[`docs/proposal.tex`](docs/proposal.tex): a document is compressed into a
*bounded* (≤256-vector) "latent bottleneck" instead of one vector per token,
so late-interaction scoring stays cheap enough for edge deployment while
keeping listwise/MaxSim-style precision. Trained by distilling from a
discriminative Cross-Encoder teacher (`Qwen/Qwen3-Reranker-8B`) against a
student built on `Qwen/Qwen3-0.6B` (hidden size 1024, 28 layers, tied
embeddings).

## Directory layout

```
sub_project/
  docs/proposal.tex              # the design doc this implementation follows
  configs/default.yaml           # all hyperparameters + paths
  src/
    config.py                    # dataclasses for model/teacher/data/loss/train config, YAML loader
    data/
      build_index.py             # joins 3 raw jsonl sources -> data/unified_examples.jsonl
      idf.py                     # builds the token-level IDF table -> data/idf_table.pt
      dataset.py                 # Dataset + Collator reading the teacher-scored jsonl at train time
    model/
      bidirectional.py           # monkeypatch: causal Qwen3 -> bidirectional encoder
      query_adapter.py           # ResidualQueryAdapter (query-side tower)
      document_compressor.py     # DocumentCompressor (document-side tower: backbone + cross-attn + gate)
      reranker.py                # LatentBottleneckReranker: wires both towers + MaxSim scorer
    distill/
      teacher_qwen_reranker.py   # Qwen3-Reranker-8B yes/no-logit scoring wrapper
      precompute_teacher_scores.py  # CLI: scores every needed (query, doc) pair, resumable
    losses.py                    # margin-MSE, IDF-weighted Chamfer, consistency, sparsity, combiner
    train.py                     # accelerate-based training loop CLI
    eval/
      inspect_bottleneck.py      # interpretability: what does each bottleneck slot encode?
      sample_beir_compress.py    # sanity-checks the compressor out-of-domain on BEIR corpora
  scripts/
    run_build_index.sh / run_build_idf.sh / run_precompute_teacher.sh / run_train.sh
```

## Data model

Three raw JSONL sources, line-joined by `doc_id` (not positionally aligned —
different doc counts per part file):

| Source | Fields | Notes |
|---|---|---|
| `full_run/keypoints_part*.jsonl` | `doc_id, url, title, body, keypoints: [str]` | canonical document text, 126,236 docs |
| `full_run/keypoint_queries_part*.jsonl` | `doc_id, body, keypoints: [{keypoint, queries: [{query, query_style: direct\|indirect}]}]` | every keypoint has exactly 1 direct + 1 indirect query, 122,100 docs |
| `negative_query/negative_queries_part*.jsonl` | `doc_id, keypoints: [{keypoint, negative_queries: [5], hard_negative_queries: [5]}]` | no body text, join on `doc_id` |

`src/data/build_index.py` joins these into `data/unified_examples.jsonl`,
one line per **`(doc_id, keypoint)`** training unit:

```json
{"doc_id": ..., "keypoint": ..., "body": ...,
 "direct_query": ..., "indirect_query": ...,
 "negative_queries": [...], "hard_negative_queries": [...]}
```

This unit aligns naturally with the Consistency loss (needs a direct/indirect
pair on the same document) and Margin-MSE (needs positive vs. negative
queries against the same document).

**Adaptation from the proposal:** Eq. 85-87 vary the *document* (`D+`/`D-`)
for a fixed query. The actual data instead varies the *query* for a fixed
document (no negative documents are provided), so the implementation uses
the symmetric form `Δs = S(Q+, D) − S(Q−, D)` — same calibration objective,
just matched to the available negative-*query* data.

## Model architecture

### Query tower — `ResidualQueryAdapter` (`src/model/query_adapter.py`)

Intentionally shallow (caps edge NPU latency): the query only goes through
the backbone's **tied embedding lookup**, padded/truncated to 32 tokens with
a repurposed unused special token (`<|vision_pad|>`, config: `mask_token`) —
never the full backbone. A small 2-layer bidirectional Transformer encoder
`F_Q` (8 heads) refines the embeddings; a residual skip + LayerNorm preserves
exact-match lexical signal (Eq. 1-2):

```
Q_emb    = Embedding(PadWithMask(Q, 32))
Q_latent = LayerNorm(Q_emb + F_Q(Q_emb))
```

### Document tower — `DocumentCompressor` (`src/model/document_compressor.py`)

Three phases (Eq. 3-6), composed in `forward`:

1. **Contextualization** — the full Qwen3-0.6B backbone, made bidirectional
   (`src/model/bidirectional.py` monkeypatches `Qwen3Model._update_causal_mask`
   to a padding-only mask, dropping the causal/triangular component — the
   standard "bidirectionalize a decoder LM" trick, à la LLM2Vec; safe since
   the backbone is fully unfrozen and fine-tuned here anyway) → `h_d`
   (contextualized tokens, one per real input token).
2. **Cross-Attention Extraction** — `M=256` learnable summary query vectors
   (`summary_queries`, a free `nn.Parameter`) attend over `h_d` via a stack
   of pre-norm `CrossAttnBlock`s (cross-attention + FFN, both with residual
   adds) → `b_d` (pre-gate bottleneck, 256×1024).
3. **Entropy-Driven Adaptive Gating** — `Linear(1024→1) + Sigmoid` on `b_d`
   → retention probability `π_k ∈ (0,1)` per slot; soft-masked bottleneck
   `b_d_tilde = π ⊙ b_d` is what downstream scoring/losses consume for
   differentiability (hard pruning at `π > 0.5` is an inference-time-only
   option, not used in training).

`encode_document` returns a dict with `h_d`, `doc_attention_mask`, `b_d`,
`pi`, `b_d_tilde` — every one of these is read by either `losses.py` or
`src/eval/inspect_bottleneck.py`.

### Scoring — `LatentBottleneckReranker.maxsim` (`src/model/reranker.py`)

Asymmetric late-interaction (MaxSim), raw dot product scaled by `1/sqrt(d)`
(attention-style scaling — needed since Margin-MSE targets are bounded
teacher logit differences, Eq. 7):

```
S(Q, D) = Σ_i max_j (q_i · b_j) / sqrt(d)
```

Bounded to `32 × 256` dot products per document — the source of the
sub-millisecond edge-latency claim.

## Two-teacher distillation

- **Generative teacher** (not implemented in this repo — produced upstream):
  a 70B-class LLM reads each document and emits the `keypoint` →
  `direct_query` / `indirect_query` pairs already sitting in
  `keypoint_queries_part*.jsonl`, plus the negative/hard-negative queries in
  `negative_queries_part*.jsonl`.
- **Discriminative teacher** — `Qwen/Qwen3-Reranker-8B`
  (`src/distill/teacher_qwen_reranker.py`), used in its documented
  **yes/no-logit classification** mode, not generation: format
  `<Instruct>: ...\n<Query>: {q}\n<Document>: {d}` inside the model's
  documented chat scaffold, read the final-position logits for the literal
  tokens `"yes"`/`"no"`, softmax between them → `P(yes)` as a calibrated
  relevance score in `(0, 1)`.
- `src/distill/precompute_teacher_scores.py` scores every `(query, doc)`
  pair the training set needs — direct, indirect, and a subsampled prefix of
  negative/hard-negative queries (`--neg-per-keypoint` / `--hardneg-per-keypoint`,
  default 2 each, to bound the number of 8B-model forward passes) — and
  writes `data/unified_examples.scored.jsonl`. Chunked and resumable: re-running
  after an interrupt skips however many output lines already exist.

## Losses (`src/losses.py`)

All four combined in `total_loss`, weighted by `configs/default.yaml`'s
`loss.*` section:

| Loss | Formula (adapted) | Purpose |
|---|---|---|
| `margin_mse_loss` | `MSE(Δs, Δs*)` over every (positive query × negative query) pair, `Δs = S(Q+,D) − S(Q−,D)` | distill the teacher's calibrated relevance margins |
| `idf_weighted_chamfer_loss` | `coverage + β·precision`, both over cosine distance between `h_d` and pre-gate `b_d` | geometric tether against "teacher blind spot": forces the bottleneck to physically retain high-entropy lexical detail, not just teacher-friendly intent |
| `consistency_loss` | `MSE(S(direct,D), S(indirect,D))` | direct/indirect queries about the same fact shouldn't fracture the latent space |
| `sparsity_loss` | `mean_over_batch( Σ_k π_k )` | entropy-driven budget: pushes gates toward 0 for sparse/short docs, loosened by the other losses' gradients for dense ones |

Two notes baked into the code (see `losses.py` docstring):

- Cosine similarity is scale-invariant to any positive scalar, so
  `cos(h_j, π_k·b_k) ≡ cos(h_j, b_k)` exactly for every `π_k` coming out of
  a sigmoid — the Chamfer loss computes cosine distances on the **pre-gate**
  `b_d` (mathematically identical to using `b_d_tilde`, but avoids dividing
  by near-zero norms as gates are pushed toward 0) and applies `π_k` exactly
  where Eq. 94 puts it: as the explicit outer weight in the precision term.
- `sparsity_loss` sums gates over whatever unique documents are in the
  batch — the proposal's literal `Σπ_{D+} + Σπ_{D-}` doesn't apply since
  this dataset has no negative *documents*.

## Training (`src/train.py`)

`accelerate launch -m src.train --config configs/default.yaml` — bf16 +
multi-GPU via `accelerate`, AdamW, linear warmup/decay, gradient
accumulation, periodic checkpointing (`torch.save` of the full model state
dict) and metric logging (stdout + CSV + TensorBoard via
`accelerator.log`). One batch = `B` distinct `(doc_id, keypoint)` rows;
`Collator` (`src/data/dataset.py`) deduplicates document encoding when
multiple sampled rows in a batch share a `doc_id`, and gathers that doc's
`b_d_tilde` back out per-row via `doc_index` before scoring every
query (direct/indirect/neg/hard-neg) against it.

Parameterized paths (`encode_document`/`encode_query`) are invoked through
`model(mode=..., ...)` rather than a direct attribute call, so that under
`DistributedDataParallel` the gradient-bucket hooks DDP installs on
`__call__` actually fire — DDP doesn't proxy custom methods, so e.g.
`model.encode_document(...)` directly on the wrapped model raises
`AttributeError`.

## End-to-end pipeline

```bash
# 1. Join the 3 raw sources into one training-unit-per-line jsonl
./scripts/run_build_index.sh

# 2. Build the token-level IDF table (CPU, multiprocessing, ~126K docs)
./scripts/run_build_idf.sh

# 3. Score every needed (query, doc) pair with the 8B teacher (GPU, resumable)
./scripts/run_precompute_teacher.sh

# 4. Train (GPU, accelerate handles multi-GPU)
./scripts/run_train.sh
```

## Evaluation / interpretability (`src/eval/`)

- **`inspect_bottleneck.py`** — for a given document, ranks the 256 bottleneck
  slots by gate `π`, and for the top-k slots reports (a) the nearest token(s)
  in the full vocabulary embedding table, and (b) the nearest token among the
  document's own tokens — both via plain cosine similarity against the
  **pre-gate** `b_d` (gate-invariant: `π_k` only rescales `b_d` by a positive
  scalar, so cosine similarity is unaffected, whereas `b_d_tilde` goes to
  exactly zero/undefined direction for a fully-pruned slot). The document
  view deliberately compares against **raw input-embedding rows**, not the
  post-backbone `h_d`: `h_d` and `b_d` share the backbone's "rogue"/outlier
  dimensions (both live in the same transformer's residual stream), which
  dominate raw cosine similarity and collapse every token to ~1.0 against a
  single slot; raw embeddings don't carry those dimensions. Also exposes a
  `--compare-norms` mode (L2-norm statistics + histogram of `b_d` vs `h_d`,
  to sanity-check whether the two populations even live at comparable
  magnitude) and a `--probe-token` mode (bar chart: one chosen vocabulary
  word's cosine similarity against a document's top-k gated slots). Can
  render results as a console table, a PNG heatmap, or a colored-text HTML
  page (single- or multi-document).
- **`sample_beir_compress.py`** — batch sanity-check of the trained compressor
  on out-of-domain BEIR corpora (pulled from the `BeIR/<dataset>` Hugging
  Face Hub mirror, since BEIR's own hardcoded UKP download URL isn't always
  reachable). Reports per-document gate statistics across a random sample
  and can render a multi-doc HTML visualization for a subset of the sample.

## Known simplifications / not implemented

- No cross-document in-batch negatives (would need either live teacher calls
  on `O(B²)` pairs, or a heuristic zero-relevance assumption) — left out of
  v1.
- No inference-time hard-pruning code path (the gate `π` is computed and
  trained against, but nothing currently drops low-`π` slots to realize the
  latency win at serving time).
- `_cos()` in `inspect_bottleneck.py` is plain `torch.nn.functional.cosine_similarity`
  (no per-dimension standardization) by deliberate choice — simpler code, at
  the cost of the "rogue dimension" collapse described above whenever both
  sides of a comparison are taken from the backbone's post-transformer
  hidden-state space (e.g. `b_d` vs `h_d` directly).
