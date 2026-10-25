# Latent Bottleneck Reranker

Compress every document into nonoverlapping 1,024-token chunks, with 32
gated summary vectors per chunk. A shallow query adapter scores the entire
document by MaxSim over all real chunk summaries. A learned positive scale
and bias convert that score into a sigmoid relevance prediction.

The shared document/keypoint backbone is Qwen/Qwen3-Embedding-0.6B.
The model trains jointly with soft-target BCE, normalized token coverage,
main-keypoint matching, and summed-gate sparsity. The equations and detailed
implementation are in [docs/compression.md](docs/compression.md).

## Environment

Use Python 3.10 or newer and install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The Qwen3 bidirectional attention adapter is tested with Transformers
4.52.3, which is pinned in the requirements. Full training needs access to
the student backbone and the prepared training data. Teacher precomputation
separately loads the 8B reranker.

Optional inspection plots use `matplotlib`; BEIR sampling uses
`datasets`. The tests use tiny local models and require no model downloads.

## Training data and main keypoints

Each scored JSONL row retains the existing query fields:

```json
{
  "doc_id": "doc-001",
  "body": "The full document, including all chunks.",
  "keypoint": "The fact that generated this row's queries.",
  "main_keypoint": "The document's designated main factual content.",
  "direct_query": "A direct query",
  "indirect_query": "An indirect query",
  "negative_queries": ["An ordinary negative query"],
  "hard_negative_queries": ["A hard negative query"],
  "teacher_scores": {
    "direct": 0.94,
    "indirect": 0.87,
    "negative_queries": [0.03],
    "hard_negative_queries": [0.18]
  },
  "teacher_metadata": {
    "document_scope": "full_document",
    "source": "external full-document teacher",
    "policy": "Judge the complete document for each query"
  }
}
```

The numbers and text above illustrate the schema. Target arrays must match
the corresponding query arrays, and every target must be finite and in
[0, 1]. Direct and indirect queries each receive their own soft target.

Use one main keypoint per document. If several query rows refer to the same
document, they must agree on its body and main keypoint. The loader scans
the file to check this association while keeping document bodies lazy.
Queries may still be generated from several different row-level keypoints.

You can supply the main keypoint in a separate JSON/JSONL file:

```json
{"doc_id": "doc-001", "keypoint": "The document's designated main keypoint."}
```

A JSON list of these records or a mapping from document ID to keypoint text
also works. Set `data.main_keypoints_path` or pass:

```bash
accelerate launch -m src.train \
  --config configs/default.yaml \
  --main-keypoints-path data/main_keypoints.jsonl
```

An explicit `main_keypoint` or sidecar resolves documents with multiple
row-level keypoints. The older `keypoint` field is accepted as the main
keypoint only when all rows for that document agree across the complete
file. Ambiguous legacy data raises an error.

## Prepare and train

```bash
# 1. Join existing canonical bodies, queries, and negative queries.
./scripts/run_build_index.sh

# 2. Build nonnegative IDF weights from complete original documents.
./scripts/run_build_idf.sh

# 3. Score complete query-document pairs with the discriminative teacher.
./scripts/run_precompute_teacher.sh

# 4. Train the student with all four losses.
./scripts/run_train.sh --main-keypoints-path data/main_keypoints.jsonl
```

The joiner consumes the existing upstream JSONL sources; keypoint and query
generation remain external. Teacher precomputation preserves extra row
fields, including an explicit main keypoint.

Teacher targets describe whole documents, after all their chunk summaries
are combined by the student. The teacher wrapper refuses prompts beyond its
configured context budget instead of truncating them. Its default total
prompt budget is 32,768 tokens, including query, instructions, and special
tokens. Documents beyond that budget need targets from an explicitly
defined long-document teacher policy. The teacher's
[official model card](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
describes its relevance scoring and supported context.

Generated scores include `teacher_metadata` identifying the model,
instruction, context budget, and full-document scope. Training rejects
missing full-document provenance and inconsistent teacher settings. For
external targets, record their full-document scope and source/policy as
illustrated above. Prefix-only targets must be regenerated; changing their
metadata does not make them full-document judgments.

## Model and loss configuration

Key settings in [configs/default.yaml](configs/default.yaml):

| Setting | Default |
|---|---|
| `model.doc_chunk_size` | 1024 tokens |
| `model.num_summary_tokens` | 32 per chunk |
| `model.n_cross_attn_layers` | 1 complete summary-refinement block |
| `model.query_max_len` | 32 query slots |
| `loss.lambda_coverage` | 1.0 |
| `loss.lambda_keypoint` | 1.0 |
| `loss.lambda_sparsity` | 0.01 |

Each refinement block performs pre-norm residual cross-attention, FFN1,
summary self-attention, and FFN2. Every document token is retained in a
chunk; only the last chunk is padded.

The training components are:

- **BCE:** match each teacher probability with
  `sigmoid(softplus(score_alpha) * MaxSim + score_bias)`.
- **Coverage:** within each chunk, average nearest-summary cosine distances
  with weights `log((N + 1) / (df + 1))`, normalized by their total mass.
  A zero-total-weight chunk contributes zero.
- **Keypoint:** encode every keypoint token with the shared full backbone;
  use one minus the mean tokenwise maximum cosine against all gated
  document summaries. This branch has no pooling or query-length truncation.
- **Sparsity:** sum gates over all chunks of each document, then average
  over unique documents.

Coverage averages over unique chunks. Keypoint loss and sparsity average
once per unique document, even when it has multiple query rows in a batch.
Missing negative queries and padded document summary slots are masked.
Cosine calculations run in float32, including under bf16 autocast.

Positive gate scaling cancels inside cosine away from its epsilon floor.
Consequently, the requested keypoint objective aligns summary directions
without directly resisting gate shrinkage. Soft gating also retains every
slot physically; pruning and a persistent serving index are separate work.

## Inference

After loading a model and its corresponding tokenizer:

```python
import torch

from src.data.dataset import pad_query_ids

# model is a LatentBottleneckReranker loaded with the run's config/checkpoint.
model.eval()
device = next(model.parameters()).device
document_inputs = tokenizer(
    document_text, add_special_tokens=False, truncation=False,
    return_tensors="pt",
).to(device)
mask_id = tokenizer.convert_tokens_to_ids(model.cfg.mask_token)
query_ids = torch.tensor([
    pad_query_ids(tokenizer, query_text, model.cfg.query_max_len, mask_id)
], device=device)

with torch.no_grad():
    document = model.encode_document(**document_inputs)
    query = model.encode_query(query_ids)
    score = model.maxsim(
        query, document["document_vectors"], document["document_summary_mask"]
    )
    probability = model.scores_to_probabilities(score)
```

`encode_document` automatically chunks full document input rows.
`document_vectors` and `document_summary_mask` are the packed
whole-document representation. The older output names `h_d`, `b_d`,
`pi`, and `b_d_tilde` remain **chunkwise** for coverage and diagnostics.
In theory notation, `b_d` is ungated $S_r$ and `b_d_tilde` is gated
$B_r$. Reuse packed document vectors across queries when caching them.
Keypoint text is needed for training, not for serving.

Training writes `checkpoint-<step>.pt`, `final_model.pt`,
`config.yaml`, tokenizer files, `metrics.csv`, and TensorBoard logs.
Weight files contain the full model state, including calibration parameters.
They do not resume optimizer or scheduler state.
