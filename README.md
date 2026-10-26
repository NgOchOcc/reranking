# Latent Bottleneck Reranker

The revised design uses a **frozen `Qwen/Qwen3-Embedding-8B` encoder with its
native pretrained attention**. A trainable compressor produces **256 summary
vectors per 1,024-token chunk**. Each vector has 4,096 components. Documents
are encoded independently of queries, allowing their summaries to be cached.

Direct queries, indirect queries, and keypoints share a trainable query adapter
with a **64-token limit** and the encoder's frozen input embedding table.
Document scores use MaxSim over all chunk summaries. Training combines only
teacher soft-target BCE and IDF-weighted token coverage. There is no keypoint
matching loss, gating head, or sparsity loss.

**These documents specify the revised design. The checked-in Python code,
scripts, and YAML still implement the earlier 0.6B model with 32 gated vectors
per chunk and four losses.** Follow the implementation migration guide before
running training for the revised design.

## Documentation

- [Compression theory](docs/compression.md): architecture, supervision, and the
  two training losses.
- [Implementation guide](docs/implementation.md): current code, required changes,
  data preparation, tensor interfaces, and training artifacts.
- [Evaluation](docs/evaluation.md): post-training vector selection, Precision@1,
  Mean Reciprocal Rank (MRR), and token coverage.

## Training supervision

Use the existing document-linked keypoints and generated queries. Keypoints
become positive queries for their source document, using the same query path
as direct and indirect queries. Include all three pair types:

| Pair | Purpose |
|---|---|
| `(positive query or keypoint, relevant document)` | Match relevant pairs |
| `(positive query or keypoint, negative document)` | Distinguish candidate documents |
| `(negative query, source document)` | Reject ordinary and hard-negative queries |

The frozen `Qwen/Qwen3-Reranker-8B` supplies a separate, precomputed relevance
probability for every selected pair. BCE matches those probabilities; pair
labels guide sampling and do not replace the teacher targets with hard labels.
Exclude duplicate bodies and known relevant documents from negative-document
sampling. Preserve full-document teacher provenance and mask unavailable targets.

Coverage uses the frozen encoder's own contextual token vectors, so a second
embedding-model copy is unnecessary. Encode each unique document once per batch
and compute its chunk coverage once, regardless of how many queries reference it.

## Revised configuration targets

These are target settings, not the current contents of
[configs/default.yaml](configs/default.yaml):

| Setting | Revised value |
|---|---|
| Embedding encoder | `Qwen/Qwen3-Embedding-8B`, frozen including its input embedding table |
| Encoder attention | Native pretrained behavior |
| Hidden-state width | 4,096 |
| Chunk size | 1,024 tokens, without whole-document truncation |
| Summary count | 256 per chunk before selection |
| Query limit | 64 tokens, including keypoints and negative queries |
| Relevance teacher | `Qwen/Qwen3-Reranker-8B` |
| Relevance loss | Soft-target BCE, coefficient 1 |
| Token coverage loss | IDF-weighted cosine coverage, coefficient 1 |
| Post-training cosine thresholds | `0, 0.25, 0.5, 0.75, 0.9` |

Train with all 256 summary slots. After training, select vectors separately
within each chunk using the largest gap in sorted token-coverage counts
across the five thresholds, then concatenate the retained vectors across
the document. Compare with the unpruned baseline using
Precision@1, MRR, token coverage, and retained-vector counts. Validate the
selection procedure before evaluating the held-out test set.

## Environment

Use Python 3.10 or newer and install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

The existing tests use tiny local models without downloading pretrained
weights. They validate the earlier implementation; they do not establish
correctness of the revised design. The implementation guide lists the
required migration checks and identifies existing pipeline entrypoints.
