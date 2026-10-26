# Compressor implementation guide

The architecture and objectives are defined in [compression.md](compression.md).
This guide maps that design to the Python repository. The evaluation protocol
is in [evaluation.md](evaluation.md).

**Implementation status:** this documentation revision does not change the
Python code or checked-in YAML. The frozen 8B encoder, revised relevance
training, and coverage-based vector selection below require implementation.
Existing commands still run the earlier model described in the status table.

## 1. Target design and migration status

| Component | Revised target | Checked-in implementation |
|---|---|---|
| Embedding backbone | `Qwen/Qwen3-Embedding-8B`, native width 4096 | `Qwen/Qwen3-Embedding-0.6B`, width 1024 |
| Encoder training | Frozen backbone and input embedding table; evaluation mode; no gradient recording | Shared backbone is optimized with the other model parameters |
| Coverage reference | Reuse the same frozen encoder's contextual token states | Current coverage uses trainable student token states |
| Summary count | 256 slots per chunk before post-training selection | 32 slots per chunk |
| Keypoints | Additional positive queries through the ordinary query path | Separate full-backbone keypoint encoding and matching loss |
| Query length | 64 token slots for all query types, including keypoints | 32 slots for ordinary queries; full-length keypoints in a separate encoder branch |
| Encoder attention | Native pretrained attention behavior | Bidirectional attention patch |
| Relevance training | Teacher soft-target BCE for all selected query/document pairs | Teacher soft-target BCE against each query row's source document |
| Training coverage | Keep IDF-weighted token coverage against frozen encoder states | IDF-weighted token coverage against trainable encoder states |
| Gates | Direct summary vectors, with no learned multiplicative gate in the revised scoring path | Sigmoid gate head and gated summaries |
| Sparsity | Removed | Summed-gate sparsity loss |
| Vector selection | Per-chunk coverage-gap cutoff across thresholds `0, 0.25, 0.5, 0.75, 0.9` | Inspection utilities expose gates; no coverage-gap selection pipeline |
| Evaluation | Precision@1, MRR, and token coverage | Existing inspection scripts do not implement the complete evaluation protocol |

Do not treat proposed fields or output names in this guide as an existing
API. A configuration-only change cannot implement this migration.

### Repository map

| Responsibility | Files |
|---|---|
| Configuration | [src/config.py](../src/config.py), [configs/default.yaml](../configs/default.yaml) |
| Joining upstream document/query rows | [src/data/build_index.py](../src/data/build_index.py) |
| Lazy dataset and batching | [src/data/dataset.py](../src/data/dataset.py) |
| IDF weights | [src/data/idf.py](../src/data/idf.py) |
| Reranker teacher and precomputation | [teacher_qwen_reranker.py](../src/distill/teacher_qwen_reranker.py), [precompute_teacher_scores.py](../src/distill/precompute_teacher_scores.py) |
| Attention behavior | [src/model/bidirectional.py](../src/model/bidirectional.py) |
| Query encoding | [src/model/query_adapter.py](../src/model/query_adapter.py) |
| Summary extraction | [src/model/document_compressor.py](../src/model/document_compressor.py) |
| Document packing and scoring | [src/model/reranker.py](../src/model/reranker.py) |
| Losses and training | [src/losses.py](../src/losses.py), [src/train.py](../src/train.py) |
| Existing inspection utilities | [inspect_bottleneck.py](../src/eval/inspect_bottleneck.py), [sample_beir_compress.py](../src/eval/sample_beir_compress.py) |

## 2. Environment and configuration

Use Python 3.10 or newer and install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

The current attention patch is tested with the pinned Transformers 4.52.3.
That is evidence for the current implementation, not validation of the full
8B migration. Optional inspection plots use `matplotlib`; BEIR sampling uses
`datasets`. Existing focused tests use tiny local models without downloading
the pretrained backbone.

The following are target requirements, not a runnable replacement YAML:

| Setting | Target requirement |
|---|---|
| Backbone name | `Qwen/Qwen3-Embedding-8B` |
| Hidden width | 4096 throughout the query adapter, summary blocks, normalization, and scoring |
| Backbone freeze | All backbone parameters, including the shared input embedding table |
| Summary capacity | `model.num_summary_tokens: 256` per chunk |
| Query slots | `model.query_max_len: 64` for ordinary queries and keypoints |
| Query adapter structure | Retain `n_query_adapter_layers: 2` and `n_query_adapter_heads: 8` |
| Compressor structure | Retain `n_cross_attn_layers: 1` complete refinement block and `n_cross_attn_heads: 8` for its cross-attention and self-attention |
| Keypoint matching weight | Remove the loss and its configuration field |
| Gate and sparsity settings | Remove gate-dependent behavior and sparsity configuration |
| Relevance objective | Teacher soft-target BCE; no InfoNCE term |
| Coverage objective | Keep IDF-weighted token coverage |
| Encoder attention | Native pretrained attention; remove the bidirectional patch from this path |
| Post-training thresholds | `0, 0.25, 0.5, 0.75, 0.9` |

The chunk size remains 1024 tokens; query length becomes 64 slots. These are
separate from vector width and summary count. A 256-vector budget does not
imply a 256-dimensional representation. The 8B checkpoint's
[official configuration](https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/main/config.json)
specifies the native hidden width of 4096.

`Config.from_yaml` merges known dataclass fields with supplied values. New
fields require corresponding dataclass support; adding invented keys to YAML
is insufficient. The current YAML overrides the Python epoch default from
three to five. Current optimizer and batch defaults have not been tuned for
the larger backbone or wider trainable modules.

Increasing width from 1024 to 4096 also enlarges the existing full-width
attention and feed-forward layers. Measure trainable parameter count, device
memory, and latency before choosing a batch size or claiming serving speed.

## 3. Load and freeze one shared embedding encoder

Load the original 8B checkpoint and record its revision, tokenizer,
preprocessing, attention policy, and output layer. Use native pretrained
attention without the current loader's bidirectional patch. Record that
choice in run metadata. Freezing weights does not reverse an attention
modification, so skipping the patch is a separate migration step.

The same frozen encoder can provide document contextual states and coverage
reference states. Reuse its detached document states when both uses have
identical token IDs, masks, attention behavior, and output layer. A second
frozen copy is only necessary if an implementation deliberately uses a
different reference encoding policy or cannot share the model instance.

Freeze before constructing the optimizer. The following illustrates the
required behavior; the current training class does not enforce it yet:

```python
backbone.requires_grad_(False)
backbone.eval()

# Run only the backbone under no_grad; keep summary extraction differentiable.
with torch.no_grad():
    h_d = backbone(
        input_ids=chunk_input_ids,
        attention_mask=chunk_attention_mask,
        use_cache=False,
    ).last_hidden_state

summaries = compressor_from_states(h_d, chunk_attention_mask)  # proposed API
```

Do not put the trainable compressor or query adapter inside that `no_grad`
block. The frozen lookup table supplies query token vectors, while the
adapter's Transformer layers and normalization remain trainable. Because
`model.train()` recursively changes module modes, preserve the backbone's
evaluation mode explicitly throughout training.

The backbone is currently registered through several shared module
references. Build optimizer groups from unique trainable parameters after
freezing, and ensure the shared embedding table is excluded. A separate
teacher optimizer or a second embedding checkpoint is unnecessary.

## 4. Training examples and pair construction

Continue using the existing document-linked `keypoint`, `direct_query`,
`indirect_query`, `negative_queries`, and `hard_negative_queries` fields.
Keypoint and query generation remain upstream of this repository.

Treat each nonempty keypoint as another query associated with its source
document. Encode it through exactly the same query path used for ordinary
queries. Remove its separate token-matching branch and loss. The old
`encode_keypoint` API is not the target keypoint-query path.

The revised batch must represent these pair types explicitly:

| Pair | Role |
|---|---|
| `(direct query, source document)` | Positive |
| `(indirect query, source document)` | Positive |
| `(keypoint query, source document)` | Positive |
| `(positive query, negative document)` | Negative |
| `(negative query, source document)` | Negative |
| `(hard negative query, source document)` | Hard negative |

Other relevant documents remain positives, even when their IDs differ from
the source document. Exclude known relevant documents, duplicate bodies, and
the source document itself from negative-document candidates. Repeated rows
for one document do not make that document a negative. Pair roles describe
sampling and known associations; all pairs receive their own teacher soft
targets rather than hard labels of one or zero.

Negative queries are negative relative to a particular document, not every
document. Keep their document association in the batch. Include explicit BCE
terms for both `(query, negative document)` and `(negative query, document)`.
A selected negative candidate that receives a high teacher score keeps that
soft target; never overwrite it with zero to match the sampling role.

For each valid pair, compute the student's full-document MaxSim score,
apply the shared positive scale and bias, and pass the resulting logit to
numerically stable BCE-with-logits against the teacher probability. Flatten
all selected pair types into one mean over valid pairs. Missing targets
contribute neither a loss nor a count; they are not zero-probability targets.
There is no separate InfoNCE or keypoint-matching term.

The total objective is teacher soft-target BCE plus IDF-weighted coverage,
with the existing coverage coefficient of 1.0. More pairs of a given type
increase that type's share of the BCE mean; report the sampling counts.

### Deduplication and batch interfaces

The current dataset indexes byte offsets and loads rows on demand. It checks
that repeated document IDs agree on the body. Retain those checks and encode
each distinct document only once per batch, including selected negative
documents. Reuse the encoded summaries across all associated query pairs.

A proposed revised batch needs:

- Unique document chunks, token masks, and chunk-to-document ownership.
- Query token IDs, query type, and stable query IDs, including keypoints.
- Positive query/document associations and explicit negative pair masks.
- Candidate document indices for cross-document comparisons.
- Pair-specific teacher probabilities and target-validity masks.

These fields are not implemented by the current `Collator`. Its existing
`keypoint_input_ids`, `keypoint_attention_mask`, and `keypoint_doc_index`
serve the old full-backbone matching branch and require migration.

Set `query_max_len` to 64 for every query type, including keypoints. Truncate
longer query inputs to this limit and fill unused slots with
`<|vision_pad|>`, following the existing query tokenizer's slot behavior.
All 64 slots participate in query encoding and MaxSim. This replaces the
old untruncated full-backbone keypoint branch. Missing query entries require
separate masks and must never become training negatives.

## 5. Reranker targets and coverage weights

### Reranker teacher

The relevance teacher remains `Qwen/Qwen3-Reranker-8B`. Its current wrapper
formats the instruction, query, and full document in one prompt, runs one
causal-LM forward pass without gradients, and computes the two-class
softmax of final-position `yes` and `no` logits.

Precompute a target for every pair that enters BCE, including keypoint
queries and selected negative-document pairs. A target for
`(query, source document)` cannot be reused for `(query, another document)`.
Keypoint positives also need their own teacher probability rather than an
automatic target of one. The reranker teacher is used only to obtain targets
and is not loaded in the compressor training loop.

The current precomputation CLI scores direct, indirect, negative, and hard
negative queries only against their row's full source body. It does not yet
score keypoints as queries or selected cross-document pairs.

The default teacher budget is 32,768 tokens for the entire formatted input.
The wrapper rejects oversized pairs instead of silently truncating them.
Longer documents require targets from an explicit long-document policy.
Record model, revision, instruction, context budget, document scope, and
source/policy for every target set.

Current training requires `teacher_metadata.document_scope` to be
`full_document` and rejects inconsistent provenance. Preserve that validation
for every teacher-target branch. Old prefix scores require
regeneration; changing their metadata does not restore missing evidence.

Precomputation resumes by output line count and verifies teacher settings
before appending. Input order and negative limits must still match. Cache
new pair targets by query identity, full candidate content/version, and
teacher settings.

### IDF weights for coverage training

The existing IDF builder tokenizes complete original documents and counts
each token ID once per document, after deduplicating `doc_id` values. It
rejects conflicting bodies and an empty corpus. Chunking does not change
the document-frequency unit.

The weight is `log((N + 1) / (df + 1))`, with nonnegative values. A ubiquitous
token receives zero. Version-2 artifacts record the tokenizer, document
count, formula, and `full_documents` scope. Training validates those fields,
vocabulary size, and finite nonnegative weights.

Regenerate or validate the IDF artifact against the selected 8B tokenizer
identity. Use the frozen contextual states as the coverage reference and
compute cosine similarity in float32 with safe norms. Keep the established
per-chunk normalization; zero-total-weight chunks contribute zero. Compute
coverage once per unique chunk, so additional query pairs do not repeat it.
The post-training coverage counts are separate from this IDF-weighted
training objective.

## 6. Summary extraction and scoring interfaces

The current summary block performs residual pre-norm cross-attention,
FFN1, summary self-attention, then FFN2. Learned initial summary queries are
shared across chunks. Preserve the distinction between contextual encoder
states and compressed summary vectors.

The [architecture diagram](compression.md#2-query-encoding-and-document-scoring)
shows these blocks at the revised dimensions. Retain the existing block
structures while changing their widths and budgets:

| Component | Revised architecture |
|---|---|
| Query lookup | Frozen shared embedding table; `(B, 64)` IDs become `(B, 64, 4096)` vectors |
| Query adapter | Two pre-norm Transformer encoder layers; eight self-attention heads per layer; FFN width 16384, GELU, dropout 0 |
| Query output | `LayerNorm(X_Q + F_Q(X_Q))`; includes an outer residual in addition to each layer's internal residuals |
| Summary initialization | One trainable `(256, 4096)` matrix, reused for every chunk |
| Summary block | One block by default: cross-attention, FFN1, self-attention, FFN2; eight heads in both attention modules; pre-norm and a residual at each stage |
| Compressor FFNs | Separate `Linear(4096, 16384) -> GELU -> Linear(16384, 4096)` networks; dropout 0 |
| Document scoring | Concatenate chunk summaries; form query-to-summary scaled dot products; take the maximum per query slot and sum all 64 maxima |

The query adapter adds no positional encoding and uses bidirectional
self-attention over all 64 slots, including fill slots. This adapter behavior
is separate from the frozen document encoder's native attention. In summary
cross-attention, use separate trainable query and key/value LayerNorm modules
and mask padded chunk tokens. Keep those normalization and projection layers
outside the frozen encoder's `no_grad` region. The current block structure
has no additional final compressor LayerNorm.

The revised output contract should expose these tensors. Names are proposed
interfaces to implement, not a claim that the current outputs already obey
this contract. Here `Nc` is the number of unique chunks, `U` the number of
unique documents, and `Jmax` the maximum chunk count in a document batch.

| Output | Shape | Revised meaning |
|---|---|---|
| `h_d` | `(Nc, 1024, 4096)` | Detached frozen encoder states, reused for coverage |
| `doc_attention_mask` | `(Nc, 1024)` | Real token mask, including the final partial chunk |
| `b_d` | `(Nc, 256, 4096)` | Direct trainable summary vectors |
| `chunk_doc_index` | `(Nc,)` | Chunk owner in the unique-document table |
| `document_vectors` | `(U, Jmax * 256, 4096)` | Concatenated direct summaries with packing padding |
| `document_summary_mask` | `(U, Jmax * 256)` | Valid summary positions before selection |

Each chunk produces 256 vectors. A document with `J` chunks therefore has
`J * 256` vectors before post-training selection. Select separately within
each chunk, then concatenate its retained vectors with those of the other
chunks. Keep original chunk and slot IDs in selected caches.

`document_vectors` should contain direct summaries. Remove the learned
`pi` factor from relevance scoring, and remove the obsolete `pi` and
`b_d_tilde` target interfaces. Merely setting the sparsity coefficient to
zero leaves gate-dependent scoring active in the existing code.

MaxSim still compares each query slot against all real retained summaries
of a candidate document, takes that slot's maximum, then sums across query
slots. Document packing padding must be masked before the maximum. Removing
a summary means physically excluding it or masking its similarity to
negative infinity; replacing it with zero can change rankings incorrectly.

Retain the existing shared positive scale and bias for BCE calibration:
`logit = softplus(score_alpha) * MaxSim + score_bias`, using its numerical
positive floor. Apply sigmoid only when reporting relevance probabilities;
feed raw logits into BCE-with-logits. Use raw MaxSim scores for ranking to
avoid sigmoid saturation producing numerical ties.

## 7. Training loop, artifacts, and post-training selection

Retain Accelerate's single wrapped forward for all trainable query and
compressor branches. Current training uses AdamW, gradient accumulation,
gradient clipping, and linear warmup/decay. The scheduler horizon is
`ceil(len(prepared_loader) / grad_accum_steps) * epochs`, after distributed
loader preparation. Incomplete batches are retained.

Update loss aggregation and metric names to the BCE-plus-coverage objective.
Remove keypoint-matching and sparsity loss calls, optimizer parameters, and
logs. Keypoint-query relevance examples still contribute to the ordinary
relevance objective. Compute coverage loss once per unique
chunk rather than once per query pair.

Current training saves the following artifacts:

| Artifact | Current contents and migration requirement |
|---|---|
| `config.yaml` | Effective dataclass configuration and CLI overrides; add revised design settings |
| Tokenizer files | Student tokenizer saved with `save_pretrained`; use the selected 8B tokenizer |
| `metrics.csv`, TensorBoard | Current losses; replace obsolete terms with actual revised objectives |
| `checkpoint-<step>.pt`, `final_model.pt` | Full model `state_dict`; the new architecture requires new compatible weights |

Current checkpoints omit optimizer, scheduler, RNG, and data-order state;
full training resume is not implemented. For the revised model, also record
the frozen backbone revision and encoding policy. Coverage-based selection
must save each chunk's selected slot IDs, threshold, retained count, winning
gap, cutoff rule, and reference metadata.

After training, compute summary coverage at `tau = 0, 0.25, 0.5, 0.75, 0.9`.
For each chunk and threshold, count the real tokens whose cosine similarity
with each summary is at least `tau`. Sort the 256 counts descending, breaking
ties by original slot ID. Evaluate the 255 adjacent drops in that sorted
list; a cutoff after position `p` retains the first `p` vectors.

Choose the largest drop across all thresholds and cutoff positions for that
chunk. Ties retain the most vectors, then choose the lowest threshold. If
every drop at every threshold is zero, retain all 256 vectors and record
`tau = 0`. Concatenate the retained vectors across document chunks. There
is no trained gate or sparsity objective to consult. See
[evaluation.md](evaluation.md) for equations and reporting. Report the full
baseline of 256 vectors per chunk alongside selected representations.

## 8. Commands and validation

Existing scripts remain entry points for the earlier implementation:

```bash
./scripts/run_build_index.sh
./scripts/run_build_idf.sh
./scripts/run_precompute_teacher.sh
./scripts/run_train.sh
```

They do not yet train the revised design. Data-building, IDF, and teacher
precomputation CLIs have their own arguments; editing a training YAML does
not configure those separate commands. Migrate the code and data contracts
before using these scripts to train the revised design.

Before running the revised model, verify backbone weights remain unchanged
after an optimizer step, frozen states have no gradients, and the encoder
stays in evaluation mode. Verify compressor and query-adapter gradients
remain nonzero on a meaningful batch. Check shared embeddings are frozen,
all tensor widths agree, and all requested positive/negative directions
participate in the loss without introducing known false negatives.

Validate chunk/token masks, unique-document reuse, selected-vector masks,
and any teacher-target provenance. Check that keypoints enter the query
path, no keypoint-matching or sparsity term remains, and scoring contains no
gate multiplication. Existing tests do not establish these revised behaviors.
Full 8B training, retrieval quality, post-training vector selection, MRR,
export, and measured serving latency require their own completed validation.
