# Evaluating the frozen-encoder compressor

This is the evaluation specification for the revised design in
[`compression.md`](compression.md). The encoder is a frozen
`Qwen/Qwen3-Embedding-0.6B` with width 1,024, including its input embedding table.
The compressor produces 64 summary vectors per chunk before postprocessing.
Training keeps soft-target BCE and IDF-weighted coverage, with no InfoNCE,
gating, or sparsity objective. Keypoints are queries alongside direct and
indirect queries; there is no separate keypoint matching loss.

The Python configuration, model outputs, losses, and inspection scripts still
implement the earlier architecture and require updates before this procedure
can run. No revised selection API is assumed here. Earlier checkpoints require
their own matching architecture.

## Evaluation manifest and held-out data

Record the following before evaluating a checkpoint:

| Input | Required information |
|---|---|
| Compressor | Checkpoint, architecture version, effective configuration, tokenizer |
| Frozen encoder | `Qwen/Qwen3-Embedding-0.6B`, exact revision, 1,024-dimensional output layer |
| Encoding behavior | Native pretrained attention, preprocessing, chunk size, masks, precision |
| Vector budget | 64 per chunk before selection |
| Candidates | Stable unique `doc_id` values and complete document bodies |
| Queries | `query_id`, text, type, source document, all labeled relevant document IDs |
| Sampling | Seed, document count, query count per document and query type |
| Query budget | 64 slots for every query type, including keypoints |
| Matching thresholds | $\tau\in\{0,0.25,0.5,0.75,0.9\}$ |
| Selection | Exact largest-gap algorithm, scope, tie/fallback rules, retained original slot IDs |
| Split policy | Document/duplicate groups, train/validation/test assignments |

Split documents and duplicate bodies before selecting queries. A document and
all of its associated direct, indirect, and keypoint queries belong to the
same split. Deduplicate repeated document IDs and reject conflicting bodies.
Use document IDs in a stable order to resolve equal retrieval scores.

Choose checkpoints and selection-policy details using training and validation
data. Freeze the complete algorithm before the final test run. A fixed rule
may choose a threshold separately for each test chunk from its own token-summary
similarities, without looking at test relevance labels or ranking quality.
Reporting the full prespecified threshold diagnostics is allowed; changing
the rule to favor test-set metrics is not a validation procedure.

Load the encoder revision and tokenizer recorded with the run. Keep every
encoder parameter frozen and run evaluation with gradient recording disabled
and the model in evaluation mode. Load only the trainable components that
belong to the revised architecture, including the query adapter and relevance
calibration parameters if present.

## Encode documents and retain the reference tokens

Tokenize complete documents without silently truncating their bodies. Keep
real-token masks and the exact token IDs, chunk boundaries, and document
ownership used by the compressor. Exclude batch padding from all matching,
counts, and denominators. Reject empty documents and nonfinite activations.

The same frozen encoder supplies compressor inputs and contextual reference
tokens $h_{r,j}$. Reuse available hidden states or cached states; an additional
model copy is unnecessary.
Recomputation must use the same weights, tokenizer, preprocessing, hidden
layer, attention behavior, chunk boundaries, and masks as the compressor input.

Use the checkpoint's native pretrained attention behavior for both compressor
inputs and the coverage reference. The revised design removes the current
bidirectional attention patch. The reference consists of contextual hidden
states, not raw input-embedding table rows or pooled sentence embeddings. The
reranker teacher supplies relevance probabilities; it does not supply these
token references.

Retain original summary-slot IDs and document/chunk ownership before any
selection. Each chunk owns 64 original summaries. Summary vectors are used
directly: no learned gate scales them. A document with $J_D$ chunks therefore
has $64J_D$ vectors before selection.

## Match tokens and summaries at each threshold

A matching group $g$ is one chunk: compare its real token positions with
its own 64 original summaries. Compute selection and coverage independently
for each chunk. Do not match a token with summaries from another chunk, even
when both chunks belong to the same document.

For contextual token $h_{g,j}$ and original summary $s_{g,k}$, calculate

$$
C_{g,jk}=\frac{h_{g,j}^{\top}s_{g,k}}
{\max(\lVert h_{g,j}\rVert_2,\epsilon)
 \max(\lVert s_{g,k}\rVert_2,\epsilon)}.
$$

Compute cosine similarities in float32 with a recorded positive normalization
epsilon. Validate finite inputs and record zero or near-zero vector counts.
The match matrix for each requested threshold is

$$
A_{g,jk}(\tau)=\mathbf 1[C_{g,jk}\geq\tau],
\qquad \tau\in\{0,0.25,0.5,0.75,0.9\}.
$$

The comparison is **greater than or equal to** the threshold. A token can
match several summaries; each token position remains a distinct observation.
For every original summary, report its token-coverage count and fraction:

$$
n_{g,k}(\tau)=\sum_{j=1}^{\ell_g}A_{g,jk}(\tau),
\qquad
u_{g,k}(\tau)=\frac{n_{g,k}(\tau)}{\ell_g}.
$$

These are postprocessing measurements, not learned gate scores. They are
unweighted token counts, separate from IDF-weighted training coverage.

### Postprocessing selection by the largest gap

Apply the following rule independently to each chunk $g$. At every threshold
$\tau$, sort all 64 original slots by descending $n_{g,k}(\tau)$, breaking
equal-count ties by ascending original slot ID. Let $a_{g,\tau,p}$ be the
original slot ID at one-based sorted position $p$ and $v_{g,\tau,p}$ its count.
Define the 63 adjacent drops by

$$
\Delta_{g,\tau,p}=v_{g,\tau,p}-v_{g,\tau,p+1},
\qquad p=1,\ldots,63.
$$

For each threshold, find its largest drop. If several positions have that
drop, choose the largest position so that more vectors survive:

$$
d_{g,\tau}=\max_p\Delta_{g,\tau,p},
\qquad
p_{g,\tau}=\max\{p:\Delta_{g,\tau,p}=d_{g,\tau}\}.
$$

Choose the threshold lexicographically by **largest drop, then most retained
vectors, then lowest threshold**:

$$
\tau_g^*=\underset{\tau\in\{0,0.25,0.5,0.75,0.9\}}{\arg\max}
\big(d_{g,\tau},\ p_{g,\tau},\ -\tau\big).
$$

When the winning drop is positive, retain exactly the sorted prefix
$\mathcal K_g=\{a_{g,\tau_g^*,1},\ldots,
a_{g,\tau_g^*,p_{g,\tau_g^*}}\}$. The cutoff is the boundary between positions
$p_{g,\tau_g^*}$ and $p_{g,\tau_g^*}+1$; record both adjacent counts. If **all**
gaps at **all** thresholds are zero, keep all 64 slots, record $\tau_g^*=0$,
and record no cutoff. This rule always retains at least one vector per chunk.

Equivalent pseudocode, with one-based positions and stable original IDs:

```text
for tau in [0, .25, .5, .75, .9]:
    counts = number of real tokens with cosine(token, slot) >= tau, per slot
    order[tau] = sort original slots by (-counts[slot], original_slot_id)
    gaps[p] = counts[order[tau][p]] - counts[order[tau][p + 1]], p = 1..63
    drop[tau] = max(gaps)
    prefix[tau] = largest p where gaps[p] == drop[tau]
if max(drop.values()) == 0:
    return all 64 original slots, selected_tau=0, cutoff=null
chosen_tau = argmax_tau (drop[tau], prefix[tau], -tau)
return order[chosen_tau][1..prefix[chosen_tau]], selected_tau=chosen_tau
```

Always derive every threshold's counts from the original summaries. This is
an adaptive per-chunk choice among cosine thresholds, not a percentage-removal
sweep. Save all counts, sorted IDs, gaps, winning threshold, cutoff boundary,
and retained/removed original IDs. Different chunks can choose different
thresholds and retain different numbers of vectors. Equal positive counts
are not split at a positive-gap cutoff.

Concatenate the retained vectors from all chunks in original chunk/slot order.
Remove discarded vectors physically or mask them to negative infinity in
MaxSim. Do not replace discarded vectors or packing padding with zeros: a
zero could beat a valid negative score. Combine selection and padding masks
using the correct document ownership. Include all 64 slots per chunk as
the unselected baseline; neither representation has a gate multiplier.

## Rank candidates for direct, indirect, and keypoint queries

Select reproducible held-out candidates and collect direct queries, indirect
queries, and row keypoint text as three query types. All use the same query
path and 64-token budget. Record truncation and mask padding with the run's
tokenizer; keypoints have no auxiliary target or separate full-length path.

Deduplicate query text within each type and merge known relevant-document
IDs. If identical text occurs under different types, preserve the type labels
and document how it contributes to the combined result. Sample without
replacement per source document and type; record shortfalls. Fix the sampled
queries, candidate set, and relevance judgments across settings and models.

Each query must have at least one labeled relevant candidate. Include all
known relevant candidates, rather than assuming only the source document can
answer a query. If only source-document labels are available, report that
limitation. Negative queries require their own positive relevance judgments
before entering these metrics.

Use the trained query adapter with the frozen shared input embedding table.
Use all 64 configured query slots and their training-time padding behavior.
For query vectors $q_i$ and the final retained summary set $\mathcal S_D$ of
document $D$, score

$$
S(Q,D)=\sum_{i=1}^{64}
\max_{s\in\mathcal S_D}\frac{q_i^{\top}s}{\sqrt{1024}}.
$$

Concatenate the retained summaries from every chunk. The maximum spans all
those retained summaries belonging to the document. There is no gate
multiplier. Preserve the training score definition; do not substitute
cosine retrieval scoring just because selection uses cosine similarities.
Rank complete candidate documents by descending raw score and resolve exact
ties by ascending stable document ID. A shared positive score scale, bias,
and sigmoid preserve ordering and are unnecessary for ranking.

### Precision@1 and mean reciprocal rank

Let $R_q$ be query $q$'s relevant candidate IDs and $D_{q,r}$ its document at
one-based rank $r$. Precision@1 is

$$
\operatorname{P@1}=\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}\mathbf 1[D_{q,1}\in R_q].
$$

Define the first relevant rank and reciprocal rank by

$$
r_q=
\begin{cases}
\min\{r\geq1:D_{q,r}\in R_q\},&\text{if a relevant document is retrieved},\\
\infty,&\text{otherwise},
\end{cases}
\qquad
\operatorname{RR}_q=
\begin{cases}
1/r_q,&\text{if a relevant document is retrieved},\\
0,&\text{if none is retrieved}.
\end{cases}
$$

Then

$$
\operatorname{MRR}=\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}\operatorname{RR}_q.
$$

Use complete candidate rankings for the primary MRR. If only the first $L$
results are evaluated, label the metric MRR@$L$ and set reciprocal rank to
zero when no relevant document appears in those $L$ results. An empty result
has P@1 and reciprocal rank zero; record the failure reason. Keep query-label
or candidate-manifest errors distinct from a valid retrieval miss.

Report P@1 and MRR overall and separately for direct, indirect, and keypoint
queries, with counts and numerators. The combined result is the mean over
all evaluated query records; any equal-weight average over query types must
be labeled separately. An absent query type has no metric, rather than zero.
Save full rankings, raw scores, first relevant rank, reciprocal rank, top-1
hit, and relevant IDs for every query and selection setting.

## Measure coverage after selection

For each real token, use the retained summaries from its declared matching
group to calculate

$$
c_{g,j}(\tau)=\mathbf 1\!\left[
\max_{k\in\mathcal K_g}C_{g,jk}\geq\tau\right],
\qquad
\operatorname{Coverage}_g(\tau)=\frac{1}{\ell_g}
\sum_{j=1}^{\ell_g}c_{g,j}(\tau).
$$

Count a token once even if several retained summaries match it. Report
**fixed-threshold coverage at all five thresholds** before and after the final
adaptive selection. Keep $\mathcal K_g$ unchanged during this measurement sweep
so before/after comparisons use the same criterion.

Also report **adaptive coverage** using each chunk's selected $\tau_g^*$, with
its matching baseline evaluated at that same threshold. Label this aggregate
explicitly: it mixes per-chunk thresholds and is not fixed-threshold coverage.
Coverage measures vector agreement; it does not establish factual preservation
by itself.

Aggregate document coverage using real-token counts. Corpus micro coverage
is total covered tokens divided by total real tokens; document macro coverage
is the mean document coverage. Report both explicitly. An unweighted mean of
chunk percentages gives short final chunks disproportionate influence.

## Export complete matching records and comparison results

For every real token, including uncovered tokens, save document/chunk IDs,
token position in the document and chunk, token ID/string, measurement
threshold, the chunk's selected threshold, every original above-threshold
summary match and cosine score, the retained subset of those matches, best
retained slot/score (ties use ascending original slot ID), and the coverage flag. Keep stable
original slot IDs. Token strings are subword pieces; preserve positions and
IDs, and use original character offsets for rendering when available.

For every original summary, including discarded ones, save its original slot
ID, owner, threshold, coverage count/fraction, retained flag, and every matched
token position/ID/string/score. Export all qualifying matrix entries, rather
than only each token's best match. Preserve selection diagnostics and the
number of vectors before and after selection for each matching group and
document. An optional full cosine matrix supports later threshold inspection.

Compare the unselected baseline against the final adaptive selector. Report
retained vector counts/fractions and P@1/MRR overall and by query type, with
changes from baseline. Add fixed-threshold token micro and document macro
coverage before/after at each of the five thresholds; report adaptive coverage
separately. Record the distribution of selected thresholds and retained counts
across chunks. Save the manifest, fixed queries, candidate IDs, relevance
labels, matching records, and rankings with the tables.

Existing inspection scripts use earlier outputs and raw input-table comparisons.
Update and validate the model and evaluation interfaces before interpreting
their output as this contextual coverage, selection, or retrieval evaluation.
