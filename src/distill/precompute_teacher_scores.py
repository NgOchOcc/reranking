"""Precompute Qwen3-Reranker-8B teacher scores for every (query, document)
pair needed for training, and write a new, resumable, "scored" JSONL that
train.py consumes directly (see data/dataset.py).

To bound teacher-call cost, negatives/hard-negatives are subsampled per
keypoint (first --neg-per-keypoint / --hardneg-per-keypoint entries of each
list; the generation order has no special meaning so a fixed prefix is as
good as a random sample and keeps runs reproducible without needing a seed).

Processes the input in chunks of --chunk-records records: within a chunk,
every (query, doc) pair needed by those records is flattened into one big
list and scored in --score-batch-size batches (maximizing teacher-GPU
utilization), then results are reassembled and the chunk is appended to
--out immediately -- so an interrupted run can resume by skipping however
many *output* lines already exist (same convention as the existing
generate_*_msmarco.py scripts in this repo).

Usage (run on a GPU node):
    python -m src.distill.precompute_teacher_scores \
        --unified data/unified_examples.jsonl \
        --out data/unified_examples.scored.jsonl \
        --teacher Qwen/Qwen3-Reranker-8B \
        --score-batch-size 32 --chunk-records 256
"""
import argparse
import json
import os

from tqdm import tqdm

from ..config import TeacherConfig
from .teacher_qwen_reranker import QwenRerankerTeacher


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _iter_records(path, skip=0):
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if line.strip():
                yield json.loads(line)


def _score_chunk(teacher, records, score_batch_size):
    """Flattens all (query, doc) pairs in `records` into one list, scores in
    batches, and attaches a `teacher_scores` dict to each record in place."""
    pairs_q, pairs_d, locs = [], [], []
    for r_idx, rec in enumerate(records):
        rec["teacher_scores"] = {}
        for slot in ("direct_query", "indirect_query"):
            pairs_q.append(rec[slot])
            pairs_d.append(rec["body"])
            locs.append((r_idx, slot.replace("_query", ""), None))
        for slot in ("negative_queries", "hard_negative_queries"):
            rec["teacher_scores"][slot] = [None] * len(rec[slot])
            for q_idx, q in enumerate(rec[slot]):
                pairs_q.append(q)
                pairs_d.append(rec["body"])
                locs.append((r_idx, slot, q_idx))

    scores = []
    for start in range(0, len(pairs_q), score_batch_size):
        batch_q = pairs_q[start:start + score_batch_size]
        batch_d = pairs_d[start:start + score_batch_size]
        scores.extend(teacher.score_batch(batch_q, batch_d))

    for (r_idx, slot, q_idx), score in zip(locs, scores):
        if q_idx is None:
            records[r_idx]["teacher_scores"][slot] = score
        else:
            records[r_idx]["teacher_scores"][slot][q_idx] = score
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default="data/unified_examples.jsonl")
    ap.add_argument("--out", default="data/unified_examples.scored.jsonl")
    ap.add_argument("--teacher", default="Qwen/Qwen3-Reranker-8B")
    ap.add_argument("--instruction", default="Given a web search query, retrieve relevant passages that answer the query")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neg-per-keypoint", type=int, default=2)
    ap.add_argument("--hardneg-per-keypoint", type=int, default=2)
    ap.add_argument("--score-batch-size", type=int, default=32)
    ap.add_argument("--chunk-records", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="cap total records processed, for smoke tests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)
    n_done = _count_lines(args.out)
    if n_done:
        print(f"[precompute_teacher] resuming: {n_done} records already scored in {args.out}", flush=True)

    teacher_cfg = TeacherConfig(
        teacher_name=args.teacher, instruction=args.instruction, max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    teacher = QwenRerankerTeacher(teacher_cfg, device=args.device)

    n_unified = _count_lines(args.unified)
    total_target = min(args.limit, n_unified) if args.limit is not None else n_unified
    n_total = n_done
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    chunk = []
    with open(args.out, "a") as out_f:
        pbar = tqdm(desc="[precompute_teacher] scoring", initial=n_done, total=total_target)
        for rec in _iter_records(args.unified, skip=n_done):
            if total_target is not None and n_total >= total_target:
                break
            rec["negative_queries"] = rec["negative_queries"][: args.neg_per_keypoint]
            rec["hard_negative_queries"] = rec["hard_negative_queries"][: args.hardneg_per_keypoint]
            chunk.append(rec)
            if len(chunk) >= args.chunk_records:
                _score_chunk(teacher, chunk, args.score_batch_size)
                for r in chunk:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()
                n_total += len(chunk)
                pbar.update(len(chunk))
                chunk = []
        if chunk:
            _score_chunk(teacher, chunk, args.score_batch_size)
            for r in chunk:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            n_total += len(chunk)
            pbar.update(len(chunk))
        pbar.close()

    print(f"[precompute_teacher] done, {n_total} total scored records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
