#!/bin/bash
# Scores every (query, doc) pair needed for training with Qwen3-Reranker-8B
# and writes the resumable, chunked "scored" JSONL that train.py consumes.
# Run on a GPU node -- re-run after an interrupt and it resumes automatically
# by counting existing lines in --out.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.distill.precompute_teacher_scores \
    --unified data/unified_examples.jsonl \
    --out data/unified_examples.scored.jsonl \
    --teacher Qwen/Qwen3-Reranker-8B \
    --neg-per-keypoint 2 \
    --hardneg-per-keypoint 2 \
    --score-batch-size 32 \
    --chunk-records 256 \
    "$@"
