#!/bin/bash
# Builds the token-level IDF table over the full document corpus (used by the
# Chamfer loss's coverage term). CPU-only, multiprocessing over the student
# tokenizer; takes a while over the full ~126K-doc corpus.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.data.idf \
    --keypoints-dir ../data/msmarco-document/full_run \
    --tokenizer Qwen/Qwen3-0.6B \
    --out data/idf_table.pt \
    --num-workers 8 \
    "$@"
