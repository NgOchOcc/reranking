#!/bin/bash
# Joins full_run (keypoints + positive queries) with negative_query by doc_id
# into one unified, resumable-friendly JSONL (one line per (doc_id, keypoint)).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.data.build_index \
    --keypoints-dir ../data/msmarco-document/full_run \
    --negative-dir ../data/msmarco-document/negative_query \
    --out data/unified_examples.jsonl \
    "$@"
