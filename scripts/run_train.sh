#!/bin/bash
# Trains the latent-bottleneck reranker (backbone + query adapter + document
# compressor, jointly) against the teacher-scored data + IDF table. Run on a
# GPU node; use `accelerate launch` directly for multi-GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

accelerate launch -m src.train --config configs/default.yaml "$@"
