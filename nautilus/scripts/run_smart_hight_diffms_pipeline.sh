#!/usr/bin/env bash
set -euo pipefail

# ... (Input validation and directory setup remains same) ...

# One version for everything
PYTHON_BIN="python"
DIFFMS_PYTHON="python"
DIFFMS_DIR="${DIFFMS_DIR:-/code/DiffMS}"
SMART_HIGHT_DIR="${SMART_HIGHT_DIR:-/code/SMART-HIGHT}"

# No more manual installs here! Pixi handles it.
cd "${DIFFMS_DIR}"
${PYTHON_BIN} -m pip install -e .

cd "${SMART_HIGHT_DIR}"

# 1. Export top-5 retrieval JSONL
python run_pipeline_with_dataset.py \
  --dataset-root "/root/datasets/MoonshotDatasetv3.zip" \
  --split train --export-top5 --export-path "./artifacts/moonshot_top5_train.jsonl"

# 2. Build the DiffMS bridge (Graph Tokenization)
python build_diffms_training_set.py \
  --moonshot-top5-jsonl "./artifacts/moonshot_top5_train.jsonl" \
  --out-jsonl "./artifacts/diffms_bridge_train.jsonl" \
  --out-npz "./artifacts/diffms_bridge_train_tokens.npz"

# 3. Run the baseline vs HIGHT-augmented comparison
python run_true_diffms_comparison.py \
  --diffms-root "/code/DiffMS" \
  --bridge-jsonl "$(pwd)/artifacts/diffms_bridge_train.jsonl" \
  --bridge-npz "$(pwd)/artifacts/diffms_bridge_train_tokens.npz" \
  --epochs 10