#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting SMART-HIGHT + DiffMS pipeline"

# ---- Required inputs ----
: "${MOONSHOT_DATASET_ZIP:?Set MOONSHOT_DATASET_ZIP (example: MoonshotDatasetv3.zip)}"
: "${SMART_HIGHT_REPO:?Set SMART_HIGHT_REPO (ssh/https git URL)}"
: "${DIFFMS_REPO:?Set DIFFMS_REPO (ssh/https git URL)}"

# ---- Optional inputs with defaults ----
SMART_HIGHT_BRANCH="${SMART_HIGHT_BRANCH:-main}"
DIFFMS_BRANCH="${DIFFMS_BRANCH:-master}"
PIPELINE_MODE="${PIPELINE_MODE:-smoke}" # smoke | full
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="${WORKDIR:-/code}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/gurusmart/experiments}"
DATASET_DIR="${DATASET_DIR:-/workspace/datasets}"
K_TOP="${K_TOP:-5}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SIMILARITY="${SIMILARITY:-cosine}"
GPU_COUNT="${GPU_COUNT:-2}"

SMART_HIGHT_DIR="${WORKDIR}/SMART-HIGHT"
DIFFMS_DIR="${WORKDIR}/DiffMS"
DATASET_PATH="${DATASET_DIR}/${MOONSHOT_DATASET_ZIP}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TS}_${PIPELINE_MODE}"
mkdir -p "${RUN_DIR}" "${WORKDIR}" "${DATASET_DIR}"

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Dataset path: ${DATASET_PATH}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[ERROR] Dataset zip not found at ${DATASET_PATH}"
  exit 1
fi

if [[ ! -d "${SMART_HIGHT_DIR}" ]]; then
  echo "[INFO] Cloning SMART-HIGHT (${SMART_HIGHT_BRANCH})"
  git clone --branch "${SMART_HIGHT_BRANCH}" "${SMART_HIGHT_REPO}" "${SMART_HIGHT_DIR}"
else
  echo "[INFO] Reusing existing ${SMART_HIGHT_DIR}"
fi

if [[ ! -d "${DIFFMS_DIR}" ]]; then
  echo "[INFO] Cloning DiffMS (${DIFFMS_BRANCH})"
  git clone --branch "${DIFFMS_BRANCH}" "${DIFFMS_REPO}" "${DIFFMS_DIR}"
else
  echo "[INFO] Reusing existing ${DIFFMS_DIR}"
fi

echo "[INFO] Installing runtime dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install numpy scipy pandas matplotlib rdkit
"${PYTHON_BIN}" -m pip install torch-geometric pytorch-lightning hydra-core

# DiffMS requires Python 3.9; keep earlier pipeline steps on PYTHON_BIN.
DIFFMS_PYTHON=""
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -xq "diffms"; then
    echo "[INFO] Creating Conda env 'diffms' (python=3.9, rdkit=2024.09.4)"
    conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9
  fi
  conda activate diffms
  DIFFMS_PYTHON="python"
elif command -v python3.9 >/dev/null 2>&1; then
  DIFFMS_PYTHON="python3.9"
else
  echo "[ERROR] DiffMS requires Python 3.9, but no conda env support or python3.9 binary was found." >&2
  echo "[ERROR] Keep image/auth/dataset setup unchanged; add conda (recommended) or python3.9 to the image." >&2
  exit 1
fi

"${DIFFMS_PYTHON}" -m pip install --upgrade pip
"${DIFFMS_PYTHON}" -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
"${DIFFMS_PYTHON}" -m pip install -e "${DIFFMS_DIR}"

cd "${SMART_HIGHT_DIR}"

if [[ "${PIPELINE_MODE}" == "smoke" ]]; then
  MAX_SAMPLES=64
  MAX_COUNT=2000
  EPOCHS=1
  SEEDS=(0)
else
  MAX_SAMPLES=0
  MAX_COUNT=0
  EPOCHS=20
  SEEDS=(0 1 2)
fi

ART_DIR="${RUN_DIR}/artifacts"
mkdir -p "${ART_DIR}"

echo "[INFO] Exporting top-k retrieval jsonl (train/val)"
for SPLIT in train val; do
  SPLIT_ARGS=()
  if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
    SPLIT_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  "${PYTHON_BIN}" run_pipeline_with_dataset.py \
    --dataset-root "${DATASET_PATH}" \
    --split "${SPLIT}" \
    --k "${K_TOP}" \
    --similarity "${SIMILARITY}" \
    --export-top5 \
    --export-path "${ART_DIR}/moonshot_top5_${SPLIT}.jsonl" \
    --seed 0 \
    "${SPLIT_ARGS[@]}"
done

echo "[INFO] Building bridge artifacts (train/val)"
for SPLIT in train val; do
  "${PYTHON_BIN}" build_diffms_training_set.py \
    --moonshot-top5-jsonl "${ART_DIR}/moonshot_top5_${SPLIT}.jsonl" \
    --out-jsonl "${ART_DIR}/diffms_bridge_${SPLIT}.jsonl" \
    --out-npz "${ART_DIR}/diffms_bridge_${SPLIT}_tokens.npz" \
    --hidden-dim "${HIDDEN_DIM}"
done

echo "[INFO] Merging bridge artifacts"
"${PYTHON_BIN}" merge_bridge_artifacts.py \
  --train-jsonl "${ART_DIR}/diffms_bridge_train.jsonl" \
  --train-npz "${ART_DIR}/diffms_bridge_train_tokens.npz" \
  --val-jsonl "${ART_DIR}/diffms_bridge_val.jsonl" \
  --val-npz "${ART_DIR}/diffms_bridge_val_tokens.npz" \
  --out-jsonl "${ART_DIR}/diffms_bridge_all.jsonl" \
  --out-npz "${ART_DIR}/diffms_bridge_all_tokens.npz"

echo "[INFO] Running true DiffMS comparison"
COMPARE_ARGS=(
  --diffms-root "${DIFFMS_DIR}"
  --python "${DIFFMS_PYTHON}"
  --bridge-jsonl "${ART_DIR}/diffms_bridge_all.jsonl"
  --bridge-npz "${ART_DIR}/diffms_bridge_all_tokens.npz"
  --dataset msg
  --epochs "${EPOCHS}"
  --gpus "${GPU_COUNT}"
  --out-json "${ART_DIR}/true_diffms_comparison.json"
)

for SEED in "${SEEDS[@]}"; do
  COMPARE_ARGS+=(--seeds "${SEED}")
done

if [[ "${MAX_COUNT}" -gt 0 ]]; then
  COMPARE_ARGS+=(--max-count "${MAX_COUNT}")
fi

"${PYTHON_BIN}" run_true_diffms_comparison.py "${COMPARE_ARGS[@]}"

echo "[INFO] Summarizing results"
"${PYTHON_BIN}" summarize_true_diffms_comparison.py \
  --runs-json "${ART_DIR}/true_diffms_comparison.json" \
  --out-json "${ART_DIR}/true_diffms_comparison_summary.json"

echo "[INFO] Completed. Artifacts available at ${ART_DIR}"
