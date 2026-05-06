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
${PYTHON_BIN} -km pip install -e .

cd "${SMART_HIGHT_DIR}"
# ... (Rest of the pipeline execution) ...