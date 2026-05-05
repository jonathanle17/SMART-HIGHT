#!/usr/bin/env bash
set -euo pipefail

# ... (Input validation and directory setup remains same) ...

# One version for everything
PYTHON_BIN="python"
DIFFMS_PYTHON="python"

# No more manual installs here! Pixi handles it.
cd "/code/DiffMS"
${PYTHON_BIN} -m pip install -e .

cd "/code/SMART-HIGHT"
# ... (Rest of the pipeline execution) ...