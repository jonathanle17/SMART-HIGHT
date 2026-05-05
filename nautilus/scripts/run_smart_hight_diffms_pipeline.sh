#!/usr/bin/env bash
set -euo pipefail

# ... (Input validation and directory setup remains same) ...

# One version for everything
PYTHON_BIN="python"
DIFFMS_PYTHON="python"

# No more manual installs here! Pixi handles it.
cd "git@github.com:coleygroup/DiffMS.git"
${PYTHON_BIN} -m pip install -e .

cd "git@github.com:jonathanle17/SMART-HIGHT.git"
# ... (Rest of the pipeline execution) ...