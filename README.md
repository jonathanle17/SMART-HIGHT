# SMART-HIGHT

This repository connects **spectrum-driven retrieval** (ranked structure candidates) to **HIGHT-style graph tokenization**, then uses those tokens as **extra conditioning** when training and evaluating **DiffMS**. The main deliverable is a reproducible comparison of **standard DiffMS** versus **DiffMS augmented with retrieval graph tokens**.

---

## End-to-end flow

### 1) From Spectre / Marina (or equivalent) to ranked candidates

Upstream tools such as **Spectre** and **Marina** (or any pipeline that outputs “spectrum → ranked SMILES”) produce, for each query spectrum, a short list of predicted structures with scores. This repo does not run those models directly. Instead, it expects the same kind of information in **MoonshotDatasetv3** form: a zip with an index of queries (ground-truth SMILES, splits) and a **retrieval library** of candidate SMILES.

The script `run_pipeline_with_dataset.py` walks that dataset, treats each ground-truth SMILES as the query, and **re-ranks the retrieval pool** by fingerprint similarity (cosine or Tanimoto). With `--export-top5`, it writes one JSON line per sample: ground truth, split, optional spectrum placeholders, and the **top‑k candidate SMILES** with scores. That file is the bridge between “retrieval outputs” and everything below.

### 2) Graph tokenization (HIGHT implementation)

`build_diffms_training_set.py` reads the top‑k JSONL. For every candidate SMILES, `MoonshotHightAdapter` (`moonshot_to_hight_adapter.py`) turns the molecule into a graph and encodes it into a **fixed-width token sequence**:

- If a **real HIGHT** encoder and checkpoint are available (`--hight-checkpoint`), it uses the HIGHT graph tower (`encode_mol` on a HIGHT `MolGraph`).
- Otherwise it uses the same **RDKit graph + optional Laplacian positional encodings** path as the rest of this repo, with a **dummy graph encoder** so shapes and the pipeline stay correct without the full HIGHT checkpoint.

Outputs are a **metadata JSONL** (sample id, split, candidates, tensor indices) and an **NPZ** of padded token tensors and masks aligned row-for-row with that JSONL. Those two artifacts together are the **DiffMS bridge dataset**.

### 3) Merging splits

If you built separate train / val / test bridge files, `merge_bridge_artifacts.py` concatenates the JSONL and NPZ into a single pair (e.g. `diffms_bridge_all.jsonl` + `diffms_bridge_all_tokens.npz`) for one DiffMS run over all splits.

### 4) DiffMS training and metric comparison

`run_true_diffms_comparison.py` lives in **SMART-HIGHT** but **subprocesses into your local DiffMS clone**. It repeatedly launches DiffMS’s `src.spec2mol_main` with Hydra overrides that point at the bridge files and sweep **conditioning modes**:

| Mode              | Role |
|-------------------|------|
| `baseline`        | Standard DiffMS conditioning (no HIGHT bridge). |
| `hight_augmented` | Condition on the graph tokens from the bridge (your devised setting). |
| `hight_shuffled`  | Control: same tensors, shuffled alignment to check that gains are not spurious. |

For each **seed** and **mode**, DiffMS writes training/validation/test metrics (for example top‑k accuracy and related scores logged as `test/acc_at_*`, Tanimoto metrics, etc.). Results are collected into `artifacts/true_diffms_comparison.json` (paths configurable).

`summarize_true_diffms_comparison.py` reads that JSON, aggregates **mean ± std** per mode, and computes **baseline vs `hight_augmented`** absolute and **relative (%)** deltas on the main test metrics.

---

## What you need on your machine

**SMART-HIGHT (this repo)**

- Python 3.12+ recommended; a virtual environment under the repo root is fine (examples below use `.venv`).
- **Required:** `torch`, `rdkit`, `numpy`.
- **Optional:** `torch-geometric` and `scipy` if you rely on **Laplacian PE** in the fallback graph path (`build_diffms_training_set.py` enables LapPE by default; you can disable with `--no-lap-pe` if you want to avoid those deps).

**DiffMS (separate clone)**

- Install and run according to the DiffMS project (conda/RDKit, `pip install -e .`, MSG or CANOPUS data layout).
- The comparison script needs a Python that can `import torch` and `pytorch_lightning` (the DiffMS env is the right choice).

---

## How to run (fresh clone)

Replace paths with your own dataset zip, DiffMS root, and venv Python.

### 1) Environment (SMART-HIGHT)

```powershell
cd C:\path\to\SMART-HIGHT
python -m venv .venv
.\.venv\Scripts\pip install torch rdkit numpy
# Optional, for LapPE in the fallback encoder path:
.\.venv\Scripts\pip install torch-geometric scipy
```

### 2) Export top‑5 retrieval JSONL per split

```powershell
.\.venv\Scripts\python.exe .\run_pipeline_with_dataset.py `
  --dataset-root "C:\path\to\MoonshotDatasetv3.zip" `
  --split train `
  --max-samples 200 `
  --k 5 `
  --similarity cosine `
  --export-top5 `
  --export-path ".\artifacts\moonshot_top5_train.jsonl" `
  --seed 0
```

Repeat with `--split val` / `test` and different `--export-path` values.

### 3) Build the DiffMS bridge (JSONL + NPZ)

```powershell
.\.venv\Scripts\python.exe .\build_diffms_training_set.py `
  --moonshot-top5-jsonl ".\artifacts\moonshot_top5_train.jsonl" `
  --out-jsonl ".\artifacts\diffms_bridge_train.jsonl" `
  --out-npz ".\artifacts\diffms_bridge_train_tokens.npz" `
  --hidden-dim 128
```

With a real HIGHT checkpoint:

```powershell
.\.venv\Scripts\python.exe .\build_diffms_training_set.py `
  --moonshot-top5-jsonl ".\artifacts\moonshot_top5_train.jsonl" `
  --out-jsonl ".\artifacts\diffms_bridge_train.jsonl" `
  --out-npz ".\artifacts\diffms_bridge_train_tokens.npz" `
  --hidden-dim 128 `
  --hight-checkpoint "C:\path\to\hight_checkpoint.pt"
```

### 4) Merge splits (if applicable)

```powershell
.\.venv\Scripts\python.exe .\merge_bridge_artifacts.py `
  --train-jsonl ".\artifacts\diffms_bridge_train.jsonl" `
  --train-npz ".\artifacts\diffms_bridge_train_tokens.npz" `
  --val-jsonl ".\artifacts\diffms_bridge_val.jsonl" `
  --val-npz ".\artifacts\diffms_bridge_val_tokens.npz" `
  --out-jsonl ".\artifacts\diffms_bridge_all.jsonl" `
  --out-npz ".\artifacts\diffms_bridge_all_tokens.npz"
```

### 5) Run true DiffMS baseline vs HIGHT-augmented comparison

From **SMART-HIGHT**, pointing at your **DiffMS** install and bridge files (absolute paths avoid Hydra cwd issues):

```powershell
C:\path\to\DiffMS\.venv\Scripts\python.exe .\run_true_diffms_comparison.py `
  --diffms-root "C:\path\to\DiffMS" `
  --python "C:\path\to\DiffMS\.venv\Scripts\python.exe" `
  --bridge-jsonl "C:\path\to\SMART-HIGHT\artifacts\diffms_bridge_all.jsonl" `
  --bridge-npz "C:\path\to\SMART-HIGHT\artifacts\diffms_bridge_all_tokens.npz" `
  --dataset msg `
  --seeds 0 1 2 `
  --epochs 1 `
  --out-json ".\artifacts\true_diffms_comparison.json"
```

Omit `--max-count` for the full MSG split (slow on CPU). For a faster smoke test, use a **large** cap; very small values can empty val/test and break DiffMS:

```powershell
C:\path\to\DiffMS\.venv\Scripts\python.exe .\run_true_diffms_comparison.py `
  --diffms-root "C:\path\to\DiffMS" `
  --python "C:\path\to\DiffMS\.venv\Scripts\python.exe" `
  --bridge-jsonl "C:\path\to\SMART-HIGHT\artifacts\diffms_bridge_all.jsonl" `
  --bridge-npz "C:\path\to\SMART-HIGHT\artifacts\diffms_bridge_all_tokens.npz" `
  --dataset msg `
  --seeds 0 1 2 `
  --epochs 1 `
  --max-count 8000 `
  --out-json ".\artifacts\true_diffms_comparison.json"
```

### 6) Summarize accuracy deltas (baseline vs devised model)

```powershell
C:\path\to\DiffMS\.venv\Scripts\python.exe .\summarize_true_diffms_comparison.py `
  --runs-json ".\artifacts\true_diffms_comparison.json" `
  --out-json ".\artifacts\true_diffms_comparison_summary.json"
```

The summary JSON includes `baseline_vs_hight_augmented` with **relative_delta_percent** for metrics such as `test/acc_at_1` and `test/acc_at_5`.

---

## Optional: lightweight multitask probe (this repo only)

`train_diffms_multitask.py` runs a small **local** baseline / multi-task / shuffled control without the full DiffMS stack. Use it for quick sanity checks; treat `run_true_diffms_comparison.py` as the authoritative **DiffMS-native** metric comparison.

---

## Troubleshooting

- **`ModuleNotFoundError: rdkit`** — install RDKit into the SMART-HIGHT venv: `pip install rdkit`.
- **LapPE / scipy errors** — install `scipy` and `torch-geometric`, or build the bridge with `--no-lap-pe`.
- **`run_true_diffms_comparison.py` complains about torch / Lightning** — pass `--python` pointing to the DiffMS environment’s `python.exe`.
- **`--gpus` > 0 but no CUDA** — use `--gpus 0` or install a CUDA build of PyTorch and working drivers.
