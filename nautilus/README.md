# Nautilus Execution for SMART-HIGHT + DiffMS

This folder lets you run the full pipeline on Nautilus with Kubernetes Jobs:

1. Export top-k retrieval candidates from Moonshot data.
2. Build HIGHT bridge tensors (`JSONL` + `NPZ`).
3. Run DiffMS in baseline / hight_augmented / hight_shuffled modes.
4. Write comparison and summary JSON outputs to your PVC.

## Prerequisites

- Access to Nautilus/NRP with `kubectl` configured to `guru-research` namespace.
- The shared `smart-datasets` PVC available with `MoonshotDatasetv3.zip`.
- Your private PVC created from `volumes/andre-smart-hight-pvc.yaml`.

## One-time setup

```bash
kubectl apply -f nautilus/volumes/andre-smart-hight-pvc.yaml
```

## Configure manifests

Edit both files in `nautilus/jobs/` before running:

- `smart-hight-diffms-smoke.yaml`
- `smart-hight-diffms-full.yaml`

Update at minimum:

- `metadata.name`
- `volumes[].persistentVolumeClaim.claimName` for your user PVC
- `SMART_HIGHT_REPO` (your repo URL)
- `SMART_HIGHT_BRANCH` (your branch)
- `MOONSHOT_DATASET_ZIP` if your dataset zip differs

## Run smoke test

```bash
kubectl apply -f nautilus/jobs/smart-hight-diffms-smoke.yaml
kubectl logs -f job/andre-smart-hight-diffms-smoke
```

When done:

```bash
kubectl get pods
kubectl describe job andre-smart-hight-diffms-smoke
```

Expected output location on PVC:

`/root/gurusmart/experiments/<timestamp>_smoke/artifacts/`

## Run full experiment

```bash
kubectl apply -f nautilus/jobs/smart-hight-diffms-full.yaml
kubectl logs -f job/andre-smart-hight-diffms-full
```

Expected output location on PVC:

`/root/gurusmart/experiments/<timestamp>_full/artifacts/`

Contains:

- `true_diffms_comparison.json`
- `true_diffms_comparison_summary.json`
- bridge files used for training/evaluation

## Runner script behavior

`nautilus/scripts/run_smart_hight_diffms_pipeline.sh`:

- Clones `SMART-HIGHT` and `DiffMS` into `/code` if missing.
- Installs Python dependencies and `DiffMS` editable package.
- Executes train/val export, bridge generation, merge, and DiffMS comparison.
- Supports modes:
  - `smoke`: 1 seed, 1 epoch, reduced counts
  - `full`: 3 seeds, 20 epochs, full-size run
