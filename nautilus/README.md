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
- A **docker-registry pull secret** for `gitlab-registry.nrp-nautilus.io` (see below), unless your namespace already provides one (then set `imagePullSecrets` in the job YAML to that secret name).
- If `SMART_HIGHT_REPO` is an **SSH** URL (`git@github.com:...`), a deploy key (or similar) on the PVC at `/root/gurusmart/.ssh/id_rsa`, or mounted at `/ssh/id_rsa` (add a `Secret` volume + `volumeMount` for `/ssh` if you use that layout).

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
- `spec.template.spec.imagePullSecrets[0].name` if your registry secret uses a different name than `gitlab-registry-nrp-pull`

### Registry pull secret (private container image)

The job image is hosted on GitLab’s NRP registry. Create a pull secret once in your namespace (use credentials your project issued for that registry; do not commit them to git):

```bash
kubectl create secret docker-registry gitlab-registry-nrp-pull \
  --docker-server=gitlab-registry.nrp-nautilus.io \
  --docker-username='<username-or-token-name>' \
  --docker-password='<password-or-token>' \
  --namespace=guru-research
```

If you already have a secret (for example from cluster onboarding), edit both job manifests so `imagePullSecrets` references that name instead of `gitlab-registry-nrp-pull`.

### Git SSH key layout

The job copies `id_rsa` from, in order:

1. `/root/gurusmart/.ssh/` on your user PVC (recommended), or
2. `/ssh/` if you mount a key there (add a `volume` from a `Secret` and `volumeMount` at `mountPath: /ssh`).

If `SMART_HIGHT_REPO` is SSH and no key is found, the container exits immediately with a clear error instead of failing later at `git clone`. To avoid SSH keys entirely, switch `SMART_HIGHT_REPO` to an HTTPS URL and supply a token via a Kubernetes `Secret` (not implemented in these manifests by default).

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
