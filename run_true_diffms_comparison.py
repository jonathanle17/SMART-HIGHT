from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _read_metrics_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        for k, v in row.items():
            if not k or not v:
                continue
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _run_one(
    python_exe: str,
    diffms_root: Path,
    run_name: str,
    dataset_name: str,
    mode: str,
    seed: int,
    bridge_jsonl: Path,
    bridge_npz: Path,
    n_epochs: int,
    max_count: int | None,
    dataset_rel_base: str | None,
    test_samples: int,
    val_samples: int,
    gpus: int,
    progress_bar: bool,
    hydra_extra: list[str],
) -> tuple[int, Path]:
    run_dir = diffms_root / "outputs" / "moonshot_compare" / run_name
    cmd = [
        python_exe,
        "-m",
        "src.spec2mol_main",
        f"general.name={run_name}",
        "general.wandb=disabled",
        f"general.gpus={gpus}",
        f"dataset={dataset_name}",
    ]
    if max_count is not None:
        cmd.append(f"dataset.max_count={max_count}")
    cmd.extend(
        [
            f"dataset.conditioning_mode={mode}",
            f"dataset.hight_bridge_jsonl={bridge_jsonl.as_posix()}",
            f"dataset.hight_bridge_npz={bridge_npz.as_posix()}",
            "dataset.hight_fusion=add",
            f"train.n_epochs={n_epochs}",
            f"train.seed={seed}",
            "train.num_workers=0",
            f"general.test_samples_to_generate={test_samples}",
            f"general.val_samples_to_generate={val_samples}",
            "hydra.run.dir=outputs/moonshot_compare/${general.name}",
        ]
    )
    # Hydra job cwd is outputs/.../<run_name>; configs use ../../../ to reach repo root.
    if dataset_rel_base:
        rel = dataset_rel_base.replace("\\", "/").strip("/")
        cmd.extend(
            [
                f"dataset.datadir=../../../{rel}",
                f"dataset.split_file=../../../{rel}/split.tsv",
                f"dataset.labels_file=../../../{rel}/labels.tsv",
                f"dataset.spec_folder=../../../{rel}/spec_files",
                f"dataset.subform_folder=../../../{rel}/subformulae/default_subformulae",
            ]
        )
    if progress_bar:
        cmd.append("train.progress_bar=true")
    cmd.extend(hydra_extra)
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    existing = env.get("PYTHONPATH", "")
    add_paths = f"{diffms_root}{os.pathsep}{diffms_root / 'src'}"
    env["PYTHONPATH"] = add_paths if not existing else f"{add_paths}{os.pathsep}{existing}"
    res = subprocess.run(cmd, cwd=str(diffms_root), env=env)
    return res.returncode, run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run true DiffMS baseline vs HIGHT-augmented comparison.")
    parser.add_argument("--diffms-root", type=str, default="C:/Users/andre/DiffMS")
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable that has DiffMS deps (torch, lightning, torch_geometric, rdkit). "
        "Default: interpreter running this script.",
    )
    parser.add_argument("--bridge-jsonl", type=str, default="C:/Users/andre/SMART-HIGHT/artifacts/diffms_bridge_all.jsonl")
    parser.add_argument("--bridge-npz", type=str, default="C:/Users/andre/SMART-HIGHT/artifacts/diffms_bridge_all_tokens.npz")
    parser.add_argument("--dataset", type=str, default="msg", choices=["msg", "canopus"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Cap spectra loaded (first N paths from spec_folder glob). "
        "Too small can leave val/test empty and crash. Default: no cap (msg.yaml null).",
    )
    parser.add_argument("--out-json", type=str, default="artifacts/true_diffms_comparison.json")
    parser.add_argument(
        "--dataset-rel-base",
        type=str,
        default="data/fp2mol/raw/msg",
        help="MSG (or canopus) folder relative to DiffMS repo root when data is not in data/msg. "
        "Use forward slashes. Example: data/fp2mol/raw/msg",
    )
    parser.add_argument(
        "--no-dataset-override",
        action="store_true",
        help="Use only msg.yaml / canopus.yaml paths (no Hydra dataset path overrides).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="DiffMS molecules to sample per example during test (lower = faster).",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="DiffMS molecules to sample per example during val sampling steps.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs for DiffMS (general.gpus). Requires CUDA torch; 0 = CPU.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Turn on Lightning tqdm (train.progress_bar=true). Needs DiffMS Trainer wired to cfg.train.progress_bar.",
    )
    parser.add_argument(
        "--hydra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides (repeatable), e.g. --hydra train.log_every_n_steps=10",
    )
    args = parser.parse_args()

    python_exe = args.python or sys.executable
    chk = subprocess.run(
        [python_exe, "-c", "import torch; import pytorch_lightning"],
        capture_output=True,
        text=True,
    )
    if chk.returncode != 0:
        print(
            "The chosen Python cannot import torch / pytorch_lightning.\n"
            f"  python: {python_exe}\n"
            "Create a DiffMS env (see DiffMS README: conda + rdkit, pip install torch, pip install -e .)\n"
            "or pass --python path\\to\\that\\python.exe",
            file=sys.stderr,
        )
        if chk.stderr:
            print(chk.stderr, file=sys.stderr)
        return 1

    if args.gpus > 0:
        cuda_chk = subprocess.run(
            [
                python_exe,
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ],
            capture_output=True,
            text=True,
        )
        if cuda_chk.returncode != 0:
            print(
                "You passed --gpus > 0 but torch.cuda.is_available() is False.\n"
                "Install a CUDA build of PyTorch and NVIDIA drivers, or use --gpus 0.",
                file=sys.stderr,
            )
            return 1

    diffms_root = Path(args.diffms_root)
    # Hydra cwd is under DiffMS outputs; relative paths must be absolute.
    bridge_jsonl = Path(args.bridge_jsonl).resolve()
    bridge_npz = Path(args.bridge_npz).resolve()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    modes = ["baseline", "hight_augmented", "hight_shuffled"]
    records = []
    ds_override = None if args.no_dataset_override else args.dataset_rel_base
    for seed in args.seeds:
        for mode in modes:
            run_name = f"moonshot_{mode}_s{seed}"
            code, run_dir = _run_one(
                python_exe=python_exe,
                diffms_root=diffms_root,
                run_name=run_name,
                dataset_name=args.dataset,
                mode=mode,
                seed=seed,
                bridge_jsonl=bridge_jsonl,
                bridge_npz=bridge_npz,
                n_epochs=args.epochs,
                max_count=args.max_count,
                dataset_rel_base=ds_override,
                test_samples=args.test_samples,
                val_samples=args.val_samples,
                gpus=args.gpus,
                progress_bar=args.progress_bar,
                hydra_extra=list(args.hydra),
            )
            metrics_csv = run_dir / "logs" / run_name / run_name / "version_0" / "metrics.csv"
            if not metrics_csv.exists():
                alt = list(run_dir.glob("**/metrics.csv"))
                metrics_csv = alt[-1] if alt else metrics_csv
            metrics = _read_metrics_csv(metrics_csv)
            records.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "return_code": code,
                    "run_dir": str(run_dir),
                    "metrics_csv": str(metrics_csv),
                    "metrics": metrics,
                }
            )

    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote run records to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
