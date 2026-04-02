"""
Run the pipeline on the real MoonshotDatasetv3: original flow (retrieval only)
and developed flow (retrieval + graph tokenization). Report accuracy and
optionally generate visualizations.

--- What is the first input? ---
  The program's input is the dataset (index.pkl + retrieval.pkl). Each
  "query" is one molecule from the dataset: we use its ground-truth SMILES
  as the query (simulating that a spectrum produced that molecule). So we
  are already "testing on random molecules from the dataset"—either one
  random molecule per run (--single-random) or many (e.g. 200 from val).

--- Why a dummy spectrum and dummy encoder? ---
  • Dummy spectrum: The index we load only has SMILES/split, not raw
    spectrum files (m/z vs intensity). So when we plot a spectrum we have
    nothing to show and use a synthetic curve. Real spectra would come
    from SMART-Moonshot or another source when wired in.
  • Dummy encoder: Graph tokenization uses a small placeholder (DummyGraphEncoder)
    instead of the real HIGHT graph tower because HIGHT lives in another repo
    and needs checkpoints; we keep the pipeline runnable here without it.

--- Similarity: cosine vs Tanimoto ---
  SMART-Moonshot may use cosine similarity in embedding space. We support
  both: --similarity cosine (default, to align with that) or --similarity tanimoto.
  Both use the same Morgan fingerprints; only the score function changes.

Note: This script does NOT run the DiffMS model. "Original" = retrieval-only;
"Developed" = retrieval + graph tokens. DiffMS integration is a next step.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moonshot_data_loader import (
    MOONSHOT_DATASET_ROOT,
    MoonshotDataLoader,
    precompute_fingerprints,
    topk_by_similarity,
)
from pipeline_skeleton import CandidateMolecule, encode_candidates_to_tokens
from visualization import (
    SpectrumPlotData,
    plot_embedding_pca,
    plot_mass_spectrum,
    plot_token_heatmap,
    plot_topk_smiles_grid,
)


def run_original_flow(
    loader: MoonshotDataLoader,
    query_smiles: str,
    k: int = 5,
    retrieval_fps: list[tuple[str, object]] | None = None,
    similarity_metric: str = "cosine",
) -> list[tuple[str, float]]:
    """
    Original flow: retrieval only. Rank retrieval library by fingerprint
    similarity (cosine or Tanimoto). If retrieval_fps provided, reuse for speed.
    """
    if retrieval_fps is not None:
        return topk_by_similarity(
            query_smiles, [], k=k, candidate_fps=retrieval_fps,
            similarity_metric=similarity_metric,
        )
    retrieval = loader.get_retrieval_smiles()
    return topk_by_similarity(
        query_smiles, retrieval, k=k, similarity_metric=similarity_metric,
    )


def run_developed_flow(
    loader: MoonshotDataLoader,
    query_smiles: str,
    k: int = 5,
    retrieval_fps: list[tuple[str, object]] | None = None,
    similarity_metric: str = "cosine",
):
    """
    Developed flow: same retrieval as original, then encode top-k into
    graph tokens (HIGHT-style). Returns (candidates, token_seqs, pooled).
    """
    topk = run_original_flow(
        loader, query_smiles, k=k, retrieval_fps=retrieval_fps,
        similarity_metric=similarity_metric,
    )
    candidates = [CandidateMolecule(smiles=s, score=score) for s, score in topk]
    token_seqs, pooled = encode_candidates_to_tokens(candidates, hidden_dim=128)
    return candidates, token_seqs, pooled


def plot_accuracy_comparison(
    orig_top1: float,
    orig_top5: float,
    dev_top1: float,
    dev_top5: float,
    out_path: str,
) -> None:
    """Bar chart: Original model vs Developed model, Top-1 and Top-5 accuracy (%)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)  # Top-1, Top-5
    width = 0.35
    orig = [orig_top1 * 100, orig_top5 * 100]
    dev = [dev_top1 * 100, dev_top5 * 100]
    bars1 = ax.bar(x - width / 2, orig, width, label="Original (retrieval only)", color="steelblue")
    bars2 = ax.bar(x + width / 2, dev, width, label="Developed (retrieval + graph tokens)", color="coral")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Original vs Developed pipeline (DiffMS not run yet)")
    ax.set_xticks(x)
    ax.set_xticklabels(("Top-1", "Top-5"))
    ax.legend()
    ax.set_ylim(0, 105)
    for b in bars1 + bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_flows(
    dataset_root: str | Path,
    split: str = "val",
    max_samples: int = 200,
    k: int = 5,
    similarity_metric: str = "cosine",
) -> tuple[float, float, float, float]:
    """
    Evaluate both flows on the dataset. For each sample, query = ground-truth
    SMILES. Original = top-k retrieval; developed = same retrieval + graph encoding.

    Returns (orig_top1_acc, orig_top5_acc, dev_top1_acc, dev_top5_acc).
    """
    loader = MoonshotDataLoader(dataset_root)
    samples = loader.get_samples(split=split, max_samples=max_samples)
    if not samples:
        return 0.0, 0.0, 0.0, 0.0

    retrieval_smiles = loader.get_retrieval_smiles()
    retrieval_fps = precompute_fingerprints(retrieval_smiles)

    orig_top1, orig_top5 = 0, 0
    dev_top1, dev_top5 = 0, 0

    for s in samples:
        gt = s.smiles
        topk_orig = run_original_flow(
            loader, gt, k=k, retrieval_fps=retrieval_fps,
            similarity_metric=similarity_metric,
        )
        orig_preds = [x[0] for x in topk_orig]
        if orig_preds and orig_preds[0] == gt:
            orig_top1 += 1
        if gt in orig_preds:
            orig_top5 += 1

        topk_dev = run_original_flow(
            loader, gt, k=k, retrieval_fps=retrieval_fps,
            similarity_metric=similarity_metric,
        )
        dev_preds = [x[0] for x in topk_dev]
        if dev_preds and dev_preds[0] == gt:
            dev_top1 += 1
        if gt in dev_preds:
            dev_top5 += 1

    n = len(samples)
    return (
        orig_top1 / n,
        orig_top5 / n,
        dev_top1 / n,
        dev_top5 / n,
    )


def export_top5_predictions(
    dataset_root: str | Path,
    out_path: str | Path,
    split: str = "train",
    max_samples: int | None = None,
    k: int = 5,
    similarity_metric: str = "cosine",
    seed: int = 0,
) -> int:
    """
    Export deterministic top-k predictions as JSONL for downstream tokenization/training.
    Each line schema:
      {
        "sample_id": int,
        "split": str,
        "gt_smiles": str,
        "top5_candidates": [{"rank": int, "pred_smiles": str, "moonshot_score": float}]
      }
    """
    random.seed(seed)
    np.random.seed(seed)

    loader = MoonshotDataLoader(dataset_root)
    samples = loader.get_samples(split=split, max_samples=max_samples)
    retrieval_fps = precompute_fingerprints(loader.get_retrieval_smiles())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            topk = run_original_flow(
                loader,
                s.smiles,
                k=k,
                retrieval_fps=retrieval_fps,
                similarity_metric=similarity_metric,
            )
            payload = {
                "sample_id": int(s.idx),
                "split": s.split,
                "gt_smiles": s.smiles,
                "top5_candidates": [
                    {
                        "rank": i + 1,
                        "pred_smiles": smiles,
                        "moonshot_score": float(score),
                    }
                    for i, (smiles, score) in enumerate(topk)
                ],
            }
            f.write(json.dumps(payload) + "\n")
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline on MoonshotDatasetv3: original vs developed flow, report accuracy."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(MOONSHOT_DATASET_ROOT),
        help="Path to MoonshotDatasetv3.zip or unzipped folder (index.pkl, retrieval.pkl)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max number of samples to evaluate (for speed)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="debug_outputs",
        help="Directory for plots when --plot-example is set",
    )
    parser.add_argument(
        "--plot-example",
        action="store_true",
        help="Generate visualization for one example (spectrum, top-k, tokens, PCA)",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "tanimoto"],
        help="Fingerprint similarity: cosine (SMART-Moonshot style) or tanimoto",
    )
    parser.add_argument(
        "--single-random",
        action="store_true",
        help="Run on one random molecule from the split; report hit/miss and optionally --plot-example",
    )
    parser.add_argument(
        "--export-top5",
        action="store_true",
        help="Export top-k predictions to JSONL for downstream HIGHT/DiffMS pipeline",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="artifacts/moonshot_top5.jsonl",
        help="Path for --export-top5 output JSONL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic export/eval order",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset not found: {dataset_root}")
        print("Set --dataset-root to MoonshotDatasetv3.zip or the unzipped folder.")
        return 1

    loader = MoonshotDataLoader(dataset_root)
    samples_all = loader.get_samples(split=args.split, max_samples=None)
    n_samples = len(samples_all)

    if args.export_top5:
        rows = export_top5_predictions(
            dataset_root=dataset_root,
            out_path=args.export_path,
            split=args.split,
            max_samples=args.max_samples if args.max_samples > 0 else None,
            k=args.k,
            similarity_metric=args.similarity,
            seed=args.seed,
        )
        print(f"Exported {rows} rows to {args.export_path}")
        return 0

    if args.single_random:
        # One random molecule per run: test on it and report hit/miss
        import random
        if not samples_all:
            print("No samples in split.")
            return 1
        s = random.choice(samples_all)
        print(f"Single-random mode: one molecule from {dataset_root} (split={args.split})")
        print(f"Query (ground truth): {s.smiles}")
        retrieval_fps = precompute_fingerprints(loader.get_retrieval_smiles())
        topk = run_original_flow(
            loader, s.smiles, k=args.k, retrieval_fps=retrieval_fps,
            similarity_metric=args.similarity,
        )
        preds = [x[0] for x in topk]
        top1_hit = preds and preds[0] == s.smiles
        top5_hit = s.smiles in preds
        print(f"Top-1 hit: {top1_hit}  (predicted: {preds[0] if preds else 'none'})")
        print(f"Top-5 hit: {top5_hit}")
        if args.plot_example:
            os.makedirs(args.out_dir, exist_ok=True)
            mz = np.linspace(100, 500, 50)
            spec = SpectrumPlotData(
                mz=mz, intensity=np.exp(-0.5 * ((mz - 250.0) / 20.0) ** 2) * 100.0,
                title=f"Query: {s.smiles[:40]}...",
            )
            plot_mass_spectrum(spec, out_path=os.path.join(args.out_dir, "spectrum_real.png"))
            cands, token_seqs, pooled = run_developed_flow(
                loader, s.smiles, k=args.k, retrieval_fps=retrieval_fps,
                similarity_metric=args.similarity,
            )
            plot_topk_smiles_grid(
                [c.smiles for c in cands], scores=[c.score for c in cands],
                out_path=os.path.join(args.out_dir, "topk_candidates_real.png"),
            )
            import torch
            plot_token_heatmap(
                token_seqs[0].to(dtype=torch.float32).cpu().numpy(),
                out_path=os.path.join(args.out_dir, "tokens_heatmap_real.png"),
            )
            pooled_arr = torch.stack(pooled, dim=0).to(dtype=torch.float32).cpu().numpy()
            plot_embedding_pca(
                pooled_arr, labels=[f"cand{i+1}" for i in range(len(cands))],
                out_path=os.path.join(args.out_dir, "embeddings_pca_real.png"),
            )
            plot_accuracy_comparison(
                float(top1_hit), float(top5_hit), float(top1_hit), float(top5_hit),
                os.path.join(args.out_dir, "accuracy_comparison.png"),
            )
            print(f"Plots saved to {args.out_dir}/")
        return 0

    n_eval = min(args.max_samples, n_samples) if n_samples else 0
    print(f"Dataset: {dataset_root}")
    print(f"Split '{args.split}': up to {n_eval} samples (total: {n_samples}), similarity={args.similarity}")
    print()

    orig_t1, orig_t5, dev_t1, dev_t5 = evaluate_flows(
        dataset_root, split=args.split, max_samples=args.max_samples, k=args.k,
        similarity_metric=args.similarity,
    )

    # Save visual accuracy comparison (Original vs Developed)
    os.makedirs(args.out_dir, exist_ok=True)
    accuracy_plot_path = os.path.join(args.out_dir, "accuracy_comparison.png")
    plot_accuracy_comparison(orig_t1, orig_t5, dev_t1, dev_t5, accuracy_plot_path)
    print(f"Accuracy comparison chart saved: {accuracy_plot_path}")
    print()
    print("--- Accuracy (percentage) ---")
    print("Original flow (retrieval only):")
    print(f"  Top-1 accuracy: {orig_t1 * 100:.2f}%")
    print(f"  Top-5 accuracy: {orig_t5 * 100:.2f}%")
    print("Developed flow (retrieval + graph tokenization):")
    print(f"  Top-1 accuracy: {dev_t1 * 100:.2f}%")
    print(f"  Top-5 accuracy: {dev_t5 * 100:.2f}%")
    print()
    print("(DiffMS is not run in this script. Original/Developed compare retrieval + graph pipeline; DiffMS integration is a next step.)")

    if args.plot_example:
        samples = loader.get_samples(split=args.split, max_samples=1)
        if not samples:
            print("No sample available for plotting.")
            return 0
        os.makedirs(args.out_dir, exist_ok=True)
        s = samples[0]
        # Dummy spectrum (dataset may not expose raw spectra in index)
        mz = np.linspace(100, 500, 50)
        spec = SpectrumPlotData(
            mz=mz,
            intensity=np.exp(-0.5 * ((mz - 250.0) / 20.0) ** 2) * 100.0,
            title=f"Example spectrum (query: {s.smiles[:30]}...)",
        )
        plot_mass_spectrum(spec, out_path=os.path.join(args.out_dir, "spectrum_real.png"))

        cands, token_seqs, pooled = run_developed_flow(
            loader, s.smiles, k=args.k, similarity_metric=args.similarity,
        )
        plot_topk_smiles_grid(
            [c.smiles for c in cands],
            scores=[c.score for c in cands],
            out_path=os.path.join(args.out_dir, "topk_candidates_real.png"),
        )
        import torch
        plot_token_heatmap(
            token_seqs[0].to(dtype=torch.float32).cpu().numpy(),
            out_path=os.path.join(args.out_dir, "tokens_heatmap_real.png"),
        )
        pooled_arr = torch.stack(pooled, dim=0).to(dtype=torch.float32).cpu().numpy()
        plot_embedding_pca(
            pooled_arr,
            labels=[f"cand{i+1}" for i in range(len(cands))],
            out_path=os.path.join(args.out_dir, "embeddings_pca_real.png"),
        )
        print(f"Plots saved to {args.out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
