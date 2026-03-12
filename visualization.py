from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA


@dataclass
class SpectrumPlotData:
    """Lightweight container for an MS spectrum to plot."""

    mz: np.ndarray  # shape [N]
    intensity: np.ndarray  # shape [N]
    title: str = "Input spectrum"


def plot_mass_spectrum(
    spec: SpectrumPlotData,
    out_path: Optional[str] = None,
) -> None:
    """Plot m/z vs intensity as a simple stick spectrum."""
    mz = np.asarray(spec.mz, dtype=float)
    inten = np.asarray(spec.intensity, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.vlines(mz, 0.0, inten, color="black", linewidth=1.0)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(spec.title)
    ax.set_xlim(mz.min() - 5, mz.max() + 5)

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_topk_smiles_grid(
    smiles: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    out_path: Optional[str] = None,
    n_cols: int = 5,
    mols_per_row: int = 5,
) -> None:
    """Draw a grid of the top‑k candidate molecules."""
    mols: List[Chem.Mol] = []
    legends: List[str] = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        if scores is not None and i < len(scores):
            legends.append(f"#{i+1}  score={scores[i]:.3f}")
        else:
            legends.append(f"#{i+1}")

    if not mols:
        raise ValueError("No valid SMILES to plot.")

    # RDKit grid image
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(n_cols, mols_per_row),
        subImgSize=(250, 250),
        legends=legends,
        useSVG=False,
    )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    else:
        # Show via matplotlib for interactive use
        fig, ax = plt.subplots(figsize=(3 * min(len(mols), n_cols), 3))
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    title: str = "Graph/token embeddings (PCA projection)",
) -> None:
    """
    Project high‑dimensional embeddings to 2D with PCA and scatter plot them.

    embeddings: array [N, D]
    labels: optional text labels for each point (e.g. 'gt', 'cand1', ...)
    """
    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Need at least 2 embeddings with shape [N, D] to run PCA.")

    pca = PCA(n_components=2)
    xy = pca.fit_transform(arr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xy[:, 0], xy[:, 1], c="tab:blue", s=40)

    if labels is not None and len(labels) == arr.shape[0]:
        for (x, y), lab in zip(xy, labels):
            ax.text(x, y, lab, fontsize=8, ha="center", va="center")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_token_heatmap(
    token_embeddings: np.ndarray,
    out_path: Optional[str] = None,
    title: str = "Token embeddings (heatmap, first dims)",
    max_tokens: int = 64,
    max_dims: int = 64,
) -> None:
    """
    Visualize token embeddings as a small heatmap for qualitative inspection.

    token_embeddings: array [T, D]
    """
    arr = np.asarray(token_embeddings, dtype=float)
    t = min(max_tokens, arr.shape[0])
    d = min(max_dims, arr.shape[1])
    sub = arr[:t, :d]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(sub, aspect="auto", cmap="viridis")
    ax.set_xlabel("Embedding dim (truncated)")
    ax.set_ylabel("Token index (truncated)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

