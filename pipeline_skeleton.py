from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from debug_smiles_to_graph_tokens import MolData, smiles2graph, DummyGraphEncoder, graph_dict_to_mol
from visualization import (
    SpectrumPlotData,
    plot_embedding_pca,
    plot_mass_spectrum,
    plot_token_heatmap,
    plot_topk_smiles_grid,
)


@dataclass
class CandidateMolecule:
    smiles: str
    score: float


@dataclass
class PipelineOutputs:
    spectrum_plot: Optional[str]
    topk_grid_plot: Optional[str]
    token_heatmap_plot: Optional[str]
    embedding_pca_plot: Optional[str]


def dummy_spectrum_example() -> SpectrumPlotData:
    """
    Placeholder that fabricates a toy spectrum.

    In the real pipeline, you will replace this function with logic that:
      - reads one MS spectrum from MoonshotDatasetv3, or
      - uses SMART‑Moonshot's internal loaders.
    """
    mz = np.linspace(100, 500, 50)
    inten = np.exp(-0.5 * ((mz - 250.0) / 20.0) ** 2) * 100.0
    return SpectrumPlotData(mz=mz, intensity=inten, title="Dummy spectrum (for debugging)")


def select_topk_candidates_from_moonshot(
    k: int = 5,
    query_smiles: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> List[CandidateMolecule]:
    """
    Top‑k candidates: from real dataset (if dataset_root + query_smiles) via
    Tanimoto retrieval, or hard‑coded SMILES for debugging.
    """
    if dataset_root and query_smiles:
        try:
            from moonshot_data_loader import MoonshotDataLoader, topk_by_similarity
            loader = MoonshotDataLoader(dataset_root)
            topk = topk_by_similarity(
                query_smiles, loader.get_retrieval_smiles(), k=k,
                similarity_metric="cosine",
            )
            return [CandidateMolecule(smiles=s, score=score) for s, score in topk]
        except Exception:
            pass
    example_smiles = [
        "C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "CCN(CC)CC",
    ]
    return [CandidateMolecule(smiles=s, score=float(k - i)) for i, s in enumerate(example_smiles[:k])]


def encode_candidates_to_tokens(
    candidates: Sequence[CandidateMolecule],
    hidden_dim: int = 512,
    use_lap_pe: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Use the existing SMILES -> graph -> DummyGraphEncoder pipeline to obtain
    per‑candidate token embeddings and global pooled embeddings.
    """
    encoder = DummyGraphEncoder(hidden_dim=hidden_dim, use_lap_pe=use_lap_pe, seed=0)

    token_seqs: List[torch.Tensor] = []
    pooled: List[torch.Tensor] = []

    for cand in candidates:
        graph_dict = smiles2graph(cand.smiles, motif=False, add_lap_pe=use_lap_pe)
        mol: MolData = graph_dict_to_mol(graph_dict)
        with torch.no_grad():
            feats = encoder(mol, compute_type=dtype)  # [T, H]
        token_seqs.append(feats)
        pooled.append(feats[-1])  # last token is global in DummyGraphEncoder

    return token_seqs, pooled


def run_debug_pipeline(
    output_dir: str = "debug_outputs",
    k: int = 5,
    dataset_root: Optional[str] = None,
    query_smiles: Optional[str] = None,
) -> PipelineOutputs:
    """
    Run the workflow: (1) spectrum plot, (2) top‑k candidates, (3) graph tokens,
    (4) visualizations. If dataset_root and query_smiles are set, uses real
    MoonshotDatasetv3 retrieval; otherwise uses dummy data.
    """
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) Spectrum (dummy if no dataset; real spectrum loading can be added later)
    spec = dummy_spectrum_example()
    spec_path = os.path.join(output_dir, "spectrum.png")
    plot_mass_spectrum(spec, out_path=spec_path)

    # 2) Top‑k candidates (real retrieval if dataset_root + query_smiles)
    cands = select_topk_candidates_from_moonshot(
        k=k, query_smiles=query_smiles, dataset_root=dataset_root
    )
    smiles = [c.smiles for c in cands]
    scores = [c.score for c in cands]
    grid_path = os.path.join(output_dir, "topk_candidates.png")
    plot_topk_smiles_grid(smiles, scores=scores, out_path=grid_path)

    # 3) Graph tokens
    token_seqs, pooled = encode_candidates_to_tokens(cands, hidden_dim=128)
    # Stack pooled embeddings into an array [K, H] and convert to float32 for NumPy
    pooled_arr = (
        torch.stack(pooled, dim=0)
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )

    # Heatmap for the first candidate's token embeddings (cast to float32 for NumPy)
    heatmap_path = os.path.join(output_dir, "tokens_heatmap.png")
    from visualization import plot_token_heatmap as _plot_token_heatmap

    _plot_token_heatmap(
        token_seqs[0].to(dtype=torch.float32).cpu().numpy(),
        out_path=heatmap_path,
    )

    # 4) PCA of pooled embeddings
    pca_path = os.path.join(output_dir, "embeddings_pca.png")
    labels = [f"cand{i+1}" for i in range(len(cands))]
    plot_embedding_pca(pooled_arr, labels=labels, out_path=pca_path)

    return PipelineOutputs(
        spectrum_plot=spec_path,
        topk_grid_plot=grid_path,
        token_heatmap_plot=heatmap_path,
        embedding_pca_plot=pca_path,
    )


if __name__ == "__main__":
    outputs = run_debug_pipeline()
    print("Debug pipeline completed. Generated plots:")
    print(f"  spectrum:      {outputs.spectrum_plot}")
    print(f"  top‑k grid:    {outputs.topk_grid_plot}")
    print(f"  token heatmap: {outputs.token_heatmap_plot}")
    print(f"  PCA:           {outputs.embedding_pca_plot}")

