"""
Optional wrapper to use the real HIGHT graph encoder (LFhase/HIGHT) in this pipeline.
If HIGHT is not installed or checkpoint is missing, the pipeline falls back to
DummyGraphEncoder.

To use:
  1. Clone HIGHT:  git clone https://github.com/LFhase/HIGHT.git
  2. Install:     cd HIGHT && pip install -e . && pip install -r requirements.txt
  3. Download checkpoint from https://huggingface.co/lfhase/HIGHT
  4. Set HIGHT_ROOT and optionally HIGHT_CHECKPOINT, or pass checkpoint_path to get_hight_encoder().
  5. In pipeline_skeleton or run_pipeline_with_dataset, use encode_candidates_to_tokens(..., encoder="hight")
     or call get_hight_encoder() and use it instead of DummyGraphEncoder.

HIGHT expects its own graph format (MolGraph from llava.datasets.data_utils with motif/super node).
This module uses HIGHT's MolGraph(smiles) so no format conversion is needed.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

_HIGHT_ENCODER: Optional[Callable[..., Any]] = None
_HIGHT_AVAILABLE: Optional[bool] = None


def is_hight_available() -> bool:
    """True if HIGHT can be imported and encoder is usable."""
    global _HIGHT_AVAILABLE
    if _HIGHT_AVAILABLE is not None:
        return _HIGHT_AVAILABLE
    try:
        from llava.model.llava_graph_arch import LlavaMetaForCausalLM  # type: ignore
        from llava.datasets.data_utils import MolGraph  # type: ignore
        _HIGHT_AVAILABLE = True
    except Exception:
        _HIGHT_AVAILABLE = False
    return _HIGHT_AVAILABLE


def get_hight_encoder(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> Optional[Callable[[Any], Any]]:
    """
    Return a callable that takes a HIGHT MolGraph and returns graph token embeddings,
    or None if HIGHT is not available or checkpoint cannot be loaded.

    The callable has signature: encode(mol) -> tensor [num_tokens, hidden_dim].
    Our pipeline would do: mol = MolGraph(smiles); feats = encode(mol).

    Requires: HIGHT repo installed, checkpoint path set (env HIGHT_CHECKPOINT or argument).
    Build the model using HIGHT's scripts (e.g. builder.build_graph_tower + load checkpoint),
    then pass the model's encode_mol here. This stub returns None until you wire that.
    """
    global _HIGHT_ENCODER
    if not is_hight_available():
        return None
    # You would load the model from checkpoint_path / HIGHT_CHECKPOINT using
    # HIGHT's own loading (see their eval or train scripts), then return
    # a lambda that calls model.encode_mol(mol, compute_type=...).
    # For now we return None so the pipeline keeps using DummyGraphEncoder.
    return None


def smiles_to_hight_mol(smiles: str):
    """Build HIGHT MolGraph from SMILES. Requires HIGHT to be installed."""
    if not is_hight_available():
        raise RuntimeError("HIGHT is not installed. Install from https://github.com/LFhase/HIGHT")
    from llava.datasets.data_utils import MolGraph
    return MolGraph(smiles)
