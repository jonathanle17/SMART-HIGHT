from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from debug_smiles_to_graph_tokens import DummyGraphEncoder, graph_dict_to_mol, smiles2graph
from hight_encoder_wrapper import get_hight_encoder, is_hight_available, smiles_to_hight_mol


@dataclass
class CandidateTokenOutput:
    pred_smiles: str
    moonshot_score: float
    rank: int
    hight_tokens: torch.Tensor  # [T, D]
    token_mask: torch.Tensor  # [T]
    num_nodes: int
    num_edges: int


class MoonshotHightAdapter:
    """
    Adapter that converts candidate SMILES into HIGHT-like token sequences.
    If real HIGHT encoder is unavailable, falls back to DummyGraphEncoder.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        use_lap_pe: bool = True,
        dtype: torch.dtype = torch.float32,
        hight_checkpoint: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.use_lap_pe = use_lap_pe
        self.dtype = dtype
        self.device = device
        self._real_encoder = get_hight_encoder(checkpoint_path=hight_checkpoint, device=device)
        self._dummy_encoder = DummyGraphEncoder(hidden_dim=hidden_dim, use_lap_pe=use_lap_pe, seed=0)

    @property
    def using_real_hight(self) -> bool:
        return self._real_encoder is not None and is_hight_available()

    def _encode_smiles(self, smiles: str) -> tuple[torch.Tensor, int, int]:
        if self.using_real_hight:
            mol = smiles_to_hight_mol(smiles)
            with torch.no_grad():
                tokens = self._real_encoder(mol)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.as_tensor(tokens)
            tokens = tokens.to(dtype=self.dtype, device="cpu")
            num_nodes = int(getattr(mol, "num_nodes", tokens.shape[0]))
            edge_index = getattr(mol, "edge_index", None)
            num_edges = int(edge_index.shape[1]) if edge_index is not None else 0
            return tokens, num_nodes, num_edges

        graph_dict = smiles2graph(smiles, motif=False, add_lap_pe=self.use_lap_pe)
        mol = graph_dict_to_mol(graph_dict)
        with torch.no_grad():
            tokens = self._dummy_encoder(mol, compute_type=self.dtype)
        tokens = tokens.to(dtype=self.dtype, device="cpu")
        num_nodes = int(graph_dict.get("num_nodes", 0))
        edge_index = graph_dict.get("edge_index")
        num_edges = int(edge_index.shape[1]) if edge_index is not None else 0
        return tokens, num_nodes, num_edges

    def encode_topk(self, candidates: Sequence[dict]) -> List[CandidateTokenOutput]:
        out: List[CandidateTokenOutput] = []
        for idx, cand in enumerate(candidates):
            smiles = str(cand["pred_smiles"])
            score = float(cand.get("moonshot_score", 0.0))
            rank = int(cand.get("rank", idx + 1))
            tokens, num_nodes, num_edges = self._encode_smiles(smiles)
            mask = torch.ones(tokens.shape[0], dtype=torch.bool)
            out.append(
                CandidateTokenOutput(
                    pred_smiles=smiles,
                    moonshot_score=score,
                    rank=rank,
                    hight_tokens=tokens,
                    token_mask=mask,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            )
        return out


def pad_candidate_tokens(
    encoded: Sequence[CandidateTokenOutput],
    max_candidates: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert variable-length candidate tokens to fixed tensors.
    Returns:
      token_tensor: [K, T, D]
      mask_tensor:  [K, T]
    """
    if not encoded:
        return torch.zeros((max_candidates, 1, 1)), torch.zeros((max_candidates, 1), dtype=torch.bool)

    dim = encoded[0].hight_tokens.shape[1]
    max_t = max(x.hight_tokens.shape[0] for x in encoded)
    token_tensor = torch.zeros((max_candidates, max_t, dim), dtype=encoded[0].hight_tokens.dtype)
    mask_tensor = torch.zeros((max_candidates, max_t), dtype=torch.bool)

    for i in range(min(max_candidates, len(encoded))):
        t = encoded[i].hight_tokens
        m = encoded[i].token_mask
        token_tensor[i, : t.shape[0], :] = t
        mask_tensor[i, : m.shape[0]] = m
    return token_tensor, mask_tensor
