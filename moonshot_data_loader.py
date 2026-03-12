"""
Load MoonshotDatasetv3 from disk or from MoonshotDatasetv3.zip (index.pkl,
retrieval.pkl) and expose samples + retrieval library. No SMART-Moonshot imports.
"""
from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np


# Default: .zip file (user said the file is MoonshotDatasetv3.zip)
MOONSHOT_DATASET_ROOT = Path("C:/Users/andre/Downloads/MoonshotDatasetv3.zip")


@dataclass
class MoonshotSample:
    """One entry from the index: ground-truth SMILES and metadata."""
    idx: int
    smiles: str
    split: str  # 'train', 'val', 'test'
    raw: Dict[str, Any]  # full entry for debugging


class MoonshotDataLoader:
    """
    Load index.pkl and retrieval.pkl from MoonshotDatasetv3 folder or from
    MoonshotDatasetv3.zip. Exposes samples and retrieval library for top-k.
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"MoonshotDatasetv3 not found: {self.root}. "
                "Use path to MoonshotDatasetv3.zip or the unzipped folder with index.pkl."
            )
        self._is_zip = self.root.suffix.lower() == ".zip"
        self._index: Optional[Dict[str, Any]] = None
        self._retrieval_smiles: Optional[List[str]] = None

    def _open_path(self, subpath: str) -> BinaryIO:
        """Open a file: from zip (root or MoonshotDatasetv3/) or from folder. Caller must close if not using 'with'."""
        if self._is_zip:
            with zipfile.ZipFile(self.root, "r") as zf:
                for prefix in ("", "MoonshotDatasetv3/", "MoonshotDatasetv3\\"):
                    name = f"{prefix}{subpath}".replace("\\", "/")
                    if name in zf.namelist():
                        return BytesIO(zf.read(name))
            raise FileNotFoundError(f"{subpath} not found inside {self.root}")
        p = self.root / subpath
        if not p.exists():
            raise FileNotFoundError(str(p))
        return open(p, "rb")

    def _load_index(self) -> Dict[str, Any]:
        if self._index is None:
            with self._open_path("index.pkl") as f:
                self._index = pickle.load(f)
        return self._index

    def get_index(self) -> Dict[str, Any]:
        """Raw index: keys are idx (int or str), values are dicts (split, smiles/ SMILES, etc.)."""
        return self._load_index()

    def get_samples(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[MoonshotSample]:
        """
        Return list of MoonshotSample (idx, smiles, split).
        Optional filter by split ('train'/'val'/'test') and cap count.
        """
        idx_map = self._load_index()
        samples: List[MoonshotSample] = []
        for k, v in idx_map.items():
            if not isinstance(v, dict):
                continue
            s = v.get("split", "train")
            if split is not None and s != split:
                continue
            smiles = v.get("smiles") or v.get("SMILES")
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            idx = int(k) if isinstance(k, (int, float)) else int(k) if str(k).isdigit() else len(samples)
            samples.append(MoonshotSample(idx=idx, smiles=smiles.strip(), split=s, raw=v))
            if max_samples is not None and len(samples) >= max_samples:
                break
        return samples

    def get_retrieval_smiles(self) -> List[str]:
        """
        Return list of SMILES from retrieval.pkl (library for top-k search).
        If retrieval.pkl is missing or wrong format, falls back to unique SMILES from index.
        """
        if self._retrieval_smiles is not None:
            return self._retrieval_smiles
        try:
            with self._open_path("retrieval.pkl") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                self._retrieval_smiles = [s for s in obj if isinstance(s, str) and s.strip()]
            elif isinstance(obj, dict) and "smiles" in obj:
                self._retrieval_smiles = [s for s in obj["smiles"] if isinstance(s, str) and s.strip()]
            else:
                self._retrieval_smiles = [v for v in obj.values() if isinstance(v, str) and v.strip()]
        except Exception:
            self._retrieval_smiles = []
        if not self._retrieval_smiles:
            # Fallback: use all unique SMILES from index as retrieval set
            self._retrieval_smiles = []
            seen: set[str] = set()
            for s in self.get_samples(split=None, max_samples=None):
                if s.smiles not in seen:
                    seen.add(s.smiles)
                    self._retrieval_smiles.append(s.smiles)
        return self._retrieval_smiles or []

    def get_sample_by_idx(self, idx: int) -> Optional[MoonshotSample]:
        """Return one sample by index key, or None if missing/invalid."""
        idx_map = self._load_index()
        key = idx
        if key not in idx_map and str(key) in idx_map:
            key = str(key)
        if key not in idx_map:
            return None
        v = idx_map[key]
        if not isinstance(v, dict):
            return None
        smiles = v.get("smiles") or v.get("SMILES")
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        return MoonshotSample(
            idx=int(key) if isinstance(key, (int, float)) else idx,
            smiles=smiles.strip(),
            split=v.get("split", "train"),
            raw=v,
        )


def precompute_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> List[Tuple[str, Any]]:
    """
    Precompute Morgan fingerprints for a list of SMILES. Returns list of
    (smiles, fp) for valid molecules only. Use this once for the retrieval
    list and pass to topk_by_tanimoto as candidate_fps to avoid recomputing.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out: List[Tuple[str, Any]] = []
    for s in smiles_list:
        if not s or not isinstance(s, str):
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        out.append((s, mfpgen.GetFingerprint(mol)))
    return out


def topk_by_similarity(
    query_smiles: str,
    candidate_smiles: List[str],
    k: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
    candidate_fps: Optional[List[Tuple[str, Any]]] = None,
    similarity_metric: str = "cosine",
) -> List[Tuple[str, float]]:
    """
    Rank candidates by fingerprint similarity to query (Morgan FP).
    similarity_metric: "cosine" (match SMART-Moonshot style) or "tanimoto".
    Returns list of (smiles, score) sorted descending by score, length <= k.

    If candidate_fps is provided (list of (smiles, fp)), reuse those for speed.
    """
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator

    if similarity_metric.lower() == "cosine":
        sim_fn = DataStructs.CosineSimilarity
    else:
        sim_fn = DataStructs.TanimotoSimilarity

    qmol = Chem.MolFromSmiles(query_smiles)
    if qmol is None:
        return []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    qfp = mfpgen.GetFingerprint(qmol)

    scored: List[Tuple[str, float]] = []
    if candidate_fps is not None:
        for s, fp in candidate_fps:
            if s == query_smiles:
                continue
            sim = float(sim_fn(qfp, fp))
            scored.append((s, sim))
    else:
        for s in candidate_smiles:
            if not s or s == query_smiles:
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            fp = mfpgen.GetFingerprint(mol)
            sim = float(sim_fn(qfp, fp))
            scored.append((s, sim))

    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def topk_by_tanimoto(
    query_smiles: str,
    candidate_smiles: List[str],
    k: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
    candidate_fps: Optional[List[Tuple[str, Any]]] = None,
) -> List[Tuple[str, float]]:
    """Convenience wrapper: topk_by_similarity(..., similarity_metric="tanimoto")."""
    return topk_by_similarity(
        query_smiles, candidate_smiles, k=k, radius=radius, n_bits=n_bits,
        candidate_fps=candidate_fps, similarity_metric="tanimoto",
    )
