from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from moonshot_to_hight_adapter import MoonshotHightAdapter, pad_candidate_tokens


def _safe_spectrum_features(row: dict[str, Any]) -> list[float]:
    """
    Prefer existing spectrum features if present; otherwise produce a stable fallback.
    """
    for key in ("spectrum_features", "spec", "spectrum", "features"):
        val = row.get(key)
        if isinstance(val, (list, tuple)) and val:
            return [float(x) for x in val]

    gt = str(row.get("gt_smiles", ""))
    # Deterministic fallback: simple char-hash histogram.
    hist = np.zeros(64, dtype=np.float32)
    for ch in gt:
        hist[ord(ch) % 64] += 1.0
    norm = float(np.linalg.norm(hist) + 1e-8)
    return (hist / norm).tolist()


def build_training_set(
    moonshot_top5_jsonl: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    hidden_dim: int = 128,
    use_lap_pe: bool = True,
    hight_checkpoint: str | None = None,
) -> int:
    adapter = MoonshotHightAdapter(
        hidden_dim=hidden_dim,
        use_lap_pe=use_lap_pe,
        hight_checkpoint=hight_checkpoint,
        dtype=torch.float32,
    )

    in_path = Path(moonshot_top5_jsonl)
    out_jsonl_path = Path(out_jsonl)
    out_npz_path = Path(out_npz)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    tokens_all = []
    masks_all = []
    rows_out = []

    with in_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            top5 = row.get("top5_candidates", [])
            encoded = adapter.encode_topk(top5)
            token_tensor, mask_tensor = pad_candidate_tokens(encoded, max_candidates=5)
            tokens_all.append(token_tensor.numpy())
            masks_all.append(mask_tensor.numpy())

            rows_out.append(
                {
                    "sample_id": int(row.get("sample_id", i)),
                    "split": row.get("split", "train"),
                    "gt_smiles": row.get("gt_smiles", ""),
                    "spectrum_features": _safe_spectrum_features(row),
                    "top5_candidates": [
                        {
                            "rank": item.rank,
                            "pred_smiles": item.pred_smiles,
                            "moonshot_score": item.moonshot_score,
                            "candidate_graph_meta": {
                                "num_nodes": item.num_nodes,
                                "num_edges": item.num_edges,
                            },
                        }
                        for item in encoded
                    ],
                    "token_tensor_index": len(tokens_all) - 1,
                }
            )

    with out_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    # Pad across samples to a global max token length so NPZ has fixed-size arrays.
    max_t = max(arr.shape[1] for arr in tokens_all)
    k = tokens_all[0].shape[0]
    d = tokens_all[0].shape[2]
    n = len(tokens_all)
    tokens_padded = np.zeros((n, k, max_t, d), dtype=tokens_all[0].dtype)
    masks_padded = np.zeros((n, k, max_t), dtype=masks_all[0].dtype)
    for i, (tok, msk) in enumerate(zip(tokens_all, masks_all)):
        t_len = tok.shape[1]
        tokens_padded[i, :, :t_len, :] = tok
        masks_padded[i, :, :t_len] = msk

    np.savez_compressed(
        out_npz_path,
        hight_tokens=tokens_padded,  # [N, K, T, D]
        token_masks=masks_padded,  # [N, K, T]
    )
    return len(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DiffMS-ready dataset from Moonshot top-5 + HIGHT tokens.")
    parser.add_argument("--moonshot-top5-jsonl", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, default="artifacts/diffms_bridge.jsonl")
    parser.add_argument("--out-npz", type=str, default="artifacts/diffms_bridge_tokens.npz")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--no-lap-pe", action="store_true")
    parser.add_argument("--hight-checkpoint", type=str, default=None)
    args = parser.parse_args()

    rows = build_training_set(
        moonshot_top5_jsonl=args.moonshot_top5_jsonl,
        out_jsonl=args.out_jsonl,
        out_npz=args.out_npz,
        hidden_dim=args.hidden_dim,
        use_lap_pe=not args.no_lap_pe,
        hight_checkpoint=args.hight_checkpoint,
    )
    print(f"Built bridge dataset with {rows} rows")
    print(f"Metadata: {args.out_jsonl}")
    print(f"Tokens:   {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
