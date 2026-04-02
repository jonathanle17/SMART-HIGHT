from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_bridge_artifacts(
    train_jsonl: str | Path,
    train_npz: str | Path,
    val_jsonl: str | Path,
    val_npz: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
) -> None:
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)

    train_tok = np.load(train_npz)
    val_tok = np.load(val_npz)
    train_tokens = train_tok["hight_tokens"]
    train_masks = train_tok["token_masks"]
    val_tokens = val_tok["hight_tokens"]
    val_masks = val_tok["token_masks"]

    max_t = max(train_tokens.shape[2], val_tokens.shape[2])
    d = train_tokens.shape[3]
    train_pad = np.zeros((train_tokens.shape[0], train_tokens.shape[1], max_t, d), dtype=train_tokens.dtype)
    val_pad = np.zeros((val_tokens.shape[0], val_tokens.shape[1], max_t, d), dtype=val_tokens.dtype)
    train_pad[:, :, : train_tokens.shape[2], :] = train_tokens
    val_pad[:, :, : val_tokens.shape[2], :] = val_tokens

    train_mask_pad = np.zeros((train_masks.shape[0], train_masks.shape[1], max_t), dtype=train_masks.dtype)
    val_mask_pad = np.zeros((val_masks.shape[0], val_masks.shape[1], max_t), dtype=val_masks.dtype)
    train_mask_pad[:, :, : train_masks.shape[2]] = train_masks
    val_mask_pad[:, :, : val_masks.shape[2]] = val_masks

    out_tokens = np.concatenate([train_pad, val_pad], axis=0)
    out_masks = np.concatenate([train_mask_pad, val_mask_pad], axis=0)

    offset = len(train_rows)
    for row in val_rows:
        row["token_tensor_index"] = int(row["token_tensor_index"]) + offset

    out_rows = train_rows + val_rows
    out_jsonl_path = Path(out_jsonl)
    out_npz_path = Path(out_npz)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    np.savez_compressed(out_npz_path, hight_tokens=out_tokens, token_masks=out_masks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge train+val bridge artifacts into one file pair.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out-jsonl", default="artifacts/diffms_bridge_all.jsonl")
    parser.add_argument("--out-npz", default="artifacts/diffms_bridge_all_tokens.npz")
    args = parser.parse_args()

    merge_bridge_artifacts(
        train_jsonl=args.train_jsonl,
        train_npz=args.train_npz,
        val_jsonl=args.val_jsonl,
        val_npz=args.val_npz,
        out_jsonl=args.out_jsonl,
        out_npz=args.out_npz,
    )
    print(f"Merged metadata: {args.out_jsonl}")
    print(f"Merged tokens:   {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
