from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class BridgeRow:
    split: str
    spectrum_features: np.ndarray
    top5_scores: np.ndarray
    token_index: int
    label_top5_hit: float


class BridgeDataset(Dataset):
    def __init__(self, meta_jsonl: str | Path, token_npz: str | Path, split: str) -> None:
        self.rows: list[BridgeRow] = []
        with Path(meta_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("split", "train") != split:
                    continue
                cands = obj.get("top5_candidates", [])
                scores = np.array([float(c.get("moonshot_score", 0.0)) for c in cands], dtype=np.float32)
                if scores.size < 5:
                    scores = np.pad(scores, (0, 5 - scores.size))
                gt = str(obj.get("gt_smiles", ""))
                label_top5_hit = 1.0 if any(str(c.get("pred_smiles", "")) == gt for c in cands) else 0.0
                self.rows.append(
                    BridgeRow(
                        split=split,
                        spectrum_features=np.array(obj.get("spectrum_features", []), dtype=np.float32),
                        top5_scores=scores[:5],
                        token_index=int(obj["token_tensor_index"]),
                        label_top5_hit=label_top5_hit,
                    )
                )
        tok = np.load(token_npz)
        self.hight_tokens = tok["hight_tokens"]  # [N, K, T, D]
        self.token_masks = tok["token_masks"]  # [N, K, T]

        # Fix feature dimension by right-padding to max size in split
        self.spec_dim = max((r.spectrum_features.shape[0] for r in self.rows), default=64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        spec = r.spectrum_features
        if spec.shape[0] < self.spec_dim:
            spec = np.pad(spec, (0, self.spec_dim - spec.shape[0]))
        spec = torch.tensor(spec, dtype=torch.float32)
        score = torch.tensor(r.top5_scores, dtype=torch.float32)
        tok = torch.tensor(self.hight_tokens[r.token_index], dtype=torch.float32)
        msk = torch.tensor(self.token_masks[r.token_index], dtype=torch.bool)
        y = torch.tensor(r.label_top5_hit, dtype=torch.float32)
        return spec, score, tok, msk, y


class BaselineHead(nn.Module):
    def __init__(self, spec_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec_dim + 5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, spec: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spec, scores], dim=-1)
        return self.net(x).squeeze(-1)


class HightAuxHead(nn.Module):
    def __init__(self, token_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(token_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: [B, K, T, D], mask: [B, K, T]
        m = mask.float().unsqueeze(-1)
        pooled = (tokens * m).sum(dim=2) / (m.sum(dim=2) + 1e-6)  # [B, K, D]
        pooled = pooled.mean(dim=1)  # [B, D]
        h = F.relu(self.proj(pooled))
        return self.out(h).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_mode(
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    aux_weight: float,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = next(iter(train_loader))
    spec_dim = sample[0].shape[-1]
    token_dim = sample[2].shape[-1]

    baseline = BaselineHead(spec_dim=spec_dim).to(device)
    aux = HightAuxHead(token_dim=token_dim).to(device)
    params = list(baseline.parameters()) + (list(aux.parameters()) if mode != "baseline" else [])
    opt = torch.optim.Adam(params, lr=lr)

    def evaluate(loader: DataLoader) -> tuple[float, float]:
        baseline.eval()
        aux.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for spec, scores, tok, msk, y in loader:
                spec, scores, tok, msk, y = spec.to(device), scores.to(device), tok.to(device), msk.to(device), y.to(device)
                logits_base = baseline(spec, scores)
                loss = F.binary_cross_entropy_with_logits(logits_base, y)
                logits = logits_base
                if mode != "baseline":
                    tok_in = tok
                    if mode == "shuffled_top5":
                        idx = torch.randperm(tok_in.shape[1], device=tok_in.device)
                        tok_in = tok_in[:, idx, :, :]
                    logits_aux = aux(tok_in, msk)
                    loss = loss + aux_weight * F.binary_cross_entropy_with_logits(logits_aux, y)
                    logits = logits + aux_weight * logits_aux
                losses.append(float(loss.item()))
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += int((preds == y).sum().item())
                total += int(y.shape[0])
        return float(np.mean(losses) if losses else 0.0), float(correct / max(total, 1))

    for _ in range(epochs):
        baseline.train()
        aux.train()
        for spec, scores, tok, msk, y in train_loader:
            spec, scores, tok, msk, y = spec.to(device), scores.to(device), tok.to(device), msk.to(device), y.to(device)
            logits_base = baseline(spec, scores)
            loss = F.binary_cross_entropy_with_logits(logits_base, y)
            if mode != "baseline":
                tok_in = tok
                if mode == "shuffled_top5":
                    idx = torch.randperm(tok_in.shape[1], device=tok_in.device)
                    tok_in = tok_in[:, idx, :, :]
                logits_aux = aux(tok_in, msk)
                loss = loss + aux_weight * F.binary_cross_entropy_with_logits(logits_aux, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    val_loss, val_acc = evaluate(val_loader)
    return {"val_loss": val_loss, "val_acc": val_acc}


def run_experiments(
    meta_jsonl: str | Path,
    token_npz: str | Path,
    out_json: str | Path,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    aux_weight: float,
) -> dict:
    train_ds = BridgeDataset(meta_jsonl, token_npz, split="train")
    val_ds = BridgeDataset(meta_jsonl, token_npz, split="val")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Expected non-empty train/val splits in bridge dataset.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    modes = ["baseline", "multi_task_hight", "shuffled_top5"]
    all_results: dict[str, list[dict[str, float]]] = {m: [] for m in modes}
    for seed in seeds:
        for mode in modes:
            all_results[mode].append(
                train_one_mode(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    mode=mode,
                    aux_weight=aux_weight,
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                )
            )

    summary = {}
    for mode, rows in all_results.items():
        losses = [r["val_loss"] for r in rows]
        accs = [r["val_acc"] for r in rows]
        summary[mode] = {
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses)),
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs)),
            "per_seed": rows,
        }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline/multi-task/shuffled control on bridge dataset.")
    parser.add_argument("--meta-jsonl", type=str, required=True)
    parser.add_argument("--token-npz", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="artifacts/diffms_multitask_results.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    args = parser.parse_args()

    summary = run_experiments(
        meta_jsonl=args.meta_jsonl,
        token_npz=args.token_npz,
        out_json=args.out_json,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_weight=args.aux_weight,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
