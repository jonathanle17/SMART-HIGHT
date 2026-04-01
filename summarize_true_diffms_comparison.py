from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# DiffMS logs K_ACC as test/acc_at_{k} (see src/metrics/diffms_metrics.py).
PREFERRED_KEYS = [
    "test/acc_at_1",
    "test/acc_at_5",
    "test/tanimoto_at_1",
    "test/tanimoto_at_5",
    "test/NLL",
    "test/E_CE",
    "test/validity",
]


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize true DiffMS baseline vs HIGHT runs.")
    parser.add_argument("--runs-json", type=str, default="artifacts/true_diffms_comparison.json")
    parser.add_argument("--out-json", type=str, default="artifacts/true_diffms_comparison_summary.json")
    args = parser.parse_args()

    runs = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in runs:
        if int(r.get("return_code", 1)) != 0:
            continue
        mode = str(r["mode"])
        metrics = r.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                grouped[mode][k].append(float(v))

    summary: dict[str, dict] = {}
    for mode, m in grouped.items():
        summary[mode] = {k: _mean_std(vs) for k, vs in m.items()}

    # baseline vs hight_augmented deltas for preferred keys
    deltas = {}
    base = summary.get("baseline", {})
    aug = summary.get("hight_augmented", {})
    for key in PREFERRED_KEYS:
        if key in base and key in aug:
            b = base[key]["mean"]
            a = aug[key]["mean"]
            deltas[key] = {
                "baseline_mean": b,
                "hight_augmented_mean": a,
                "absolute_delta": a - b,
                "relative_delta_percent": ((a - b) / abs(b) * 100.0) if b != 0 else float("nan"),
            }

    out = {"summary": summary, "baseline_vs_hight_augmented": deltas}
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
