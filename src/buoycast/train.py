"""Train and evaluate: persistence vs linear regression vs a small MLP.

Usage:
    uv run buoycast-train            # full run: 5 seeds, writes artifacts/
    uv run buoycast-train --quick    # tiny run for CI, writes to a temp dir
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import TEST_START, VAL_START, build_all, load_hourly, split_by_time
from .models import LinearModel, persistence, save_bundle, train_mlp
from .paths import ARTIFACTS, SAMPLE_PARQUET

HORIZONS = (3, 6)


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).mean()))


def scores(pred: np.ndarray, test: pd.DataFrame) -> dict[str, float]:
    true = test["target"].to_numpy()
    return {"mae": mae(pred, true), "rmse": rmse(pred, true)}


def per_station_mae(pred: np.ndarray, test: pd.DataFrame) -> dict[str, float]:
    err = pd.Series(np.abs(pred - test["target"].to_numpy()), index=test.index)
    return {s: float(v) for s, v in err.groupby(test["station"].to_numpy()).mean().items()}


def run(data: Path, out_dir: Path, seeds: int, epochs: int) -> dict:
    hourly = load_hourly(data)
    result: dict = {
        "split": {"val_start": str(VAL_START), "test_start": str(TEST_START)},
        "seeds": seeds,
        "horizons": {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        train, val, test = split_by_time(build_all(hourly, h), h)
        entry: dict = {"rows": {"train": len(train), "val": len(val), "test": len(test)}}
        entry["test_stations"] = sorted(test["station"].unique())
        entry["persistence"] = scores(persistence(test), test)

        linear = LinearModel().fit(train)
        entry["linear"] = scores(linear.predict(test), test)

        runs, station_runs, first = [], [], None
        for seed in range(seeds):
            fc = train_mlp(train, val, seed=seed, epochs=epochs)
            pred = fc.predict(test)
            runs.append(scores(pred, test))
            station_runs.append(per_station_mae(pred, test))
            first = first or fc
        entry["mlp"] = {
            k: {
                "mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs])),
            }
            for k in ("mae", "rmse")
        }
        entry["per_station_mae"] = {
            "persistence": per_station_mae(persistence(test), test),
            "linear": per_station_mae(linear.predict(test), test),
            "mlp": {
                s: float(np.mean([r[s] for r in station_runs])) for s in station_runs[0]
            },
        }
        result["horizons"][str(h)] = entry
        assert first is not None
        save_bundle(out_dir / f"model_h{h}.pt", first, linear, h)  # seed 0 is served
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def format_table(result: dict) -> str:
    lines = [
        "| Horizon | Model | MAE (m) | RMSE (m) |",
        "|---|---|---|---|",
    ]
    for h, e in result["horizons"].items():
        lines.append(f"| {h} h | Persistence | {e['persistence']['mae']:.3f} | "
                     f"{e['persistence']['rmse']:.3f} |")
        lines.append(f"| {h} h | Linear regression | {e['linear']['mae']:.3f} | "
                     f"{e['linear']['rmse']:.3f} |")
        m = e["mlp"]
        lines.append(
            f"| {h} h | MLP (mean of {result['seeds']} seeds) | "
            f"{m['mae']['mean']:.3f} +/- {m['mae']['std']:.3f} | "
            f"{m['rmse']['mean']:.3f} +/- {m['rmse']['std']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=SAMPLE_PARQUET)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, temp output dir")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out_dir = ARTIFACTS
    if args.quick:
        args.seeds, args.epochs = 1, 2
        out_dir = Path(tempfile.mkdtemp(prefix="buoycast-"))
    start = time.time()
    result = run(args.data, out_dir, args.seeds, args.epochs)
    print(format_table(result))
    print(f"\nwrote {out_dir} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
