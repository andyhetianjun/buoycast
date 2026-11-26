"""Does adding nearby airport weather reports help the wave forecast?

Usage: uv run buoycast-extra            # 5 seeds, writes artifacts/extra_data.json
       uv run buoycast-extra --quick    # 1 seed, 2 epochs, prints only

Both variants use exactly the same rows (hours where the buoy and its airport both
have data), the same split and the same settings. The only difference is whether the
five airport inputs are given to the model.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .features import load_hourly, split_by_time
from .models import LinearModel, persistence, train_mlp
from .nnja import EXTRA_NAMES, NEAR_AIRPORT, build_with_extra, load_extra
from .paths import ARTIFACTS
from .train import mae, rmse

HORIZONS = (3, 6)
N_BOOT = 1000


def paired_mae_difference(
    test: pd.DataFrame, with_extra: np.ndarray, without: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Mean change in absolute error (with extra minus without), and a 95% range from days."""
    true = test["target"].to_numpy()
    diff = np.abs(with_extra - true) - np.abs(without - true)
    codes, _ = pd.factorize(test.index.floor("1D"))
    k = codes.max() + 1
    per_day = np.bincount(codes, weights=diff, minlength=k)
    count = np.bincount(codes, minlength=k)
    w = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=N_BOOT)
    boot = (w @ per_day) / (w @ count)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run(seeds: int = 5, epochs: int = 40) -> dict:
    hourly, extra = load_hourly(), load_extra()
    out: dict = {
        "airports": NEAR_AIRPORT,
        "extra_inputs": list(EXTRA_NAMES),
        "seeds": seeds,
        "wind_correlation": {
            s: float(hourly[s]["wspd"].corr(extra[s]["wspd"], min_periods=1000)) for s in extra
        },
        "horizons": {},
    }
    for h in HORIZONS:
        data = build_with_extra(hourly, extra, h)
        train, val, test = split_by_time(data, h)
        entry: dict = {
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
            "stations": sorted(test["station"].unique()),
            "persistence": {"mae": mae(persistence(test), test["target"].to_numpy())},
        }
        true = test["target"].to_numpy()
        entry["persistence"]["rmse"] = rmse(persistence(test), true)
        preds: dict[str, list[np.ndarray]] = {"buoy_only": [], "buoy_plus_airport": []}
        for name, cols in (("buoy_only", ()), ("buoy_plus_airport", tuple(EXTRA_NAMES))):
            lin = LinearModel(cols).fit(train)
            entry[f"{name}_linear"] = {
                "mae": mae(lin.predict(test), true),
                "rmse": rmse(lin.predict(test), true),
            }
            for seed in range(seeds):
                fc = train_mlp(train, val, seed=seed, epochs=epochs, extra=cols)
                preds[name].append(fc.predict(test))
            runs = [(mae(p, true), rmse(p, true)) for p in preds[name]]
            entry[f"{name}_mlp"] = {
                "mae": {
                    "mean": float(np.mean([r[0] for r in runs])),
                    "std": float(np.std([r[0] for r in runs])),
                },
                "rmse": {
                    "mean": float(np.mean([r[1] for r in runs])),
                    "std": float(np.std([r[1] for r in runs])),
                },
            }
        base = np.mean(preds["buoy_only"], axis=0)
        plus = np.mean(preds["buoy_plus_airport"], axis=0)
        change, lo, hi = paired_mae_difference(test, plus, base)
        entry["mae_change_with_airport"] = {"mean": change, "range_95": [lo, hi]}
        entry["per_station_mae"] = {
            s: {
                "buoy_only": mae(base[m], true[m]),
                "buoy_plus_airport": mae(plus[m], true[m]),
                "persistence": mae(persistence(test)[m], true[m]),
                "n": int(m.sum()),
            }
            for s in entry["stations"]
            for m in [(test["station"] == s).to_numpy()]
        }
        out["horizons"][str(h)] = entry
    return out


def format_table(result: dict) -> str:
    lines = [
        "| Horizon | Model | Buoy only MAE (m) | Buoy + airport MAE (m) |",
        "|---|---|---|---|",
    ]
    for h, e in result["horizons"].items():
        lines.append(f"| {h} h | Persistence | {e['persistence']['mae']:.3f} | same |")
        lines.append(
            f"| {h} h | Linear regression | {e['buoy_only_linear']['mae']:.3f} | "
            f"{e['buoy_plus_airport_linear']['mae']:.3f} |"
        )
        a, b = e["buoy_only_mlp"]["mae"], e["buoy_plus_airport_mlp"]["mae"]
        lines.append(
            f"| {h} h | MLP | {a['mean']:.3f} +/- {a['std']:.3f} | "
            f"{b['mean']:.3f} +/- {b['std']:.3f} |"
        )
        c = e["mae_change_with_airport"]
        lines.append(
            f"\n{h} h: change in MLP MAE with airport data {c['mean']:+.4f} m "
            f"(95% range {c['range_95'][0]:+.4f} to {c['range_95'][1]:+.4f}), "
            f"{e['rows']['test']} test rows"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.quick:
        args.seeds, args.epochs = 1, 2
    result = run(args.seeds, args.epochs)
    print(format_table(result))
    if not args.quick:
        path = ARTIFACTS / "extra_data.json"
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
