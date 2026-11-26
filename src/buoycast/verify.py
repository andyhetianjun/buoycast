"""Score the forecasts properly: by lead time, by wave-height bin and by station.

Usage: uv run buoycast-verify

For every forecast in the test period (July to December 2024) this computes the
mean absolute error (MAE), root mean squared error (RMSE) and bias of each method,
and a skill score against persistence ("the waves stay as they are now"):

    skill = 1 - (error of the model) / (error of persistence)

Skill above 0 means better than persistence, 0 means no better, below 0 means worse.
Each skill number comes with a 95% range from resampling whole days (so that hours
within the same storm are not treated as independent). Forecasts are grouped by the
wave height at the time the forecast is issued, which is known in advance, unlike the
height that actually happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_all, load_hourly, split_by_time
from .models import load_bundle as load_models
from .models import persistence
from .paths import SAMPLE_PARQUET, VERIFICATION_JSON, WEIGHTS

METHODS = ("persistence", "linear", "mlp")
HORIZONS = (3, 6)
# Bins of the wave height at issue time (metres).
BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]
MIN_FORECASTS = 200  # fewer than this and I call a result "too few to say"
N_BOOT = 1000


def bin_label(lo: float, hi: float) -> str:
    return f"{lo:g} to {hi:g} m" if hi < 99 else f"over {lo:g} m"


def forecast_table(
    data: Path = SAMPLE_PARQUET, weights: Path = WEIGHTS, horizon: int = 3
) -> pd.DataFrame:
    """One row per test-period forecast with the observation and each method's forecast."""
    hourly = load_hourly(data)
    _, _, test = split_by_time(build_all(hourly, horizon), horizon)
    mlp, lin = load_models(weights / f"model_h{horizon}.pt")
    out = pd.DataFrame(
        {
            "station": test["station"].to_numpy(),
            "issued": test.index,
            "height_now": test["wvht_lag0"].to_numpy(),
            "observed": test["target"].to_numpy(),
            "persistence": persistence(test),
            "linear": lin.predict(test),
            "mlp": mlp.predict(test),
        }
    )
    out["valid"] = out["issued"] + pd.Timedelta(hours=horizon)
    return out


def _errors(df: pd.DataFrame, method: str) -> np.ndarray:
    return df[method].to_numpy() - df["observed"].to_numpy()


def scores(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Metrics for each method on this group of forecasts, plus skill with a 95% range."""
    per_method = {}
    for m in METHODS:
        e = _errors(df, m)
        per_method[m] = {
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean()),
        }
    for m in METHODS:
        if m != "persistence":
            ref = per_method["persistence"]
            per_method[m]["skill_mae"] = _skill(per_method[m]["mae"], ref["mae"])
            per_method[m]["skill_rmse"] = _skill(per_method[m]["rmse"], ref["rmse"])
    lo, hi = skill_interval(df, "mlp", rng)
    verdict = verdict_for(len(df), lo, hi)
    mean_obs = float(df["observed"].mean())
    return {
        "n": len(df),
        "mean_observed": mean_obs,
        "mlp_mae_pct_of_mean": per_method["mlp"]["mae"] / mean_obs if mean_obs > 0 else None,
        "methods": per_method,
        "mlp_skill_mae_95": [lo, hi],
        "verdict": verdict,
    }


def _skill(error: float, reference: float) -> float | None:
    """1 - error / reference, or None when the reference is perfect (nothing to beat)."""
    return None if reference == 0 else 1 - error / reference


def skill_interval(df: pd.DataFrame, method: str, rng: np.random.Generator) -> tuple[float, float]:
    """95% range of the MAE skill, from resampling days with replacement."""
    day = df["issued"].dt.floor("1D").to_numpy()
    codes, _ = pd.factorize(day)
    k = codes.max() + 1
    err_m = np.bincount(codes, weights=np.abs(_errors(df, method)), minlength=k)
    err_p = np.bincount(codes, weights=np.abs(_errors(df, "persistence")), minlength=k)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=N_BOOT).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = 1 - (w @ err_m) / (w @ err_p)
    skill = skill[np.isfinite(skill)]
    if len(skill) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(skill, 2.5)), float(np.percentile(skill, 97.5))


def verdict_for(n: int, lo: float, hi: float) -> str:
    if n < MIN_FORECASTS or np.isnan(lo):
        return "too few forecasts to say"
    if lo > 0:
        return "better than persistence"
    if hi < 0:
        return "worse than persistence"
    return "no clear difference from persistence"


def verify(data: Path = SAMPLE_PARQUET, weights: Path = WEIGHTS, seed: int = 0) -> dict:
    result: dict = {
        "period": "2024-07-01 to 2024-12-31 (UTC)",
        "methods": list(METHODS),
        "note": (
            "mlp is the model served by the API (seed 0). Skill = 1 - error / persistence error. "
            "Heights are grouped by the wave height when the forecast is issued."
        ),
        "leads": {},
    }
    for h in HORIZONS:
        rng = np.random.default_rng(seed + h)
        t = forecast_table(data, weights, h)
        by_height = []
        for lo, hi in BINS:
            g = t[(t["height_now"] >= lo) & (t["height_now"] < hi)]
            if len(g):
                by_height.append({"bin": bin_label(lo, hi), **scores(g, rng)})
        by_station = [
            {"station": s, **scores(g, rng)} for s, g in t.groupby("station", observed=True)
        ]
        result["leads"][str(h)] = {
            "overall": scores(t, rng),
            "by_height": by_height,
            "by_station": by_station,
        }
    return result


def format_report(result: dict) -> str:
    lines = []
    for h, lead in result["leads"].items():
        lines.append(f"\n{h} h ahead")
        lines.append(
            "| Group | n | Persistence MAE | MLP MAE | MLP bias | Skill (95% range) | Verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        groups = (
            [("all", lead["overall"])]
            + [(g["bin"], g) for g in lead["by_height"]]
            + [(g["station"], g) for g in lead["by_station"]]
        )
        for name, g in groups:
            m, p = g["methods"]["mlp"], g["methods"]["persistence"]
            lo, hi = g["mlp_skill_mae_95"]
            lines.append(
                f"| {name} | {g['n']} | {p['mae']:.3f} | {m['mae']:.3f} | {m['bias']:+.3f} | "
                f"{m['skill_mae']:+.1%} ({lo:+.1%} to {hi:+.1%}) | {g['verdict']} |"
            )
    return "\n".join(lines)


def main() -> None:
    result = verify()
    VERIFICATION_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(format_report(result))
    print(f"\nwrote {VERIFICATION_JSON}")


if __name__ == "__main__":
    main()
