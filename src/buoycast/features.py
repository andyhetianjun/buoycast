"""Turn hourly buoy tables into model features and targets.

Every feature at time t only uses readings from t or earlier. The target is the
wave height `horizon` hours after t. tests/test_features.py checks this by
corrupting the future and confirming the features do not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import SAMPLE_PARQUET

WVHT_LAGS = (0, 1, 2, 3, 6, 12, 24)
WSPD_LAGS = (0, 1, 3, 6)
FFILL_LIMIT = 2  # fill gaps of at most 2 hours, using the last past value only

VAL_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")


def load_hourly(path: Path = SAMPLE_PARQUET) -> dict[str, pd.DataFrame]:
    """Load the sample and put each station on a complete hourly grid (gaps are NaN)."""
    df = pd.read_parquet(path)
    out: dict[str, pd.DataFrame] = {}
    for station, part in df.groupby("station", observed=True):
        part = part.drop(columns="station").set_index("time").sort_index()
        grid = pd.date_range(part.index[0], part.index[-1], freq="1h", name="time")
        out[str(station)] = part.astype("float64").reindex(grid)
    return out


def feature_names() -> list[str]:
    names = [f"wvht_lag{k}" for k in WVHT_LAGS]
    names += [f"wspd_lag{k}" for k in WSPD_LAGS]
    names += ["gst_lag0", "wind_u", "wind_v", "wind_u_lag3", "wind_v_lag3"]
    names += ["dpd_lag0", "dpd_lag3", "pres", "pres_d3", "pres_d12"]
    names += ["hour_sin", "hour_cos"]
    return names


def build_features(
    hourly: pd.DataFrame, horizon: int, require_target: bool = True
) -> pd.DataFrame:
    """Features for every hour of one station, plus `target` (wvht at t+horizon).

    Rows with any missing feature are dropped, and so are rows with a missing target
    unless `require_target` is False (used to forecast the newest hour, whose target
    does not exist yet).
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 hour")
    h = hourly.ffill(limit=FFILL_LIMIT)  # forward fill only looks backwards in time
    feats: dict[str, pd.Series] = {}
    for k in WVHT_LAGS:
        feats[f"wvht_lag{k}"] = h["wvht"].shift(k)
    for k in WSPD_LAGS:
        feats[f"wspd_lag{k}"] = h["wspd"].shift(k)
    feats["gst_lag0"] = h["gst"]
    # wdir is the direction the wind blows from; u is eastward, v is northward.
    rad = np.deg2rad(h["wdir"])
    u = -h["wspd"] * np.sin(rad)
    v = -h["wspd"] * np.cos(rad)
    feats["wind_u"], feats["wind_v"] = u, v
    feats["wind_u_lag3"], feats["wind_v_lag3"] = u.shift(3), v.shift(3)
    feats["dpd_lag0"] = h["dpd"]
    feats["dpd_lag3"] = h["dpd"].shift(3)
    feats["pres"] = h["pres"]
    feats["pres_d3"] = h["pres"] - h["pres"].shift(3)
    feats["pres_d12"] = h["pres"] - h["pres"].shift(12)
    hour = h.index.hour
    feats["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=h.index)
    feats["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=h.index)

    out = pd.DataFrame(feats)[feature_names()]
    out["target"] = hourly["wvht"].shift(-horizon)  # raw series: never filled forward
    # Only rows where an observation really exists at t are usable for forecasting.
    out = out[hourly["wvht"].notna()]
    return out.dropna(subset=None if require_target else feature_names())


def build_all(
    hourly: dict[str, pd.DataFrame], horizon: int, stations: list[str] | None = None
) -> pd.DataFrame:
    """Stack features of all stations, with a `station` column."""
    frames = []
    for station in stations or sorted(hourly):
        f = build_features(hourly[station], horizon)
        f.insert(0, "station", station)
        frames.append(f)
    return pd.concat(frames)


def split_by_time(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test by date.

    A row at time t has its label at t + horizon, so a row only joins a split if
    its label time is inside that split too. Rows straddling a boundary are dropped.
    """
    gap = pd.Timedelta(hours=horizon)
    t = df.index
    train = df[t + gap < VAL_START]
    val = df[(t >= VAL_START) & (t + gap < TEST_START)]
    test = df[t >= TEST_START]
    return train, val, test
