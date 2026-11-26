"""Live observations from NDBC and forecasts from the committed trained models.

NDBC publishes the last 45 days of each buoy as a plain text file that is updated
every hour: https://www.ndbc.noaa.gov/data/realtime2/<station>.txt. This module
downloads those files, turns them into the same hourly table the training used,
and runs the saved models over them.

The models were trained on six buoys (2022 to 2023). The Canadian buoys below are
not among them, so for those the station id input is all zeros and the forecast
is less trustworthy. `hindcast_scores` measures that on the recent past instead of
assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from .features import build_features
from .models import LinearModel, MLPForecaster, load_bundle, persistence
from .parse import parse_stdmet, to_hourly
from .paths import STATIONS, WEIGHTS

BASE = "https://www.ndbc.noaa.gov/data/realtime2"
HORIZONS = (3, 6)
FRESH_HOURS = 6  # a station whose newest wave reading is older than this is "offline"

# Buoys run by Environment and Climate Change Canada that NDBC also serves
# (positions from NDBC's active stations list).
CANADIAN: dict[str, dict[str, object]] = {
    "46131": {"name": "Sentry Shoal, BC (Canada)", "lat": 49.91, "lon": -124.98},
    "46146": {"name": "Halibut Bank, BC (Canada)", "lat": 49.34, "lon": -123.73},
    "46206": {"name": "La Perouse Bank, BC (Canada)", "lat": 48.84, "lon": -126.0},
    "46207": {"name": "East Dellwood, BC (Canada)", "lat": 50.87, "lon": -129.92},
    "46004": {"name": "Middle Nomad, Pacific (Canada)", "lat": 50.93, "lon": -136.1},
    "44139": {"name": "Banquereau Banks, NS (Canada)", "lat": 44.24, "lon": -57.1},
    "44137": {"name": "East Scotia Slope, NS (Canada)", "lat": 42.26, "lon": -62.03},
    "44150": {"name": "La Have Bank, NS (Canada)", "lat": 42.5, "lon": -64.02},
}
LIVE_STATIONS: dict[str, dict[str, object]] = {**STATIONS, **CANADIAN}
TRAINED = frozenset(STATIONS)


def fetch_station(station: str, client: httpx.Client) -> pd.DataFrame | None:
    """Hourly table for one buoy from its realtime file, or None if it cannot be had."""
    try:
        resp = client.get(f"{BASE}/{station}.txt", timeout=30)
        resp.raise_for_status()
        raw = parse_stdmet(resp.text)
        hourly = to_hourly(raw)
    except (httpx.HTTPError, ValueError, KeyError):
        return None
    if hourly.empty:
        return None
    # to_hourly leaves holes as NaN rows on a full hourly grid; make sure it is complete.
    return hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq="1h", name="time"))


def fetch_all(stations: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Download every station; the ones that fail are simply missing from the result."""
    out: dict[str, pd.DataFrame] = {}
    with httpx.Client(follow_redirects=True, headers={"User-Agent": "buoycast"}) as client:
        for sid in stations or list(LIVE_STATIONS):
            frame = fetch_station(sid, client)
            if frame is not None:
                out[sid] = frame
    return out


@dataclass
class Models:
    by_horizon: dict[int, tuple[MLPForecaster, LinearModel]]

    @classmethod
    def load(cls) -> Models:
        return cls({h: load_bundle(WEIGHTS / f"model_h{h}.pt") for h in HORIZONS})


def forecast_frame(
    station: str, hourly: pd.DataFrame, models: Models, horizon: int
) -> pd.DataFrame:
    """One row per hour that has all the inputs, issued at `issued`, valid `horizon` h later.

    The newest rows have no outcome yet; older rows can be compared with what happened.
    """
    feats = build_features(hourly, horizon, require_target=False)
    if feats.empty:
        return pd.DataFrame(columns=["issued", "valid", "persistence", "linear", "mlp", "observed"])
    feats = feats.copy()
    feats.insert(0, "station", station)
    mlp, lin = models.by_horizon[horizon]
    out = pd.DataFrame(
        {
            "issued": feats.index,
            "valid": feats.index + pd.Timedelta(hours=horizon),
            "persistence": persistence(feats),
            "linear": lin.predict(feats),
            "mlp": mlp.predict(feats),
            "observed": feats["target"].to_numpy(),
        }
    )
    return out


def latest(frame: pd.DataFrame) -> pd.Timestamp:
    """Time of the newest hour with a wave height reading."""
    return frame["wvht"].dropna().index[-1]


def hindcast_scores(tables: dict[str, dict[int, pd.DataFrame]], days: int = 30) -> dict:
    """Mean absolute error of each method over the last `days` days, per horizon and overall.

    These forecasts were made by the committed models on data from after their
    training period, using only readings up to each issue time.
    """
    result: dict = {}
    for h in HORIZONS:
        per_station = {}
        pooled = []
        for sid, by_h in tables.items():
            t = by_h[h].dropna(subset=["observed"])
            if t.empty:
                continue
            t = t[t["issued"] >= t["issued"].max() - pd.Timedelta(days=days)]
            if len(t) < 24:
                continue
            per_station[sid] = {
                "n": len(t),
                **{m: float(np.abs(t[m] - t["observed"]).mean()) for m in METHODS},
            }
            pooled.append(t)
        if not pooled:
            continue
        allt = pd.concat(pooled)
        result[str(h)] = {
            "n": len(allt),
            **{m: float(np.abs(allt[m] - allt["observed"]).mean()) for m in METHODS},
            "stations": per_station,
        }
    return result


METHODS = ("persistence", "linear", "mlp")
