"""FastAPI app: stations, recent observations, forecasts and a small backtest.

The data is the committed historical sample, so "now" is the last hour in the
sample (or any hour passed as `as_of`), not the live ocean.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from . import geo
from .features import build_features, load_hourly
from .models import LinearModel, MLPForecaster, load_bundle
from .paths import (
    ARTIFACTS,
    QC_FLAGS_PARQUET,
    QC_SUMMARY_JSON,
    SAMPLE_PARQUET,
    STATIONS,
    VERIFICATION_JSON,
    WEB_DIR,
)

HORIZONS = (3, 6)
POLYGON_BODY = Body(..., description="a GeoJSON Polygon")
FLAG_CODES = {"1": "pass", "2": "not evaluated", "3": "suspect", "4": "fail", "9": "missing"}


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x: float) -> float | None:
    return None if pd.isna(x) else round(float(x), 3)


def parse_time(value: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except ValueError as exc:
        raise HTTPException(422, f"bad time: {value}") from exc
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.floor("1h")


def create_app(data: Path = SAMPLE_PARQUET, artifacts: Path = ARTIFACTS) -> FastAPI:
    app = FastAPI(title="buoycast", version="0.1.0")
    hourly = load_hourly(data)
    models: dict[int, tuple[MLPForecaster, LinearModel]] = {}
    for h in HORIZONS:
        path = artifacts / f"model_h{h}.pt"
        if path.exists():
            models[h] = load_bundle(path)

    def station_frame(station: str) -> pd.DataFrame:
        if station not in hourly:
            raise HTTPException(404, f"unknown station {station}")
        return hourly[station]

    @lru_cache(maxsize=32)
    def features_with_target(station: str, horizon: int) -> pd.DataFrame:
        return build_features(hourly[station], horizon)

    @app.get("/stations")
    def stations() -> list[dict]:
        out = []
        for sid, meta in STATIONS.items():
            if sid not in hourly:
                continue
            obs = hourly[sid]["wvht"].dropna()
            out.append(
                {
                    "id": sid,
                    **meta,
                    "first_obs": _iso(obs.index[0]),
                    "last_obs": _iso(obs.index[-1]),
                    "last_wvht": _num(obs.iloc[-1]),
                }
            )
        return out

    def station_points() -> pd.DataFrame:
        return geo.station_table(hourly)

    def station_rows(df: pd.DataFrame) -> list[dict]:
        cols = ["id", "name", "lat", "lon"] + (["distance_km"] if "distance_km" in df else [])
        return [
            {k: (round(v, 2) if k == "distance_km" else v) for k, v in r.items()}
            for r in df[cols].to_dict("records")
        ]

    @app.get("/stations/bbox")
    def stations_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[dict]:
        """Stations inside a box. If min_lon is larger than max_lon the box crosses 180 degrees."""
        try:
            return station_rows(
                geo.within_bbox(station_points(), min_lon, min_lat, max_lon, max_lat)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/stations/nearest")
    def stations_nearest(lat: float, lon: float, k: int = Query(1, ge=1, le=10)) -> list[dict]:
        """The k stations closest to a position, with distance in km."""
        try:
            return station_rows(geo.nearest(station_points(), lat, lon, k))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/stations/within")
    def stations_within(polygon: dict = POLYGON_BODY) -> list[dict]:
        """Stations inside a polygon given as GeoJSON (coordinates are [lon, lat])."""
        try:
            return station_rows(geo.within_polygon(station_points(), polygon))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise HTTPException(422, f"bad polygon: {exc}") from exc

    @app.get("/observations")
    def observations(
        station: str,
        hours: int = Query(48, ge=1, le=24 * 30),
        end: str | None = None,
    ) -> dict:
        df = station_frame(station)
        end_ts = parse_time(end) if end else df.index[-1]
        window = df.loc[end_ts - pd.Timedelta(hours=hours - 1) : end_ts]
        rows = [
            {
                "time": _iso(t),
                "wvht": _num(r.wvht),
                "wspd": _num(r.wspd),
                "pres": _num(r.pres),
            }
            for t, r in window.iterrows()
        ]
        return {
            "station": station,
            "unit": {"wvht": "m", "wspd": "m/s", "pres": "hPa"},
            "rows": rows,
        }

    @app.get("/forecast/{station}")
    def forecast(station: str, as_of: str | None = None) -> dict:
        df = station_frame(station)
        if not models:
            raise HTTPException(503, "no trained models found; run buoycast-train")
        past = df if as_of is None else df.loc[: parse_time(as_of)]
        feats = build_features(past, 3, require_target=False)
        if feats.empty:
            raise HTTPException(404, "not enough recent observations for a forecast")
        issued = feats.index[-1] if as_of is None else parse_time(as_of)
        if issued not in feats.index:
            raise HTTPException(404, f"no complete observations at {_iso(issued)}")
        row = feats.loc[[issued]].assign(station=station)
        out = []
        for h, (mlp, lin) in sorted(models.items()):
            out.append(
                {
                    "horizon_hours": h,
                    "valid_time": _iso(issued + pd.Timedelta(hours=h)),
                    "mlp": _num(mlp.predict(row)[0]),
                    "linear": _num(lin.predict(row)[0]),
                    "persistence": _num(row["wvht_lag0"].iloc[0]),
                }
            )
        return {
            "station": station,
            "as_of": _iso(issued),
            "current_wvht": _num(row["wvht_lag0"].iloc[0]),
            "unit": "m",
            "forecasts": out,
        }

    @app.get("/backtest/{station}")
    def backtest(
        station: str,
        horizon: int = Query(3, ge=3, le=6),
        hours: int = Query(72, ge=6, le=24 * 14),
        end: str | None = None,
    ) -> dict:
        station_frame(station)
        if horizon not in models:
            raise HTTPException(404, f"no model for horizon {horizon}")
        feats = features_with_target(station, horizon)
        end_ts = parse_time(end) if end else hourly[station].index[-1]
        valid = feats.index + pd.Timedelta(hours=horizon)
        keep = feats[(valid <= end_ts) & (valid > end_ts - pd.Timedelta(hours=hours))]
        mlp, lin = models[horizon]
        keep = keep.assign(station=station)
        mlp_pred, lin_pred = mlp.predict(keep), lin.predict(keep)
        rows = [
            {
                "valid_time": _iso(t + pd.Timedelta(hours=horizon)),
                "observed": _num(r.target),
                "mlp": _num(m),
                "linear": _num(li),
                "persistence": _num(r.wvht_lag0),
            }
            for (t, r), m, li in zip(keep.iterrows(), mlp_pred, lin_pred, strict=True)
        ]
        return {"station": station, "horizon_hours": horizon, "unit": "m", "rows": rows}

    @lru_cache(maxsize=1)
    def qc_flags() -> pd.DataFrame:
        if not QC_FLAGS_PARQUET.exists():
            raise HTTPException(503, "no QC flags found; run buoycast-qc")
        df = pd.read_parquet(QC_FLAGS_PARQUET)
        df["station"] = df["station"].astype(str)
        return df

    @app.get("/verification")
    def verification() -> dict:
        """Test-period scores by lead time, wave-height bin and station (buoycast-verify)."""
        if not VERIFICATION_JSON.exists():
            raise HTTPException(503, "no verification report found; run buoycast-verify")
        return json.loads(VERIFICATION_JSON.read_text())

    @app.get("/qc/summary")
    def qc_summary() -> dict:
        if not QC_SUMMARY_JSON.exists():
            raise HTTPException(503, "no QC summary found; run buoycast-qc")
        return json.loads(QC_SUMMARY_JSON.read_text())

    @app.get("/qc/{station}")
    def qc_station(
        station: str,
        hours: int = Query(72, ge=1, le=24 * 30),
        end: str | None = None,
    ) -> dict:
        """Per-test flags (1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing)."""
        df = station_frame(station)
        end_ts = parse_time(end) if end else df.index[-1]
        flags = qc_flags()
        flags = flags[flags["station"] == station].set_index("time")
        window = flags.loc[end_ts - pd.Timedelta(hours=hours - 1) : end_ts].drop(columns="station")
        window = window[[c for c in window.columns if not c.endswith("_overall")]]
        rows = [
            {"time": _iso(t), **{c: int(v) for c, v in r.items()}} for t, r in window.iterrows()
        ]
        return {"station": station, "flag_codes": FLAG_CODES, "rows": rows}

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def serve() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
