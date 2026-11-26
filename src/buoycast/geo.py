"""Positions: distances, simple spatial queries and GeoParquet / GeoJSON export.

A "points table" in this module is any pandas table with `lat` and `lon` columns in
degrees (WGS84, the same coordinates GPS uses). It can be the six moored stations or
every fix of a drifting buoy, so the same queries work for both.

Usage: uv run buoycast-export [--out exports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely.geometry import Polygon, shape

from .features import load_hourly
from .paths import ROOT, SAMPLE_PARQUET, STATIONS

EARTH_RADIUS_KM = 6371.0


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (haversine formula), for numbers or arrays."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(np.asarray(lon2) - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _check_point(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"not a valid position: lat={lat}, lon={lon}")


def station_table(hourly: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """The stations with position and the first and last hour with a wave reading."""
    hourly = hourly if hourly is not None else load_hourly()
    rows = []
    for sid, meta in STATIONS.items():
        if sid not in hourly:
            continue
        obs = hourly[sid]["wvht"].dropna()
        rows.append({"id": sid, **meta, "first_obs": obs.index[0], "last_obs": obs.index[-1]})
    return pd.DataFrame(rows)


def within_bbox(
    points: pd.DataFrame, min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> pd.DataFrame:
    """Rows inside the box, edges included. If min_lon > max_lon the box crosses 180 degrees."""
    _check_point(min_lat, min_lon)
    _check_point(max_lat, max_lon)
    if min_lat > max_lat:
        raise ValueError("min_lat is larger than max_lat")
    in_lat = points["lat"].between(min_lat, max_lat)
    if min_lon <= max_lon:
        in_lon = points["lon"].between(min_lon, max_lon)
    else:
        in_lon = (points["lon"] >= min_lon) | (points["lon"] <= max_lon)
    return points[in_lat & in_lon]


def as_polygon(polygon) -> Polygon:
    """A shapely polygon from a GeoJSON geometry, a list of (lon, lat) pairs or a polygon."""
    if isinstance(polygon, dict):
        polygon = shape(polygon["geometry"] if polygon.get("type") == "Feature" else polygon)
    elif not isinstance(polygon, Polygon):
        polygon = Polygon([(float(x), float(y)) for x, y in polygon])
    if polygon.geom_type != "Polygon":
        raise ValueError(f"expected a polygon, got {polygon.geom_type}")
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("the polygon is empty or crosses itself")
    return polygon


def within_polygon(points: pd.DataFrame, polygon) -> pd.DataFrame:
    """Rows inside the polygon (a point on its edge counts as inside)."""
    poly = as_polygon(polygon)
    pts = shapely.points(points["lon"].to_numpy(), points["lat"].to_numpy())
    return points[shapely.covers(poly, pts)]


def nearest(points: pd.DataFrame, lat: float, lon: float, k: int = 1) -> pd.DataFrame:
    """The k rows closest to a position, nearest first, with a `distance_km` column."""
    _check_point(lat, lon)
    if k < 1:
        raise ValueError("k must be at least 1")
    dist = distance_km(lat, lon, points["lat"].to_numpy(), points["lon"].to_numpy())
    return points.assign(distance_km=dist).sort_values("distance_km", kind="stable").head(k)


def to_geojson(points: pd.DataFrame, properties: list[str]) -> dict:
    """A GeoJSON FeatureCollection of points. Times become ISO strings, NaN becomes null."""
    features = []
    for row in points.itertuples(index=False):
        props = {}
        for name in properties:
            v = getattr(row, name)
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif isinstance(v, float | np.floating):
                v = None if np.isnan(v) else round(float(v), 4)
            elif isinstance(v, np.integer):
                v = int(v)
            props[name] = v
        geometry = {
            "type": "Point",
            "coordinates": [round(float(row.lon), 5), round(float(row.lat), 5)],
        }
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def write_geoparquet(df: pd.DataFrame, path: Path) -> None:
    """Write a table of points as GeoParquet 1.0: WKB point geometry plus the 'geo' metadata.

    `lat` and `lon` stay as ordinary columns too.
    """
    geometry = shapely.to_wkb(shapely.points(df["lon"].to_numpy(), df["lat"].to_numpy()))
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    table = table.append_column("geometry", pa.array(geometry, type=pa.binary()))
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "bbox": [
                    float(df["lon"].min()),
                    float(df["lat"].min()),
                    float(df["lon"].max()),
                    float(df["lat"].max()),
                ],
            }
        },
    }
    meta = {**(table.schema.metadata or {}), b"geo": json.dumps(geo).encode()}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")


def export_all(out: Path) -> dict[str, int]:
    """Write stations, observations, forecasts and drifter tracks. Returns rows per file."""
    from . import drifters, qc_run, verify  # imported here: they need the sample files

    out.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly()
    stations = station_table(hourly)
    counts: dict[str, int] = {}

    def save_json(name: str, obj: dict) -> None:
        (out / name).write_text(json.dumps(obj) + "\n")
        counts[name] = len(obj["features"])

    def save_parquet(name: str, df: pd.DataFrame) -> None:
        write_geoparquet(df, out / name)
        counts[name] = len(df)

    save_json("stations.geojson", to_geojson(stations, ["id", "name", "first_obs", "last_obs"]))
    save_parquet("stations.parquet", stations)

    pos = stations.set_index("id")[["lat", "lon"]]
    flags = qc_run.station_flags()
    obs = pd.read_parquet(SAMPLE_PARQUET).assign(station=lambda d: d["station"].astype(str))
    obs = obs.join(pos, on="station")
    obs["wvht_qc"] = flags["wvht_overall"].to_numpy()  # same rows, same order as the sample
    save_parquet("observations.parquet", obs)

    fc = []
    for lead in verify.HORIZONS:
        t = verify.forecast_table(horizon=lead).assign(lead_hours=lead)
        fc.append(t.join(pos, on="station"))
    forecasts = pd.concat(fc, ignore_index=True)
    save_parquet("forecasts.parquet", forecasts)
    last = forecasts[forecasts["lead_hours"] == 3].sort_values("issued").groupby("station").tail(1)
    save_json(
        "forecasts_latest.geojson",
        to_geojson(
            last,
            [
                "station",
                "issued",
                "valid",
                "lead_hours",
                "observed",
                "persistence",
                "linear",
                "mlp",
            ],
        ),
    )

    tracks = drifters.load_drifters()
    fixes = pd.concat(
        [f.reset_index().assign(drifter=name) for name, f in tracks.items()], ignore_index=True
    )
    save_parquet("drifters.parquet", fixes)
    lines = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [round(x, 4), round(y, 4)] for x, y in zip(f["lon"], f["lat"], strict=True)
                ],
            },
            "properties": {"drifter": name, "fixes": len(f)},
        }
        for name, f in tracks.items()
    ]
    save_json("drifter_tracks.geojson", {"type": "FeatureCollection", "features": lines})
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "exports")
    args = ap.parse_args()
    for name, n in export_all(args.out).items():
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()
