"""Compare buoycast with NOAA's operational wave model (GFS-Wave, built on WAVEWATCH III).

Usage: uv run buoycast-operational [--download]

NOAA publishes GFS-Wave output for each model run ("cycle", four a day at 00, 06, 12 and
18 UTC). Besides the big GRIB2 grids it publishes a small text bulletin for each buoy
location. I read those bulletins instead of GRIB2 files: they hold the model's own point
output at the buoy, so there is no grid interpolation for me to get wrong.

For a cycle at time C and a lead of h hours, the operational forecast for time C+h is
compared with buoycast's forecast issued at C for C+h, against the same buoy reading.
Both are scored with the same error code as the rest of the project.

Source: s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/CC/wave/station/bulls.tCCz/gfswave.<buoy>.bull
(NOAA Open Data Dissemination on AWS, public, no account).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .paths import OPERATIONAL_CSV, OPERATIONAL_JSON, RAW_DIR, STATIONS
from .verify import HORIZONS, _errors, _skill, forecast_table

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CYCLES = (0, 6, 12, 18)
START = pd.Timestamp("2024-10-01", tz="UTC")
DAYS = 28
METHODS = ("persistence", "mlp", "gfswave")
N_BOOT = 1000

_CYCLE = re.compile(r"Cycle\s*:\s*(\d{8})\s+(\d+)\s+UTC")
_ROW = re.compile(r"^\s*\|\s*(\d+)\s+(\d+)\s*\|\s*([\d.]+)\s")


def parse_bulletin(text: str) -> pd.DataFrame:
    """Read one buoy bulletin into rows of (cycle, valid, hs).

    `hs` is the total significant wave height in metres (the "Hst" column). Rows only carry
    a day of the month and an hour, so the month and year come from the cycle line and a
    day number smaller than the cycle's day means the next month.
    """
    m = _CYCLE.search(text)
    if m is None:
        raise ValueError("no cycle line in bulletin")
    cycle = pd.Timestamp(m.group(1), tz="UTC") + pd.Timedelta(hours=int(m.group(2)))
    valid, hs = [], []
    year, month = cycle.year, cycle.month
    prev_day = cycle.day
    for line in text.splitlines():
        r = _ROW.match(line)
        if r is None:
            continue
        day, hour = int(r.group(1)), int(r.group(2))
        if day < prev_day:
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        prev_day = day
        valid.append(pd.Timestamp(year=year, month=month, day=day, hour=hour, tz="UTC"))
        hs.append(float(r.group(3)))
    return pd.DataFrame({"cycle": cycle, "valid": valid, "hs": hs})


def bulletin_url(cycle: pd.Timestamp, station: str) -> str:
    c = f"{cycle.hour:02d}"
    return f"{BASE}/gfs.{cycle:%Y%m%d}/{c}/wave/station/bulls.t{c}z/gfswave.{station}.bull"


def extract_leads(bulletin: pd.DataFrame, station: str, leads=HORIZONS) -> pd.DataFrame:
    """Keep only the forecasts for the lead times we score."""
    out = bulletin.copy()
    out["lead"] = ((out["valid"] - out["cycle"]) / pd.Timedelta(hours=1)).astype(int)
    out = out[out["lead"].isin(leads)]
    out.insert(0, "station", station)
    return out[["station", "cycle", "lead", "valid", "hs"]].reset_index(drop=True)


def fetch(cycle: pd.Timestamp, station: str, client: httpx.Client) -> str | None:
    """One bulletin, cached in data/raw. Returns None if NOAA has no file (404)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"gfswave_{station}_{cycle:%Y%m%d%H}.bull"
    if cache.exists():
        return cache.read_text()
    resp = client.get(bulletin_url(cycle, station), timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    cache.write_text(resp.text)
    return resp.text


def download_sample(stations: list[str] | None = None) -> pd.DataFrame:
    """Download the bulletins for the sample weeks and keep just the scored leads."""
    stations = stations or list(STATIONS)
    cycles = [START + pd.Timedelta(days=d, hours=h) for d in range(DAYS) for h in CYCLES]
    parts, missing = [], {}
    with httpx.Client(follow_redirects=True) as client:
        for s in stations:
            for c in cycles:
                text = fetch(c, s, client)
                if text is None:
                    missing[s] = missing.get(s, 0) + 1
                    continue
                parts.append(extract_leads(parse_bulletin(text), s))
    for s, n in missing.items():
        print(f"{s}: no GFS-Wave point output ({n} of {len(cycles)} bulletins missing)")
    return pd.concat(parts, ignore_index=True)


def align(buoycast: pd.DataFrame, gfs: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Rows where both systems made a forecast for the same buoy, issue time and valid time.

    `buoycast` is a verify.forecast_table (columns station, issued, valid, observed,
    persistence, mlp). `gfs` has station, cycle, lead, valid, hs. A buoycast forecast
    issued at C matches the GFS-Wave cycle that started at C.
    """
    g = gfs[gfs["lead"] == lead].rename(columns={"cycle": "issued", "hs": "gfswave"})
    g = g[["station", "issued", "valid", "gfswave"]].copy()
    b = buoycast.copy()
    b["station"] = b["station"].astype(str)
    out = b.merge(g, on=["station", "issued", "valid"], how="inner")
    return out.dropna(subset=["observed", "persistence", "mlp", "gfswave"]).reset_index(drop=True)


def group_scores(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """MAE, RMSE and bias for each method, plus how much better buoycast is than GFS-Wave."""
    methods = {}
    for m in METHODS:
        e = _errors(df, m)
        methods[m] = {
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean()),
        }
    lo, hi = _diff_interval(df, rng)
    return {
        "n": len(df),
        "methods": methods,
        "mlp_skill_vs_gfswave": _skill(methods["mlp"]["mae"], methods["gfswave"]["mae"]),
        "mlp_skill_vs_gfswave_95": [lo, hi],
    }


def _diff_interval(df: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """95% range of 1 - (buoycast MAE / GFS-Wave MAE), resampling whole days."""
    codes, _ = pd.factorize(df["issued"].dt.floor("1D").to_numpy())
    k = codes.max() + 1
    a = np.bincount(codes, weights=np.abs(_errors(df, "mlp")), minlength=k)
    b = np.bincount(codes, weights=np.abs(_errors(df, "gfswave")), minlength=k)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=N_BOOT).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1 - (w @ a) / (w @ b)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def compare(gfs: pd.DataFrame, seed: int = 0) -> dict:
    result: dict = {
        "period": f"{START:%Y-%m-%d} for {DAYS} days, cycles 00, 06, 12 and 18 UTC",
        "source": "NOAA GFS-Wave point bulletins (noaa-gfs-bdp-pds on AWS)",
        "stations": sorted(gfs["station"].unique()),
        "leads": {},
    }
    for h in HORIZONS:
        rng = np.random.default_rng(seed + h)
        joined = align(forecast_table(horizon=h), gfs, h)
        by_station = [{"station": s, **group_scores(g, rng)} for s, g in joined.groupby("station")]
        result["leads"][str(h)] = {"overall": group_scores(joined, rng), "by_station": by_station}
    return result


def format_report(result: dict) -> str:
    lines = []
    for h, lead in result["leads"].items():
        lines += [
            f"\n{h} h ahead",
            "| Group | n | Persistence MAE | buoycast MAE | GFS-Wave MAE | "
            "GFS-Wave bias | buoycast bias | buoycast vs GFS-Wave (95% range) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, g in [("all", lead["overall"])] + [(x["station"], x) for x in lead["by_station"]]:
            m = g["methods"]
            lo, hi = g["mlp_skill_vs_gfswave_95"]
            lines.append(
                f"| {name} | {g['n']} | {m['persistence']['mae']:.3f} | {m['mlp']['mae']:.3f} | "
                f"{m['gfswave']['mae']:.3f} | {m['gfswave']['bias']:+.3f} | "
                f"{m['mlp']['bias']:+.3f} | "
                f"{g['mlp_skill_vs_gfswave']:+.1%} ({lo:+.1%} to {hi:+.1%}) |"
            )
    return "\n".join(lines)


def load_sample(path: Path = OPERATIONAL_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": str}, parse_dates=["cycle", "valid"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="fetch the bulletins from NOAA first")
    args = ap.parse_args()
    if args.download:
        gfs = download_sample()
        OPERATIONAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        gfs.to_csv(OPERATIONAL_CSV, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"wrote {OPERATIONAL_CSV} ({len(gfs)} rows)")
    result = compare(load_sample())
    OPERATIONAL_JSON.write_text(json.dumps(result, indent=1) + "\n")
    print(format_report(result))
    print(f"\nwrote {OPERATIONAL_JSON}")


if __name__ == "__main__":
    main()
