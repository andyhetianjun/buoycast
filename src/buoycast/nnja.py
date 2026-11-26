"""Load nearby airport weather reports from the open NNJA-AI archive.

NNJA-AI (https://github.com/brightbandtech/nnja-ai) is a public bucket of the
observations NOAA feeds its weather models, stored as one Parquet file per day.
It is read here without any account. I use the METAR airport reports
(dataset NC000007), because a few airports sit near some of my buoys and their
wind and pressure might help the wave forecast.

The archive has no wave heights, so this is extra *inputs*, not extra targets.

Usage: uv run buoycast-nnja
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .paths import NNJA_PARQUET, RAW_DIR, YEARS

BUCKET_PATH = "nnja-ai/data/v1/conv/adpsfc/NC000007/OBS_DATE={d}/gdas.{n}.NC000007.parquet"
COLUMNS = ["OBS_TIMESTAMP", "MTRID.RPID", "MTRWND.WSPD", "MTRWND.WDIR", "MTRPRS.ALSE"]

# Airport (ICAO id) nearest to each buoy that has one within about 90 km.
# 46005 is 500 km from land and 44005 has no test rows, so neither gets one.
NEAR_AIRPORT = {"44027": "KBHB", "45005": "KCLE", "45012": "KROC", "46041": "KUIL"}


def day_path(day: date) -> str:
    return BUCKET_PATH.format(d=day.isoformat(), n=day.strftime("%Y%m%d"))


def tidy(raw: pd.DataFrame, airports: set[str]) -> pd.DataFrame:
    """Keep the wanted airports and rename the columns.

    Wind speed is m/s, direction degrees, pressure is the altimeter setting in hPa.
    """
    df = raw.rename(
        columns={
            "MTRID.RPID": "airport",
            "OBS_TIMESTAMP": "time",
            "MTRWND.WSPD": "wspd",
            "MTRWND.WDIR": "wdir",
            "MTRPRS.ALSE": "pres",
        }
    )
    df = df[df["airport"].isin(airports)].copy()
    df["pres"] = df["pres"] / 100.0
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["airport", "time", "wspd", "wdir", "pres"]]


def to_hourly(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per airport per hour.

    A report taken at 12:53 is labelled 13:00, so a value stamped t was always known
    by t. If an hour has several reports, the last one is kept.
    """
    obs = obs.copy()
    obs["time"] = obs["time"].dt.ceil("1h")
    obs = obs.sort_values("time").drop_duplicates(["airport", "time"], keep="last")
    return obs.reset_index(drop=True)


def fetch_day(day: date, airports: set[str], fs: pafs.GcsFileSystem) -> pd.DataFrame:
    cache = RAW_DIR / "nnja" / f"{day.isoformat()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        with fs.open_input_file(day_path(day)) as f:
            raw = pq.read_table(f, columns=COLUMNS).to_pandas()
        out = tidy(raw, airports)
    except FileNotFoundError:
        out = tidy(pd.DataFrame(columns=COLUMNS), airports)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def download(years: tuple[int, ...] = YEARS, airports: set[str] | None = None) -> pd.DataFrame:
    airports = airports or set(NEAR_AIRPORT.values())
    fs = pafs.GcsFileSystem(anonymous=True)
    days = []
    d = date(years[0], 1, 1)
    while d <= date(years[-1], 12, 31):
        days.append(d)
        d += timedelta(days=1)
    with ThreadPoolExecutor(max_workers=12) as pool:
        parts = list(pool.map(lambda x: fetch_day(x, airports, fs), days))
    return to_hourly(pd.concat(parts, ignore_index=True))


def load_extra(path=NNJA_PARQUET) -> dict[str, pd.DataFrame]:
    """Hourly airport frames keyed by *buoy* id, on a complete hourly grid (gaps are NaN)."""
    df = pd.read_parquet(path)
    out: dict[str, pd.DataFrame] = {}
    for buoy, airport in NEAR_AIRPORT.items():
        part = df[df["airport"] == airport].set_index("time").sort_index()
        if part.empty:
            continue
        grid = pd.date_range(part.index[0], part.index[-1], freq="1h", name="time")
        out[buoy] = part[["wspd", "wdir", "pres"]].astype("float64").reindex(grid)
    return out


def main() -> None:
    df = download()
    NNJA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.astype({c: "float32" for c in ("wspd", "wdir", "pres")}).round(2).assign(
        airport=lambda x: x["airport"].astype("category")
    ).to_parquet(NNJA_PARQUET, index=False, compression="zstd")
    counts = df.groupby("airport", observed=True).size().to_dict()
    print(f"wrote {NNJA_PARQUET} ({len(df)} rows) {counts}")
    print(f"missing pressure: {float(np.mean(df['pres'].isna())):.1%}")


if __name__ == "__main__":
    main()
