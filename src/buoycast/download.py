"""Download NDBC yearly files and write the committed hourly sample.

Usage: uv run buoycast-download
"""

from __future__ import annotations

import httpx
import pandas as pd

from .parse import parse_stdmet, to_hourly
from .paths import RAW_DIR, SAMPLE_PARQUET, STATIONS, YEARS

BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet"


def fetch(station: str, year: int, client: httpx.Client) -> bytes:
    """Fetch one year file, caching the gzip in data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{station}h{year}.txt.gz"
    if cache.exists():
        return cache.read_bytes()
    resp = client.get(f"{BASE}/{station}h{year}.txt.gz", timeout=60)
    resp.raise_for_status()
    cache.write_bytes(resp.content)
    return resp.content


def build_sample() -> pd.DataFrame:
    frames = []
    with httpx.Client(follow_redirects=True) as client:
        for station in STATIONS:
            parts = []
            for year in YEARS:
                print(f"{station} {year}")
                parts.append(parse_stdmet(fetch(station, year, client)))
            # Keep only hours with a wave reading (gaps are restored when loading).
            hourly = to_hourly(pd.concat(parts)).dropna(subset=["wvht"]).round(2)
            hourly.insert(0, "station", station)
            frames.append(hourly.reset_index())
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = build_sample()
    SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df["station"] = df["station"].astype("category")
    df.astype({c: "float32" for c in df.columns if c not in ("station", "time")}).to_parquet(
        SAMPLE_PARQUET, index=False, compression="zstd"
    )
    print(f"wrote {SAMPLE_PARQUET} ({len(df)} rows)")


if __name__ == "__main__":
    main()
