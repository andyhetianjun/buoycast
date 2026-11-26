"""A small sample of moving platforms: surface drifters from the Global Drifter Program.

A drifter floats with the current and reports its position and sea surface
temperature, so every row has a different latitude and longitude. The data
comes from NOAA AOML's public ERDDAP server (6-hourly series).

Usage: uv run buoycast-drifters
"""

from __future__ import annotations

import io

import httpx
import pandas as pd

from .paths import DRIFTERS_PARQUET

ERDDAP = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.csv"
# Drifters that spent July to December 2024 in the north-east Pacific, near stations 46041
# and 46005. Picked from a query for drifters inside 35-50 N, 140-120 W in that season.
DRIFTER_IDS = (
    "300534064906390",
    "300534064906470",
    "300534064906520",
    "300534064906880",
    "300534064906900",
)
START, END = "2024-07-01T00:00:00Z", "2024-12-31T23:59:59Z"


def parse_erddap(text: str) -> pd.DataFrame:
    """ERDDAP CSV (header line then a units line) to a tidy frame."""
    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"ID": "drifter", "latitude": "lat", "longitude": "lon"})
    df["drifter"] = df["drifter"].astype(str)
    return df[["drifter", "time", "lat", "lon", "sst"]]


def fetch(drifter: str, client: httpx.Client) -> pd.DataFrame:
    query = f'ID,time,latitude,longitude,sst&ID="{drifter}"&time>={START}&time<={END}'
    resp = client.get(f"{ERDDAP}?{query}".replace(">", "%3E").replace("<", "%3C"), timeout=90)
    resp.raise_for_status()
    return parse_erddap(resp.text)


def load_drifters(path=DRIFTERS_PARQUET) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    return {
        str(k): g.drop(columns="drifter").set_index("time").sort_index()
        for k, g in df.groupby("drifter", observed=True)
    }


def main() -> None:
    with httpx.Client() as client:
        df = pd.concat([fetch(d, client) for d in DRIFTER_IDS], ignore_index=True)
    DRIFTERS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DRIFTERS_PARQUET, index=False, compression="zstd")
    print(f"wrote {DRIFTERS_PARQUET} ({len(df)} rows, {df['drifter'].nunique()} drifters)")


if __name__ == "__main__":
    main()
