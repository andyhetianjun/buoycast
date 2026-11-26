"""Parse NDBC standard meteorological files and build a clean hourly table."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

# Column renames across the years the format has been used.
_RENAMES = {"WD": "WDIR", "BAR": "PRES", "YYYY": "YY"}

# NDBC marks a missing reading by filling the field with 9s. Each column has its own code.
MISSING_CODES: dict[str, float] = {
    "WDIR": 999,
    "WSPD": 99.0,
    "GST": 99.0,
    "WVHT": 99.0,
    "DPD": 99.0,
    "APD": 99.0,
    "MWD": 999,
    "PRES": 9999.0,
    "ATMP": 999.0,
    "WTMP": 999.0,
    "DEWP": 999.0,
}

KEEP = ["wvht", "dpd", "apd", "wspd", "gst", "wdir", "pres", "atmp", "wtmp"]


def parse_stdmet(source: str | bytes | Path) -> pd.DataFrame:
    """Read one stdmet file (text, gzip bytes or a path) into a UTC-indexed frame.

    Missing-value codes are turned into NaN. Rows are kept at their original
    time resolution (10 or 60 minutes depending on the station and year).
    """
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, bytes):
        if source[:2] == b"\x1f\x8b":
            source = gzip.decompress(source)
        source = source.decode("ascii", errors="replace")

    lines = [ln for ln in source.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")
    header = lines[0].lstrip("#").split()
    # The unit line (#yr mo dy ...) exists in newer files and must be skipped.
    body = [ln for ln in lines[1:] if not ln.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", header=None, names=header)
    df = df.rename(columns=_RENAMES)

    year = df["YY"].astype(int)
    year = year.where(year >= 100, year + 1900)
    minute = df["mm"] if "mm" in df.columns else 0
    stamp = pd.to_datetime(
        {
            "year": year,
            "month": df["MM"],
            "day": df["DD"],
            "hour": df["hh"],
            "minute": minute,
        },
        utc=True,
    )
    out = pd.DataFrame(index=pd.DatetimeIndex(stamp, name="time"))
    for col, code in MISSING_CODES.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, copy=True)
            # Compare with a tolerance: some files print 99.00 or 99.0.
            values[np.isclose(values, code)] = np.nan
            out[col.lower()] = values
    for col in KEEP:
        if col not in out.columns:
            out[col] = np.nan
    return out[KEEP].sort_index()


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Average to hourly values on a complete hourly grid.

    Wind direction is averaged as a vector (via sin and cos), since 359 and 1
    degrees should average to 0, not 180. Hours with no readings stay NaN.
    """
    raw = raw[~raw.index.duplicated(keep="last")]
    rad = np.deg2rad(raw["wdir"])
    parts = raw.drop(columns="wdir").assign(_s=np.sin(rad), _c=np.cos(rad))
    hourly = parts.resample("1h").mean()
    hourly["wdir"] = np.rad2deg(np.arctan2(hourly.pop("_s"), hourly.pop("_c"))) % 360
    hourly = hourly[KEEP]
    # Drop leading and trailing hours with no wave reading at all.
    valid = hourly["wvht"].dropna().index
    if len(valid) == 0:
        return hourly.iloc[0:0]
    return hourly.loc[valid[0] : valid[-1]]
