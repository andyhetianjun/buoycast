"""Automatic quality checks on sensor readings, in the style of the QARTOD manuals.

QARTOD (Quality Assurance of Real-Time Oceanographic Data) is a set of public
guides from NOAA's IOOS programme that describe simple, explainable tests for
ocean sensors. Each test gives every reading one flag:

    1 pass       2 not evaluated (not enough neighbours to judge)
    3 suspect    4 fail          9 missing

The tests here work on one variable of one platform at a time and use real
elapsed time, not row counts, so they behave the same on hourly buoy data and on
a drifter that reports every six hours:

    gross_range     value outside what the sensor or the ocean can produce
    spike           value far from the line between its two neighbours
    rate_of_change  value moved too fast since the previous reading
    flat_line       value has not changed for too long (a stuck sensor)
    gap             the reading arrived after a long silence

A platform that moves (drifter, ship) also gets two position tests: the
position must be on Earth, and the speed implied by consecutive fixes must be
believable.

The thresholds in `BUOY_CONFIG` are my own rough choices, not the official
limits for any station. They are in one place so they are easy to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import distance_km

PASS, NOT_EVALUATED, SUSPECT, FAIL, MISSING = 1, 2, 3, 4, 9
TESTS = ("gross_range", "spike", "rate_of_change", "flat_line", "gap")
TRACK_TESTS = ("location_range", "track_speed")


@dataclass(frozen=True)
class VarConfig:
    """Thresholds for one variable. Pairs are (suspect, fail)."""

    fail_range: tuple[float, float]  # outside this: fail
    suspect_range: tuple[float, float]  # outside this: suspect
    spike: tuple[float, float]  # |value - line between neighbours|
    rate: tuple[float, float]  # |change| per hour
    flat_hours: tuple[float, float]  # unchanged for this long
    flat_eps: float  # changes up to this size count as "unchanged"
    gap_hours: tuple[float, float]  # silence before this reading
    max_gap_hours: float = 3.0  # neighbours further apart than this are not compared


BUOY_CONFIG: dict[str, VarConfig] = {
    "wvht": VarConfig((0, 25), (0, 15), (1.0, 2.5), (1.5, 4.0), (12, 24), 0.0, (6, 48)),
    "wspd": VarConfig((0, 50), (0, 35), (8, 16), (8, 16), (12, 24), 0.0, (6, 48)),
    "gst": VarConfig((0, 70), (0, 45), (12, 24), (12, 24), (12, 24), 0.0, (6, 48)),
    "dpd": VarConfig((0, 30), (0, 25), (10, 20), (12, 24), (12, 24), 0.0, (6, 48)),
    "pres": VarConfig((850, 1090), (940, 1060), (5, 12), (8, 20), (12, 24), 0.0, (6, 48)),
    "atmp": VarConfig((-60, 60), (-40, 45), (5, 10), (8, 15), (24, 72), 0.0, (6, 48)),
    "wtmp": VarConfig((-2, 40), (-2, 35), (3, 6), (3, 6), (48, 120), 0.0, (6, 48)),
}

# Drifters report every 6 hours, so "neighbours" can be up to 12 hours apart.
DRIFTER_CONFIG: dict[str, VarConfig] = {
    "sst": VarConfig(
        (-2, 40), (-2, 35), (1.5, 3.0), (0.5, 1.0), (48, 120), 0.0, (18, 72), max_gap_hours=12.0
    ),
}


@dataclass(frozen=True)
class TrackConfig:
    """Position tests. Speeds are km/h as (suspect, fail)."""

    speed_kmh: tuple[float, float] = (10.0, 25.0)  # a drifter; a ship would need more
    max_gap_hours: float = 12.0


def _hours(index: pd.DatetimeIndex) -> np.ndarray:
    return (index - index[0]).total_seconds().to_numpy() / 3600.0 if len(index) else np.array([])


def _grade(value: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """PASS, SUSPECT or FAIL by size; NaN stays NaN so callers can mark it."""
    out = np.full(value.shape, PASS, dtype=np.int8)
    out[value > limits[0]] = SUSPECT
    out[value > limits[1]] = FAIL
    return out


def _blank(series: pd.Series, code: int = NOT_EVALUATED) -> pd.Series:
    """A flag series for `series`: `code` where a reading exists, MISSING where it does not."""
    out = pd.Series(code, index=series.index, dtype="int8")
    out[series.isna()] = MISSING
    return out


def gross_range(s: pd.Series, cfg: VarConfig) -> pd.Series:
    out = _blank(s, PASS)
    v = s.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    flags = out.to_numpy().copy()
    flags[ok & ((v < cfg.suspect_range[0]) | (v > cfg.suspect_range[1]))] = SUSPECT
    flags[ok & ((v < cfg.fail_range[0]) | (v > cfg.fail_range[1]))] = FAIL
    return pd.Series(flags, index=s.index, dtype="int8")


def spike(s: pd.Series, cfg: VarConfig) -> pd.Series:
    """Distance from the straight line between the previous and next reading, for lone outliers."""
    out = _blank(s)
    v = s.dropna()
    if len(v) < 3:
        return out
    t = _hours(v.index)
    x = v.to_numpy(dtype=float)
    t0, t1, t2 = t[:-2], t[1:-1], t[2:]
    ref = x[:-2] + (x[2:] - x[:-2]) * (t1 - t0) / (t2 - t0)
    dev = np.abs(x[1:-1] - ref)
    # A point is a spike only if it sticks out more than its neighbours differ from each
    # other. Without this, the readings either side of a spike get flagged too, and a
    # plain step (which the rate test is for) would be flagged as a spike.
    dev[dev <= np.abs(x[2:] - x[:-2])] = 0.0
    flags = _grade(dev, cfg.spike)
    flags[(t2 - t0) > 2 * cfg.max_gap_hours] = NOT_EVALUATED
    out.loc[v.index[1:-1]] = flags
    return out


def rate_of_change(s: pd.Series, cfg: VarConfig) -> pd.Series:
    """Change per hour since the previous reading."""
    out = _blank(s)
    v = s.dropna()
    if len(v) < 2:
        return out
    t = _hours(v.index)
    x = v.to_numpy(dtype=float)
    dt = np.diff(t)
    rate = np.abs(np.diff(x)) / dt
    flags = _grade(rate, cfg.rate)
    flags[dt > cfg.max_gap_hours] = NOT_EVALUATED
    out.loc[v.index[1:]] = flags
    return out


def flat_line(s: pd.Series, cfg: VarConfig) -> pd.Series:
    """How long the value has stayed the same, counted up to and including this reading."""
    out = _blank(s)
    v = s.dropna()
    if v.empty:
        return out
    t = _hours(v.index)
    x = v.to_numpy(dtype=float)
    same = np.r_[False, (np.abs(np.diff(x)) <= cfg.flat_eps) & (np.diff(t) <= cfg.max_gap_hours)]
    # A new run starts wherever a reading is not "the same as the last".
    start_idx = np.maximum.accumulate(np.where(~same, np.arange(len(x)), 0))
    duration = t - t[start_idx]
    flags = np.full(len(x), PASS, dtype=np.int8)
    flags[duration >= cfg.flat_hours[0]] = SUSPECT
    flags[duration >= cfg.flat_hours[1]] = FAIL
    out.loc[v.index] = flags
    return out


def gap(s: pd.Series, cfg: VarConfig) -> pd.Series:
    """Flag a reading that arrives after a long silence in this variable."""
    out = _blank(s)
    v = s.dropna()
    if len(v) < 2:
        return out
    silence = np.diff(_hours(v.index))
    out.loc[v.index[1:]] = _grade(silence, cfg.gap_hours)
    out.loc[v.index[0]] = PASS
    return out


_TEST_FUNCS = {
    "gross_range": gross_range,
    "spike": spike,
    "rate_of_change": rate_of_change,
    "flat_line": flat_line,
    "gap": gap,
}


def combine(flags: list[pd.Series]) -> pd.Series:
    """Overall flag: the worst of pass/suspect/fail; missing if there is no reading."""
    stack = np.stack([f.to_numpy() for f in flags])
    worst = np.where(np.isin(stack, (PASS, SUSPECT, FAIL)), stack, PASS).max(axis=0)
    worst = np.where((stack == MISSING).all(axis=0), MISSING, worst)
    return pd.Series(worst.astype("int8"), index=flags[0].index)


def run_variable(s: pd.Series, cfg: VarConfig) -> pd.DataFrame:
    """All five tests on one variable, plus the combined flag in column `overall`."""
    cols = {name: fn(s, cfg) for name, fn in _TEST_FUNCS.items()}
    out = pd.DataFrame(cols)
    out["overall"] = combine(list(cols.values()))
    return out


def run_platform(df: pd.DataFrame, config: dict[str, VarConfig] = BUOY_CONFIG) -> pd.DataFrame:
    """Flags for every configured variable present in `df`, as columns `<var>_<test>`."""
    parts = []
    for var, cfg in config.items():
        if var in df.columns:
            parts.append(run_variable(df[var], cfg).add_prefix(f"{var}_"))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def run_track(df: pd.DataFrame, cfg: TrackConfig = TrackConfig()) -> pd.DataFrame:
    """Position tests for a moving platform with `lat` and `lon` columns."""
    lat, lon = df["lat"], df["lon"]
    present = lat.notna() & lon.notna()
    loc = pd.Series(PASS, index=df.index, dtype="int8")
    loc[(lat.abs() > 90) | (lon.abs() > 180)] = FAIL
    loc[~present] = MISSING

    speed = pd.Series(NOT_EVALUATED, index=df.index, dtype="int8")
    speed[~present] = MISSING
    ok = df[present & (loc == PASS)]
    if len(ok) >= 2:
        hours = _hours(ok.index)
        dt = np.diff(hours)
        km = distance_km(
            ok["lat"].to_numpy()[:-1],
            ok["lon"].to_numpy()[:-1],
            ok["lat"].to_numpy()[1:],
            ok["lon"].to_numpy()[1:],
        )
        flags = _grade(km / dt, cfg.speed_kmh)
        flags[dt > cfg.max_gap_hours] = NOT_EVALUATED
        speed.loc[ok.index[1:]] = flags
        speed.loc[ok.index[0]] = NOT_EVALUATED
    return pd.DataFrame({"location_range": loc, "track_speed": speed})


def split_column(col: str) -> tuple[str, str]:
    """'wvht_flat_line' -> ('wvht', 'flat_line'); track columns belong to 'position'."""
    if col in TRACK_TESTS:
        return "position", col
    for test in (*TESTS, "overall"):
        if col.endswith("_" + test):
            return col[: -len(test) - 1], test
    return col, ""


def summarize(flags: pd.DataFrame) -> list[dict]:
    """Per variable and test: how many readings got each flag."""
    rows = []
    for col in flags.columns:
        var, test = split_column(col)
        counts = flags[col].value_counts()
        rows.append(
            {
                "variable": var,
                "test": test,
                "n_pass": int(counts.get(PASS, 0)),
                "n_not_evaluated": int(counts.get(NOT_EVALUATED, 0)),
                "n_suspect": int(counts.get(SUSPECT, 0)),
                "n_fail": int(counts.get(FAIL, 0)),
                "n_missing": int(counts.get(MISSING, 0)),
            }
        )
    return rows
