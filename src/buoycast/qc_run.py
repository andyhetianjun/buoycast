"""Run the quality checks on the committed samples and store the flags next to the data.

Usage: uv run buoycast-qc

Writes:
  data/sample/qc_flags.parquet      one row per row of hourly.parquet, one flag column per test
  data/sample/drifters_qc.parquet   the drifter sample with its flags
  artifacts/qc_summary.json         how many readings each test caught
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import qc
from .drifters import load_drifters
from .features import load_hourly
from .paths import (
    DRIFTERS_PARQUET,
    DRIFTERS_QC_PARQUET,
    QC_FLAGS_PARQUET,
    QC_SUMMARY_JSON,
    SAMPLE_PARQUET,
)


def station_flags(data=SAMPLE_PARQUET) -> pd.DataFrame:
    """Flags for every station, on the same (station, time) rows as the sample file."""
    sample = pd.read_parquet(data, columns=["station", "time"])
    parts = []
    for station, frame in load_hourly(data).items():
        flags = qc.run_platform(frame)
        flags.insert(0, "station", station)
        parts.append(flags.reset_index())
    flags = pd.concat(parts, ignore_index=True)
    flags["station"] = flags["station"].astype("category")
    key = sample.assign(station=sample["station"].astype(str))
    out = key.merge(flags.assign(station=flags["station"].astype(str)), on=["station", "time"])
    out["station"] = out["station"].astype("category")
    return out


def drifter_flags(path=DRIFTERS_PARQUET) -> pd.DataFrame:
    """Drifter rows with variable flags and position flags added as columns."""
    parts = []
    for name, frame in load_drifters(path).items():
        flags = pd.concat([qc.run_platform(frame, qc.DRIFTER_CONFIG), qc.run_track(frame)], axis=1)
        parts.append(frame.join(flags).assign(drifter=name).reset_index())
    return pd.concat(parts, ignore_index=True)


FAULTS = ("spike", "flat_line", "out_of_range")
INJECT_VARS = ("wvht", "wspd", "pres")


def injection_test(data=SAMPLE_PARQUET, seed: int = 0, n: int = 40) -> dict:
    """Plant known faults in real series and count how many the tests find.

    This only says whether the tests notice obvious faults I created. It does not say
    how well they would do on the faults real sensors produce.
    """
    rng = np.random.default_rng(seed)
    found = {f: [0, 0] for f in FAULTS}  # [detected, planted]
    false_alarm = [0, 0]  # [flagged, checked] on untouched readings
    hourly = load_hourly(data)
    for frame in hourly.values():
        for var in INJECT_VARS:
            cfg = qc.BUOY_CONFIG[var]
            clean = frame[var]
            valid = np.flatnonzero(clean.notna().to_numpy())
            if len(valid) < 500:
                continue
            idx = rng.choice(valid[30:-30], size=3 * n, replace=False)
            groups = {"spike": idx[:n], "flat_line": idx[n : 2 * n], "out_of_range": idx[2 * n :]}
            for fault, points in groups.items():
                x = clean.copy()
                points = _keep_apart(points, 40)
                for p in points:
                    if fault == "spike":
                        x.iloc[p] += 2.5 * cfg.spike[1]
                    elif fault == "out_of_range":
                        x.iloc[p] = cfg.fail_range[1] * 1.5
                    else:  # a stuck sensor: 30 hours at the value it had
                        x.iloc[p : p + 30] = x.iloc[p]
                flags = qc.run_variable(x, cfg)
                for p in points:
                    # Read the test at the point where it should react.
                    where = p + 29 if fault == "flat_line" else p
                    got = flags[fault if fault != "out_of_range" else "gross_range"].iloc[where]
                    found[fault][0] += int(got in (qc.SUSPECT, qc.FAIL))
                    found[fault][1] += 1
                touched = np.zeros(len(x), dtype=bool)
                for p in points:
                    touched[max(0, p - 2) : p + 33] = True
                ok = clean.notna().to_numpy() & ~touched
                false_alarm[0] += int((flags["overall"].to_numpy()[ok] >= qc.SUSPECT).sum())
                false_alarm[1] += int(ok.sum())
    track = _track_injection(rng)
    return {
        "drifter_planted": track[1],
        "drifter_detected": track[0],
        "planted": {f: v[1] for f, v in found.items()},
        "detected": {f: v[0] for f, v in found.items()},
        "recall": {f: round(v[0] / v[1], 3) for f, v in found.items()},
        "untouched_readings_flagged": false_alarm[0],
        "untouched_readings_checked": false_alarm[1],
    }


def _track_injection(rng: np.random.Generator, n: int = 20) -> tuple[dict, dict]:
    """Move single drifter fixes 3 degrees north and add 6 C to single SST readings."""
    detected = {"position_jump": 0, "sst_spike": 0}
    planted = {"position_jump": 0, "sst_spike": 0}
    for frame in load_drifters().values():
        valid = np.arange(5, len(frame) - 5)
        points = _keep_apart(rng.choice(valid, size=2 * n, replace=False), 8)
        for i, p in enumerate(points):
            x = frame.copy()
            if i % 2 == 0:
                x.iloc[p, x.columns.get_loc("lat")] += 3.0
                got = qc.run_track(x)["track_speed"].iloc[p]
                key = "position_jump"
            else:
                x.iloc[p, x.columns.get_loc("sst")] += 6.0
                got = qc.run_variable(x["sst"], qc.DRIFTER_CONFIG["sst"])["spike"].iloc[p]
                key = "sst_spike"
            planted[key] += 1
            detected[key] += int(got in (qc.SUSPECT, qc.FAIL))
    return detected, planted


def _keep_apart(points: np.ndarray, gap: int) -> list[int]:
    """Sorted points with at least `gap` between them, so planted faults do not overlap."""
    kept: list[int] = []
    for p in sorted(int(x) for x in points):
        if not kept or p - kept[-1] >= gap:
            kept.append(p)
    return kept


def main() -> None:
    flags = station_flags()
    flags.to_parquet(QC_FLAGS_PARQUET, index=False, compression="zstd")
    drifters = drifter_flags()
    drifters.to_parquet(DRIFTERS_QC_PARQUET, index=False, compression="zstd")

    cols = [c for c in flags.columns if c not in ("station", "time")]
    per_station = {
        str(s): qc.summarize(g[cols]) for s, g in flags.groupby("station", observed=True)
    }
    summary = {
        "stations": {
            "rows": len(flags),
            "by_test": qc.summarize(flags[cols]),
            "by_station": per_station,
        },
        "drifters": {
            "rows": len(drifters),
            "by_test": qc.summarize(drifters[[c for c in drifters.columns if _is_flag(c)]]),
        },
        "injection_test": injection_test(),
    }
    QC_SUMMARY_JSON.write_text(json.dumps(summary, indent=1) + "\n")
    print(format_summary(summary))


def _is_flag(col: str) -> bool:
    return col in qc.TRACK_TESTS or qc.split_column(col)[1] != ""


def format_summary(summary: dict) -> str:
    lines = ["| Variable | Test | Suspect | Fail | Not evaluated |", "|---|---|---|---|---|"]
    for r in summary["stations"]["by_test"]:
        if r["test"] and r["test"] != "overall":
            lines.append(
                f"| {r['variable']} | {r['test']} | {r['n_suspect']} | {r['n_fail']} | "
                f"{r['n_not_evaluated']} |"
            )
    lines.append(f"\nstation rows: {summary['stations']['rows']}")
    lines.append(f"drifter rows: {summary['drifters']['rows']}")
    for r in summary["drifters"]["by_test"]:
        if r["test"] != "overall":
            lines.append(
                f"drifter {r['variable']} {r['test']}: suspect {r['n_suspect']} fail {r['n_fail']}"
            )
    lines.append(f"injection test: {summary['injection_test']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
