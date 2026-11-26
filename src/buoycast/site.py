"""Build the static demo site: the map page plus all its data as plain JSON files.

Usage: uv run buoycast-site [--out site]

Nothing is computed in the browser or on a server. This script runs the models and
the quality checks once, over the committed sample, and writes the results next to
a copy of the map page (web/). Any static host can serve the folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import drifters, qc, verify
from .features import TEST_START, load_hourly
from .paths import (
    QC_SUMMARY_JSON,
    ROOT,
    STATIONS,
    VERIFICATION_JSON,
    WEB_DIR,
)

SITE_DIR = ROOT / "site"
QC_VARS = ("wvht", "wspd", "pres", "wtmp")  # shown as strips under the chart
EVENT_VARS = ("wvht", "wspd", "gst", "pres", "atmp", "wtmp")
TEST_NAMES = {
    "gross_range": "gross range",
    "spike": "spike",
    "rate_of_change": "rate of change",
    "flat_line": "flat line",
    "gap": "gap",
}


def _series(s: pd.Series, digits: int) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), digits) for v in s.to_numpy()]


def _flag_string(flags: pd.Series) -> str:
    """One character per hour: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing."""
    return "".join(str(int(v)) for v in flags.to_numpy())


def flag_events(flags: pd.DataFrame, limit: int = 40) -> tuple[list[dict], int]:
    """Runs of consecutive suspect/fail readings, worst and longest first.

    Each event has the index of its first hour in the station's hourly grid.
    """
    events = []
    for var in EVENT_VARS:
        for test in TEST_NAMES:
            col = flags.get(f"{var}_{test}")
            if col is None:
                continue
            bad = col.to_numpy() >= qc.SUSPECT
            bad &= col.to_numpy() != qc.MISSING
            start = None
            for i, b in enumerate([*bad, False]):
                if b and start is None:
                    start = i
                elif not b and start is not None:
                    worst = int(col.to_numpy()[start:i].max())
                    events.append(
                        {"start": start, "hours": i - start, "var": var,
                         "test": TEST_NAMES[test], "flag": worst}
                    )  # fmt: skip
                    start = None
    events.sort(key=lambda e: (-e["flag"], -e["hours"], e["start"]))
    return events[:limit], len(events)


def station_payload(station: str, frame: pd.DataFrame, leads: dict[int, pd.DataFrame]) -> dict:
    flags = qc.run_platform(frame)
    start = frame.index[0]
    events, n_events = flag_events(flags)
    payload = {
        "id": station,
        **STATIONS[station],
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hours": len(frame),
        "wvht": _series(frame["wvht"], 2),
        "wspd": _series(frame["wspd"], 1),
        "pres": _series(frame["pres"], 1),
        "wtmp": _series(frame["wtmp"], 1),
        "qc": {v: _flag_string(flags[f"{v}_overall"]) for v in QC_VARS},
        "events": events,
        "n_events": n_events,
    }
    forecasts: dict[str, dict] = {}
    offset = None
    for lead, table in leads.items():
        t = table[table["station"] == station]
        if t.empty:
            continue
        idx = ((t["valid"] - start) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        offset = int(idx.min()) if offset is None else min(offset, int(idx.min()))
        forecasts[str(lead)] = {"idx": idx, "t": t}
    if forecasts:
        out = {"offset": offset}
        for lead, f in forecasts.items():
            size = len(frame) - offset
            arrays = {m: [None] * size for m in ("mlp", "linear", "persistence")}
            for m in arrays:
                for i, v in zip(f["idx"], f["t"][m].to_numpy(), strict=True):
                    arrays[m][i - offset] = round(float(v), 2)
            out[f"h{lead}"] = arrays
        payload["forecast"] = out
    return payload


def build_site(out: Path = SITE_DIR) -> dict[str, int]:
    """Write the site folder. Returns the size in bytes of each data file."""
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy(WEB_DIR / name, out / name)
    built = datetime.now(UTC).strftime("%Y-%m-%d")
    (out / "config.js").write_text(
        f'window.BUOYCAST_STATIC = true;\nwindow.BUOYCAST_BUILT = "{built}";\n'
    )

    hourly = load_hourly()
    leads = {h: verify.forecast_table(horizon=h) for h in verify.HORIZONS}
    index = []
    for station, frame in hourly.items():
        payload = station_payload(station, frame, leads)
        (out / "data" / f"{station}.json").write_text(json.dumps(payload, separators=(",", ":")))
        index.append(
            {
                "id": station,
                "name": STATIONS[station]["name"],
                "lat": STATIONS[station]["lat"],
                "lon": STATIONS[station]["lon"],
                "start": payload["start"],
                "hours": payload["hours"],
                "has_forecast": "forecast" in payload,
            }
        )
    tracks = []
    for name, frame in drifters.load_drifters().items():
        flags = pd.concat([qc.run_platform(frame, qc.DRIFTER_CONFIG), qc.run_track(frame)], axis=1)
        flagged = int(
            (flags.drop(columns=[c for c in flags if c.endswith("_overall")]) >= 3).sum().sum()
        )
        tracks.append(
            {
                "id": name,
                "coords": [
                    [round(x, 3), round(y, 3)]
                    for x, y in zip(frame["lat"], frame["lon"], strict=True)
                ],
                "fixes": len(frame),
                "flagged": flagged,
                "first": frame.index[0].strftime("%Y-%m-%d"),
                "last": frame.index[-1].strftime("%Y-%m-%d"),
            }
        )
    meta = {
        "built": built,
        "test_start": TEST_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations": index,
        "drifters": tracks,
        "verification": json.loads(VERIFICATION_JSON.read_text()),
        "qc": json.loads(QC_SUMMARY_JSON.read_text()),
    }
    (out / "data" / "index.json").write_text(json.dumps(meta, separators=(",", ":")))
    return {p.name: p.stat().st_size for p in sorted((out / "data").glob("*.json"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SITE_DIR)
    args = ap.parse_args()
    sizes = build_site(args.out)
    total = sum(sizes.values())
    print(f"wrote {args.out} ({len(sizes)} data files, {total / 1e6:.1f} MB before compression)")


if __name__ == "__main__":
    main()
