"""The `buoycast` command: forecast a buoy's wave height from its live NDBC data.

    buoycast forecast 46041        # 3 h and 6 h forecasts from the latest readings
    buoycast stations              # buoys it knows about (including Canadian ones)

The trained models ship inside the package, so nothing else is needed but internet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

QC_NAMES = {1: "pass", 2: "not checked", 3: "suspect", 4: "fail", 9: "no reading"}


def _frame_from_file(path: Path) -> pd.DataFrame:
    from .parse import parse_stdmet, to_hourly

    hourly = to_hourly(parse_stdmet(path))
    return hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq="1h", name="time"))


def forecast_report(station: str, frame: pd.DataFrame) -> str:
    from . import live, qc

    models = live.Models.load()
    last = live.latest(frame)
    frame = frame.loc[:last]
    age = (pd.Timestamp.now(tz="UTC") - last) / pd.Timedelta(hours=1)
    meta = live.LIVE_STATIONS.get(station)
    name = f"{station} ({meta['name']})" if meta else station
    lines = [f"Buoy {name}", f"Latest wave reading: {last:%Y-%m-%d %H:%M} UTC ({age:.1f} h ago)"]
    if age > live.FRESH_HOURS:
        lines.append("Warning: this buoy has not reported recently, so the forecasts are old.")
    if station not in live.TRAINED:
        lines.append(
            "Note: the model was not trained on this buoy, so its forecast is less reliable."
        )
    flags = qc.run_platform(frame).iloc[-1]
    now = ", ".join(
        f"{label} {frame[col].iloc[-1]:.1f} {unit} ({QC_NAMES[int(flags[f'{col}_overall'])]})"
        for col, label, unit in (
            ("wvht", "waves", "m"),
            ("wspd", "wind", "m/s"),
            ("pres", "pressure", "hPa"),
        )
        if pd.notna(frame[col].iloc[-1])
    )
    lines.append(f"Now: {now}")
    rows = []
    for h in live.HORIZONS:
        t = live.forecast_frame(station, frame, models, h)
        t = t[t["issued"] == last]
        if t.empty:
            continue
        r = t.iloc[0]
        rows.append(
            f"  {r['valid']:%Y-%m-%d %H:%M} UTC (+{h} h)   MLP {r['mlp']:.2f} m   "
            f"linear {r['linear']:.2f} m   persistence {r['persistence']:.2f} m"
        )
    if not rows:
        lines.append("No forecast: the model needs 24 hours of readings without long gaps.")
    else:
        lines.append("Forecast wave height:")
        lines += rows
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="buoycast", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast one buoy from its live NDBC data")
    f.add_argument("station", help="NDBC station id, e.g. 46041")
    f.add_argument("--file", type=Path, help="use a saved realtime file instead of downloading")
    sub.add_parser("stations", help="list the buoys the demo covers")
    args = ap.parse_args(argv)

    from . import live

    if args.cmd == "stations":
        for sid, meta in live.LIVE_STATIONS.items():
            trained = "trained on" if sid in live.TRAINED else "not in training"
            print(f"{sid}  {meta['name']}  [{trained}]")
        return 0

    if args.file:
        frame = _frame_from_file(args.file)
    else:
        import httpx

        with httpx.Client(follow_redirects=True, headers={"User-Agent": "buoycast"}) as client:
            frame = live.fetch_station(args.station, client)
        if frame is None:
            print(f"Could not get live data for station {args.station}.", file=sys.stderr)
            return 1
    if frame["wvht"].notna().sum() == 0:
        print(f"Station {args.station} has no wave height readings.", file=sys.stderr)
        return 1
    print(forecast_report(args.station, frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
