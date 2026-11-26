import json

import httpx
import numpy as np
import pandas as pd

from buoycast import live
from buoycast.site import build_site

HEADER = (
    "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE\n"
    "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft\n"
)


def realtime_text(hours: int, end: pd.Timestamp) -> str:
    """A fake NDBC realtime file: newest row first, 'MM' for missing, wave height at :50."""
    lines = []
    for k in range(hours):
        t = end - pd.Timedelta(hours=k)
        wave = 2.0 + np.sin(k / 6)
        lines.append(
            f"{t:%Y %m %d %H} 00 180 8.0 10.0   MM    MM    MM  MM 1010.0  9.0  9.0 MM MM MM MM"
        )
        wave_time = t - pd.Timedelta(minutes=10)
        lines.append(
            f"{wave_time:%Y %m %d %H %M} 180 8.0 10.0 {wave:5.1f}     8   5.0 200 1010.0  9.0  9.0 MM MM MM MM"
        )
    return HEADER + "\n".join(lines) + "\n"


def frame(hours: int, end: pd.Timestamp) -> pd.DataFrame:
    raw = live.parse_stdmet(realtime_text(hours, end))
    return live.to_hourly(raw)


def now_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("1h")


def test_realtime_file_parses_and_missing_marker_becomes_nan():
    f = frame(30, now_hour())
    assert f["wvht"].notna().sum() >= 29
    assert f["wvht"].max() < 4  # no 'MM' turned into a number
    assert f["pres"].between(1000, 1020).all()


def test_fetch_failure_gives_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not found")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert live.fetch_station("46041", client) is None


def test_forecast_frame_includes_the_newest_hour_without_an_outcome():
    models = live.Models.load()
    f = frame(60, now_hour())
    t = live.forecast_frame("46041", f, models, 3)
    newest = t.iloc[-1]
    assert newest["issued"] == live.latest(f)
    assert np.isnan(newest["observed"])
    assert newest["valid"] == newest["issued"] + pd.Timedelta(hours=3)
    assert t[["persistence", "linear", "mlp"]].notna().all().all()


def test_live_site_has_a_current_forecast_and_extends_past_the_last_reading(tmp_path):
    end = now_hour()
    frames = {"46041": frame(80, end), "46146": frame(25, end)}
    build_site(tmp_path, use_live=True, fetch=lambda: frames)
    meta = json.loads((tmp_path / "data" / "index.json").read_text())
    assert meta["site"]["mode"] == "live"
    by_id = {s["id"]: s for s in meta["stations"]}
    assert by_id["46041"]["current_forecast"] and by_id["46041"]["trained"]
    assert by_id["46146"]["current_forecast"] and not by_id["46146"]["trained"]
    assert "46041" in meta["hindcast"]["3"]["stations"]
    d = json.loads((tmp_path / "data" / "46041.json").read_text())
    assert len(d["forecast"]["h3"]["mlp"]) + d["forecast"]["offset"] == d["hours"] + 3
    assert "mode" in (tmp_path / "config.js").read_text()


def test_stale_buoy_is_marked_offline(tmp_path):
    old = now_hour() - pd.Timedelta(hours=20)
    build_site(tmp_path, use_live=True, fetch=lambda: {"46041": frame(60, old)})
    meta = json.loads((tmp_path / "data" / "index.json").read_text())
    assert not meta["stations"][0]["online"]
    assert not meta["stations"][0]["current_forecast"]


def test_site_falls_back_to_the_sample_when_nothing_downloads(tmp_path):
    build_site(tmp_path, use_live=True, fetch=lambda: {})
    meta = json.loads((tmp_path / "data" / "index.json").read_text())
    assert meta["site"]["mode"] == "sample-fallback"
    assert len(meta["stations"]) == 6 and len(meta["drifters"]) == 5


def test_cli_forecast_from_a_saved_file(tmp_path, capsys):
    from buoycast.cli import main

    path = tmp_path / "46041.txt"
    path.write_text(realtime_text(60, now_hour()))
    assert main(["forecast", "46041", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "(+3 h)" in out and "(+6 h)" in out and "not trained" not in out
    assert main(["forecast", "46146", "--file", str(path)]) == 0
    assert "not trained" in capsys.readouterr().out
