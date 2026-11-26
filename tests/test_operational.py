import numpy as np
import pandas as pd
import pytest

from buoycast import operational as op

BULLETIN = """  Location : 44027      (44.28N  67.31W)
  Model    : spectral resolution for points
  Cycle    : 20241230 12 UTC
 +-------+-----------+-----------------+-----------------+
 | day & |  Hst  n x |    Hs   Tp  dir |    Hs   Tp  dir |
 |  hour |  (m)  - - |    (m)  (s) (d) |    (m)  (s) (d) |
 +-------+-----------+-----------------+-----------------+
 | 30 12 | 1.00  3   |   0.90  7.7 353 |   0.53  5.2 233 |
 | 30 15 | 1.20  4   |   0.90  7.7 353 | * 0.53  5.2 233 |
 | 30 18 | 1.50  4   |   0.90  7.7 353 |                 |
 | 31  0 | 2.00  2   |   0.90  7.7 353 |                 |
 |  1  0 | 2.50  2   |   0.90  7.7 353 |                 |
 |  1  6 | 3.10  2   |   0.90  7.7 353 |                 |
 +-------+-----------+-----------------+-----------------+
"""


def test_parse_reads_total_height_and_valid_time():
    df = op.parse_bulletin(BULLETIN)
    assert list(df["hs"]) == [1.0, 1.2, 1.5, 2.0, 2.5, 3.1]
    assert (df["cycle"] == pd.Timestamp("2024-12-30 12:00", tz="UTC")).all()
    assert df["valid"].iloc[1] == pd.Timestamp("2024-12-30 15:00", tz="UTC")


def test_parse_rolls_over_the_month_and_the_year():
    df = op.parse_bulletin(BULLETIN)
    assert df["valid"].iloc[3] == pd.Timestamp("2024-12-31 00:00", tz="UTC")
    assert df["valid"].iloc[4] == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert df["valid"].is_monotonic_increasing


def test_parse_rejects_text_that_is_not_a_bulletin():
    with pytest.raises(ValueError):
        op.parse_bulletin("<html>not found</html>")


def test_extract_leads_keeps_only_the_scored_hours():
    df = op.extract_leads(op.parse_bulletin(BULLETIN), "44027")
    assert list(df["lead"]) == [3, 6]
    assert list(df["hs"]) == [1.2, 1.5]
    assert (df["station"] == "44027").all()


def test_bulletin_url_uses_the_cycle_date_and_hour():
    url = op.bulletin_url(pd.Timestamp("2024-10-05 06:00", tz="UTC"), "46041")
    assert url.endswith("gfs.20241005/06/wave/station/bulls.t06z/gfswave.46041.bull")


def _buoycast(issued, station="44027", lead=3):
    n = len(issued)
    issued = pd.DatetimeIndex(issued)
    return pd.DataFrame(
        {
            "station": pd.Categorical([station] * n),
            "issued": issued,
            "valid": issued + pd.Timedelta(hours=lead),
            "observed": np.full(n, 2.0),
            "persistence": np.full(n, 1.5),
            "mlp": np.full(n, 1.9),
        }
    )


def _gfs(cycles, hs, station="44027", lead=3):
    cycles = pd.DatetimeIndex(cycles)
    return pd.DataFrame(
        {
            "station": station,
            "cycle": cycles,
            "lead": lead,
            "valid": cycles + pd.Timedelta(hours=lead),
            "hs": hs,
        }
    )


def test_align_matches_only_the_same_station_issue_time_and_lead():
    t = pd.date_range("2024-10-01", periods=4, freq="6h", tz="UTC")
    b = _buoycast(t)
    gfs = pd.concat(
        [
            _gfs(t[:3], [1.0, 1.0, 1.0]),  # the last cycle is missing
            _gfs(t, [9.0] * 4, station="46041"),  # other buoy
            _gfs(t, [9.0] * 4, lead=6),  # other lead
        ]
    )
    out = op.align(b, gfs, 3)
    assert len(out) == 3
    assert (out["gfswave"] == 1.0).all()
    assert list(out["issued"]) == list(t[:3])


def test_align_ignores_hours_that_are_not_a_cycle_time():
    t = pd.date_range("2024-10-01", periods=12, freq="1h", tz="UTC")
    out = op.align(_buoycast(t), _gfs(t[[0, 6]], [1.0, 1.0]), 3)
    assert list(out["issued"]) == [t[0], t[6]]


def test_align_drops_missing_values():
    t = pd.date_range("2024-10-01", periods=3, freq="6h", tz="UTC")
    out = op.align(_buoycast(t), _gfs(t, [1.0, np.nan, 1.0]), 3)
    assert len(out) == 2


def test_scores_use_forecast_minus_observed_and_credit_the_better_forecast():
    t = pd.date_range("2024-10-01", periods=112, freq="6h", tz="UTC")
    joined = op.align(_buoycast(t), _gfs(t, [1.2] * 112), 3)
    g = op.group_scores(joined, np.random.default_rng(0))
    assert g["n"] == 112
    assert g["methods"]["gfswave"]["bias"] == pytest.approx(-0.8)
    assert g["methods"]["mlp"]["mae"] == pytest.approx(0.1)
    assert g["mlp_skill_vs_gfswave"] == pytest.approx(1 - 0.1 / 0.8)
    assert g["mlp_skill_vs_gfswave_95"][0] > 0


def test_a_worse_buoycast_gets_negative_skill_against_gfswave():
    t = pd.date_range("2024-10-01", periods=112, freq="6h", tz="UTC")
    joined = op.align(_buoycast(t), _gfs(t, [2.0] * 112), 3)  # GFS-Wave is perfect
    g = op.group_scores(joined, np.random.default_rng(0))
    assert g["methods"]["gfswave"]["mae"] == 0
    assert g["mlp_skill_vs_gfswave"] is None  # nothing to beat: no ratio


def test_the_committed_sample_is_small_and_well_formed():
    gfs = op.load_sample()
    assert len(gfs) < 2000
    assert set(gfs["lead"]) == {3, 6}
    assert ((gfs["valid"] - gfs["cycle"]) / pd.Timedelta(hours=1)).eq(gfs["lead"]).all()
    assert gfs["hs"].between(0, 30).all()


def test_the_comparison_runs_on_the_committed_sample():
    result = op.compare(op.load_sample())
    for h in ("3", "6"):
        overall = result["leads"][h]["overall"]
        assert overall["n"] > 100
        assert set(overall["methods"]) == {"persistence", "mlp", "gfswave"}
