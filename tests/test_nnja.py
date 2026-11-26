import numpy as np
import pandas as pd
import pytest

from buoycast import nnja
from buoycast.extra_run import paired_mae_difference
from buoycast.features import build_features, load_hourly


def raw_reports():
    return pd.DataFrame(
        {
            "OBS_TIMESTAMP": ["2024-07-01 00:51:00+00:00", "2024-07-01 01:53:00+00:00"] * 2
            + ["2024-07-01 01:56:00+00:00"],
            "MTRID.RPID": ["KCLE", "KCLE", "KXXX", "KXXX", "KCLE"],
            "MTRWND.WSPD": [5.0, 6.0, 1.0, 1.0, 7.0],
            "MTRWND.WDIR": [10.0, 20.0, 0.0, 0.0, 30.0],
            "MTRPRS.ALSE": [102200.0, 102210.0, 100000.0, 100000.0, 102220.0],
        }
    )


def test_tidy_keeps_only_wanted_airports_and_converts_pressure_to_hpa():
    out = nnja.tidy(raw_reports(), {"KCLE"})
    assert set(out["airport"]) == {"KCLE"}
    assert out["pres"].max() == pytest.approx(1022.2)


def test_hourly_label_is_never_earlier_than_the_report():
    hourly = nnja.to_hourly(nnja.tidy(raw_reports(), {"KCLE"}))
    # 00:51 -> 01:00, and the two reports in the 01:53 to 01:56 hour keep the last one.
    assert hourly["time"].tolist() == list(
        pd.to_datetime(["2024-07-01 01:00", "2024-07-01 02:00"], utc=True)
    )
    assert hourly["wspd"].tolist() == [5.0, 7.0]


def test_cached_day_is_read_without_network(tmp_path, monkeypatch):
    from datetime import date

    monkeypatch.setattr(nnja, "RAW_DIR", tmp_path)
    cache = tmp_path / "nnja" / "2024-07-01.parquet"
    cache.parent.mkdir()
    nnja.tidy(raw_reports(), {"KCLE"}).to_parquet(cache)
    got = nnja.fetch_day(date(2024, 7, 1), {"KCLE"}, fs=None)
    assert len(got) == 3


def test_airport_features_do_not_look_ahead():
    idx = pd.date_range("2024-07-01", periods=60, freq="1h", tz="UTC", name="time")
    rng = np.random.default_rng(0)
    a = pd.DataFrame(
        {
            "wspd": rng.random(60) * 10,
            "wdir": rng.random(60) * 360,
            "pres": 1010 + rng.random(60),
        },
        index=idx,
    )
    before = nnja.extra_features(a)
    changed = a.copy()
    changed.iloc[30:] = 999  # rewrite everything from hour 30 on
    after = nnja.extra_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:30], after.iloc[:30])


def test_committed_airport_sample_loads_on_hourly_grids():
    extra = nnja.load_extra()
    assert set(extra) == set(nnja.NEAR_AIRPORT)
    for frame in extra.values():
        assert frame.index.is_monotonic_increasing
        assert (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
        assert frame["pres"].dropna().between(900, 1100).all()


def test_rows_with_airport_inputs_are_complete_and_a_subset_of_the_buoy_rows():
    hourly, extra = load_hourly(), nnja.load_extra()
    with_extra = nnja.build_with_extra(hourly, extra, 3, ["45005"])
    plain = build_features(hourly["45005"], 3)
    assert not with_extra[list(nnja.EXTRA_NAMES)].isna().any().any()
    assert 0 < len(with_extra) <= len(plain)
    assert with_extra.index.isin(plain.index).all()


def test_paired_difference_has_the_right_sign():
    idx = pd.date_range("2024-07-01", periods=24 * 30, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    test = pd.DataFrame({"target": rng.random(len(idx))}, index=idx)
    worse = test["target"].to_numpy() + 0.2
    better = test["target"].to_numpy() + 0.1
    change, lo, hi = paired_mae_difference(test, better, worse)
    assert change == pytest.approx(-0.1) and hi < 0
