import numpy as np
import pandas as pd

from buoycast import qc

CFG = qc.VarConfig(
    fail_range=(0, 20),
    suspect_range=(0, 10),
    spike=(1.0, 3.0),
    rate=(2.0, 5.0),
    flat_hours=(5, 10),
    flat_eps=0.0,
    gap_hours=(3, 12),
)


def hourly(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="1h", tz="UTC", name="time")
    return pd.Series(values, index=idx, dtype="float64")


def test_gross_range_grades_and_marks_missing():
    s = hourly([1, 11, 25, -1, np.nan])
    flags = qc.gross_range(s, CFG).tolist()
    assert flags == [qc.PASS, qc.SUSPECT, qc.FAIL, qc.FAIL, qc.MISSING]


def test_spike_flags_only_the_odd_point():
    s = hourly([1.0, 1.1, 5.0, 1.2, 1.3, 1.4])
    flags = qc.spike(s, CFG).tolist()
    assert flags[2] == qc.FAIL
    assert flags[1] == qc.PASS and flags[3] == qc.PASS
    assert flags[0] == qc.NOT_EVALUATED and flags[-1] == qc.NOT_EVALUATED


def test_spike_ignores_a_smooth_trend():
    s = hourly(np.arange(0, 12, 1.0))  # 1 per hour, steady
    assert (qc.spike(s, CFG).iloc[1:-1] == qc.PASS).all()


def test_spike_is_not_evaluated_across_a_long_gap():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 12:00"], utc=True)
    s = pd.Series([1.0, 9.0, 1.0], index=idx)
    assert qc.spike(s, CFG).iloc[1] == qc.NOT_EVALUATED


def test_rate_of_change_uses_hours_not_rows():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True)
    s = pd.Series([0.0, 3.0, 6.0], index=idx)  # 3 per hour, then 1.5 per hour
    flags = qc.rate_of_change(s, CFG).tolist()
    assert flags == [qc.NOT_EVALUATED, qc.SUSPECT, qc.PASS]


def test_flat_line_counts_elapsed_time():
    s = hourly([1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4])
    flags = qc.flat_line(s, CFG).tolist()
    assert flags[2] == qc.PASS  # first reading of the run
    assert flags[7] == qc.SUSPECT  # 5 hours unchanged
    assert flags[12] == qc.FAIL  # 10 hours unchanged
    assert flags[13] == qc.PASS  # moved again


def test_flat_line_run_does_not_bridge_a_gap():
    v = [3.0] * 4 + [np.nan] * 20 + [3.0] * 4
    flags = qc.flat_line(hourly(v), CFG)
    assert flags.max() == qc.PASS or (flags[flags != qc.MISSING] == qc.PASS).all()


def test_gap_flags_the_reading_after_a_silence():
    v = [1.0, 1.0, np.nan, np.nan, 1.0, np.nan * 1, np.nan] + [np.nan] * 12 + [1.0]
    flags = qc.gap(hourly(v), CFG)
    assert flags.iloc[4] == qc.PASS  # 3 hours: not more than the limit
    assert flags.iloc[-1] == qc.FAIL  # 20 hours of silence
    assert flags.iloc[2] == qc.MISSING


def test_overall_flag_is_the_worst_test():
    s = hourly([1.0, 1.1, 5.0, 1.2, 1.3, 1.4])
    out = qc.run_variable(s, CFG)
    assert out["overall"].iloc[2] == qc.FAIL
    assert out["overall"].iloc[1] == qc.PASS


def test_overall_is_missing_only_when_there_is_no_reading():
    out = qc.run_variable(hourly([1.0, np.nan, 1.0, 1.0]), CFG)
    assert out["overall"].iloc[1] == qc.MISSING
    assert out["overall"].iloc[0] != qc.MISSING


def test_real_station_series_gets_a_flag_for_every_row():
    from buoycast.features import load_hourly

    frame = load_hourly()["46041"].iloc[:2000]
    flags = qc.run_platform(frame)
    assert len(flags) == len(frame)
    assert flags["wvht_overall"].isin([1, 3, 4, 9]).all()
    assert (flags["wvht_overall"] == qc.MISSING).equals(frame["wvht"].isna())


def drift(lat, lon, hours=6):
    idx = pd.date_range("2024-07-01", periods=len(lat), freq=f"{hours}h", tz="UTC", name="time")
    return pd.DataFrame({"lat": lat, "lon": lon}, index=idx)


def test_distance_between_known_points():
    # One degree of latitude is about 111 km.
    assert abs(qc.distance_km(10.0, 20.0, 11.0, 20.0) - 111.2) < 0.5


def test_track_speed_for_a_moving_platform():
    lat = [48.00, 48.02, 48.04, 51.0, 51.02]  # the 4th fix jumps about 330 km in 6 hours
    flags = qc.run_track(drift(lat, [-130.0] * 5))
    assert flags["track_speed"].tolist()[1:3] == [qc.PASS, qc.PASS]
    assert flags["track_speed"].iloc[3] == qc.FAIL
    assert flags["track_speed"].iloc[0] == qc.NOT_EVALUATED


def test_location_off_the_planet_fails():
    flags = qc.run_track(drift([48.0, 95.0, 48.1], [-130.0, -130.0, -200.0]))
    assert flags["location_range"].tolist() == [qc.PASS, qc.FAIL, qc.FAIL]


def test_variable_tests_work_while_the_position_changes():
    frame = drift(np.linspace(40, 50, 30), np.linspace(-140, -130, 30))
    frame["sst"] = np.linspace(12, 15, 30)
    frame.iloc[10, frame.columns.get_loc("sst")] += 6
    flags = qc.run_platform(frame, qc.DRIFTER_CONFIG)
    assert flags["sst_spike"].iloc[10] in (qc.SUSPECT, qc.FAIL)
    assert (flags["sst_gross_range"] == qc.PASS).all()


def test_summary_counts_add_up():
    s = hourly([1.0, 1.1, 5.0, 1.2, np.nan, 1.4])
    rows = qc.summarize(qc.run_variable(s, CFG).add_prefix("x_"))
    for r in rows:
        total = sum(v for k, v in r.items() if k.startswith("n_"))
        assert total == 6
    spike_row = next(r for r in rows if r["test"] == "spike")
    assert spike_row["n_fail"] == 1


def test_stored_flags_line_up_with_the_sample():
    sample = pd.read_parquet("data/sample/hourly.parquet", columns=["station", "time"])
    flags = pd.read_parquet("data/sample/qc_flags.parquet", columns=["station", "time"])
    assert sample.equals(flags)
