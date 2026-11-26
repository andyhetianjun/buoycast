import numpy as np
import pandas as pd
import pytest

from buoycast.features import (
    TEST_START,
    VAL_START,
    build_features,
    feature_names,
    split_by_time,
)


def synthetic(n: int = 24 * 40, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="time")
    t = np.arange(n)
    return pd.DataFrame(
        {
            "wvht": 1.5 + np.sin(t / 20) + rng.normal(0, 0.05, n),
            "dpd": 8 + rng.normal(0, 0.3, n),
            "apd": 5 + rng.normal(0, 0.1, n),
            "wspd": 6 + 2 * np.sin(t / 15),
            "gst": 8 + 2 * np.sin(t / 15),
            "wdir": (t * 3.0) % 360,
            "pres": 1012 + 5 * np.cos(t / 40),
            "atmp": 10 + rng.normal(0, 1, n),
            "wtmp": 12 + rng.normal(0, 0.1, n),
        },
        index=idx,
    )


def test_target_is_future_wave_height():
    df = synthetic()
    feats = build_features(df, horizon=3)
    t = feats.index[100]
    assert feats.loc[t, "target"] == df.loc[t + pd.Timedelta(hours=3), "wvht"]
    assert feats.loc[t, "wvht_lag0"] == df.loc[t, "wvht"]
    assert feats.loc[t, "wvht_lag24"] == df.loc[t - pd.Timedelta(hours=24), "wvht"]


def test_features_do_not_use_the_future():
    """Scramble everything after a cutoff: features at or before it must not change."""
    df = synthetic()
    cutoff = df.index[300]
    corrupted = df.copy()
    rng = np.random.default_rng(1)
    after = corrupted.index > cutoff
    corrupted.loc[after, :] = rng.uniform(-50, 500, size=(after.sum(), df.shape[1]))
    a = build_features(df, 3, require_target=False)
    b = build_features(corrupted, 3, require_target=False)
    cols = feature_names()
    pd.testing.assert_frame_equal(a.loc[:cutoff, cols], b.loc[:cutoff, cols])


def test_gaps_are_not_filled_from_the_future():
    df = synthetic()
    gap = df.index[200:210]
    df.loc[gap, "wvht"] = np.nan
    feats = build_features(df, 3)
    # Hours with no observation never become rows, and the target is never filled.
    assert not feats.index.isin(gap).any()
    assert feats["target"].notna().all()
    assert not (feats.index + pd.Timedelta(hours=3)).isin(gap).any()


def test_missing_feature_rows_are_dropped():
    df = synthetic()
    df.loc[df.index[50:60], "pres"] = np.nan
    feats = build_features(df, 3)
    assert feats[feature_names()].notna().all().all()
    assert df.index[56] not in feats.index


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        build_features(synthetic(), 0)


def test_split_has_no_label_overlap():
    df = synthetic(n=24 * 400, start="2023-06-01")
    feats = build_features(df, 6)
    feats.insert(0, "station", "x")
    train, val, test = split_by_time(feats, 6)
    gap = pd.Timedelta(hours=6)
    assert len(train) and len(val) and len(test)
    assert (train.index + gap).max() < VAL_START
    assert val.index.min() >= VAL_START and (val.index + gap).max() < TEST_START
    assert test.index.min() >= TEST_START
    assert train.index.max() < val.index.min() < test.index.min()
