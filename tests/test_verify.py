import json

import numpy as np
import pandas as pd
import pytest

from buoycast import verify
from buoycast.paths import METRICS_JSON


def frame(observed, persistence, linear=None, mlp=None, days=40):
    n = len(observed)
    return pd.DataFrame(
        {
            "station": "x",
            "issued": pd.date_range("2024-07-01", periods=n, freq="1h", tz="UTC"),
            "height_now": persistence,
            "observed": observed,
            "persistence": persistence,
            "linear": linear if linear is not None else persistence,
            "mlp": mlp if mlp is not None else persistence,
        }
    )


def test_perfect_forecast_has_skill_one_and_no_error():
    obs = np.abs(np.sin(np.arange(2000) / 7.0)) + 0.5
    per = np.r_[obs[0], obs[:-1]]
    s = verify.scores(frame(obs, per, mlp=obs), np.random.default_rng(0))
    m = s["methods"]["mlp"]
    assert m["mae"] == 0 and m["bias"] == 0
    assert m["skill_mae"] == pytest.approx(1.0)
    assert s["verdict"] == "better than persistence"


def test_the_reference_method_has_zero_skill_against_itself():
    obs = np.abs(np.cos(np.arange(1500) / 5.0)) + 1
    per = obs + 0.1
    s = verify.scores(frame(obs, per), np.random.default_rng(0))
    assert s["mlp_skill_mae_95"][0] == pytest.approx(0, abs=1e-9)
    assert s["verdict"] == "no clear difference from persistence"


def test_bias_sign_is_forecast_minus_observed():
    obs = np.full(500, 2.0)
    s = verify.scores(frame(obs, obs, mlp=obs - 0.3), np.random.default_rng(0))
    assert s["methods"]["mlp"]["bias"] == pytest.approx(-0.3)
    assert s["methods"]["mlp"]["rmse"] == pytest.approx(0.3)


def test_a_worse_model_is_called_worse():
    rng = np.random.default_rng(1)
    obs = 1 + rng.random(3000)
    per = obs + rng.normal(0, 0.05, 3000)
    bad = obs + rng.normal(0, 0.5, 3000)
    s = verify.scores(frame(obs, per, mlp=bad), np.random.default_rng(0))
    assert s["mlp_skill_mae_95"][1] < 0
    assert s["verdict"] == "worse than persistence"


def test_small_groups_are_not_judged():
    assert verify.verdict_for(50, 0.1, 0.3) == "too few forecasts to say"


def test_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    obs = 1 + rng.random(4000)
    per = obs + rng.normal(0, 0.2, 4000)
    mlp = obs + rng.normal(0, 0.15, 4000)
    s = verify.scores(frame(obs, per, mlp=mlp), np.random.default_rng(0))
    lo, hi = s["mlp_skill_mae_95"]
    assert lo < s["methods"]["mlp"]["skill_mae"] < hi


def test_real_forecasts_match_the_training_metrics():
    """Same test rows, same persistence MAE as the training script reported."""
    metrics = json.loads(METRICS_JSON.read_text())["horizons"]["3"]
    t = verify.forecast_table(horizon=3)
    assert len(t) == metrics["rows"]["test"]
    mae = float(np.abs(t["persistence"] - t["observed"]).mean())
    assert mae == pytest.approx(metrics["persistence"]["mae"], abs=1e-6)
    assert (t["valid"] - t["issued"] == pd.Timedelta(hours=3)).all()


def test_report_groups_cover_every_forecast():
    result = verify.verify()
    for lead in result["leads"].values():
        total = lead["overall"]["n"]
        assert sum(g["n"] for g in lead["by_height"]) == total
        assert sum(g["n"] for g in lead["by_station"]) == total
