import pytest
from fastapi.testclient import TestClient

from buoycast.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_stations(client):
    rows = client.get("/stations").json()
    ids = {r["id"] for r in rows}
    assert {"44005", "46041", "45012"} <= ids
    assert all(-90 < r["lat"] < 90 and r["first_obs"] < r["last_obs"] for r in rows)


def test_observations_window(client):
    r = client.get("/observations", params={"station": "46041", "hours": 24})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 24
    assert rows[0]["time"] < rows[-1]["time"]


def test_forecast_shape(client):
    r = client.get("/forecast/46041", params={"as_of": "2024-07-10T12:00:00Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["as_of"] == "2024-07-10T12:00:00Z"
    assert [f["horizon_hours"] for f in body["forecasts"]] == [3, 6]
    first = body["forecasts"][0]
    assert first["valid_time"] == "2024-07-10T15:00:00Z"
    assert first["mlp"] >= 0 and first["persistence"] == body["current_wvht"]


def test_forecast_uses_only_the_past(client):
    """The forecast issued at t must not change if later data is ignored."""
    a = client.get("/forecast/45012", params={"as_of": "2024-08-01T00:00:00Z"}).json()
    b = client.get("/forecast/45012", params={"as_of": "2024-08-01T00:20:00Z"}).json()
    assert a == b  # times are floored to the hour


def test_unknown_station_and_bad_time(client):
    assert client.get("/forecast/00000").status_code == 404
    assert client.get("/forecast/46041", params={"as_of": "not-a-date"}).status_code == 422
    assert client.get("/forecast/46041", params={"as_of": "2010-01-01"}).status_code == 404


def test_backtest_lines_up_with_observations(client):
    end = "2024-07-10T12:00:00Z"
    bt = client.get("/backtest/46041", params={"horizon": 3, "hours": 12, "end": end}).json()
    obs = client.get("/observations", params={"station": "46041", "hours": 48, "end": end}).json()
    seen = {r["time"]: r["wvht"] for r in obs["rows"]}
    assert bt["rows"]
    for row in bt["rows"]:
        assert seen[row["valid_time"]] == row["observed"]


def test_qc_flags_for_a_window(client):
    r = client.get("/qc/46005", params={"hours": 48, "end": "2024-08-01T00:00:00Z"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert 0 < len(rows) <= 48
    assert "wvht_spike" in rows[0] and "pres_flat_line" in rows[0]
    assert all(v in (1, 2, 3, 4, 9) for k, v in rows[0].items() if k != "time")


def test_qc_summary_lists_every_test(client):
    body = client.get("/qc/summary").json()
    tests = {r["test"] for r in body["stations"]["by_test"]}
    assert {"gross_range", "spike", "rate_of_change", "flat_line", "gap"} <= tests


def test_verification_report(client):
    body = client.get("/verification").json()
    assert set(body["leads"]) == {"3", "6"}
    overall = body["leads"]["3"]["overall"]
    assert overall["n"] > 10000 and overall["methods"]["mlp"]["skill_mae"] > 0
