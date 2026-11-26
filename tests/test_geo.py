import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from buoycast import geo
from buoycast.api import create_app


@pytest.fixture(scope="module")
def stations() -> pd.DataFrame:
    return geo.station_table()


def ids(df: pd.DataFrame) -> set[str]:
    return set(df["id"])


def test_distance_known_values():
    assert geo.distance_km(10.0, 20.0, 10.0, 20.0) == 0
    assert geo.distance_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)
    # Across the date line the short way round is used.
    assert geo.distance_km(0.0, 179.5, 0.0, -179.5) == pytest.approx(111.19, abs=0.1)


def test_bbox_picks_the_pacific_stations(stations):
    got = geo.within_bbox(stations, -135, 40, -120, 50)
    assert ids(got) == {"46041", "46005"}


def test_bbox_edges_are_included(stations):
    row = stations[stations["id"] == "44027"].iloc[0]
    got = geo.within_bbox(stations, row.lon, row.lat, row.lon, row.lat)
    assert ids(got) == {"44027"}


def test_bbox_across_the_date_line():
    pts = pd.DataFrame({"id": ["a", "b", "c"], "lat": [0.0, 0.0, 0.0], "lon": [179.0, -179.0, 0.0]})
    assert ids(geo.within_bbox(pts, 170, -5, -170, 5)) == {"a", "b"}


def test_bbox_rejects_nonsense(stations):
    with pytest.raises(ValueError):
        geo.within_bbox(stations, -10, 50, 10, 40)  # min_lat above max_lat
    with pytest.raises(ValueError):
        geo.within_bbox(stations, -10, 0, 200, 10)  # longitude 200


def test_polygon_around_the_great_lakes(stations):
    lakes = [(-84, 41), (-76, 41), (-76, 44.5), (-84, 44.5)]
    assert ids(geo.within_polygon(stations, lakes)) == {"45005", "45012"}


def test_polygon_can_be_concave():
    # A "C" shape: the cavity between the arms is outside, even though it is inside the bounding box.
    c_shape = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 3), (3, 3), (3, 1), (0, 1)]
    pts = pd.DataFrame(
        {"id": ["bottom arm", "cavity", "spine"], "lat": [0.5, 2.0, 2.0], "lon": [1.0, 1.5, 3.5]}
    )
    assert ids(geo.within_polygon(pts, c_shape)) == {"bottom arm", "spine"}


def test_polygon_from_geojson_and_bad_input(stations):
    gj = {
        "type": "Polygon",
        "coordinates": [[[-84, 41], [-76, 41], [-76, 44.5], [-84, 44.5], [-84, 41]]],
    }
    assert ids(geo.within_polygon(stations, gj)) == {"45005", "45012"}
    with pytest.raises(ValueError):
        geo.within_polygon(stations, [(0, 0), (1, 1), (1, 0), (0, 1)])  # bow-tie crosses itself
    with pytest.raises(ValueError):
        geo.within_polygon(stations, {"type": "Point", "coordinates": [0, 0]})


def test_nearest_station(stations):
    near = geo.nearest(stations, 43.0, -70.0)  # off the coast of New Hampshire
    assert near.iloc[0]["id"] == "44005"
    assert near.iloc[0]["distance_km"] == pytest.approx(74.3, abs=0.5)
    top3 = geo.nearest(stations, 47.0, -125.0, k=3)
    assert top3["distance_km"].is_monotonic_increasing
    assert top3.iloc[0]["id"] == "46041"


def test_nearest_rejects_bad_input(stations):
    with pytest.raises(ValueError):
        geo.nearest(stations, 91, 0)
    with pytest.raises(ValueError):
        geo.nearest(stations, 0, 0, k=0)


def test_queries_work_on_a_moving_track():
    track = pd.DataFrame(
        {"lat": np.linspace(45, 50, 6), "lon": np.linspace(-135, -125, 6), "sst": range(6)}
    )
    inside = geo.within_bbox(track, -130, 40, -120, 55)
    assert inside["sst"].tolist() == [3, 4, 5]
    assert geo.nearest(track, 45, -135).iloc[0]["sst"] == 0


def test_geojson_points(stations):
    gj = geo.to_geojson(stations, ["id", "first_obs"])
    f = gj["features"][0]
    assert f["geometry"]["type"] == "Point"
    lon, lat = f["geometry"]["coordinates"]
    assert (lat, lon) == (stations.iloc[0].lat, stations.iloc[0].lon)
    assert f["properties"]["first_obs"].endswith("Z")
    json.dumps(gj)  # must be plain JSON


def test_geoparquet_is_readable_by_geopandas(tmp_path, stations):
    gpd = pytest.importorskip("geopandas")
    path = tmp_path / "s.parquet"
    geo.write_geoparquet(stations, path)
    got = gpd.read_parquet(path)
    assert "CRS84" in got.crs.name  # longitude, latitude in WGS 84
    assert len(got) == len(stations)
    assert (got.geometry.x.to_numpy() == stations["lon"].to_numpy()).all()
    assert (got.geometry.y.to_numpy() == stations["lat"].to_numpy()).all()


def test_export_writes_every_file(tmp_path):
    gpd = pytest.importorskip("geopandas")
    counts = geo.export_all(tmp_path)
    assert counts["stations.geojson"] == 6
    assert counts["forecasts.parquet"] > 30000
    fixes = gpd.read_parquet(tmp_path / "drifters.parquet")
    assert fixes.geometry.x.nunique() > 100  # positions change from row to row
    forecasts = gpd.read_parquet(tmp_path / "forecasts.parquet")
    assert {"mlp", "observed", "lead_hours"} <= set(forecasts.columns)
    tracks = json.loads((tmp_path / "drifter_tracks.geojson").read_text())
    assert tracks["features"][0]["geometry"]["type"] == "LineString"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_api_bbox_nearest_and_polygon(client):
    r = client.get(
        "/stations/bbox", params=dict(min_lon=-135, min_lat=40, max_lon=-120, max_lat=50)
    )
    assert {s["id"] for s in r.json()} == {"46041", "46005"}
    r = client.get("/stations/nearest", params=dict(lat=43.0, lon=-70.0, k=2))
    assert r.json()[0]["id"] == "44005" and r.json()[0]["distance_km"] < r.json()[1]["distance_km"]
    poly = {
        "type": "Polygon",
        "coordinates": [[[-84, 41], [-76, 41], [-76, 44.5], [-84, 44.5], [-84, 41]]],
    }
    r = client.post("/stations/within", json=poly)
    assert {s["id"] for s in r.json()} == {"45005", "45012"}


def test_api_rejects_bad_queries(client):
    assert client.get("/stations/nearest", params=dict(lat=95, lon=0)).status_code == 422
    assert (
        client.get(
            "/stations/bbox", params=dict(min_lon=0, min_lat=10, max_lon=5, max_lat=0)
        ).status_code
        == 422
    )
    assert client.post("/stations/within", json={"type": "Polygon"}).status_code == 422
