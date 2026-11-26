# Maps and spatial queries

Everything in buoycast has a position, but until now the position only lived in a
small Python dictionary and in the map page. This makes positions a first-class
part of the data: you can export it in the standard formats that mapping tools read,
and ask simple "where" questions.

## What you can do

Export everything as GeoParquet and GeoJSON:

```
uv run buoycast-export            # writes exports/ (not committed, about 3 MB)
```

| File | What | Rows |
|---|---|---|
| `stations.geojson`, `stations.parquet` | the six buoys | 6 |
| `observations.parquet` | every hourly reading, with the buoy's position and its wave height quality flag | 117,495 |
| `forecasts.parquet` | every test-period forecast from all three methods at 3 and 6 hours, next to what was observed | 35,084 |
| `forecasts_latest.geojson` | the last 3 hour forecast in the sample for each buoy | 5 |
| `drifters.parquet` | the moving drifter fixes, each with its own position | 3,680 |
| `drifter_tracks.geojson` | the drifter paths as lines | 5 |

**GeoJSON** is a text format for shapes on a map (QGIS, geojson.io and Leaflet all
open it). **GeoParquet** is the same idea in the compact, fast Parquet format:
an ordinary table plus a `geometry` column holding each point, and a note saying
which column that is. Coordinates are longitude and latitude in WGS 84, the
system GPS uses.

I wrote the GeoParquet file with pyarrow and shapely (the writer is `write_geoparquet`
in `src/buoycast/geo.py`, about 25 lines) rather than adding a big geospatial
dependency to the project. To check that it really is valid GeoParquet, the tests
read it back with GeoPandas, which is installed as a development tool only.

Reading it back:

```python
import geopandas as gpd

fc = gpd.read_parquet("exports/forecasts.parquet")
fc[fc.lead_hours == 3].plot(column="mlp")
```

## Spatial queries

Three questions, in `src/buoycast/geo.py` and in the API:

| Question | Python | API |
|---|---|---|
| Which stations are inside this box? | `within_bbox(points, min_lon, min_lat, max_lon, max_lat)` | `GET /stations/bbox?min_lon=-135&min_lat=40&max_lon=-120&max_lat=50` |
| Which stations are inside this shape? | `within_polygon(points, polygon)` | `POST /stations/within` with a GeoJSON polygon |
| Which stations are closest to this point? | `nearest(points, lat, lon, k=3)` | `GET /stations/nearest?lat=43&lon=-70&k=3` |

For example the box above returns 46041 and 46005, the two Pacific buoys. The
point 43 N, 70 W (off New Hampshire) is 74 km from 44005, which is the nearest.
Distances use the great-circle formula (haversine), so they are right across long
distances and across the date line, unlike plain x and y arithmetic on degrees.

The functions take any table with `lat` and `lon` columns, so they also work on
the drifter, where every row is somewhere different:

```python
from buoycast import geo, drifters

fixes = drifters.load_drifters()["300534064906390"].reset_index()
geo.within_bbox(fixes, -135, 45, -130, 50)  # the fixes inside that box
geo.nearest(fixes, 46.0, -130.0, k=1)  # the fix closest to a point
```

Details I thought about:

- A point exactly on the edge of a box or polygon counts as inside.
- A box with `min_lon` larger than `max_lon` is read as crossing longitude 180.
- A polygon that crosses itself, or an impossible position like latitude 95, is
  rejected with a clear message (HTTP 422 in the API).
- Polygons may be concave (a C shape); the test checks that a point in the cavity
  is outside.

## Tests

```
uv run pytest tests/test_geo.py -q    # 16 tests
```

They cover the distance formula against known values, box edges and the date
line, a concave polygon, a bad polygon, nearest-first ordering, the queries on a
moving track, that the GeoJSON is plain JSON, that GeoPandas can read the
GeoParquet back with the right coordinates, and the three API routes.

## Limits

- Only point data. There are no coastlines, regions or lines apart from the drifter
  path export.
- The nearest-station search checks every station one by one. That is fine for six,
  and would need a spatial index (such as an R-tree) for many thousands.
- The polygon test is flat on longitude and latitude, so a polygon that crosses
  longitude 180 or surrounds a pole would not be handled correctly.
- I did not compare the files against other GeoParquet readers than GeoPandas.
