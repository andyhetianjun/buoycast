# The static demo

<https://andyhetianjun.github.io/buoycast/> is the same map page as
`buoycast-serve`, but it needs no server. Everything it shows was computed once, when
the site was built, from the sample data committed in this repository. The page says so
at the top. The data runs from 2022 to 2024, so it is not live, and forecasts exist
only for the test months (July to December 2024).

What is on it: the six buoys on a map; click one for its observed versus forecast
chart and the forecast table; a strip of quality flags for the same hours; a list of
flagged stretches you can click to jump to; the drifting buoy tracks (purple); the
verification report; and the quality-check counts. Clicking the map away from a
marker tells you the nearest station and how far it is (calculated in the browser).

## Build it

```
uv run buoycast-site                       # writes site/
cd site && python3 -m http.server 8765     # then open http://127.0.0.1:8765
```

`src/buoycast/site.py` runs the saved models and the quality checks and writes one
JSON file per station (about 0.7 MB each, before compression) plus an index with the
verification report. The page files are copied from `web/`. All paths are relative,
so it works under the `/buoycast/` path on GitHub Pages. `tests/test_site.py` checks
that, and that the stored forecasts match the ones from the forecast table.

`.github/workflows/pages.yml` builds and deploys it on every push to `main`. I have
not been able to run that workflow yet (it runs on GitHub), but I did build the
site locally and load it in a headless browser to check the chart, the flags and
the report render with no errors.

## Limits

- The map tiles and the Leaflet library load from the internet.
- Because the forecasts are precomputed, the page cannot forecast an hour that is
  not in the sample, which the server version can.
