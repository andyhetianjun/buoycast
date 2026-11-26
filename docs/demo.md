# The static demo (live version)

The page at <https://andyhetianjun.github.io/buoycast/> is rebuilt every hour by
`.github/workflows/pages.yml` with `uv run buoycast-site --live`. That command:

1. downloads `https://www.ndbc.noaa.gov/data/realtime2/<station>.txt` for the six
   training buoys and eight Canadian buoys (`src/buoycast/live.py`);
2. parses and averages it to hourly with the same code as training, then runs the five
   quality checks;
3. runs the committed models (`src/buoycast/weights/`, 32 KB each) to get 3 h and 6 h
   forecasts for every hour that has enough history, including the newest hour;
4. writes the same static JSON files as below, plus a 30-day "recent track record".

Nothing fetched is committed. If no station can be downloaded, the build uses the
committed sample instead and the banner says so. `last updated` and the source are
shown at the top of the page. Test it without waiting for the schedule:
`uv run buoycast-site --live && cd site && python3 -m http.server 8765`.

Things to know: GitHub can delay scheduled runs by many minutes, and pauses them in
repositories with no activity for 60 days. Canadian buoys only have 24 hours of
history in NDBC's file, so each gets one forecast and no track record. Their
station id was never seen in training, so the model gets an all-zero station input.
`tests/test_live.py` covers the parsing, the newest-hour forecast, the offline
marker, the fallback and the command, with fake files (no network in tests).

The rest of this page describes the sample-data version, which is what `uv run buoycast-site`
(without `--live`) still builds.

In either mode the map page is the same as
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
- Because the forecasts are precomputed, the page cannot compute a forecast on the
  fly, which the server version does for any hour in the sample.
