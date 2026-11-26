# Data quality checks

Buoy sensors break in boring ways: they get stuck on one number, they report a
reading that is far too big, or they go quiet for a day. The first version of
buoycast only handled the one case I knew about (the "no data" codes like 99 and
999). This adds automatic checks that give every reading a flag, so a bad
reading can be seen and set aside instead of quietly going into the model.

The checks follow the public QARTOD guides from NOAA's IOOS programme. I only
read the guides' descriptions of the tests; the numbers I use for "too big" or
"too fast" are my own rough choices (see `BUOY_CONFIG` in `src/buoycast/qc.py`),
not official limits for any station.

## What each reading gets

Each reading gets a flag from each test:

| Flag | Meaning |
|---|---|
| 1 | pass |
| 2 | not evaluated (for example the first reading, which has no neighbour) |
| 3 | suspect |
| 4 | fail |
| 9 | missing |

The five tests:

| Test | What it asks |
|---|---|
| gross range | is the value outside what a sensor or the ocean can give? (a 30 m wave) |
| spike | does one reading stick out from the line between the readings before and after it? |
| rate of change | did the value move faster than it plausibly can? |
| flat line | has the value stayed exactly the same for too long? (a stuck sensor) |
| gap | did this reading arrive after a long silence? |

Time is measured in real hours, not rows, so the same code works on hourly buoy
data and on a drifter that reports every 6 hours. The flags for the committed
sample are stored in `data/sample/qc_flags.parquet`, one row for each row of
`hourly.parquet`.

## Results on the real sample

Command: `uv run buoycast-qc` (writes `artifacts/qc_summary.json`, takes a few
seconds). The sample has 117,495 station-hours from 6 buoys over 2022 to 2024.
Counts below are readings, for all variables with at least one flag:

| Variable | Gross range | Spike | Rate of change | Flat line | Gap |
|---|---|---|---|---|---|
| wave height | 0 | 13 | 8 | 1 | 21 |
| wind speed | 0 | 1 | 10 | 0 | 21 |
| pressure | 0 | 0 | 5 | 60 | 32 |
| water temperature | 0 | 5 | 39 | 190 | 46 |
| dominant wave period | 0 | 95 | 86 | 45 | 464 |

(Each cell is suspect plus fail. The full table, including the other variables
and per-station counts, is in `artifacts/qc_summary.json`.)

Two things stood out:

- **A real stuck sensor.** Station 46005's pressure reads exactly 999.0 hPa for
  72 hours (2022-05-29 00:00 to 2022-05-31 23:00). 999 hPa is a believable pressure, so the
  gross range test cannot see it; the flat line test marked 60 of those hours.
- **The wave period is the noisy one.** The dominant wave period jumps between
  neighbouring readings in normal conditions, so it collects most of the spike
  and rate flags. My thresholds for it are probably too tight and I did not tune
  them. Most of its flags are not necessarily faults.

I did not check the flags by hand against the NDBC's own quality notes, so I
cannot say how many are real faults. The gap flags in particular are partly an
artefact of how I built the sample: it only keeps hours that have a wave reading.

## A moving platform

Buoys on moorings never change position, but a drifter does, so every row has a
different latitude and longitude. A drifter also gets two position tests: is the
position on Earth at all, and is the speed between two fixes believable (more than
10 km/h is suspect and more than 25 km/h fails for a drifter; a ship would need
different limits).

The sample is five surface drifters from NOAA's Global Drifter Program that
floated through the north-east Pacific from July to December 2024 (3,680 six-hourly
rows, `data/sample/drifters.parquet`, downloaded with `uv run buoycast-drifters`).
The flags are stored with the rows in `data/sample/drifters_qc.parquet`.

The result: zero flags on all of it. That is not surprising. This version of the
drifter data has already been quality controlled and interpolated to 6 hours by
the programme that publishes it, so there is little left to find. It does at
least show that the code runs on a moving track, and nothing is wrongly flagged.

## Does it find faults at all?

Since the real data has few faults, I also planted fake ones and counted how many
the tests found (`injection_test` in `qc_run.py`, seed 0, included in
`artifacts/qc_summary.json`):

| Planted fault | Planted | Found |
|---|---|---|
| a single reading 2.5 times the spike limit too high | 674 | 673 |
| a sensor stuck for 30 hours | 656 | 656 |
| a reading far outside the sensor range | 663 | 663 |
| a drifter fix moved 3 degrees north | 72 | 72 |
| a drifter temperature 6 C too high | 68 | 68 |

These faults are obvious by design, so the near-perfect scores say the code does
what I meant, not that it would catch every real fault. Smaller faults would be
harder, and I did not measure that.

## Limits

- The tests look at one variable in time. They do not compare a reading with a
  neighbouring buoy or with the other variables, and the thresholds ignore the
  season and the place (a wave height that is rare at one buoy may be normal at
  another).
- For a moving platform the thresholds do not know about ocean fronts, where the
  temperature can really change quickly over a short distance.
- Nothing downstream uses the flags yet. The model still trains on every reading,
  and I have not measured whether dropping failed readings changes the forecast.

## Try it

```
uv run buoycast-qc                  # recompute flags and the summary
uv run pytest tests/test_qc.py -q   # 17 tests
curl "http://127.0.0.1:8000/qc/46005?hours=48&end=2024-08-01T00:00:00Z"   # with buoycast-serve running
```
