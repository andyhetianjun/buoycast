# How buoycast compares with NOAA's wave forecast

I wanted to know how a small model compares with the real forecast system. buoycast is
a few thousand numbers trained on three years of buoy readings. NOAA runs a physics-based
wave model four times a day on a supercomputer. So I scored both against the same buoys,
at the same forecast times.

## The short answer

At 3 hours ahead buoycast had the smaller average error. At 6 hours ahead NOAA's model
did, and the gap is clear. This is what I expected, and the reasons are below.

Average miss in metres (MAE), 336 forecasts per lead time, three buoys, four weeks of
October 2024:

| Lead | Persistence MAE | buoycast MAE | GFS-Wave MAE | buoycast vs GFS-Wave (95% range) |
|---|---|---|---|---|
| 3 h | 0.220 | 0.176 | 0.233 | buoycast error 24.7% smaller (+11.1% to +36.7%) |
| 6 h | 0.350 | 0.274 | 0.232 | buoycast error 18.1% larger (+1.2% to +37.5% larger) |

Other error measures, same forecasts:

| Lead | RMSE: persistence / buoycast / GFS-Wave | Bias: persistence / buoycast / GFS-Wave |
|---|---|---|
| 3 h | 0.315 / 0.261 / 0.337 | -0.000 / -0.032 / -0.100 |
| 6 h | 0.512 / 0.415 / 0.346 | -0.015 / -0.053 / -0.100 |

By buoy (MAE in metres, 112 forecasts each):

| Lead | Buoy | buoycast | GFS-Wave | Clear winner? |
|---|---|---|---|---|
| 3 h | 44027 (Maine) | 0.120 | 0.183 | buoycast |
| 3 h | 46005 (Pacific) | 0.203 | 0.236 | no |
| 3 h | 46041 (Washington) | 0.203 | 0.280 | buoycast |
| 6 h | 44027 | 0.170 | 0.171 | no |
| 6 h | 46005 | 0.319 | 0.246 | GFS-Wave |
| 6 h | 46041 | 0.335 | 0.280 | no |

Command: `uv run buoycast-operational` (a few seconds; uses the committed sample and writes
`artifacts/operational.json`). `uv run buoycast-operational --download` fetches the NOAA files again.

## Why GFS-Wave wins at 6 hours

- **It simulates the sea.** It is driven by NOAA's wind forecast, so it knows about a storm that
  has not reached the buoy yet. buoycast only sees the last 24 hours of readings at one
  buoy, so it can extend a trend but cannot know what is coming.
- **Its error does not grow with time.** GFS-Wave's average miss is 0.233 m at 3 hours and
  0.232 m at 6 hours. buoycast's goes from 0.176 to 0.274 m, and so does persistence's.
  Over longer leads the gap would widen in NOAA's favour. I only trained 3 and 6 hour models.

## Why buoycast wins at 3 hours

This is not because it is a better model. The comparison is lopsided in its favour at short leads:

- buoycast is given the buoy's own readings up to the forecast time. GFS-Wave never sees
  the buoy, so it carries its own small errors (it runs about 0.10 m low here) from the start.
  Persistence also beats GFS-Wave at 3 hours (0.220 against 0.233), which shows how much
  the latest reading is worth over a few hours.
- In real use, GFS-Wave output for a cycle appears hours after the cycle time, so a user could
  not have its 3 hour forecast "at" the cycle time. buoycast could, because it only needs the
  readings. I did not model this delay in the table.

So the fair summary is: for the next few hours, "what the buoy says now" is hard to beat and a small
model on top of it helps a little. Beyond that, the physics model is better.

## How it works

NOAA publishes GFS-Wave (WAVEWATCH III run on the GFS weather forecast) in the open
`noaa-gfs-bdp-pds` bucket on AWS. Next to the large GRIB2 grids it publishes a short text
bulletin for each buoy location, with the model's own hourly wave height at that point.
I read those bulletins (`src/buoycast/operational.py`) instead of GRIB2 files. I did not try
cfgrib or eccodes: the bulletins give the model's point output directly, so I avoid grid
interpolation and a large download. About 50 KB per buoy per cycle; the 448 files I fetched are cached in
`data/raw/` (not committed). What is committed is `data/sample/gfswave_points.csv`, 896 rows (49 KB) with only the
3 and 6 hour values.

To match forecasts:

1. A GFS-Wave cycle at time C gives a forecast for C+3 h and C+6 h.
2. buoycast's forecast issued at C for the same hours comes from the existing `verify.forecast_table`.
3. The two are joined on buoy, issue time and valid time. Rows missing any value are dropped.
4. Errors are forecast minus what the buoy measured, scored with the same MAE, RMSE and bias
   code as [forecast verification](forecast-verification.md). The 95% range resamples whole days.

`tests/test_operational.py` checks the bulletin parser (including month and year rollover),
the join (wrong buoy, wrong lead, missing cycle, non-cycle hours), and the score signs.

## Limits

- Three buoys, 28 days, four cycles a day. Forecasts from neighbouring cycles overlap and
  share storms, so 336 forecasts is less evidence than it sounds. The ranges above account for this
  by resampling days, but a different month could look different.
- Only 3 of the 6 project buoys were scored. 44005 has no readings after March 2024, so it has no
  test data. The Great Lakes buoys (45005, 45012) have no point output in GFS-Wave (the bulletin files do not exist),
  so those two are not compared. A lake is also a different problem.
- The buoy measures wave height in a way that differs a little from the model's total
  significant wave height, and a single buoy position is compared with a model point. Some of
  GFS-Wave's low bias (-0.10 m) may come from that and not from the physics.
- buoycast's training period ended before July 2024 and these weeks come after, so it is a fair test for it.
  GFS-Wave was not tuned by me at all.

Data: NOAA GFS-Wave from NOAA Open Data on AWS; buoy readings from NOAA's National Data Buoy Center.
