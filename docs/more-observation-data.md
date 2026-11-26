# More observation data: nearby airport reports

The model so far only sees the buoy's own readings. I wanted to see how much a
bigger public observation archive could add, so I added a loader for the
NNJA-AI archive and tried it on the forecast. Short result: it did not help, and
the numbers below show by how much.

## What the archive is

NNJA-AI is a public set of the weather observations that NOAA feeds into its
weather models (airport reports, ship and buoy reports, balloons, satellites),
stored as one Parquet file per day in an open Google Cloud bucket
(`gs://nnja-ai`, described at <https://github.com/brightbandtech/nnja-ai>). It
needs no account. I read it with the Parquet reader in pyarrow, one day at a time,
fetching only the five columns I use.

I looked for wave heights in the surface datasets (`conv/adpsfc`, four of them:
land weather stations, mobile land stations and two airport sets). They carry wind,
pressure and temperature but no sea state, and I did not look at the other groups
(balloons, satellites). So this archive cannot give me more *targets* to
learn from. What it can give me is more *inputs*: the wind and pressure at an
airport on the coast near each buoy.

## What I added

- `src/buoycast/nnja.py` downloads the METAR airport reports (dataset
  `NC000007`) for four airports, one near each buoy that has a coast within 90 km:

  | Buoy | Airport | Distance |
  |---|---|---|
  | 44027 Jonesport ME | KBHB, Bar Harbor | about 87 km |
  | 45005 West Erie | KCLE, Cleveland | about 52 km |
  | 45012 East Lake Ontario | KROC, Rochester | about 60 km |
  | 46041 Cape Elizabeth WA | KUIL, Quillayute | about 68 km |

  The other two buoys have no airport nearby (46005 is 500 km from land, and
  44005 has no test rows), so they are not in this experiment.
- The reports from 2022 to 2024 are averaged to one row per hour and committed as
  `data/sample/nnja_metar.parquet` (90,068 rows, 0.5 MB). A report taken at 12:53
  is labelled 13:00, so a value stamped `t` was always known by `t`.
- Five extra inputs per hour: airport wind speed, its east and north parts,
  pressure, and pressure change over 3 hours. `tests/test_nnja.py` checks that
  they never look ahead in time.

To download it again (a few minutes on my connection, 1,096 small requests):

```
uv run buoycast-nnja
```

## The experiment

`uv run buoycast-extra` trains the same MLP twice on the same rows (hours where
both the buoy and its airport have data), with the same split, settings and 5
seeds. One gets only the buoy features, the other also gets the five airport
inputs. Writes `artifacts/extra_data.json`. Takes about 5 seconds.

Test period July to December 2024, four buoys, about 12,200 forecasts per lead.
MAE is the average miss in metres, MLP shown as mean and standard deviation over 5 seeds.

| Lead | Model | Buoy only | Buoy + airport |
|---|---|---|---|
| 3 h | Persistence | 0.155 | same |
| 3 h | Linear regression | 0.137 | 0.138 |
| 3 h | MLP | 0.126 +/- 0.001 | 0.125 +/- 0.000 |
| 6 h | Persistence | 0.245 | same |
| 6 h | Linear regression | 0.220 | 0.220 |
| 6 h | MLP | 0.200 +/- 0.001 | 0.201 +/- 0.001 |

The paired change in the MLP's MAE from adding the airport data (average of 5
seeds, 95% range from resampling days):

- 3 h: -0.0005 m (range -0.0012 to +0.0001)
- 6 h: -0.0006 m (range -0.0021 to +0.0009)

That is half a millimetre, and both ranges include zero. I read it as "no
measurable effect". Per buoy it goes both ways (for example at 44027 the 6 hour
MAE drops from 0.173 to 0.169 m, at 46041 it rises from 0.256 to 0.258 m), which also looks
like noise.

## Why I think it did not help

I did not test these, they are guesses:

- The buoy already measures its own wind and pressure, and those are better than
  an airport 50 to 90 km away. The airport wind correlates only moderately with the
  buoy wind (correlation 0.48 to 0.58 over 2022 to 2024, in `artifacts/extra_data.json`).
- At 3 and 6 hours, waves are mostly decided by what the sea is already doing. Wind
  from far away would matter more at longer leads, which I did not train.
- An airport on land sees land-affected wind, which is a poor proxy for wind over
  open water.

## Limits

- One archive, one dataset (METAR) and four airports. It says nothing about
  the rest of NNJA-AI, such as ship reports or satellite data.
- Only the 2022 to 2024 sample I committed. The loader takes any years in the
  archive, but I only ran these.
- The archive has gaps, so about 15% of the airport hours are missing and those
  rows are left out of both variants.
- The downloader itself talks to the internet, so the tests cover the parsing and
  the leakage check on made-up reports, and not the download.
