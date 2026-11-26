# Forecast verification

The first version of buoycast reported one error number per method. That hides
the questions a forecast user actually has: how good is it three hours out
compared with six, does it work when the sea is rough, and does it work at every
buoy? This adds a proper score card to answer those.

## The short answer

On the six months of test data (July to December 2024), the neural network (MLP)
beat the "waves stay the same" guess (persistence) in every group I checked, at
both 3 and 6 hours. It cut the average error by about 18% overall. It also
under-predicts: on average it forecasts waves a little too low, and the gap is
biggest in rough seas.

| Lead | Persistence MAE | MLP MAE | MLP bias | MLP skill vs persistence |
|---|---|---|---|---|
| 3 h | 0.163 m | 0.134 m | -0.032 m | +17.8% (95% range +15.8% to +19.7%) |
| 6 h | 0.258 m | 0.211 m | -0.043 m | +18.1% (95% range +15.5% to +20.4%) |

Command: `uv run buoycast-verify` (about 3 seconds, writes `artifacts/verification.json`).

## Words used

- **MAE** is the average size of the miss, in metres. **RMSE** is similar but
  punishes big misses more.
- **Bias** is the average of forecast minus observed. Negative means the
  forecast is usually too low.
- **Skill** is `1 - (model error) / (persistence error)`. +18% means the average
  error is 18% smaller than persistence. 0% means no better. Below 0% is worse.
- **95% range** is how far the skill number could move if I had a different
  set of days. I get it by resampling whole days with replacement 1,000 times,
  because hours in the same storm are not independent of each other.

## Where the model is reliable

Grouped by the wave height at the moment the forecast is issued (this is known
in advance, so it is a fair way to group; the table in the README that groups by the
height that actually happened looks at the outcome, so it is not).

3 hours ahead:

| Wave height now | Forecasts | Persistence MAE | MLP MAE | MLP bias | Skill (95% range) |
|---|---|---|---|---|---|
| 0 to 1 m | 7,349 | 0.108 | 0.087 | -0.032 | +19.2% (+17.2% to +21.1%) |
| 1 to 2 m | 6,512 | 0.136 | 0.115 | -0.019 | +15.1% (+12.3% to +17.9%) |
| 2 to 3 m | 1,878 | 0.230 | 0.184 | -0.023 | +20.0% (+15.4% to +24.3%) |
| over 3 m | 1,809 | 0.410 | 0.335 | -0.089 | +18.3% (+13.7% to +23.6%) |

6 hours ahead:

| Wave height now | Forecasts | Persistence MAE | MLP MAE | MLP bias | Skill (95% range) |
|---|---|---|---|---|---|
| 0 to 1 m | 7,346 | 0.170 | 0.146 | -0.034 | +14.0% (+11.3% to +16.5%) |
| 1 to 2 m | 6,507 | 0.215 | 0.180 | -0.023 | +16.5% (+13.4% to +19.9%) |
| 2 to 3 m | 1,877 | 0.379 | 0.297 | -0.036 | +21.8% (+16.8% to +26.2%) |
| over 3 m | 1,806 | 0.643 | 0.500 | -0.155 | +22.2% (+16.2% to +28.3%) |

What I take from this:

- The skill against persistence is positive in every height bin, and the ranges
  do not touch zero. Rough seas are where the model gains the most at 6 hours.
- The raw error grows quickly with wave height (a miss of half a metre at 6 hours
  when the sea is over 3 m). So the model is "better than persistence", which is
  not the same as "accurate enough for a decision".
- The model's bias is the weak spot. Over 3 m at 6 hours it forecasts 0.155 m too low
  on average, which is 31% of its average error. It was trained with a loss that
  does not punish under-forecasting extra, and big waves are rare in the training
  data. I have not tried to fix this.

## Where it is reliable by buoy

3 hours ahead (the 6 hour table is in `artifacts/verification.json` and on the
map page):

| Station | Forecasts | Persistence MAE | MLP MAE | Skill (95% range) |
|---|---|---|---|---|
| 44027 Jonesport ME | 4,405 | 0.139 | 0.113 | +18.5% (+14.4% to +22.9%) |
| 45005 West Erie | 2,799 | 0.133 | 0.100 | +24.8% (+22.1% to +27.7%) |
| 45012 East Lake Ontario | 2,532 | 0.147 | 0.112 | +23.3% (+19.7% to +27.1%) |
| 46005 West Washington | 3,401 | 0.177 | 0.151 | +14.3% (+10.8% to +17.8%) |
| 46041 Cape Elizabeth | 4,411 | 0.203 | 0.174 | +14.5% (+10.5% to +18.1%) |

The Great Lakes buoys (45005, 45012) gain most. The two Pacific buoys gain least
and also have the largest raw errors, which fits with ocean swell arriving from far
away that a model looking only at the local buoy cannot see coming. That
explanation is my guess; I have not tested it. Station 44005 has no test rows
(its data in the NDBC historical folder stops in March 2024).

## How it works

`src/buoycast/verify.py` rebuilds every test-period forecast with the saved
models (the same ones the API serves), lines it up with what happened, and scores
each method on groups of those forecasts. Each forecast is issued at hour `t` and
is for hour `t + lead`.

The report marks a group as "better than persistence" only if the whole 95% range of
its skill is above zero, "worse" if it is all below zero, "too few forecasts to say"
for fewer than 200, and "no clear difference" otherwise.

The `/verification` API endpoint returns the same JSON, and the map page shows it
in a small panel.

## Limits

- Only two lead times (3 and 6 hours), because those are the two models I
  trained. A curve over 1 to 24 hours would be more useful.
- One test period, which is the second half of 2024 and has no winter. The 95%
  ranges only reflect which days I happened to have, not what other years would do.
- The MLP being scored is one training run (seed 0), the one the API serves. The README
  table averages five seeds.
- Stations overlap in time: a storm usually hits several buoys on the same day,
  which the day resampling covers only partly.

## Tests

```
uv run pytest tests/test_verify.py -q
```

The tests check the maths on made-up data (perfect forecast has skill 1, bias
has the right sign, a clearly worse model is called worse) and that the real
forecasts line up with the numbers the training script reported.
