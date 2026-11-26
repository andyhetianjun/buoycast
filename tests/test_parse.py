import gzip

import pandas as pd

from buoycast.parse import parse_stdmet, to_hourly

NEW_FORMAT = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft
2023 01 01 00 10 350  8.0  9.7  1.20  4.76  4.00 200 1010.9  10.4   8.6  10.4 99.0 99.00
2023 01 01 00 50  10  6.0  7.0  1.40  4.55  4.22 999 1010.3   9.9   8.6  9.9 99.0 99.00
2023 01 01 01 50 999 99.0 99.0 99.00 99.00 99.00 999 9999.0 999.0 999.0 999.0 99.0 99.00
"""

OLD_FORMAT = """YY  MM DD hh WD WSPD GST WVHT  DPD  APD MWD  BAR  ATMP  WTMP  DEWP  VIS
98 01 01 00 180  5.0  6.0 2.10 8.00 5.50 999 1015.0 3.0 4.0 999.0 99.0
"""


def test_parses_new_format_and_missing_codes():
    df = parse_stdmet(NEW_FORMAT)
    assert len(df) == 3
    assert df.index.tz is not None
    assert df.index[0] == pd.Timestamp("2023-01-01 00:10", tz="UTC")
    assert df["wvht"].iloc[0] == 1.2
    # An all-9s row must be entirely NaN, not 99 m waves.
    assert df.iloc[2].isna().all()
    assert df["wvht"].max() < 50


def test_parses_old_format_with_two_digit_year():
    df = parse_stdmet(OLD_FORMAT)
    assert df.index[0] == pd.Timestamp("1998-01-01 00:00", tz="UTC")
    assert df["pres"].iloc[0] == 1015.0


def test_reads_gzip_bytes():
    df = parse_stdmet(gzip.compress(NEW_FORMAT.encode()))
    assert len(df) == 3


def test_hourly_average_and_circular_wind_direction():
    hourly = to_hourly(parse_stdmet(NEW_FORMAT))
    first = hourly.iloc[0]
    assert abs(first["wvht"] - 1.3) < 1e-9
    # 350 and 10 degrees average to north (0/360), not 180.
    assert min(first["wdir"], 360 - first["wdir"]) < 1.0


def test_hourly_grid_is_complete_and_trimmed():
    text = NEW_FORMAT.replace("2023 01 01 01 50", "2023 01 01 05 50")
    hourly = to_hourly(parse_stdmet(text))
    # The row with no wave reading is at the end, so it is trimmed away.
    assert hourly.index[-1] == pd.Timestamp("2023-01-01 00:00", tz="UTC")
    assert hourly.index.freq is not None or len(hourly) == 1
