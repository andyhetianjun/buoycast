import json
import re

import pandas as pd

from buoycast import verify
from buoycast.site import build_site


def test_site_is_self_contained_and_matches_the_forecasts(tmp_path):
    build_site(tmp_path)
    page = (tmp_path / "index.html").read_text()
    # Local assets must be relative so the site works under /<repo>/ on GitHub Pages.
    local = re.findall(r'(?:src|href)="([^"]+)"', page)
    assert all(u.startswith(("http", "style", "app", "config")) for u in local)
    assert "BUOYCAST_STATIC = true" in (tmp_path / "config.js").read_text()
    assert 'fetch("/' not in (tmp_path / "app.js").read_text()

    meta = json.loads((tmp_path / "data" / "index.json").read_text())
    assert len(meta["stations"]) == 6 and len(meta["drifters"]) == 5
    assert meta["verification"]["leads"]["3"]["overall"]["n"] > 10000
    assert all(d["flagged"] == 0 for d in meta["drifters"])
    assert meta["operational"]["leads"]["6"]["overall"]["methods"]["gfswave"]["mae"] > 0

    d = json.loads((tmp_path / "data" / "46041.json").read_text())
    assert len(d["wvht"]) == d["hours"] and all(len(v) == d["hours"] for v in d["qc"].values())
    assert set("".join(d["qc"].values())) <= set("12349")

    # A stored forecast equals the one from the forecast table, at its valid hour.
    row = verify.forecast_table(horizon=3).query("station == '46041'").iloc[1000]
    i = int((row["valid"] - pd.Timestamp(d["start"])) / pd.Timedelta(hours=1))
    assert d["forecast"]["h3"]["mlp"][i - d["forecast"]["offset"]] == round(float(row["mlp"]), 2)
    assert d["wvht"][i] is None or abs(d["wvht"][i] - row["observed"]) < 0.006
