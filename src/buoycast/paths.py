"""Project paths and the fixed list of stations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PARQUET = ROOT / "data" / "sample" / "hourly.parquet"
NNJA_PARQUET = ROOT / "data" / "sample" / "nnja_metar.parquet"
DRIFTERS_PARQUET = ROOT / "data" / "sample" / "drifters.parquet"
DRIFTERS_QC_PARQUET = ROOT / "data" / "sample" / "drifters_qc.parquet"
OPERATIONAL_CSV = ROOT / "data" / "sample" / "gfswave_points.csv"
QC_FLAGS_PARQUET = ROOT / "data" / "sample" / "qc_flags.parquet"
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
# Trained model files (about 32 KB each). Inside the package so an installed copy has them.
WEIGHTS = Path(__file__).resolve().parent / "weights"
METRICS_JSON = ARTIFACTS / "metrics.json"
QC_SUMMARY_JSON = ARTIFACTS / "qc_summary.json"
VERIFICATION_JSON = ARTIFACTS / "verification.json"
OPERATIONAL_JSON = ARTIFACTS / "operational.json"
WEB_DIR = ROOT / "web"

# Name and position from the NDBC active stations list.
STATIONS: dict[str, dict[str, object]] = {
    "44005": {"name": "Gulf of Maine, 78 NM east of Portsmouth NH", "lat": 43.201, "lon": -69.127},
    "44027": {"name": "Jonesport ME, 20 NM SE", "lat": 44.284, "lon": -67.301},
    "45005": {"name": "West Erie, 16 NM NW of Lorain OH", "lat": 41.677, "lon": -82.398},
    "45012": {
        "name": "East Lake Ontario, 20 NM NNE of Rochester NY",
        "lat": 43.621,
        "lon": -77.401,
    },
    "46041": {"name": "Cape Elizabeth, 45 NM NW of Aberdeen WA", "lat": 47.351, "lon": -124.741},
    "46005": {
        "name": "West Washington, 300 NM west of Aberdeen WA",
        "lat": 46.147,
        "lon": -131.077,
    },
}
YEARS = (2022, 2023, 2024)
