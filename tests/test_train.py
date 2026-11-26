import json

import pytest

from buoycast.paths import SAMPLE_PARQUET
from buoycast.train import format_table, run


def test_tiny_training_run_writes_metrics_and_models(tmp_path):
    """Smoke test used in CI: 1 seed, 2 epochs, on the committed sample."""
    result = run(SAMPLE_PARQUET, tmp_path, seeds=1, epochs=2)
    for h in ("3", "6"):
        e = result["horizons"][h]
        for name in ("persistence", "linear"):
            assert e[name]["mae"] > 0 and e[name]["rmse"] >= e[name]["mae"]
        assert e["mlp"]["mae"]["mean"] > 0
        assert (tmp_path / f"model_h{h}.pt").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["seeds"] == 1
    assert "Persistence" in format_table(result)


def test_training_is_reproducible(tmp_path):
    a = run(SAMPLE_PARQUET, tmp_path / "a", seeds=1, epochs=2)
    b = run(SAMPLE_PARQUET, tmp_path / "b", seeds=1, epochs=2)
    assert a["horizons"]["3"]["mlp"]["mae"]["mean"] == pytest.approx(
        b["horizons"]["3"]["mlp"]["mae"]["mean"]
    )
