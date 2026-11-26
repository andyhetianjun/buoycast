"""Baselines and the small PyTorch model.

All three predict wave height. The linear model and the MLP both work on the
change from the current wave height (target minus wvht_lag0), which makes the
persistence forecast their starting point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import feature_names
from .paths import STATIONS

STATION_IDS = sorted(STATIONS)


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature columns plus a one-hot station id, as float32."""
    x = df[feature_names()].to_numpy(dtype=np.float32)
    onehot = np.stack([(df["station"] == s).to_numpy() for s in STATION_IDS], axis=1)
    return np.concatenate([x, onehot.astype(np.float32)], axis=1)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std < 1e-6, 1.0, std))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def persistence(df: pd.DataFrame) -> np.ndarray:
    """Forecast that the wave height stays what it is now."""
    return df["wvht_lag0"].to_numpy()


class LinearModel:
    """Ordinary least squares (tiny ridge for stability) on the change in wave height."""

    def __init__(self) -> None:
        self.scaler: Scaler | None = None
        self.weights: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> LinearModel:
        x = design_matrix(df)
        self.scaler = Scaler.fit(x)
        z = np.c_[self.scaler.apply(x), np.ones(len(x), dtype=np.float32)]
        y = (df["target"] - df["wvht_lag0"]).to_numpy()
        reg = 1e-3 * np.eye(z.shape[1])
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(z.T @ z + reg, z.T @ y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None and self.weights is not None
        z = np.c_[self.scaler.apply(design_matrix(df)), np.ones(len(df), dtype=np.float32)]
        return np.maximum(df["wvht_lag0"].to_numpy() + z @ self.weights, 0.0)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPForecaster:
    """MLP plus the scaler and delta scale needed to turn its output into metres."""

    model: MLP
    scaler: Scaler
    delta_scale: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.scaler.apply(design_matrix(df)))
            delta = self.model(x).numpy() * self.delta_scale
        return np.maximum(df["wvht_lag0"].to_numpy() + delta, 0.0)


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def train_mlp(
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 512,
    lr: float = 2e-3,
) -> MLPForecaster:
    """Train on the train split; keep the epoch with the best validation MAE."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x_raw = design_matrix(train)
    scaler = Scaler.fit(x_raw)
    delta = (train["target"] - train["wvht_lag0"]).to_numpy(dtype=np.float32)
    delta_scale = float(delta.std()) or 1.0
    x = torch.from_numpy(scaler.apply(x_raw))
    y = torch.from_numpy(delta / delta_scale)

    model = MLP(x.shape[1])
    fc = MLPForecaster(model, scaler, delta_scale)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_mae, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()
        val_mae = _mae(fc.predict(val), val["target"].to_numpy())
        if val_mae < best_mae - 1e-5:
            best_mae, stale = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return fc


def save_bundle(path, fc: MLPForecaster, linear: LinearModel, horizon: int) -> None:
    assert linear.scaler is not None and linear.weights is not None
    torch.save(
        {
            "horizon": horizon,
            "mlp_state": fc.model.state_dict(),
            "mlp_in": fc.scaler.mean.shape[0],
            "mlp_mean": fc.scaler.mean,
            "mlp_std": fc.scaler.std,
            "delta_scale": fc.delta_scale,
            "lin_mean": linear.scaler.mean,
            "lin_std": linear.scaler.std,
            "lin_weights": linear.weights,
        },
        path,
    )


def load_bundle(path) -> tuple[MLPForecaster, LinearModel]:
    b = torch.load(path, weights_only=False)
    mlp = MLP(b["mlp_in"])
    mlp.load_state_dict(b["mlp_state"])
    fc = MLPForecaster(mlp, Scaler(b["mlp_mean"], b["mlp_std"]), b["delta_scale"])
    lin = LinearModel()
    lin.scaler = Scaler(b["lin_mean"], b["lin_std"])
    lin.weights = b["lin_weights"]
    return fc, lin
