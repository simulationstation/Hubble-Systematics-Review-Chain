from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params


@dataclass(frozen=True)
class H0GridPosteriorDataset:
    label: str
    H0_grid: np.ndarray
    posterior: np.ndarray
    space: str = "linear"  # "linear" (default) or "ln"

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        shared_params = set(shared_scale_params(cfg or {}))

        space = str(self.space).lower()
        if space == "ln":
            m, sd = _grid_mean_sd_ln(self.H0_grid, self.posterior)
            y = np.array([m], dtype=float)
            y0 = np.array([float(np.log(anchor.H0))], dtype=float)
            cov = np.array([float(sd**2)], dtype=float)
        elif space == "linear":
            H0_mean, H0_sd = _grid_mean_sd(self.H0_grid, self.posterior)
            y = np.array([H0_mean], dtype=float)
            y0 = np.array([float(anchor.H0)], dtype=float)
            cov = np.array([float(H0_sd**2)], dtype=float)  # diagonal variance
        else:
            raise ValueError(f"Unsupported h0_grid.space: {self.space!r}")

        cols: list[np.ndarray] = []
        names: list[str] = []
        if "delta_lnH0" in shared_params:
            if space == "ln":
                cols.append(np.ones_like(y0))
            else:
                cols.append(y0.copy())
            names.append("delta_lnH0")

        X = np.stack(cols, axis=1) if cols else np.zeros((1, 0))
        if not names:
            prior = GaussianPrior.from_sigmas([], [], mean=0.0)
            return y, y0, cov, X, prior

        # No per-probe priors on shared params; applied globally by the runner.
        prior = GaussianPrior.from_sigmas(names, [float("inf")] * len(names), mean=0.0)
        return y, y0, cov, X, prior


def load_h0_grid_posterior_dataset(probe_cfg: dict[str, Any]) -> H0GridPosteriorDataset:
    data_path = Path(probe_cfg["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[3] / data_path).resolve()
    label = str(probe_cfg.get("label", data_path.stem))
    d = json.loads(data_path.read_text())
    H0_grid = np.asarray(d["H0_grid"], dtype=float)
    posterior = np.asarray(d["posterior"], dtype=float)
    space = str(probe_cfg.get("space", "linear"))
    return H0GridPosteriorDataset(label=label, H0_grid=H0_grid, posterior=posterior, space=space)


def _grid_mean_sd(H0: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    H0 = np.asarray(H0, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if H0.shape != p.shape:
        raise ValueError("H0_grid and posterior length mismatch")
    if not (np.all(np.isfinite(H0)) and np.all(np.isfinite(p))):
        raise ValueError("Non-finite values in H0 grid or posterior")
    if not np.any(p > 0):
        raise ValueError("Posterior has no positive support")
    # Normalize on the grid using trapezoid integration.
    Z = float(np.trapezoid(p, H0))
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("Posterior normalization failed")
    p = p / Z
    m = float(np.trapezoid(p * H0, H0))
    v = float(np.trapezoid(p * (H0 - m) ** 2, H0))
    return m, float(np.sqrt(max(v, 0.0)))


def _grid_mean_sd_ln(H0: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    H0 = np.asarray(H0, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if H0.shape != p.shape:
        raise ValueError("H0_grid and posterior length mismatch")
    if not (np.all(np.isfinite(H0)) and np.all(H0 > 0) and np.all(np.isfinite(p))):
        raise ValueError("Non-finite values in H0 grid or posterior")
    if not np.any(p > 0):
        raise ValueError("Posterior has no positive support")
    Z = float(np.trapezoid(p, H0))
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("Posterior normalization failed")
    p = p / Z
    x = np.log(H0)
    m = float(np.trapezoid(p * x, H0))
    v = float(np.trapezoid(p * (x - m) ** 2, H0))
    return m, float(np.sqrt(max(v, 0.0)))
