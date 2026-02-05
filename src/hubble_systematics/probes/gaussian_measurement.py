from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params


@dataclass(frozen=True)
class GaussianMeasurementDataset:
    label: str
    quantity: str
    mean: float
    sigma: float
    space: str = "ln"  # "ln" or "linear"

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        shared_params = set(shared_scale_params(cfg or {}))

        q = str(self.quantity).lower()
        space = str(self.space).lower()
        if q not in {"h0", "rd"}:
            raise ValueError(f"Unsupported gaussian_measurement.quantity: {self.quantity!r}")
        if space not in {"ln", "linear"}:
            raise ValueError(f"Unsupported gaussian_measurement.space: {self.space!r}")

        if q == "h0":
            base = float(anchor.H0)
            shared_name = "delta_lnH0"
        else:
            base = float(anchor.rd_Mpc)
            shared_name = "delta_lnrd"

        if space == "ln":
            y = np.array([float(np.log(self.mean))], dtype=float)
            y0 = np.array([float(np.log(base))], dtype=float)
            cov = np.array([float((self.sigma / self.mean) ** 2)], dtype=float)
            X = np.zeros((1, 0), dtype=float)
            names: list[str] = []
            if shared_name in shared_params:
                X = np.ones((1, 1), dtype=float)
                names = [shared_name]
        else:
            y = np.array([float(self.mean)], dtype=float)
            y0 = np.array([float(base)], dtype=float)
            cov = np.array([float(self.sigma**2)], dtype=float)
            X = np.zeros((1, 0), dtype=float)
            names = []
            if shared_name in shared_params:
                X = y0.reshape(1, 1)  # linearized: base*exp(d) ≈ base + base*d
                names = [shared_name]

        # No per-probe priors on shared params; applied globally by the runner.
        prior = GaussianPrior.from_sigmas(names, [float("inf")] * len(names), mean=0.0)
        return y, y0, cov, X, prior


def load_gaussian_measurement_dataset(probe_cfg: dict[str, Any]) -> GaussianMeasurementDataset:
    label = str(probe_cfg.get("label", "gaussian_measurement"))
    quantity = str(probe_cfg.get("quantity", "H0"))
    mean = float(probe_cfg["mean"])
    sigma = float(probe_cfg["sigma"])
    space = str(probe_cfg.get("space", "ln"))
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("gaussian_measurement.sigma must be positive and finite")
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError("gaussian_measurement.mean must be positive and finite")
    return GaussianMeasurementDataset(label=label, quantity=quantity, mean=mean, sigma=sigma, space=space)

