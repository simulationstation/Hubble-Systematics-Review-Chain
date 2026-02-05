from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params


@dataclass(frozen=True)
class ChronometersDataset:
    z: np.ndarray
    H: np.ndarray
    sigma_H: np.ndarray

    def get_column(self, name: str) -> np.ndarray:
        if name == "z":
            return self.z
        raise KeyError(name)

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        y = np.asarray(self.H, dtype=float)
        y0 = np.asarray(anchor.H(self.z), dtype=float)
        cov = np.asarray(self.sigma_H, dtype=float) ** 2

        prior_cfg = (cfg or {}).get("priors", {}) or {}
        lvl = str(ladder_level).upper()
        shared_params = set(shared_scale_params(cfg or {}))

        cols: list[np.ndarray] = []
        names: list[str] = []

        if lvl != "L0":
            # Single fractional scaling parameter for H(z): H_obs ≈ H_anchor * (1 + frac_H).
            cols.append(y0.copy())
            names.append("frac_H")

        if "delta_lnH0" in shared_params:
            cols.append(y0.copy())
            names.append("delta_lnH0")

        X = np.stack(cols, axis=1) if cols else np.zeros((y.size, 0))
        if not names:
            prior = GaussianPrior.from_sigmas([], [], mean=0.0)
            return y, y0, cov, X, prior

        sigmas = []
        for n in names:
            if n == "frac_H":
                sigmas.append(float(prior_cfg.get("sigma_cc_frac", 0.2)))
            elif n in shared_params:
                sigmas.append(float("inf"))
            else:
                sigmas.append(float(prior_cfg.get("sigma_default", 10.0)))
        prior = GaussianPrior.from_sigmas(names, sigmas, mean=0.0)
        return y, y0, cov, X, prior


def load_chronometers_dataset(probe_cfg: dict[str, Any]) -> ChronometersDataset:
    data_path = Path(probe_cfg["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[3] / data_path).resolve()
    d = np.load(data_path)
    z = np.asarray(d["z"], dtype=float)
    H = np.asarray(d["H"], dtype=float)
    s = np.asarray(d["sigma_H"], dtype=float)

    z_min = probe_cfg.get("z_min")
    z_max = probe_cfg.get("z_max")
    mask = np.ones_like(z, dtype=bool)
    if z_min is not None:
        mask &= z >= float(z_min)
    if z_max is not None:
        mask &= z <= float(z_max)
    idx = np.where(mask)[0]

    return ChronometersDataset(z=z[idx], H=H[idx], sigma_H=s[idx])
