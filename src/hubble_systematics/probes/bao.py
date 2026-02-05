from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.design import DesignMatrix, ones
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params


_BAO_SPECS: dict[str, dict[str, Any]] = {
    "sdss_dr12_consensus_bao": {"predict": "dr12_consensus_bao", "rs_fid": 147.78},
    "sdss_dr16_lrg_bao_dmdh": {"predict": "dm_dh_over_rd"},
    "sdss_dr16_qso_bao_dmdh": {"predict": "dm_dh_over_rd"},
    "desi_2024_bao_all": {"predict": "desi_2024_all"},
}


@dataclass(frozen=True)
class BaoDataset:
    dataset: str
    z: np.ndarray
    value: np.ndarray
    obs: np.ndarray
    cov: np.ndarray

    def get_column(self, name: str) -> np.ndarray:
        if name == "z":
            return self.z
        raise KeyError(name)

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        y = np.asarray(self.value, dtype=float)
        y0 = _predict_bao(anchor, dataset=self.dataset, z=self.z, obs=self.obs)
        cov = np.asarray(self.cov, dtype=float)

        prior_cfg = (cfg or {}).get("priors", {}) or {}
        lvl = str(ladder_level).upper()
        shared_params = set(shared_scale_params(cfg or {}))

        cols: list[np.ndarray] = []
        names: list[str] = []

        if lvl != "L0":
            # Fractional scale per observable type (linearized).
            obs_types = np.unique(self.obs.astype("U"))
            for ot in obs_types:
                mask = (self.obs.astype("U") == ot).astype(float)
                cols.append((y0 * mask).astype(float))
                names.append(f"frac_{ot}")

        # Shared scale stress-test parameters.
        if "delta_lnH0" in shared_params or "delta_lnrd" in shared_params:
            obs_u = self.obs.astype("U")
            sign_H0 = np.zeros_like(y0)
            sign_rd = np.zeros_like(y0)
            for i, o in enumerate(obs_u):
                if o in {"DM_over_rs", "DH_over_rs", "DV_over_rs"}:
                    sign_H0[i] = -1.0
                    sign_rd[i] = -1.0
                elif o == "bao_Hz_rs":
                    sign_H0[i] = +1.0
                    sign_rd[i] = +1.0
                else:
                    raise ValueError(f"Unsupported BAO observable '{o}'")
            if "delta_lnH0" in shared_params:
                cols.append(sign_H0 * y0)
                names.append("delta_lnH0")
            if "delta_lnrd" in shared_params:
                cols.append(sign_rd * y0)
                names.append("delta_lnrd")

        X = np.stack(cols, axis=1) if cols else np.zeros((y.size, 0))

        if not names:
            prior = GaussianPrior.from_sigmas([], [], mean=0.0)
            return y, y0, cov, X, prior

        sigmas = []
        for n in names:
            if n.startswith("frac_"):
                sigmas.append(float(prior_cfg.get("sigma_bao_frac", 0.05)))
            elif n in shared_params:
                sigmas.append(float("inf"))
            else:
                sigmas.append(float(prior_cfg.get("sigma_default", 10.0)))
        prior = GaussianPrior.from_sigmas(names, sigmas, mean=0.0)
        return y, y0, cov, X, prior


def _predict_bao(anchor: AnchorLCDM, *, dataset: str, z: np.ndarray, obs: np.ndarray) -> np.ndarray:
    spec = _BAO_SPECS.get(dataset)
    if spec is None:
        raise ValueError(f"Unknown BAO dataset: {dataset}")

    z = np.asarray(z, dtype=float)
    obs = np.asarray(obs).astype("U")
    out = np.empty_like(z, dtype=float)

    mode = spec["predict"]
    if mode == "dr12_consensus_bao":
        rs_fid = float(spec["rs_fid"])
        for i, (zi, oi) in enumerate(zip(z, obs, strict=True)):
            if oi == "DM_over_rs":
                out[i] = anchor.cosmo.transverse_comoving_distance(np.array([zi]))[0] * (rs_fid / anchor.rd_Mpc)
            elif oi == "bao_Hz_rs":
                out[i] = anchor.H(np.array([zi]))[0] * (anchor.rd_Mpc / rs_fid)
            else:
                raise ValueError(f"Unsupported obs '{oi}' for {dataset}")
        return out

    if mode == "dm_dh_over_rd":
        for i, (zi, oi) in enumerate(zip(z, obs, strict=True)):
            if oi == "DM_over_rs":
                out[i] = anchor.DM_over_rd(np.array([zi]))[0]
            elif oi == "DH_over_rs":
                out[i] = anchor.DH_over_rd(np.array([zi]))[0]
            else:
                raise ValueError(f"Unsupported obs '{oi}' for {dataset}")
        return out

    if mode == "desi_2024_all":
        for i, (zi, oi) in enumerate(zip(z, obs, strict=True)):
            if oi == "DM_over_rs":
                out[i] = anchor.DM_over_rd(np.array([zi]))[0]
            elif oi == "DH_over_rs":
                out[i] = anchor.DH_over_rd(np.array([zi]))[0]
            elif oi == "DV_over_rs":
                out[i] = anchor.DV_over_rd(np.array([zi]))[0]
            else:
                raise ValueError(f"Unsupported obs '{oi}' for {dataset}")
        return out

    raise ValueError(f"Unknown BAO predict mode '{mode}'")


def load_bao_dataset(probe_cfg: dict[str, Any]) -> BaoDataset:
    data_path = Path(probe_cfg["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[3] / data_path).resolve()
    meta_path = probe_cfg.get("meta_path")
    dataset_name = probe_cfg.get("dataset")
    if meta_path and dataset_name is None:
        mp = Path(meta_path).expanduser()
        if not mp.is_absolute():
            mp = (Path(__file__).resolve().parents[3] / mp).resolve()
        meta = json.loads(mp.read_text())
        dataset_name = meta.get("dataset")
    if dataset_name is None:
        raise ValueError("BAO probe requires either probe.dataset or probe.meta_path with a dataset field")

    d = np.load(data_path, allow_pickle=True)
    z = np.asarray(d["z"], dtype=float)
    value = np.asarray(d["value"], dtype=float)
    obs = np.asarray(d["obs"]).astype("U")
    cov = np.asarray(d["cov"], dtype=float)

    z_min = probe_cfg.get("z_min")
    z_max = probe_cfg.get("z_max")
    mask = np.ones_like(z, dtype=bool)
    if z_min is not None:
        mask &= z >= float(z_min)
    if z_max is not None:
        mask &= z <= float(z_max)
    idx = np.where(mask)[0]
    return BaoDataset(
        dataset=str(dataset_name),
        z=z[idx],
        value=value[idx],
        obs=obs[idx],
        cov=cov[np.ix_(idx, idx)],
    )
