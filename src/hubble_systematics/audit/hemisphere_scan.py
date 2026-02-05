from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class HemisphereScanResult:
    nside: int
    frame: str
    param: str
    train_frac: float
    best_axis: dict[str, float]
    train: dict[str, float]
    test: dict[str, float]
    top_axes_train: list[dict[str, float]]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "nside": self.nside,
            "frame": self.frame,
            "param": self.param,
            "train_frac": self.train_frac,
            "best_axis": self.best_axis,
            "train": self.train,
            "test": self.test,
            "top_axes_train": self.top_axes_train,
        }


def run_hemisphere_scan(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    hemi_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> HemisphereScanResult:
    if dataset.ra_deg is None or dataset.dec_deg is None:
        raise ValueError("hemisphere_scan requires ra/dec in the dataset")

    nside = int(hemi_cfg.get("nside", 4))
    frame = str(hemi_cfg.get("frame", "galactic"))
    param = str(hemi_cfg.get("param", "global_offset_mag"))
    train_frac = float(hemi_cfg.get("train_frac", 0.7))
    use_diag = bool(hemi_cfg.get("use_diagonal_errors", True))
    z_match = bool(hemi_cfg.get("z_match", True))
    z_bins = int(hemi_cfg.get("z_match_bins", 10))
    top_k = int(hemi_cfg.get("top_k", 10))

    idx_all = np.arange(dataset.z.size)
    train_mask = rng.random(dataset.z.size) < train_frac
    idx_train = idx_all[train_mask]
    idx_test = idx_all[~train_mask]
    if idx_train.size < 10 or idx_test.size < 10:
        raise ValueError("train/test split too small; adjust train_frac or dataset selection")

    vec = _unit_vectors(dataset=dataset, frame=frame)
    # Healpix axes in the same frame.
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat = 0.5 * np.pi - theta
    lon = phi
    ax_vec = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)

    train_scores = np.empty(npix, dtype=float)
    for pidx in range(npix):
        train_scores[pidx] = _axis_score(
            idx=idx_train,
            vec=vec,
            axis=ax_vec[pidx],
            dataset=dataset,
            anchor=anchor,
            ladder_level=ladder_level,
            model_cfg=model_cfg,
            param=param,
            use_diag=use_diag,
            z_match=z_match,
            z_bins=z_bins,
            rng=rng,
        )

    best_pix = int(np.argmax(np.abs(train_scores)))
    best_axis = {"lon_deg": float(np.degrees(lon[best_pix])), "lat_deg": float(np.degrees(lat[best_pix]))}

    train_z = float(train_scores[best_pix])
    test_z = float(
        _axis_score(
            idx=idx_test,
            vec=vec,
            axis=ax_vec[best_pix],
            dataset=dataset,
            anchor=anchor,
            ladder_level=ladder_level,
            model_cfg=model_cfg,
            param=param,
            use_diag=use_diag,
            z_match=z_match,
            z_bins=z_bins,
            rng=rng,
        )
    )

    top = np.argsort(-np.abs(train_scores))[:top_k]
    top_axes = [
        {
            "lon_deg": float(np.degrees(lon[i])),
            "lat_deg": float(np.degrees(lat[i])),
            "z_train": float(train_scores[i]),
        }
        for i in top
    ]

    return HemisphereScanResult(
        nside=nside,
        frame=frame,
        param=param,
        train_frac=train_frac,
        best_axis=best_axis,
        train={"z": train_z},
        test={"z": test_z},
        top_axes_train=top_axes,
    )


def _unit_vectors(*, dataset, frame: str) -> np.ndarray:
    coords = SkyCoord(ra=dataset.ra_deg * u.deg, dec=dataset.dec_deg * u.deg, frame="icrs")
    if frame == "galactic":
        g = coords.galactic
        lon = g.l.to_value(u.rad)
        lat = g.b.to_value(u.rad)
    elif frame == "icrs":
        lon = coords.ra.to_value(u.rad)
        lat = coords.dec.to_value(u.rad)
    else:
        raise ValueError(f"Unsupported frame: {frame}")
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _axis_score(
    *,
    idx: np.ndarray,
    vec: np.ndarray,
    axis: np.ndarray,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    param: str,
    use_diag: bool,
    z_match: bool,
    z_bins: int,
    rng: np.random.Generator,
) -> float:
    dot = vec[idx] @ axis
    fore = idx[dot >= 0]
    aft = idx[dot < 0]
    if fore.size < 10 or aft.size < 10:
        return 0.0

    if z_match:
        fore, aft = _z_match_indices(dataset.z, fore, aft, n_bins=z_bins, rng=rng)
        if fore.size < 10 or aft.size < 10:
            return 0.0

    mf, vf = _fit_param(dataset, mask=_mask_from_indices(dataset.z.size, fore), anchor=anchor, ladder_level=ladder_level, model_cfg=model_cfg, param=param, use_diag=use_diag)
    ma, va = _fit_param(dataset, mask=_mask_from_indices(dataset.z.size, aft), anchor=anchor, ladder_level=ladder_level, model_cfg=model_cfg, param=param, use_diag=use_diag)
    if vf <= 0 or va <= 0:
        return 0.0
    delta = mf - ma
    sd = float(np.sqrt(vf + va))
    return float(delta / sd)


def _fit_param(dataset, *, mask: np.ndarray, anchor: AnchorLCDM, ladder_level: str, model_cfg: dict[str, Any], param: str, use_diag: bool) -> tuple[float, float]:
    sub = dataset.subset_mask(mask)
    y, y0, cov, X, prior = sub.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
    if use_diag:
        cov = sub.diag_sigma() ** 2
    fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=cov, X=X, prior=prior))
    if param not in fit.param_names:
        raise ValueError(f"Requested param '{param}' not in fit params")
    j = fit.param_names.index(param)
    return float(fit.mean[j]), float(fit.cov[j, j])


def _mask_from_indices(n: int, idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def _z_match_indices(z: np.ndarray, fore: np.ndarray, aft: np.ndarray, *, n_bins: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    edges = np.quantile(z, np.linspace(0.0, 1.0, n_bins + 1))
    fore_keep = []
    aft_keep = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (z >= lo) & (z <= hi)
        else:
            in_bin = (z >= lo) & (z < hi)
        f = fore[in_bin[fore]]
        a = aft[in_bin[aft]]
        m = min(f.size, a.size)
        if m == 0:
            continue
        fore_keep.append(rng.choice(f, size=m, replace=False))
        aft_keep.append(rng.choice(a, size=m, replace=False))
    if not fore_keep or not aft_keep:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(fore_keep), np.concatenate(aft_keep)
