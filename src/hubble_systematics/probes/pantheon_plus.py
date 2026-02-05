from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.design import DesignMatrix, bspline_basis, one_hot_levels, ones, second_difference_precision, sky_real_harmonics
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.prior_overrides import load_sigma_overrides, sigma_from_mapping
from hubble_systematics.shared_scale import shared_scale_params


@dataclass(frozen=True)
class PantheonPlusDataset:
    z: np.ndarray
    m_b_corr: np.ndarray
    cov: np.ndarray
    ra_deg: np.ndarray | None = None
    dec_deg: np.ndarray | None = None
    idsurvey: np.ndarray | None = None
    z_support_min: float | None = None
    z_support_max: float | None = None
    idsurvey_levels: np.ndarray | None = None

    def get_column(self, name: str) -> np.ndarray:
        if name == "z":
            return self.z
        if name == "m_b_corr":
            return self.m_b_corr
        if name == "idsurvey":
            if self.idsurvey is None:
                raise ValueError("idsurvey not available in this dataset")
            return self.idsurvey.astype(float)
        raise KeyError(f"Unknown column: {name}")

    def subset_leq(self, col: str, value: float) -> "PantheonPlusDataset":
        v = self.get_column(col)
        mask = v <= float(value)
        return self.subset_mask(mask)

    def subset_geq(self, col: str, value: float) -> "PantheonPlusDataset":
        v = self.get_column(col)
        mask = v >= float(value)
        return self.subset_mask(mask)

    def subset_mask(self, mask: np.ndarray) -> "PantheonPlusDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.z.shape:
            raise ValueError("mask shape mismatch")
        idx = np.where(mask)[0]
        cov = self.cov[np.ix_(idx, idx)]
        ra = self.ra_deg[idx] if self.ra_deg is not None else None
        dec = self.dec_deg[idx] if self.dec_deg is not None else None
        ids = self.idsurvey[idx] if self.idsurvey is not None else None
        return PantheonPlusDataset(
            z=self.z[idx],
            m_b_corr=self.m_b_corr[idx],
            cov=cov,
            ra_deg=ra,
            dec_deg=dec,
            idsurvey=ids,
            z_support_min=self.z_support_min,
            z_support_max=self.z_support_max,
            idsurvey_levels=self.idsurvey_levels,
        )

    def diag_sigma(self) -> np.ndarray:
        d = np.diag(self.cov)
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError("Invalid covariance diagonal")
        return np.sqrt(d)

    def build_design(
        self,
        *,
        anchor: AnchorLCDM,
        ladder_level: str,
        cfg: dict[str, Any],
    ):
        y = np.asarray(self.m_b_corr, dtype=float)
        y0 = np.asarray(anchor.mu(self.z), dtype=float)
        cov = np.asarray(self.cov, dtype=float)

        prior_cfg = (cfg or {}).get("priors", {}) or {}
        lvl = str(ladder_level).upper()

        X = DesignMatrix(X=np.zeros((y.size, 0)), names=[])

        # Shared scale stress-test params (e.g., delta_lnH0).
        shared_params = set(shared_scale_params(cfg or {}))

        def add_global_offset() -> None:
            nonlocal X
            X = X.append(DesignMatrix(X=ones(y.size), names=["global_offset_mag"]))

        def add_survey_offsets() -> None:
            nonlocal X
            if self.idsurvey is None:
                raise ValueError("idsurvey required for L2+")
            if self.idsurvey_levels is None:
                raise ValueError("idsurvey_levels required for L2+")
            ref = prior_cfg.get("survey_reference")
            dm = one_hot_levels(self.idsurvey, levels=self.idsurvey_levels, prefix="survey_offset_", reference=ref, drop_reference=True)
            X = X.append(dm)

        def add_z_spline() -> None:
            nonlocal X
            zcfg = (cfg or {}).get("z_spline", {}) or {}
            n_knots = int(zcfg.get("n_internal_knots", 6))
            xmin = float(self.z_support_min) if self.z_support_min is not None else float(np.min(self.z))
            xmax = float(self.z_support_max) if self.z_support_max is not None else float(np.max(self.z))
            dm = bspline_basis(self.z, xmin=xmin, xmax=xmax, n_internal_knots=n_knots, prefix="sn_z_spline_")
            X = X.append(dm)

        def add_sky_modes() -> None:
            nonlocal X
            if self.ra_deg is None or self.dec_deg is None:
                raise ValueError("ra_deg/dec_deg required for sky modes")
            skycfg = (cfg or {}).get("sky", {}) or {}
            lmax = int(skycfg.get("lmax", 3))
            lmin = int(skycfg.get("lmin", 2))
            frame = str(skycfg.get("frame", "galactic"))
            coords = SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg, frame="icrs")
            if frame == "galactic":
                g = coords.galactic
                lon = g.l.to_value(u.rad)
                lat = g.b.to_value(u.rad)
            elif frame == "icrs":
                lon = coords.ra.to_value(u.rad)
                lat = coords.dec.to_value(u.rad)
            else:
                raise ValueError(f"Unsupported sky.frame: {frame}")
            theta = 0.5 * np.pi - lat
            phi = lon
            dm = sky_real_harmonics(theta_rad=theta, phi_rad=phi, lmin=lmin, lmax=lmax, prefix="sky_")
            X = X.append(dm)

        def add_shared_scale() -> None:
            nonlocal X
            if "delta_lnH0" in shared_params:
                # mu(H0*e^d) = mu(H0) - (5/ln 10) d  [exact]
                dmu_dlnH0 = -5.0 / np.log(10.0)
                X = X.append(DesignMatrix(X=np.full((y.size, 1), dmu_dlnH0), names=["delta_lnH0"]))

        if lvl == "L0":
            param_names: list[str] = []
        elif lvl == "L1":
            add_global_offset()
        elif lvl == "L2":
            add_global_offset()
            add_survey_offsets()
        elif lvl == "L3":
            add_global_offset()
            add_survey_offsets()
            add_z_spline()
        elif lvl == "L4":
            add_global_offset()
            add_survey_offsets()
            add_z_spline()
            add_sky_modes()
        else:
            raise ValueError(f"Unknown ladder level: {ladder_level}")

        add_shared_scale()

        # Priors (Gaussian; with optional smoothness for z-spline).
        names = X.names
        if not names:
            prior = GaussianPrior.from_sigmas([], [], mean=0.0)
            return y, y0, cov, X.X, prior

        sigma_overrides = load_sigma_overrides(prior_cfg)
        sigma_survey_by_id = prior_cfg.get("sigma_survey_offset_mag_by_idsurvey")

        sigmas = []
        for n in names:
            if n in sigma_overrides:
                sigmas.append(float(sigma_overrides[n]))
                continue
            if n == "global_offset_mag":
                sigmas.append(float(prior_cfg.get("sigma_global_offset_mag", 10.0)))
            elif n in shared_params:
                sigmas.append(float("inf"))
            elif n.startswith("survey_offset_"):
                sid = n.removeprefix("survey_offset_")
                sigma = None
                try:
                    sigma = sigma_from_mapping(sigma_survey_by_id, int(sid))
                except Exception:
                    sigma = None
                if sigma is None:
                    sigma = float(prior_cfg.get("sigma_survey_offset_mag", 0.2))
                sigmas.append(float(sigma))
            elif n.startswith("sn_z_spline_"):
                sigmas.append(float(prior_cfg.get("sigma_z_spline_mag", 0.2)))
            elif n.startswith("sky_"):
                sigmas.append(float(prior_cfg.get("sigma_sky_mode_mag", 0.2)))
            else:
                sigmas.append(float(prior_cfg.get("sigma_default", 10.0)))

        prior = GaussianPrior.from_sigmas(names, sigmas, mean=0.0)

        # Add a second-difference smoothness penalty for z-spline coefficients (if present).
        z_idx = [i for i, n in enumerate(names) if n.startswith("sn_z_spline_")]
        if z_idx:
            sigma_d2 = float(prior_cfg.get("sigma_z_spline_d2_mag", 0.2))
            P = prior.precision_matrix()
            P_z = second_difference_precision(len(z_idx), sigma_d2=sigma_d2)
            for a, ia in enumerate(z_idx):
                for b, ib in enumerate(z_idx):
                    P[ia, ib] += P_z[a, b]
            prior = GaussianPrior.from_precision(names, P, mean=0.0)

        return y, y0, cov, X.X, prior


def load_pantheon_plus_dataset(probe_cfg: dict[str, Any]) -> PantheonPlusDataset:
    data_path = Path(probe_cfg["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parents[3] / data_path).resolve()

    d = np.load(data_path)
    z = np.asarray(d["z"], dtype=float)
    m = np.asarray(d["m_b_corr"], dtype=float)
    cov = np.asarray(d["cov"], dtype=float)
    ra = np.asarray(d["ra_deg"], dtype=float) if "ra_deg" in d else None
    dec = np.asarray(d["dec_deg"], dtype=float) if "dec_deg" in d else None
    ids = np.asarray(d["idsurvey"], dtype=int) if "idsurvey" in d else None

    z_min = probe_cfg.get("z_min")
    z_max = probe_cfg.get("z_max")
    mask = np.ones_like(z, dtype=bool)
    if z_min is not None:
        mask &= z >= float(z_min)
    if z_max is not None:
        mask &= z <= float(z_max)

    idx = np.where(mask)[0]
    cov = cov[np.ix_(idx, idx)]
    z_sel = z[idx]
    z_support_min = float(np.min(z_sel)) if z_sel.size else None
    z_support_max = float(np.max(z_sel)) if z_sel.size else None
    id_levels = np.unique(ids[idx]) if ids is not None else None
    return PantheonPlusDataset(
        z=z_sel,
        m_b_corr=m[idx],
        cov=cov,
        ra_deg=ra[idx] if ra is not None else None,
        dec_deg=dec[idx] if dec is not None else None,
        idsurvey=ids[idx] if ids is not None else None,
        z_support_min=z_support_min,
        z_support_max=z_support_max,
        idsurvey_levels=id_levels,
    )
