from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.design import (
    DesignMatrix,
    bspline_basis,
    one_hot_levels,
    ones,
    second_difference_precision,
    sky_real_harmonics,
)
from hubble_systematics.gaussian_linear_model import GaussianPrior
from hubble_systematics.shared_scale import shared_scale_params

from astropy.coordinates import SkyCoord
import astropy.units as u


@dataclass(frozen=True)
class PantheonPlusShoesLadderDataset:
    z: np.ndarray
    m_b_corr: np.ndarray
    m_b_corr_err_diag: np.ndarray
    cov: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    idsurvey: np.ndarray
    pkmjd: np.ndarray
    pkmjd_err: np.ndarray
    mwebv: np.ndarray
    host_logmass: np.ndarray
    host_logmass_err: np.ndarray
    fitprob: np.ndarray
    fitchi2: np.ndarray
    ndof: np.ndarray
    mu_shoes: np.ndarray
    mu_shoes_err_diag: np.ndarray
    ceph_dist: np.ndarray
    is_calibrator: np.ndarray
    is_hubble_flow: np.ndarray
    idsurvey_levels: np.ndarray
    idsurvey_hf_counts: dict[int, int]
    z_hf_support_min: float | None
    z_hf_support_max: float | None
    mwebv_mu: float
    mwebv_sd: float
    pkmjd_mu: float
    pkmjd_sd: float
    pkmjd_edges: np.ndarray | None

    def get_column(self, name: str) -> np.ndarray:
        if name == "z":
            return self.z
        if name in {"m_b_corr_err_diag"}:
            return self.m_b_corr_err_diag
        if name in {"m_b_corr_err_diag_hf"}:
            return np.where(self.is_hubble_flow, self.m_b_corr_err_diag, np.nan)
        if name in {"mu_shoes"}:
            return self.mu_shoes
        if name in {"mu_shoes_err_diag"}:
            return self.mu_shoes_err_diag
        if name in {"mu_shoes_err_diag_hf"}:
            return np.where(self.is_hubble_flow, self.mu_shoes_err_diag, np.nan)
        if name in {"z_hf", "z_hf_max"}:
            return np.where(self.is_hubble_flow, self.z, np.nan)
        if name in {"pkmjd_hf", "pkmjd_hf_bin"}:
            return np.where(self.is_hubble_flow, self.pkmjd, np.nan)
        if name in {"pkmjd"}:
            return self.pkmjd.astype(float)
        if name in {"pkmjd_err"}:
            return self.pkmjd_err.astype(float)
        if name in {"pkmjd_err_hf"}:
            return np.where(self.is_hubble_flow, self.pkmjd_err.astype(float), np.nan)
        if name in {"mwebv_hf"}:
            return np.where(self.is_hubble_flow, self.mwebv, np.nan)
        if name in {"fitprob"}:
            return self.fitprob.astype(float)
        if name in {"fitprob_hf"}:
            return np.where(self.is_hubble_flow, self.fitprob.astype(float), np.nan)
        if name in {"fitchi2"}:
            return self.fitchi2.astype(float)
        if name in {"fitchi2_hf"}:
            return np.where(self.is_hubble_flow, self.fitchi2.astype(float), np.nan)
        if name in {"ndof"}:
            return self.ndof.astype(float)
        if name in {"ndof_hf"}:
            return np.where(self.is_hubble_flow, self.ndof.astype(float), np.nan)
        if name in {"redchi2"}:
            nd = self.ndof.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = self.fitchi2.astype(float) / nd
            out = np.where(nd > 0.0, out, np.nan)
            return out
        if name in {"redchi2_hf"}:
            nd = self.ndof.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = self.fitchi2.astype(float) / nd
            out = np.where(nd > 0.0, out, np.nan)
            return np.where(self.is_hubble_flow, out, np.nan)
        if name in {"host_logmass_hf"}:
            return np.where(self.is_hubble_flow, self.host_logmass, np.nan)
        if name in {"host_logmass_err"}:
            return self.host_logmass_err.astype(float)
        if name in {"host_logmass_err_hf"}:
            return np.where(self.is_hubble_flow, self.host_logmass_err.astype(float), np.nan)
        if name == "idsurvey":
            return self.idsurvey.astype(float)
        if name == "mwebv":
            return self.mwebv.astype(float)
        if name == "host_logmass":
            return self.host_logmass.astype(float)
        raise KeyError(f"Unknown column: {name}")

    def subset_leq(self, col: str, value: float) -> "PantheonPlusShoesLadderDataset":
        col = str(col)
        value = float(value)
        if col in {"z_hf", "z_hf_max", "z"} or col.endswith("_hf") or col.endswith("_hf_max") or col.endswith("_hf_bin"):
            v = self.get_column(col)
            mask = self.is_calibrator | (self.is_hubble_flow & np.isfinite(v) & (v <= value))
            return self.subset_mask(mask)
        v = self.get_column(col)
        return self.subset_mask(np.asarray(v <= value, dtype=bool))

    def subset_geq(self, col: str, value: float) -> "PantheonPlusShoesLadderDataset":
        col = str(col)
        value = float(value)
        if col in {"z_hf", "z_hf_max", "z"} or col.endswith("_hf") or col.endswith("_hf_max") or col.endswith("_hf_bin"):
            v = self.get_column(col)
            mask = self.is_calibrator | (self.is_hubble_flow & np.isfinite(v) & (v >= value))
            return self.subset_mask(mask)
        v = self.get_column(col)
        return self.subset_mask(np.asarray(v >= value, dtype=bool))

    def subset_mask(self, mask: np.ndarray) -> "PantheonPlusShoesLadderDataset":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.z.shape:
            raise ValueError("mask shape mismatch")
        idx = np.where(mask)[0]
        cov = self.cov[np.ix_(idx, idx)]
        return PantheonPlusShoesLadderDataset(
            z=self.z[idx],
            m_b_corr=self.m_b_corr[idx],
            m_b_corr_err_diag=self.m_b_corr_err_diag[idx],
            cov=cov,
            ra_deg=self.ra_deg[idx],
            dec_deg=self.dec_deg[idx],
            idsurvey=self.idsurvey[idx],
            pkmjd=self.pkmjd[idx],
            pkmjd_err=self.pkmjd_err[idx],
            mwebv=self.mwebv[idx],
            host_logmass=self.host_logmass[idx],
            host_logmass_err=self.host_logmass_err[idx],
            fitprob=self.fitprob[idx],
            fitchi2=self.fitchi2[idx],
            ndof=self.ndof[idx],
            mu_shoes=self.mu_shoes[idx],
            mu_shoes_err_diag=self.mu_shoes_err_diag[idx],
            ceph_dist=self.ceph_dist[idx],
            is_calibrator=self.is_calibrator[idx],
            is_hubble_flow=self.is_hubble_flow[idx],
            idsurvey_levels=self.idsurvey_levels,
            idsurvey_hf_counts=self.idsurvey_hf_counts,
            z_hf_support_min=self.z_hf_support_min,
            z_hf_support_max=self.z_hf_support_max,
            mwebv_mu=self.mwebv_mu,
            mwebv_sd=self.mwebv_sd,
            pkmjd_mu=self.pkmjd_mu,
            pkmjd_sd=self.pkmjd_sd,
            pkmjd_edges=self.pkmjd_edges,
        )

    def diag_sigma(self) -> np.ndarray:
        d = np.diag(self.cov)
        if not (np.all(np.isfinite(d)) and np.all(d > 0)):
            raise ValueError("Invalid covariance diagonal")
        return np.sqrt(d)

    def build_design(self, *, anchor: AnchorLCDM, ladder_level: str, cfg: dict[str, Any]):
        y = np.asarray(self.m_b_corr, dtype=float)

        mu = np.asarray(anchor.mu(self.z), dtype=float)
        y0 = np.where(self.is_calibrator, np.asarray(self.ceph_dist, dtype=float), mu)
        cov = np.asarray(self.cov, dtype=float)

        prior_cfg = (cfg or {}).get("priors", {}) or {}
        mech_cfg = (cfg or {}).get("mechanisms", {}) or {}
        lvl = str(ladder_level).upper()
        shared_params = set(shared_scale_params(cfg or {}))

        hf = self.is_hubble_flow.astype(float)
        cal = self.is_calibrator.astype(float)

        X = DesignMatrix(X=np.zeros((y.size, 0)), names=[])

        def add_M() -> None:
            nonlocal X
            X = X.append(DesignMatrix(X=ones(y.size), names=["global_offset_mag"]))

        def add_calibrator_offset() -> None:
            nonlocal X
            X = X.append(DesignMatrix(X=cal.reshape(-1, 1), names=["calibrator_offset_mag"]))

        def add_survey_offsets() -> None:
            nonlocal X
            ref = prior_cfg.get("survey_reference")
            dm = one_hot_levels(self.idsurvey, levels=self.idsurvey_levels, prefix="survey_offset_", reference=ref, drop_reference=True)
            X = X.append(dm)

        def add_hf_survey_offsets() -> None:
            nonlocal X
            ref = prior_cfg.get("survey_reference")
            dm = one_hot_levels(self.idsurvey, levels=self.idsurvey_levels, prefix="hf_survey_offset_", reference=ref, drop_reference=True)
            X = X.append(DesignMatrix(X=dm.X * hf[:, None], names=dm.names))

        def add_mwebv_linear() -> None:
            nonlocal X
            x = np.asarray(self.mwebv, dtype=float)
            good = np.isfinite(x) & (x >= 0.0)
            if not np.any(good):
                raise ValueError("No finite nonnegative mwebv values for mwebv_linear")
            mu_x = float(self.mwebv_mu)
            sd_x = float(self.mwebv_sd)
            xz = (x - mu_x) / sd_x
            apply_to = str((mech_cfg.get("mwebv_apply_to") or "all")).lower()
            if apply_to == "hf":
                xz = xz * hf
            elif apply_to == "cal":
                xz = xz * cal
            elif apply_to != "all":
                raise ValueError(f"Unsupported mwebv_apply_to: {apply_to}")
            X = X.append(DesignMatrix(X=xz.reshape(-1, 1), names=["mwebv_linear_mag"]))

        def add_host_mass_step() -> None:
            nonlocal X
            thr = float(mech_cfg.get("host_mass_threshold", 10.0))
            x = np.asarray(self.host_logmass, dtype=float)
            good = np.isfinite(x) & (x > 0.0)
            step = np.zeros_like(x, dtype=float)
            step[good] = (x[good] >= thr).astype(float)
            apply_to = str((mech_cfg.get("host_mass_apply_to") or "all")).lower()
            if apply_to == "hf":
                step = step * hf
            elif apply_to == "cal":
                step = step * cal
            elif apply_to != "all":
                raise ValueError(f"Unsupported host_mass_apply_to: {apply_to}")
            X = X.append(DesignMatrix(X=step.reshape(-1, 1), names=["host_mass_step_mag"]))

        def add_pkmjd_linear() -> None:
            nonlocal X
            t = np.asarray(self.pkmjd, dtype=float)
            good = np.isfinite(t) & (t > 0.0)
            if not np.any(good):
                raise ValueError("No finite positive pkmjd values for pkmjd_linear")
            mu_t = float(self.pkmjd_mu)
            sd_t = float(self.pkmjd_sd)
            tz = np.zeros_like(t, dtype=float)
            tz[good] = (t[good] - mu_t) / sd_t
            apply_to = str((mech_cfg.get("pkmjd_apply_to") or "all")).lower()
            if apply_to == "hf":
                tz = tz * hf
            elif apply_to == "cal":
                tz = tz * cal
            elif apply_to != "all":
                raise ValueError(f"Unsupported pkmjd_apply_to: {apply_to}")
            X = X.append(DesignMatrix(X=tz.reshape(-1, 1), names=["pkmjd_linear_mag"]))

        def add_pkmjd_bins() -> None:
            nonlocal X
            bins_cfg = mech_cfg.get("pkmjd_bins", {}) or {}
            if not bool(bins_cfg.get("enable", False)):
                return
            t = np.asarray(self.pkmjd, dtype=float)
            good = np.isfinite(t) & (t > 0.0)
            if not np.any(good):
                raise ValueError("No finite positive pkmjd values for pkmjd_bins")
            edges = bins_cfg.get("edges")
            if edges is None:
                if self.pkmjd_edges is None:
                    n_bins = int(bins_cfg.get("n_bins", 6))
                    if n_bins < 2:
                        raise ValueError("pkmjd_bins.n_bins must be >=2")
                    qs = np.linspace(0.0, 1.0, n_bins + 1)
                    edges = np.quantile(t[good], qs).tolist()
                else:
                    edges = self.pkmjd_edges.tolist()
            edges = [float(x) for x in edges]
            if len(edges) < 3:
                raise ValueError("pkmjd_bins.edges must have length >=3")
            edges = sorted(edges)
            # Bin id in [0, n_bins-1]. Missing/nonpositive -> reference bin 0.
            inner = np.asarray(edges[1:-1], dtype=float)
            bid = np.zeros_like(t, dtype=int)
            bid[good] = np.digitize(t[good], inner, right=False)
            dm = one_hot_levels(bid, levels=range(len(edges) - 1), prefix="pkmjd_bin_", reference=0, drop_reference=True)
            apply_to = str((bins_cfg.get("apply_to") or "all")).lower()
            B = dm.X
            if apply_to == "hf":
                B = B * hf[:, None]
            elif apply_to == "cal":
                B = B * cal[:, None]
            elif apply_to != "all":
                raise ValueError(f"Unsupported pkmjd_bins.apply_to: {apply_to}")
            X = X.append(DesignMatrix(X=B, names=[f"pkmjd_bin_offset_{n.split('pkmjd_bin_')[1]}" for n in dm.names]))

        def add_survey_pkmjd_bins() -> None:
            nonlocal X
            bins_cfg = mech_cfg.get("survey_pkmjd_bins", {}) or {}
            if not bool(bins_cfg.get("enable", False)):
                return
            t = np.asarray(self.pkmjd, dtype=float)
            good = np.isfinite(t) & (t > 0.0)
            if not np.any(good):
                raise ValueError("No finite positive pkmjd values for survey_pkmjd_bins")

            edges = bins_cfg.get("edges")
            if edges is None:
                if self.pkmjd_edges is None:
                    n_bins = int(bins_cfg.get("n_bins", 6))
                    if n_bins < 2:
                        raise ValueError("survey_pkmjd_bins.n_bins must be >=2")
                    qs = np.linspace(0.0, 1.0, n_bins + 1)
                    edges = np.quantile(t[good], qs).tolist()
                else:
                    edges = self.pkmjd_edges.tolist()
            edges = [float(x) for x in edges]
            if len(edges) < 3:
                raise ValueError("survey_pkmjd_bins.edges must have length >=3")
            edges = sorted(edges)

            inner = np.asarray(edges[1:-1], dtype=float)
            bid = np.zeros_like(t, dtype=int)
            bid[good] = np.digitize(t[good], inner, right=False)
            n_bins = len(edges) - 1

            apply_to = str((bins_cfg.get("apply_to") or "hf")).lower()
            base = np.ones_like(hf, dtype=float)
            if apply_to == "hf":
                base = hf
            elif apply_to == "cal":
                base = cal
            elif apply_to != "all":
                raise ValueError(f"Unsupported survey_pkmjd_bins.apply_to: {apply_to}")

            min_hf_per_survey = int(bins_cfg.get("min_hf_per_survey", 20))
            surveys = [int(s) for s in self.idsurvey_levels.tolist()]
            cols: list[np.ndarray] = []
            names: list[str] = []
            for s in surveys:
                s = int(s)
                if int(self.idsurvey_hf_counts.get(s, 0)) < min_hf_per_survey:
                    continue
                in_survey = (self.idsurvey.astype(int) == s) & good
                for k in range(1, n_bins):
                    col = ((bid == k) & in_survey).astype(float) * base
                    cols.append(col)
                    names.append(f"survey_pkmjd_bin_offset_{s}_{k}")
            if cols:
                X = X.append(DesignMatrix(X=np.stack(cols, axis=1), names=names))

        def add_hf_z_spline() -> None:
            nonlocal X
            zcfg = (cfg or {}).get("z_spline", {}) or {}
            n_knots = int(zcfg.get("n_internal_knots", 6))
            z_hf = np.asarray(self.z[self.is_hubble_flow], dtype=float)
            if z_hf.size == 0:
                return
            dm_hf = bspline_basis(
                z_hf,
                xmin=float(self.z_hf_support_min if self.z_hf_support_min is not None else np.nanmin(z_hf)),
                xmax=float(self.z_hf_support_max if self.z_hf_support_max is not None else np.nanmax(z_hf)),
                n_internal_knots=n_knots,
                prefix="hf_z_spline_",
            )
            B = np.zeros((y.size, dm_hf.X.shape[1]), dtype=float)
            B[self.is_hubble_flow] = dm_hf.X
            X = X.append(DesignMatrix(X=B, names=dm_hf.names))

        def add_sky_modes() -> None:
            nonlocal X
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
                dmu_dlnH0 = -5.0 / np.log(10.0)
                X = X.append(
                    DesignMatrix(
                        X=(hf * dmu_dlnH0).reshape(-1, 1),
                        names=["delta_lnH0"],
                    )
                )

        if lvl == "L0":
            pass
        elif lvl == "L1":
            add_M()
        elif lvl == "L2":
            add_M()
            add_survey_offsets()
        elif lvl == "L3":
            add_M()
            add_survey_offsets()
            add_hf_z_spline()
        elif lvl == "L4":
            add_M()
            add_survey_offsets()
            add_hf_z_spline()
            add_sky_modes()
        else:
            raise ValueError(f"Unknown ladder level: {ladder_level}")

        if lvl != "L0":
            if bool(mech_cfg.get("calibrator_offset", False)):
                add_calibrator_offset()
            if bool(mech_cfg.get("hf_survey_offsets", False)):
                add_hf_survey_offsets()
            if bool(mech_cfg.get("mwebv_linear", False)):
                add_mwebv_linear()
            if bool(mech_cfg.get("host_mass_step", False)):
                add_host_mass_step()
            if bool(mech_cfg.get("pkmjd_linear", False)):
                add_pkmjd_linear()
            add_survey_pkmjd_bins()
            add_pkmjd_bins()

        add_shared_scale()

        names = X.names
        if not names:
            prior = GaussianPrior.from_sigmas([], [], mean=0.0)
            return y, y0, cov, X.X, prior

        mean_map = prior_cfg.get("mean_map", {}) or {}
        mu_prior = np.zeros(len(names), dtype=float)
        if isinstance(mean_map, dict):
            for k, v in mean_map.items():
                k = str(k)
                if k in names:
                    mu_prior[names.index(k)] = float(v)

        sigmas = []
        for n in names:
            if n == "global_offset_mag":
                sigmas.append(float(prior_cfg.get("sigma_global_offset_mag", 10.0)))
            elif n in shared_params:
                sigmas.append(float("inf"))
            elif n == "calibrator_offset_mag":
                sigmas.append(float(prior_cfg.get("sigma_calibrator_offset_mag", 0.2)))
            elif n.startswith("hf_survey_offset_"):
                sigmas.append(float(prior_cfg.get("sigma_hf_survey_offset_mag", 0.2)))
            elif n == "mwebv_linear_mag":
                sigmas.append(float(prior_cfg.get("sigma_mwebv_linear_mag", 0.2)))
            elif n == "host_mass_step_mag":
                sigmas.append(float(prior_cfg.get("sigma_host_mass_step_mag", 0.2)))
            elif n == "pkmjd_linear_mag":
                sigmas.append(float(prior_cfg.get("sigma_pkmjd_linear_mag", 0.2)))
            elif n.startswith("pkmjd_bin_offset_"):
                sigmas.append(float(prior_cfg.get("sigma_pkmjd_bin_offset_mag", 0.2)))
            elif n.startswith("survey_pkmjd_bin_offset_"):
                sigmas.append(float(prior_cfg.get("sigma_survey_pkmjd_bin_offset_mag", 0.2)))
            elif n.startswith("survey_offset_"):
                sigmas.append(float(prior_cfg.get("sigma_survey_offset_mag", 0.1)))
            elif n.startswith("hf_z_spline_"):
                sigmas.append(float(prior_cfg.get("sigma_z_spline_mag", 0.1)))
            elif n.startswith("sky_"):
                sigmas.append(float(prior_cfg.get("sigma_sky_mode_mag", 0.1)))
            else:
                sigmas.append(float(prior_cfg.get("sigma_default", 10.0)))
        prior = GaussianPrior(param_names=names, mean=mu_prior, sigma=np.asarray(sigmas, dtype=float), precision=None)

        # Smoothness penalty on z-spline coefficients.
        z_idx = [i for i, n in enumerate(names) if n.startswith("hf_z_spline_")]
        if z_idx:
            sigma_d2 = float(prior_cfg.get("sigma_z_spline_d2_mag", 0.1))
            P = prior.precision_matrix()
            P_z = second_difference_precision(len(z_idx), sigma_d2=sigma_d2)
            for a, ia in enumerate(z_idx):
                for b, ib in enumerate(z_idx):
                    P[ia, ib] += P_z[a, b]
            prior = GaussianPrior(param_names=names, mean=mu_prior, sigma=None, precision=P)

        return y, y0, cov, X.X, prior


def load_pantheon_plus_shoes_ladder_dataset(probe_cfg: dict[str, Any]) -> PantheonPlusShoesLadderDataset:
    raw_dat = Path(probe_cfg.get("raw_dat_path", "data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat")).expanduser()
    raw_cov = Path(probe_cfg.get("raw_cov_path", "data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov")).expanduser()
    if not raw_dat.is_absolute():
        raw_dat = (Path(__file__).resolve().parents[3] / raw_dat).resolve()
    if not raw_cov.is_absolute():
        raw_cov = (Path(__file__).resolve().parents[3] / raw_cov).resolve()

    processed_dir = Path(probe_cfg.get("processed_dir", "data/processed/pantheon_plus_shoes_ladder")).expanduser()
    if not processed_dir.is_absolute():
        processed_dir = (Path(__file__).resolve().parents[3] / processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    tag = str(probe_cfg.get("tag", "cal+hf_stat+sys_zHD"))
    processed_npz = processed_dir / f"pantheon_plus_shoes_ladder_{tag}.npz"
    processed_meta = processed_dir / f"pantheon_plus_shoes_ladder_{tag}.meta.json"

    if processed_npz.exists() and processed_meta.exists():
        npz = np.load(processed_npz)
        # Backwards compatibility: regenerate cache if required keys are missing.
        required = [
            "z",
            "m_b_corr",
            "m_b_corr_err_diag",
            "cov",
            "ra_deg",
            "dec_deg",
            "idsurvey",
            "pkmjd",
            "pkmjd_err",
            "mwebv",
            "host_logmass",
            "host_logmass_err",
            "fitprob",
            "fitchi2",
            "ndof",
            "mu_shoes",
            "mu_shoes_err_diag",
            "ceph_dist",
            "is_calibrator",
            "is_hubble_flow",
        ]
        if not all(k in npz for k in required):
            # Fall through to regenerate.
            pass
        else:
            ids = np.asarray(npz["idsurvey"], dtype=int)
            id_levels = np.unique(ids)
            is_hf = np.asarray(npz["is_hubble_flow"], dtype=bool)
            pkmjd = np.asarray(npz["pkmjd"], dtype=float)
            good_t = np.isfinite(pkmjd) & (pkmjd > 0.0)
            id_hf_counts = {int(s): int(np.sum((ids == int(s)) & is_hf & good_t)) for s in id_levels.tolist()}

            z = np.asarray(npz["z"], dtype=float)
            z_hf = z[is_hf]
            z_hf_support_min = float(np.min(z_hf)) if z_hf.size else None
            z_hf_support_max = float(np.max(z_hf)) if z_hf.size else None

            mwebv = np.asarray(npz["mwebv"], dtype=float)
            good_m = np.isfinite(mwebv) & (mwebv >= 0.0)
            if np.any(good_m):
                mwebv_mu = float(np.mean(mwebv[good_m]))
                mwebv_sd = float(np.std(mwebv[good_m]) + 1e-12)
            else:
                mwebv_mu, mwebv_sd = 0.0, 1.0

            if np.any(good_t):
                pkmjd_mu = float(np.mean(pkmjd[good_t]))
                pkmjd_sd = float(np.std(pkmjd[good_t]) + 1e-12)
                qs = np.linspace(0.0, 1.0, 6 + 1)
                pkmjd_edges = np.quantile(pkmjd[good_t], qs)
            else:
                pkmjd_mu, pkmjd_sd = 0.0, 1.0
                pkmjd_edges = None

            return PantheonPlusShoesLadderDataset(
                z=npz["z"],
                m_b_corr=npz["m_b_corr"],
                m_b_corr_err_diag=npz["m_b_corr_err_diag"],
                cov=npz["cov"],
                ra_deg=npz["ra_deg"],
                dec_deg=npz["dec_deg"],
                idsurvey=npz["idsurvey"],
                pkmjd=npz["pkmjd"],
                pkmjd_err=npz["pkmjd_err"],
                mwebv=npz["mwebv"],
                host_logmass=npz["host_logmass"],
                host_logmass_err=npz["host_logmass_err"],
                fitprob=npz["fitprob"],
                fitchi2=npz["fitchi2"],
                ndof=npz["ndof"],
                mu_shoes=npz["mu_shoes"],
                mu_shoes_err_diag=npz["mu_shoes_err_diag"],
                ceph_dist=npz["ceph_dist"],
                is_calibrator=npz["is_calibrator"].astype(bool),
                is_hubble_flow=npz["is_hubble_flow"].astype(bool),
                idsurvey_levels=id_levels.astype(int),
                idsurvey_hf_counts=id_hf_counts,
                z_hf_support_min=z_hf_support_min,
                z_hf_support_max=z_hf_support_max,
                mwebv_mu=mwebv_mu,
                mwebv_sd=mwebv_sd,
                pkmjd_mu=pkmjd_mu,
                pkmjd_sd=pkmjd_sd,
                pkmjd_edges=pkmjd_edges,
            )

    # Load table.
    df = pd.read_csv(raw_dat, sep=r"\s+", comment="#")
    n_total = int(len(df))
    if n_total <= 0:
        raise ValueError("Empty Pantheon+SH0ES table")

    include_cal = bool(probe_cfg.get("include_calibrators", True))
    include_hf = bool(probe_cfg.get("include_hubble_flow", True))
    z_col = str(probe_cfg.get("z_column", "zHD"))

    is_cal = df["IS_CALIBRATOR"].to_numpy(dtype=int) == 1
    is_hf = df["USED_IN_SH0ES_HF"].to_numpy(dtype=int) == 1
    z = df[z_col].to_numpy(dtype=float)

    z_hf_min = float(probe_cfg.get("z_hf_min", 0.023))
    z_hf_max = float(probe_cfg.get("z_hf_max", float(np.nanmax(z[is_hf]))))

    hf_sel = is_hf & (z >= z_hf_min) & (z <= z_hf_max)
    cal_sel = is_cal
    if not include_cal:
        cal_sel = np.zeros_like(cal_sel)
    if not include_hf:
        hf_sel = np.zeros_like(hf_sel)

    sel = cal_sel | hf_sel
    idx = np.where(sel)[0]
    if idx.size == 0:
        raise ValueError("Selection produced empty ladder dataset")

    cov_full = _load_cov_stream_fast(raw_cov)
    if cov_full.shape != (n_total, n_total):
        raise ValueError(f"Cov shape {cov_full.shape} does not match table N={n_total}")
    cov = cov_full[np.ix_(idx, idx)]

    # Extract required columns.
    ra = df["RA"].to_numpy(dtype=float)[idx]
    dec = df["DEC"].to_numpy(dtype=float)[idx]
    idsurvey = df["IDSURVEY"].to_numpy(dtype=int)[idx]
    pkmjd = df["PKMJD"].to_numpy(dtype=float)[idx]
    pkmjd_err = df["PKMJDERR"].to_numpy(dtype=float)[idx]
    mwebv = df["MWEBV"].to_numpy(dtype=float)[idx]
    host_logmass = df["HOST_LOGMASS"].to_numpy(dtype=float)[idx]
    host_logmass_err = df["HOST_LOGMASS_ERR"].to_numpy(dtype=float)[idx]

    m_b_corr = df["m_b_corr"].to_numpy(dtype=float)[idx]
    m_b_corr_err_diag = df["m_b_corr_err_DIAG"].to_numpy(dtype=float)[idx]
    mu_shoes = df["MU_SH0ES"].to_numpy(dtype=float)[idx]
    mu_shoes_err_diag = df["MU_SH0ES_ERR_DIAG"].to_numpy(dtype=float)[idx]
    ceph_dist = df["CEPH_DIST"].to_numpy(dtype=float)[idx]
    fitprob = df["FITPROB"].to_numpy(dtype=float)[idx]
    fitchi2 = df["FITCHI2"].to_numpy(dtype=float)[idx]
    ndof = df["NDOF"].to_numpy(dtype=float)[idx]

    is_cal_sel = is_cal[idx]
    is_hf_sel = hf_sel[idx]

    id_levels = np.unique(idsurvey)
    good_t = np.isfinite(pkmjd) & (pkmjd > 0.0)
    id_hf_counts = {int(s): int(np.sum((idsurvey == int(s)) & is_hf_sel & good_t)) for s in id_levels.tolist()}

    z_hf = z[idx][is_hf_sel]
    z_hf_support_min = float(np.min(z_hf)) if z_hf.size else None
    z_hf_support_max = float(np.max(z_hf)) if z_hf.size else None

    good_m = np.isfinite(mwebv) & (mwebv >= 0.0)
    if np.any(good_m):
        mwebv_mu = float(np.mean(mwebv[good_m]))
        mwebv_sd = float(np.std(mwebv[good_m]) + 1e-12)
    else:
        mwebv_mu, mwebv_sd = 0.0, 1.0

    if np.any(good_t):
        pkmjd_mu = float(np.mean(pkmjd[good_t]))
        pkmjd_sd = float(np.std(pkmjd[good_t]) + 1e-12)
        qs = np.linspace(0.0, 1.0, 6 + 1)
        pkmjd_edges = np.quantile(pkmjd[good_t], qs)
    else:
        pkmjd_mu, pkmjd_sd = 0.0, 1.0
        pkmjd_edges = None

    meta = {
        "source": "PantheonPlusSH0ES/DataRelease (cache copy)",
        "raw_dat": str(raw_dat),
        "raw_cov": str(raw_cov),
        "tag": tag,
        "n_total": n_total,
        "n": int(idx.size),
        "include_calibrators": include_cal,
        "include_hubble_flow": include_hf,
        "z_column": z_col,
        "z_hf_min": z_hf_min,
        "z_hf_max": z_hf_max,
        "n_cal": int(np.sum(is_cal_sel)),
        "n_hf": int(np.sum(is_hf_sel)),
        "columns": list(df.columns),
    }
    processed_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))
    np.savez_compressed(
        processed_npz,
        z=z[idx],
        m_b_corr=m_b_corr,
        m_b_corr_err_diag=m_b_corr_err_diag,
        cov=cov,
        ra_deg=ra,
        dec_deg=dec,
        idsurvey=idsurvey,
        pkmjd=pkmjd,
        pkmjd_err=pkmjd_err,
        mwebv=mwebv,
        host_logmass=host_logmass,
        host_logmass_err=host_logmass_err,
        fitprob=fitprob,
        fitchi2=fitchi2,
        ndof=ndof,
        mu_shoes=mu_shoes,
        mu_shoes_err_diag=mu_shoes_err_diag,
        ceph_dist=ceph_dist,
        is_calibrator=is_cal_sel.astype(np.uint8),
        is_hubble_flow=is_hf_sel.astype(np.uint8),
    )

    return PantheonPlusShoesLadderDataset(
        z=z[idx],
        m_b_corr=m_b_corr,
        m_b_corr_err_diag=m_b_corr_err_diag,
        cov=cov,
        ra_deg=ra,
        dec_deg=dec,
        idsurvey=idsurvey,
        pkmjd=pkmjd,
        pkmjd_err=pkmjd_err,
        mwebv=mwebv,
        host_logmass=host_logmass,
        host_logmass_err=host_logmass_err,
        fitprob=fitprob,
        fitchi2=fitchi2,
        ndof=ndof,
        mu_shoes=mu_shoes,
        mu_shoes_err_diag=mu_shoes_err_diag,
        ceph_dist=ceph_dist,
        is_calibrator=is_cal_sel.astype(bool),
        is_hubble_flow=is_hf_sel.astype(bool),
        idsurvey_levels=id_levels.astype(int),
        idsurvey_hf_counts=id_hf_counts,
        z_hf_support_min=z_hf_support_min,
        z_hf_support_max=z_hf_support_max,
        mwebv_mu=mwebv_mu,
        mwebv_sd=mwebv_sd,
        pkmjd_mu=pkmjd_mu,
        pkmjd_sd=pkmjd_sd,
        pkmjd_edges=pkmjd_edges,
    )


def _load_cov_stream_fast(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        n = int(first)
        flat = np.fromstring(f.read(), sep=" ", dtype=float)
    if flat.size != n * n:
        raise ValueError(f"Covariance file has {flat.size} entries but expected {n*n}.")
    return flat.reshape((n, n))
