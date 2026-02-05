from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior
from hubble_systematics.audit.util import cholesky_with_jitter


@dataclass(frozen=True)
class InjectionResult:
    mechanism: str
    amplitudes: list[float]
    param_of_interest: str
    rows: list[dict[str, Any]]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "amplitudes": self.amplitudes,
            "param_of_interest": self.param_of_interest,
            "rows": self.rows,
        }


def run_injection_suite(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    inj_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> InjectionResult:
    mechanism = str(inj_cfg.get("mechanism", "global_offset_mag"))
    amps = list(inj_cfg.get("amplitudes", [-0.2, -0.1, 0.0, 0.1, 0.2]))
    n_mc = int(inj_cfg.get("n_mc", 200))
    use_diag = bool(inj_cfg.get("use_diagonal_errors", True))
    poi = str(inj_cfg.get("param_of_interest", "global_offset_mag"))

    # Fit once to define a baseline "true" parameter vector.
    y_obs, y0, cov, X, prior = dataset.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
    if use_diag:
        sigma = dataset.diag_sigma()
        cov = sigma**2
    else:
        sigma = None
    fit0 = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_obs, y0=y0, cov=cov, X=X, prior=prior))
    beta_true = fit0.mean
    if poi not in fit0.param_names:
        raise ValueError(f"param_of_interest '{poi}' not in params {fit0.param_names}")
    j = fit0.param_names.index(poi)

    # Precompute noise model.
    cov_sim = cov
    if cov_sim.ndim == 2:
        L_cov = cholesky_with_jitter(cov_sim)
    else:
        L_cov = None

    y_base = y0 + X @ beta_true

    rows: list[dict[str, Any]] = []
    for amp in amps:
        amp = float(amp)
        delta = _injection_delta(n=y_base.size, dataset=dataset, mechanism=mechanism, amp=amp, inj_cfg=inj_cfg)
        vals = np.empty(n_mc, dtype=float)
        if use_diag:
            assert sigma is not None
            eps = rng.normal(0.0, sigma, size=(n_mc, sigma.size))
        else:
            assert L_cov is not None
            z = rng.normal(0.0, 1.0, size=(n_mc, y_base.size))
            eps = z @ L_cov.T
        for t in range(n_mc):
            y_sim = y_base + delta + eps[t]
            fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_sim, y0=y0, cov=cov_sim, X=X, prior=prior))
            vals[t] = float(fit.mean[j])
        rows.append(
            {
                "amp": amp,
                "mean_hat": float(np.mean(vals)),
                "std_hat": float(np.std(vals)),
                "bias": float(np.mean(vals) - beta_true[j]),
            }
        )

    return InjectionResult(mechanism=mechanism, amplitudes=[float(a) for a in amps], param_of_interest=poi, rows=rows)


def _injection_delta(*, n: int, dataset, mechanism: str, amp: float, inj_cfg: dict[str, Any]) -> np.ndarray:
    if mechanism == "global_offset_mag":
        return np.full(n, float(amp))
    if mechanism in {"y_offset", "y_shift"}:
        return np.full(n, float(amp))
    if mechanism == "calibrator_offset_mag":
        if not hasattr(dataset, "is_calibrator"):
            raise ValueError("calibrator_offset_mag requires dataset.is_calibrator")
        w = np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        if w.shape != (n,):
            raise ValueError("dataset.is_calibrator shape mismatch")
        return float(amp) * w
    if mechanism in {"hubble_flow_offset_mag", "hf_offset_mag"}:
        if not hasattr(dataset, "is_hubble_flow"):
            raise ValueError("hubble_flow_offset_mag requires dataset.is_hubble_flow")
        w = np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        if w.shape != (n,):
            raise ValueError("dataset.is_hubble_flow shape mismatch")
        return float(amp) * w
    if mechanism in {"idsurvey_offset_mag", "survey_offset_mag"}:
        if not hasattr(dataset, "idsurvey"):
            raise ValueError("idsurvey_offset_mag requires dataset.idsurvey")
        if "idsurvey" not in inj_cfg:
            raise ValueError("idsurvey_offset_mag requires inj_cfg['idsurvey']")
        sid = int(inj_cfg["idsurvey"])
        ids = np.asarray(getattr(dataset, "idsurvey"), dtype=int).reshape(-1)
        if ids.shape != (n,):
            raise ValueError("dataset.idsurvey shape mismatch")
        m = (ids == sid)
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("idsurvey_offset_mag apply_to=hf requires dataset.is_hubble_flow")
            m &= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).reshape(-1)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("idsurvey_offset_mag apply_to=cal requires dataset.is_calibrator")
            m &= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * m.astype(float)
    if mechanism in {"pkmjd_bin_offset_mag", "time_bin_offset_mag"}:
        if not hasattr(dataset, "pkmjd"):
            raise ValueError("pkmjd_bin_offset_mag requires dataset.pkmjd")
        t = np.asarray(getattr(dataset, "pkmjd"), dtype=float).reshape(-1)
        if t.shape != (n,):
            raise ValueError("dataset.pkmjd shape mismatch")
        good = np.isfinite(t) & (t > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive pkmjd values")

        edges = inj_cfg.get("edges")
        if edges is None:
            edges = getattr(dataset, "pkmjd_edges", None)
            if edges is None:
                n_bins = int(inj_cfg.get("n_bins", 6))
                if n_bins < 2:
                    raise ValueError("pkmjd_bin_offset_mag n_bins must be >=2")
                qs = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.quantile(t[good], qs).tolist()
        edges = [float(x) for x in edges]
        if len(edges) < 3:
            raise ValueError("pkmjd_bin_offset_mag edges must have length >=3")
        edges = sorted(edges)

        inner = np.asarray(edges[1:-1], dtype=float)
        bid = np.zeros_like(t, dtype=int)
        bid[good] = np.digitize(t[good], inner, right=False)
        k = int(inj_cfg.get("bin", 1))
        if k < 0 or k >= (len(edges) - 1):
            raise ValueError(f"pkmjd_bin_offset_mag bin out of range: {k} for n_bins={len(edges)-1}")
        m = (bid == k)

        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("pkmjd_bin_offset_mag apply_to=hf requires dataset.is_hubble_flow")
            m &= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).reshape(-1)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("pkmjd_bin_offset_mag apply_to=cal requires dataset.is_calibrator")
            m &= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * m.astype(float)
    if mechanism in {"survey_pkmjd_bin_offset_mag", "survey_time_bin_offset_mag"}:
        if not hasattr(dataset, "pkmjd"):
            raise ValueError("survey_pkmjd_bin_offset_mag requires dataset.pkmjd")
        if not hasattr(dataset, "idsurvey"):
            raise ValueError("survey_pkmjd_bin_offset_mag requires dataset.idsurvey")
        if "idsurvey" not in inj_cfg:
            raise ValueError("survey_pkmjd_bin_offset_mag requires inj_cfg['idsurvey']")

        sid = int(inj_cfg["idsurvey"])
        t = np.asarray(getattr(dataset, "pkmjd"), dtype=float).reshape(-1)
        ids = np.asarray(getattr(dataset, "idsurvey"), dtype=int).reshape(-1)
        if t.shape != (n,):
            raise ValueError("dataset.pkmjd shape mismatch")
        if ids.shape != (n,):
            raise ValueError("dataset.idsurvey shape mismatch")

        good = np.isfinite(t) & (t > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive pkmjd values")

        edges = inj_cfg.get("edges")
        if edges is None:
            edges = getattr(dataset, "pkmjd_edges", None)
            if edges is None:
                n_bins = int(inj_cfg.get("n_bins", 6))
                if n_bins < 2:
                    raise ValueError("survey_pkmjd_bin_offset_mag n_bins must be >=2")
                qs = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.quantile(t[good], qs).tolist()
        edges = [float(x) for x in edges]
        if len(edges) < 3:
            raise ValueError("survey_pkmjd_bin_offset_mag edges must have length >=3")
        edges = sorted(edges)

        inner = np.asarray(edges[1:-1], dtype=float)
        bid = np.zeros_like(t, dtype=int)
        bid[good] = np.digitize(t[good], inner, right=False)
        k = int(inj_cfg.get("bin", 1))
        if k < 0 or k >= (len(edges) - 1):
            raise ValueError(f"survey_pkmjd_bin_offset_mag bin out of range: {k} for n_bins={len(edges)-1}")

        m = (ids == sid) & (bid == k)

        apply_to = str(inj_cfg.get("apply_to", "cal")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("survey_pkmjd_bin_offset_mag apply_to=hf requires dataset.is_hubble_flow")
            m &= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).reshape(-1)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("survey_pkmjd_bin_offset_mag apply_to=cal requires dataset.is_calibrator")
            m &= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).reshape(-1)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * m.astype(float)
    if mechanism in {"mwebv_linear_mag", "mwebv_linear"}:
        if not hasattr(dataset, "mwebv"):
            raise ValueError("mwebv_linear_mag requires dataset.mwebv")
        x = np.asarray(getattr(dataset, "mwebv"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.mwebv shape mismatch")
        good = np.isfinite(x) & (x >= 0.0)
        if not np.any(good):
            raise ValueError("No finite nonnegative mwebv values")
        mu = float(np.mean(x[good]))
        sd = float(np.std(x[good]) + 1e-12)
        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("mwebv_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("mwebv_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"c_linear_mag", "c_linear"}:
        if not hasattr(dataset, "c"):
            raise ValueError("c_linear_mag requires dataset.c")
        x = np.asarray(getattr(dataset, "c"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.c shape mismatch")
        good = np.isfinite(x)
        if not np.any(good):
            raise ValueError("No finite c values")

        if hasattr(dataset, "c_mu") and hasattr(dataset, "c_sd"):
            mu = float(getattr(dataset, "c_mu"))
            sd = float(getattr(dataset, "c_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("c_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("c_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"x1_linear_mag", "x1_linear"}:
        if not hasattr(dataset, "x1"):
            raise ValueError("x1_linear_mag requires dataset.x1")
        x = np.asarray(getattr(dataset, "x1"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.x1 shape mismatch")
        good = np.isfinite(x)
        if not np.any(good):
            raise ValueError("No finite x1 values")

        if hasattr(dataset, "x1_mu") and hasattr(dataset, "x1_sd"):
            mu = float(getattr(dataset, "x1_mu"))
            sd = float(getattr(dataset, "x1_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("x1_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("x1_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"biascor_m_b_linear_mag", "biascor_m_b_linear"}:
        if not hasattr(dataset, "biascor_m_b"):
            raise ValueError("biascor_m_b_linear_mag requires dataset.biascor_m_b")
        x = np.asarray(getattr(dataset, "biascor_m_b"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.biascor_m_b shape mismatch")
        good = np.isfinite(x)
        if not np.any(good):
            raise ValueError("No finite biascor_m_b values")

        if hasattr(dataset, "biascor_m_b_mu") and hasattr(dataset, "biascor_m_b_sd"):
            mu = float(getattr(dataset, "biascor_m_b_mu"))
            sd = float(getattr(dataset, "biascor_m_b_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("biascor_m_b_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("biascor_m_b_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"m_b_corr_err_linear_mag", "m_b_corr_err_linear"}:
        if not hasattr(dataset, "m_b_corr_err_diag"):
            raise ValueError("m_b_corr_err_linear_mag requires dataset.m_b_corr_err_diag")
        x = np.asarray(getattr(dataset, "m_b_corr_err_diag"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.m_b_corr_err_diag shape mismatch")
        good = np.isfinite(x) & (x > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive m_b_corr_err_diag values")

        if hasattr(dataset, "m_b_corr_err_mu") and hasattr(dataset, "m_b_corr_err_sd"):
            mu = float(getattr(dataset, "m_b_corr_err_mu"))
            sd = float(getattr(dataset, "m_b_corr_err_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "hf")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("m_b_corr_err_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("m_b_corr_err_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"m_b_corr_err_raw_linear_mag", "m_b_corr_err_raw_linear"}:
        if not hasattr(dataset, "m_b_corr_err_raw"):
            raise ValueError("m_b_corr_err_raw_linear_mag requires dataset.m_b_corr_err_raw")
        x = np.asarray(getattr(dataset, "m_b_corr_err_raw"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.m_b_corr_err_raw shape mismatch")
        good = np.isfinite(x) & (x > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive m_b_corr_err_raw values")

        if hasattr(dataset, "m_b_corr_err_raw_mu") and hasattr(dataset, "m_b_corr_err_raw_sd"):
            mu = float(getattr(dataset, "m_b_corr_err_raw_mu"))
            sd = float(getattr(dataset, "m_b_corr_err_raw_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "hf")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("m_b_corr_err_raw_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("m_b_corr_err_raw_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"m_b_corr_err_vpec_linear_mag", "m_b_corr_err_vpec_linear"}:
        if not hasattr(dataset, "m_b_corr_err_vpec"):
            raise ValueError("m_b_corr_err_vpec_linear_mag requires dataset.m_b_corr_err_vpec")
        x = np.asarray(getattr(dataset, "m_b_corr_err_vpec"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.m_b_corr_err_vpec shape mismatch")
        good = np.isfinite(x) & (x > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive m_b_corr_err_vpec values")

        if hasattr(dataset, "m_b_corr_err_vpec_mu") and hasattr(dataset, "m_b_corr_err_vpec_sd"):
            mu = float(getattr(dataset, "m_b_corr_err_vpec_mu"))
            sd = float(getattr(dataset, "m_b_corr_err_vpec_sd"))
        else:
            mu = float(np.mean(x[good]))
            sd = float(np.std(x[good]) + 1e-12)

        xz = np.zeros_like(x, dtype=float)
        xz[good] = (x[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "cal")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("m_b_corr_err_vpec_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            xz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("m_b_corr_err_vpec_linear_mag apply_to=cal requires dataset.is_calibrator")
            xz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * xz
    if mechanism in {"host_mass_step_mag", "host_mass_step"}:
        if not hasattr(dataset, "host_logmass"):
            raise ValueError("host_mass_step_mag requires dataset.host_logmass")
        thr = float(inj_cfg.get("threshold", 10.0))
        x = np.asarray(getattr(dataset, "host_logmass"), dtype=float).reshape(-1)
        if x.shape != (n,):
            raise ValueError("dataset.host_logmass shape mismatch")
        good = np.isfinite(x) & (x > 0.0)
        step = np.zeros_like(x, dtype=float)
        step[good] = (x[good] >= thr).astype(float)
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("host_mass_step_mag apply_to=hf requires dataset.is_hubble_flow")
            step *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("host_mass_step_mag apply_to=cal requires dataset.is_calibrator")
            step *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * step
    if mechanism in {"pkmjd_linear_mag", "pkmjd_linear"}:
        if not hasattr(dataset, "pkmjd"):
            raise ValueError("pkmjd_linear_mag requires dataset.pkmjd")
        t = np.asarray(getattr(dataset, "pkmjd"), dtype=float).reshape(-1)
        if t.shape != (n,):
            raise ValueError("dataset.pkmjd shape mismatch")
        good = np.isfinite(t) & (t > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive pkmjd values")
        mu = float(np.mean(t[good]))
        sd = float(np.std(t[good]) + 1e-12)
        tz = np.zeros_like(t, dtype=float)
        tz[good] = (t[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "all")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("pkmjd_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            tz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("pkmjd_linear_mag apply_to=cal requires dataset.is_calibrator")
            tz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * tz
    if mechanism in {"pkmjd_err_linear_mag", "pkmjd_err_linear"}:
        if not hasattr(dataset, "pkmjd_err"):
            raise ValueError("pkmjd_err_linear_mag requires dataset.pkmjd_err")
        t = np.asarray(getattr(dataset, "pkmjd_err"), dtype=float).reshape(-1)
        if t.shape != (n,):
            raise ValueError("dataset.pkmjd_err shape mismatch")
        good = np.isfinite(t) & (t > 0.0)
        if not np.any(good):
            raise ValueError("No finite positive pkmjd_err values")

        if hasattr(dataset, "pkmjd_err_mu") and hasattr(dataset, "pkmjd_err_sd"):
            mu = float(getattr(dataset, "pkmjd_err_mu"))
            sd = float(getattr(dataset, "pkmjd_err_sd"))
        else:
            mu = float(np.mean(t[good]))
            sd = float(np.std(t[good]) + 1e-12)

        tz = np.zeros_like(t, dtype=float)
        tz[good] = (t[good] - mu) / sd
        apply_to = str(inj_cfg.get("apply_to", "cal")).lower()
        if apply_to == "hf":
            if not hasattr(dataset, "is_hubble_flow"):
                raise ValueError("pkmjd_err_linear_mag apply_to=hf requires dataset.is_hubble_flow")
            tz *= np.asarray(getattr(dataset, "is_hubble_flow"), dtype=bool).astype(float)
        elif apply_to == "cal":
            if not hasattr(dataset, "is_calibrator"):
                raise ValueError("pkmjd_err_linear_mag apply_to=cal requires dataset.is_calibrator")
            tz *= np.asarray(getattr(dataset, "is_calibrator"), dtype=bool).astype(float)
        elif apply_to != "all":
            raise ValueError(f"Unsupported apply_to: {apply_to}")
        return float(amp) * tz
    if mechanism == "z_linear_mag":
        z = np.asarray(dataset.z, dtype=float)
        z0 = float(inj_cfg.get("z0", float(np.mean(z))))
        zscale = float(inj_cfg.get("zscale", float(np.std(z) + 1e-12)))
        return float(amp) * (z - z0) / zscale
    if mechanism == "sky_dipole_mag":
        if dataset.ra_deg is None or dataset.dec_deg is None:
            raise ValueError("sky_dipole_mag requires ra_deg/dec_deg")
        frame = str(inj_cfg.get("frame", "galactic"))
        lon0 = float(inj_cfg.get("axis_lon_deg", 264.021))
        lat0 = float(inj_cfg.get("axis_lat_deg", 48.253))
        coords = SkyCoord(ra=dataset.ra_deg * u.deg, dec=dataset.dec_deg * u.deg, frame="icrs")
        if frame == "galactic":
            g = coords.galactic
            lon = g.l.to_value(u.rad)
            lat = g.b.to_value(u.rad)
            ax = SkyCoord(l=lon0 * u.deg, b=lat0 * u.deg, frame="galactic")
        elif frame == "icrs":
            lon = coords.ra.to_value(u.rad)
            lat = coords.dec.to_value(u.rad)
            ax = SkyCoord(ra=lon0 * u.deg, dec=lat0 * u.deg, frame="icrs")
        else:
            raise ValueError(f"Unsupported frame: {frame}")
        # cos(theta) between unit vectors.
        v = np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
            axis=1,
        )
        axc = ax.galactic if frame == "galactic" else ax.icrs
        lon_a = axc.l.to_value(u.rad) if frame == "galactic" else axc.ra.to_value(u.rad)
        lat_a = axc.b.to_value(u.rad) if frame == "galactic" else axc.dec.to_value(u.rad)
        va = np.array([np.cos(lat_a) * np.cos(lon_a), np.cos(lat_a) * np.sin(lon_a), np.sin(lat_a)])
        costh = v @ va
        return float(amp) * costh
    raise ValueError(f"Unknown injection mechanism: {mechanism}")
