from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hubble_systematics.audit.types import DriftNullResult
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class _CutDesign:
    y_obs: np.ndarray
    y0: np.ndarray
    X: np.ndarray
    cov: np.ndarray
    prior: Any


def run_correlated_cut_null_mc(
    *,
    dataset,
    anchor,
    ladder_level: str,
    model_cfg: dict[str, Any],
    cut_var: str,
    cut_mode: str,
    cuts: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
    drift_param: str,
    use_diagonal_errors: bool,
) -> DriftNullResult:
    cut_mode = str(cut_mode).lower()
    if cut_mode not in {"leq", "geq"}:
        raise ValueError(f"cut_mode must be one of {{leq,geq}}; got {cut_mode!r}")
    subset_fn = dataset.subset_leq if cut_mode == "leq" else dataset.subset_geq

    # Precompute designs for each cut.
    designs: list[_CutDesign] = []
    for cut in cuts:
        subset = subset_fn(cut_var, float(cut))
        y_obs, y0, cov, X, prior = subset.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
        prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
        if bool(use_diagonal_errors):
            sigma = subset.diag_sigma()
            cov = sigma**2
        designs.append(_CutDesign(y_obs=y_obs, y0=y0, X=X, cov=cov, prior=prior))

    # Fit observed per cut.
    obs_seq = []
    for cut, design in zip(cuts, designs, strict=True):
        spec = GaussianLinearModelSpec(y=design.y_obs, y0=design.y0, cov=design.cov, X=design.X, prior=design.prior)
        fit = fit_gaussian_linear_model(spec)
        if drift_param not in fit.param_names:
            raise ValueError(f"drift_param '{drift_param}' not in model params {fit.param_names}")
        j = fit.param_names.index(drift_param)
        obs_seq.append(float(fit.mean[j]))

    obs_seq = np.asarray(obs_seq, dtype=float)
    obs_metrics = _drift_metrics(obs_seq)

    # Fit full sample at max cut for a "true" parameter vector to inject.
    full_cut = float(cuts[-1]) if cut_mode == "leq" else float(cuts[0])
    subset_full = subset_fn(cut_var, full_cut)
    y_full, y0_full, cov_full, X_full, prior_full = subset_full.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior_full = apply_shared_scale_prior(prior_full, model_cfg=model_cfg)
    if bool(use_diagonal_errors):
        sigma_full = subset_full.diag_sigma()
        cov_full = sigma_full**2
    fit_full = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_full, y0=y0_full, cov=cov_full, X=X_full, prior=prior_full))
    beta_true = fit_full.mean

    # Precompute masks for each cut on the *full* subset ordering.
    v_full = subset_full.get_column(cut_var)
    # Allow NaNs in cut variables (e.g., "always-included" calibrator rows). Treat NaNs as included.
    if cut_mode == "leq":
        cut_masks = [np.isnan(v_full) | (v_full <= float(c)) for c in cuts]
    else:
        cut_masks = [np.isnan(v_full) | (v_full >= float(c)) for c in cuts]

    # Noise model: diagonal unless explicitly requested otherwise.
    if bool(use_diagonal_errors):
        sigma = subset_full.diag_sigma()

    mc_end_to_end = np.empty(n_mc, dtype=float)
    mc_max_pair = np.empty(n_mc, dtype=float)
    mc_path = np.empty(n_mc, dtype=float)

    y_base = y0_full + X_full @ beta_true
    for t in range(n_mc):
        if bool(use_diagonal_errors):
            eps = rng.normal(0.0, sigma, size=sigma.size)
            y_sim = y_base + eps
        else:
            eps = rng.multivariate_normal(mean=np.zeros_like(y_base), cov=cov_full)
            y_sim = y_base + eps

        seq = []
        for mask, design in zip(cut_masks, designs, strict=True):
            y_cut = y_sim[mask]
            y0_cut = design.y0
            X_cut = design.X
            cov_cut = design.cov
            prior = design.prior
            spec = GaussianLinearModelSpec(y=y_cut, y0=y0_cut, cov=cov_cut, X=X_cut, prior=prior)
            fit = fit_gaussian_linear_model(spec)
            if drift_param not in fit.param_names:
                raise ValueError(f"drift_param '{drift_param}' not in model params {fit.param_names}")
            j = fit.param_names.index(drift_param)
            seq.append(float(fit.mean[j]))

        seq = np.asarray(seq, dtype=float)
        m = _drift_metrics(seq)
        mc_end_to_end[t] = m["end_to_end"]
        mc_max_pair[t] = m["max_pair"]
        mc_path[t] = m["path_length"]

    p_end = float(np.mean(mc_end_to_end >= obs_metrics["end_to_end"]))
    p_max = float(np.mean(mc_max_pair >= obs_metrics["max_pair"]))
    p_path = float(np.mean(mc_path >= obs_metrics["path_length"]))

    return DriftNullResult(
        cut_var=str(cut_var),
        cuts=[float(c) for c in cuts],
        param_name=str(drift_param),
        observed=obs_metrics,
        mc={
            "end_to_end": mc_end_to_end.tolist(),
            "max_pair": mc_max_pair.tolist(),
            "path_length": mc_path.tolist(),
        },
        p_values={"end_to_end": p_end, "max_pair": p_max, "path_length": p_path},
        cut_mode=cut_mode,
    )


def _drift_metrics(seq: np.ndarray) -> dict[str, float]:
    seq = np.asarray(seq, dtype=float)
    if seq.size < 2:
        return {"end_to_end": 0.0, "max_pair": 0.0, "path_length": 0.0}
    diffs = np.abs(np.diff(seq))
    path = float(np.sum(diffs))
    end = float(abs(seq[-1] - seq[0]))
    max_pair = float(np.max(np.abs(seq[:, None] - seq[None, :])))
    return {"end_to_end": end, "max_pair": max_pair, "path_length": path}
