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


def run_correlated_cut_null_siren_gate2_event_gaussian_mc(
    *,
    dataset,
    anchor,
    cut_var: str,
    cut_mode: str,
    cuts: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> DriftNullResult:
    """Correlated-cut drift null for Gate-2 siren datasets using an event-level Gaussian surrogate.

    This is designed for `siren_gate2_grid` datasets: it uses the per-event `logL_H0` curves to
    estimate event-level ln(H0) widths, then simulates new realizations under a stationary truth
    ln(H0)_true by drawing per-event ln(H0) means with those widths and evaluating a Gaussian
    likelihood curve for each event on the shared grid.

    The combined posterior for a subset of N events is computed with selection correction:
      logpost(H0) = sum_i logL_i(H0) - N * log_alpha_grid(H0) + const

    Drift metrics are computed on the posterior mean of ln(H0), reported as delta_lnH0 relative to
    the anchor.
    """

    if not hasattr(dataset, "logL_H0") or not hasattr(dataset, "H0_grid") or not hasattr(dataset, "log_alpha_grid"):
        raise ValueError("dataset is missing required Gate-2 siren fields (logL_H0, H0_grid, log_alpha_grid)")

    cut_mode = str(cut_mode).lower()
    if cut_mode not in {"leq", "geq"}:
        raise ValueError(f"cut_mode must be one of {{leq,geq}}; got {cut_mode!r}")
    cuts = np.asarray(cuts, dtype=float).reshape(-1)
    if cuts.size < 2:
        raise ValueError("cuts must contain at least 2 values")

    # Observed sequence (exact from the selection-corrected grid posterior moments).
    subset_fn = dataset.subset_leq if cut_mode == "leq" else dataset.subset_geq
    obs_seq = []
    for c in cuts:
        sub = subset_fn(cut_var, float(c))
        mu_ln, _ = sub.lnH0_moments()
        obs_seq.append(float(mu_ln - float(np.log(anchor.H0))))
    obs_seq = np.asarray(obs_seq, dtype=float)
    obs_metrics = _drift_metrics(obs_seq)

    # Simulation truth: use the full (least strict) subset.
    full_cut = float(cuts[-1]) if cut_mode == "leq" else float(cuts[0])
    ds_full = subset_fn(cut_var, full_cut)
    lnH0_true, _ = ds_full.lnH0_moments()
    lnH0_true = float(lnH0_true)

    # Per-event widths from the raw event logL curves (no selection factor).
    H0 = np.asarray(dataset.H0_grid, dtype=float).reshape(-1)
    lnH0 = np.log(H0)
    logL_ev = np.asarray(dataset.logL_H0, dtype=float)
    if logL_ev.ndim != 2 or logL_ev.shape[1] != H0.size:
        raise ValueError("dataset.logL_H0 must have shape (n_events, n_grid)")

    # Normalize each event curve with respect to dH0 on the grid and compute Var[lnH0].
    x = logL_ev - np.nanmax(logL_ev, axis=1, keepdims=True)
    w = np.exp(x)
    Z = np.trapezoid(w, H0, axis=1)
    bad = ~(np.isfinite(Z) & (Z > 0))
    if np.any(bad):
        raise ValueError("Some event likelihood curves have invalid normalization")
    w = w / Z[:, None]
    mu = np.trapezoid(w * lnH0[None, :], H0, axis=1)
    var = np.trapezoid(w * (lnH0[None, :] - mu[:, None]) ** 2, H0, axis=1)
    sd_ev = np.sqrt(np.clip(var, 1e-12, np.inf))

    # Precompute nested-cut indices using NaN-as-included semantics.
    v = np.asarray(dataset.get_column(cut_var), dtype=float).reshape(-1)
    if v.shape[0] != logL_ev.shape[0]:
        raise ValueError("cut_var column length mismatch vs events")
    v_mod = v.copy()
    if cut_mode == "geq":
        v_mod[np.isnan(v_mod)] = np.inf
    else:
        v_mod[np.isnan(v_mod)] = -np.inf
    order = np.argsort(v_mod)
    v_sorted = v_mod[order]
    sd_sorted = sd_ev[order]

    if cut_mode == "geq":
        # Included = suffix where v_sorted >= cut.
        start = np.searchsorted(v_sorted, cuts, side="left").astype(int)
        n_sub = (v_sorted.size - start).astype(int)
        # Pad one extra row for the empty suffix case.
        idx_pick = start
        which = "suffix"
    else:
        # Included = prefix where v_sorted <= cut.
        end = np.searchsorted(v_sorted, cuts, side="right").astype(int)
        n_sub = end.astype(int)
        idx_pick = end - 1  # last included row
        which = "prefix"

    log_alpha = np.asarray(dataset.log_alpha_grid, dtype=float).reshape(-1)
    if log_alpha.shape != H0.shape:
        raise ValueError("log_alpha_grid shape mismatch vs H0_grid")

    mc_end_to_end = np.empty(n_mc, dtype=float)
    mc_max_pair = np.empty(n_mc, dtype=float)
    mc_path = np.empty(n_mc, dtype=float)

    # Chunked simulation for memory stability.
    n_events = int(v_sorted.size)
    n_cuts = int(cuts.size)
    lnH0_anchor = float(np.log(anchor.H0))

    # Use moderate chunk sizes to limit (chunk * n_events * n_grid) memory.
    chunk = int(max(1, min(4000, n_mc)))
    t = 0
    while t < n_mc:
        b = int(min(chunk, n_mc - t))
        eps = rng.normal(0.0, 1.0, size=(b, n_events))
        m = lnH0_true + eps * sd_sorted[None, :]

        # logL_sim[b, n_events, n_grid]
        dx = (lnH0[None, None, :] - m[:, :, None]) / sd_sorted[None, :, None]
        logL_sim = -0.5 * dx * dx

        if which == "suffix":
            # cumulative sums from end
            csum = np.cumsum(logL_sim[:, ::-1, :], axis=1)[:, ::-1, :]
            csum = np.concatenate([csum, np.zeros((b, 1, H0.size))], axis=1)
            logL_sums = csum[:, idx_pick, :]
        else:
            csum = np.cumsum(logL_sim, axis=1)
            csum = np.concatenate([np.zeros((b, 1, H0.size)), csum], axis=1)
            # idx_pick in [-1, n_events-1]; shift by +1 due to leading zeros row.
            logL_sums = csum[:, idx_pick + 1, :]

        logpost = logL_sums - n_sub[None, :, None].astype(float) * log_alpha[None, None, :]
        a = logpost - np.max(logpost, axis=2, keepdims=True)
        wpost = np.exp(a)
        Zpost = np.trapezoid(wpost, H0, axis=2)
        # Guard against underflow.
        Zpost = np.where(Zpost > 0, Zpost, np.nan)
        mu_ln = np.trapezoid(wpost * lnH0[None, None, :], H0, axis=2) / Zpost
        seq = mu_ln - lnH0_anchor

        end_to_end = np.abs(seq[:, -1] - seq[:, 0])
        path = np.sum(np.abs(np.diff(seq, axis=1)), axis=1)
        max_pair = np.max(np.abs(seq[:, :, None] - seq[:, None, :]), axis=(1, 2))

        mc_end_to_end[t : t + b] = end_to_end
        mc_path[t : t + b] = path
        mc_max_pair[t : t + b] = max_pair
        t += b

    p_end = float(np.mean(mc_end_to_end >= obs_metrics["end_to_end"]))
    p_max = float(np.mean(mc_max_pair >= obs_metrics["max_pair"]))
    p_path = float(np.mean(mc_path >= obs_metrics["path_length"]))

    return DriftNullResult(
        cut_var=str(cut_var),
        cuts=[float(c) for c in cuts.tolist()],
        param_name="delta_lnH0",
        observed=obs_metrics,
        mc={"end_to_end": mc_end_to_end.tolist(), "max_pair": mc_max_pair.tolist(), "path_length": mc_path.tolist()},
        p_values={"end_to_end": p_end, "max_pair": p_max, "path_length": p_path},
        cut_mode=cut_mode,
    )


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
