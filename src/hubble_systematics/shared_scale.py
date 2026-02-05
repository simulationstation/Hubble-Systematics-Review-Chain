from __future__ import annotations

from typing import Any

import numpy as np

from hubble_systematics.gaussian_linear_model import GaussianPrior


def shared_scale_params(model_cfg: dict[str, Any]) -> list[str]:
    shared = (model_cfg or {}).get("shared_scale", {}) or {}
    if not bool(shared.get("enable", False)):
        return []
    params = shared.get("params", [])
    if isinstance(params, str):
        return [params]
    return [str(p) for p in params]


def shared_scale_precision(
    *,
    param_names: list[str],
    model_cfg: dict[str, Any],
) -> np.ndarray:
    shared = (model_cfg or {}).get("shared_scale", {}) or {}
    if not bool(shared.get("enable", False)):
        return np.zeros((len(param_names), len(param_names)))

    params = shared_scale_params(model_cfg)
    if not params:
        return np.zeros((len(param_names), len(param_names)))

    sigma_default = shared.get("prior_sigma", np.inf)
    sigma_map = (shared.get("sigma", {}) or {}) if isinstance(shared.get("sigma", {}), dict) else {}

    P = np.zeros((len(param_names), len(param_names)), dtype=float)
    for p in params:
        if p not in param_names:
            continue
        j = param_names.index(p)
        sigma = float(sigma_map.get(p, sigma_default))
        if np.isfinite(sigma) and sigma > 0:
            P[j, j] += 1.0 / (sigma**2)
    return P


def apply_shared_scale_prior(prior: GaussianPrior, *, model_cfg: dict[str, Any]) -> GaussianPrior:
    """Apply global (non per-probe) priors for shared-scale stress-test parameters.

    Important: for joint/stacked fits, call this once on the stacked prior to avoid
    double-counting. Per-probe build_design methods should keep shared params
    unregularized (sigma=inf) and let the runner apply this.
    """

    P_extra = shared_scale_precision(param_names=prior.param_names, model_cfg=model_cfg)
    if not np.any(P_extra):
        return prior
    P = prior.precision_matrix() + P_extra
    return GaussianPrior(param_names=list(prior.param_names), mean=np.asarray(prior.mean, dtype=float), sigma=None, precision=P)
