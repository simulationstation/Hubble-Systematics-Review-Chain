from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hubble_systematics.anchors import AnchorLCDM
from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, fit_gaussian_linear_model
from hubble_systematics.shared_scale import apply_shared_scale_prior


@dataclass(frozen=True)
class SBCResult:
    n_rep: int
    use_diagonal_errors: bool
    param_names: list[str]
    coverage_68: dict[str, float]
    coverage_95: dict[str, float]
    zscore_mean: dict[str, float]
    zscore_std: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_rep": self.n_rep,
            "use_diagonal_errors": self.use_diagonal_errors,
            "param_names": self.param_names,
            "coverage_68": self.coverage_68,
            "coverage_95": self.coverage_95,
            "zscore_mean": self.zscore_mean,
            "zscore_std": self.zscore_std,
        }


def run_sbc(
    *,
    dataset,
    anchor: AnchorLCDM,
    ladder_level: str,
    model_cfg: dict[str, Any],
    sbc_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> SBCResult:
    n_rep = int(sbc_cfg.get("n_rep", 256))
    use_diag = bool(sbc_cfg.get("use_diagonal_errors", True))

    y_obs, y0, cov, X, prior = dataset.build_design(anchor=anchor, ladder_level=ladder_level, cfg=model_cfg)
    prior = apply_shared_scale_prior(prior, model_cfg=model_cfg)
    if use_diag:
        sigma = dataset.diag_sigma()
        cov = sigma**2
    fit0 = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_obs, y0=y0, cov=cov, X=X, prior=prior))
    beta_true = fit0.mean

    p = beta_true.size
    if p == 0:
        return SBCResult(
            n_rep=n_rep,
            use_diagonal_errors=use_diag,
            param_names=[],
            coverage_68={},
            coverage_95={},
            zscore_mean={},
            zscore_std={},
        )

    # Replicate fits under the truth with repeated noise.
    inside_68 = np.zeros((n_rep, p), dtype=bool)
    inside_95 = np.zeros((n_rep, p), dtype=bool)
    zscores = np.empty((n_rep, p), dtype=float)

    y_base = y0 + X @ beta_true
    for r in range(n_rep):
        if use_diag:
            eps = rng.normal(0.0, sigma, size=sigma.size)
        else:
            eps = rng.multivariate_normal(mean=np.zeros_like(y_base), cov=cov)
        y_sim = y_base + eps
        fit = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y_sim, y0=y0, cov=cov, X=X, prior=prior))
        sd = np.sqrt(np.diag(fit.cov))
        z = (beta_true - fit.mean) / sd
        zscores[r] = z
        inside_68[r] = np.abs(z) <= 1.0
        inside_95[r] = np.abs(z) <= 1.96

    names = fit0.param_names
    cov68 = {n: float(np.mean(inside_68[:, i])) for i, n in enumerate(names)}
    cov95 = {n: float(np.mean(inside_95[:, i])) for i, n in enumerate(names)}
    zmu = {n: float(np.mean(zscores[:, i])) for i, n in enumerate(names)}
    zsd = {n: float(np.std(zscores[:, i])) for i, n in enumerate(names)}
    return SBCResult(
        n_rep=n_rep,
        use_diagonal_errors=use_diag,
        param_names=names,
        coverage_68=cov68,
        coverage_95=cov95,
        zscore_mean=zmu,
        zscore_std=zsd,
    )
