from __future__ import annotations

import numpy as np

from hubble_systematics.gaussian_linear_model import GaussianLinearModelSpec, GaussianPrior, fit_gaussian_linear_model


def test_diag_cov_equivalence_to_dense_diag() -> None:
    rng = np.random.default_rng(0)
    n = 50
    X = np.stack([np.ones(n), rng.normal(size=n)], axis=1)
    beta_true = np.array([1.2, -0.7])
    y0 = np.zeros(n)
    sigma = 0.3 + 0.1 * rng.random(n)
    var = sigma**2
    y = y0 + X @ beta_true + rng.normal(0.0, sigma)

    prior = GaussianPrior.from_sigmas(["b0", "b1"], [10.0, 10.0])

    fit_diag = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=var, X=X, prior=prior))
    fit_dense = fit_gaussian_linear_model(GaussianLinearModelSpec(y=y, y0=y0, cov=np.diag(var), X=X, prior=prior))

    assert np.allclose(fit_diag.mean, fit_dense.mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(fit_diag.cov, fit_dense.cov, rtol=1e-10, atol=1e-10)
    assert np.isclose(fit_diag.chi2, fit_dense.chi2, rtol=1e-10, atol=1e-10)

