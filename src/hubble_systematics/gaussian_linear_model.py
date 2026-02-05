from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GaussianPrior:
    param_names: list[str]
    mean: np.ndarray
    sigma: np.ndarray | None = None
    precision: np.ndarray | None = None

    @classmethod
    def from_sigmas(cls, param_names: Sequence[str], sigma: Sequence[float] | float, mean: float = 0.0) -> "GaussianPrior":
        names = list(param_names)
        n = len(names)
        if np.isscalar(sigma):
            sig = np.full(n, float(sigma))
        else:
            sig = np.asarray(list(sigma), dtype=float)
            if sig.shape != (n,):
                raise ValueError("sigma must have shape (n_params,)")
        mu = np.full(n, float(mean))
        return cls(param_names=names, mean=mu, sigma=sig, precision=None)

    @classmethod
    def from_precision(cls, param_names: Sequence[str], precision: np.ndarray, mean: float = 0.0) -> "GaussianPrior":
        names = list(param_names)
        n = len(names)
        P = np.asarray(precision, dtype=float)
        if P.shape != (n, n):
            raise ValueError("precision must have shape (n_params, n_params)")
        mu = np.full(n, float(mean))
        return cls(param_names=names, mean=mu, sigma=None, precision=P)

    def precision_matrix(self) -> np.ndarray:
        if self.precision is not None:
            P = np.asarray(self.precision, dtype=float)
            if P.shape != (len(self.param_names), len(self.param_names)):
                raise ValueError("prior.precision shape mismatch")
            return P
        if self.sigma is None:
            return np.zeros((len(self.param_names), len(self.param_names)))
        sig = np.asarray(self.sigma, dtype=float)
        if sig.shape != (len(self.param_names),):
            raise ValueError("prior.sigma shape mismatch")
        out = np.zeros((sig.size, sig.size), dtype=float)
        finite = np.isfinite(sig) & (sig > 0)
        out[np.diag_indices_from(out)] = np.where(finite, 1.0 / (sig**2), 0.0)
        return out


@dataclass(frozen=True)
class GaussianLinearModelSpec:
    y: np.ndarray
    y0: np.ndarray
    cov: np.ndarray
    X: np.ndarray
    prior: GaussianPrior


@dataclass(frozen=True)
class GaussianLinearFit:
    param_names: list[str]
    mean: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    logdet_cov: float


def fit_gaussian_linear_model(spec: GaussianLinearModelSpec) -> GaussianLinearFit:
    y = np.asarray(spec.y, dtype=float).reshape(-1)
    y0 = np.asarray(spec.y0, dtype=float).reshape(-1)
    X = np.asarray(spec.X, dtype=float)
    cov = np.asarray(spec.cov, dtype=float)

    n = y.shape[0]
    if y0.shape != (n,):
        raise ValueError("y0 must have shape (n,)")
    if X.shape[0] != n:
        raise ValueError("X must have shape (n, p)")
    cov_is_diag = False
    if cov.ndim == 1:
        if cov.shape != (n,):
            raise ValueError("cov (diag) must have shape (n,)")
        if not (np.all(np.isfinite(cov)) and np.all(cov > 0)):
            raise ValueError("cov diagonal must be finite and >0")
        cov_is_diag = True
    else:
        if cov.shape != (n, n):
            raise ValueError("cov must have shape (n, n)")

    p = X.shape[1]
    if len(spec.prior.param_names) != p:
        raise ValueError("prior.param_names length must match X columns")

    if cov_is_diag:
        var = cov
        logdet_cov = float(np.sum(np.log(var)))

        def solve_C(v: np.ndarray) -> np.ndarray:
            return np.asarray(v, dtype=float) / var[..., None] if v.ndim == 2 else np.asarray(v, dtype=float) / var

        def quad_C(v: np.ndarray) -> float:
            v = np.asarray(v, dtype=float).reshape(-1)
            return float(np.sum((v * v) / var))

    else:
        # C^{-1} via Cholesky
        L = np.linalg.cholesky(cov)
        logdet_cov = 2.0 * float(np.sum(np.log(np.diag(L))))

        def solve_C(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            u = np.linalg.solve(L, v)
            return np.linalg.solve(L.T, u)

        def quad_C(v: np.ndarray) -> float:
            v = np.asarray(v, dtype=float).reshape(-1)
            u = solve_C(v)
            return float(v @ u)

    r = y - y0
    if p == 0:
        chi2 = quad_C(r)
        return GaussianLinearFit(
            param_names=[],
            mean=np.zeros((0,)),
            cov=np.zeros((0, 0)),
            chi2=chi2,
            dof=int(n),
            logdet_cov=logdet_cov,
        )

    Cinvr = solve_C(r)
    XtCinvr = X.T @ Cinvr
    XtCinvX = X.T @ solve_C(X)

    prior = spec.prior
    prior_mean = np.asarray(prior.mean, dtype=float).reshape(p)
    P = prior.precision_matrix()

    A = XtCinvX + P
    b = XtCinvr + P @ prior_mean

    mean = np.linalg.solve(A, b)
    cov_post = np.linalg.inv(A)

    r_post = r - X @ mean
    chi2 = quad_C(r_post)

    dof = int(n - p)
    return GaussianLinearFit(
        param_names=list(prior.param_names),
        mean=mean,
        cov=cov_post,
        chi2=chi2,
        dof=dof,
        logdet_cov=logdet_cov,
    )


def log_marginal_likelihood(spec: GaussianLinearModelSpec) -> float | None:
    """Exact log marginal likelihood for a linear-Gaussian model with a proper Gaussian prior.

    Returns None if the prior is improper/singular (e.g. infinite-variance parameters), since
    the evidence would be undefined up to an arbitrary constant and is not comparable.
    """

    y = np.asarray(spec.y, dtype=float).reshape(-1)
    y0 = np.asarray(spec.y0, dtype=float).reshape(-1)
    X = np.asarray(spec.X, dtype=float)
    cov = np.asarray(spec.cov, dtype=float)

    n = y.size
    if y0.shape != (n,) or X.shape[0] != n:
        raise ValueError("Shape mismatch in spec for log_marginal_likelihood")

    cov_is_diag = cov.ndim == 1
    if cov_is_diag:
        if cov.shape != (n,):
            raise ValueError("cov (diag) must have shape (n,)")
        if not (np.all(np.isfinite(cov)) and np.all(cov > 0)):
            raise ValueError("cov diagonal must be finite and >0")
        var = cov
        logdet_cov = float(np.sum(np.log(var)))

        def solve_C(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            return v / var[..., None] if v.ndim == 2 else v / var

    else:
        if cov.shape != (n, n):
            raise ValueError("cov must have shape (n, n)")
        L = np.linalg.cholesky(cov)
        logdet_cov = 2.0 * float(np.sum(np.log(np.diag(L))))

        def solve_C(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            u = np.linalg.solve(L, v)
            return np.linalg.solve(L.T, u)

    r = y - y0
    p = X.shape[1]

    # No-parameter special case.
    if p == 0:
        Cinvr = solve_C(r)
        rCinv_r = float(r @ Cinvr)
        return -0.5 * (n * np.log(2.0 * np.pi) + logdet_cov + rCinv_r)

    prior = spec.prior
    if len(prior.param_names) != p:
        raise ValueError("prior.param_names length must match X columns")
    mu0 = np.asarray(prior.mean, dtype=float).reshape(p)
    P = np.asarray(prior.precision_matrix(), dtype=float)
    if P.shape != (p, p):
        raise ValueError("prior.precision shape mismatch")

    signP, logdetP = np.linalg.slogdet(P)
    if signP <= 0 or not np.isfinite(logdetP):
        return None

    Cinvr = solve_C(r)
    XtCinvr = X.T @ Cinvr
    XtCinvX = X.T @ solve_C(X)
    A = XtCinvX + P
    signA, logdetA = np.linalg.slogdet(A)
    if signA <= 0 or not np.isfinite(logdetA):
        return None

    b = XtCinvr + P @ mu0
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    bAinvb = float(b @ x)
    rCinv_r = float(r @ Cinvr)
    mu0Pmu0 = float(mu0 @ (P @ mu0))

    # log p(y) = -n/2 log(2π) - 1/2 log|C| + 1/2 log|P| - 1/2 log|A|
    #            - 1/2 [ rᵀ C^{-1} r + μ0ᵀ P μ0 - bᵀ A^{-1} b ].
    return float(
        -0.5 * n * np.log(2.0 * np.pi)
        - 0.5 * logdet_cov
        + 0.5 * logdetP
        - 0.5 * logdetA
        - 0.5 * (rCinv_r + mu0Pmu0 - bAinvb)
    )
