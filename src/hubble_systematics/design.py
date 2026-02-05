from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import sph_harm_y


def ones(n: int) -> np.ndarray:
    return np.ones((n, 1), dtype=float)


@dataclass(frozen=True)
class DesignMatrix:
    X: np.ndarray
    names: list[str]

    def append(self, other: "DesignMatrix") -> "DesignMatrix":
        if self.X.shape[0] != other.X.shape[0]:
            raise ValueError("Row mismatch in design append")
        X = np.concatenate([self.X, other.X], axis=1)
        return DesignMatrix(X=X, names=[*self.names, *other.names])


def one_hot(
    values: np.ndarray,
    *,
    prefix: str,
    reference: int | str | None = None,
    drop_reference: bool = True,
) -> DesignMatrix:
    v = np.asarray(values)
    uniq = np.unique(v)
    if reference is None:
        reference = uniq[0]
    cols = []
    names = []
    for u in uniq:
        if drop_reference and u == reference:
            continue
        cols.append((v == u).astype(float))
        names.append(f"{prefix}{u}")
    if not cols:
        return DesignMatrix(X=np.zeros((v.size, 0)), names=[])
    X = np.stack(cols, axis=1)
    return DesignMatrix(X=X, names=names)


def one_hot_levels(
    values: np.ndarray,
    *,
    levels: Iterable[int | float | str],
    prefix: str,
    reference: int | float | str | None = None,
    drop_reference: bool = True,
) -> DesignMatrix:
    """One-hot encoding with a fixed, caller-specified level ordering.

    This is important for reproducibility and cross-subset operations (e.g. cross-validation),
    where `np.unique(values)` would otherwise change the set/order of columns.
    """

    v = np.asarray(values)
    lv = list(levels)
    if not lv:
        return DesignMatrix(X=np.zeros((v.size, 0)), names=[])
    if reference is None:
        reference = lv[0]
    cols = []
    names = []
    for u in lv:
        if drop_reference and u == reference:
            continue
        cols.append((v == u).astype(float))
        names.append(f"{prefix}{u}")
    if not cols:
        return DesignMatrix(X=np.zeros((v.size, 0)), names=[])
    X = np.stack(cols, axis=1)
    return DesignMatrix(X=X, names=names)


def bspline_basis(
    x: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    n_internal_knots: int,
    degree: int = 3,
    prefix: str = "z_spline_",
) -> DesignMatrix:
    if n_internal_knots < 0:
        raise ValueError("n_internal_knots must be >=0")
    x = np.asarray(x, dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(x >= xmin) and np.all(x <= xmax)):
        raise ValueError("x must be finite and within [xmin,xmax] for extrapolate=False")

    if n_internal_knots == 0:
        internal = np.array([], dtype=float)
    else:
        internal = np.linspace(xmin, xmax, n_internal_knots + 2, dtype=float)[1:-1]
    t = np.concatenate(
        [
            np.repeat(xmin, degree + 1),
            internal,
            np.repeat(xmax, degree + 1),
        ]
    )
    B = BSpline.design_matrix(x, t, degree, extrapolate=False).toarray()
    names = [f"{prefix}{i}" for i in range(B.shape[1])]
    return DesignMatrix(X=B, names=names)


def second_difference_precision(n: int, sigma_d2: float) -> np.ndarray:
    """Return D2^T D2 / sigma_d2^2 for an n-vector of spline coefficients."""

    if n <= 2:
        return np.zeros((n, n))
    if sigma_d2 <= 0 or not np.isfinite(sigma_d2):
        raise ValueError("sigma_d2 must be positive and finite")
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i : i + 3] = np.array([1.0, -2.0, 1.0])
    return (D2.T @ D2) / (sigma_d2**2)


def sky_real_harmonics(
    *,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    lmin: int,
    lmax: int,
    prefix: str = "Y_",
) -> DesignMatrix:
    """Real spherical harmonics basis (a convenient regression basis; normalization is conventional).

    Uses:
      - m = 0: Re(Y_l^0)
      - m > 0: sqrt(2) * Re(Y_l^m)
      - m < 0: sqrt(2) * Im(Y_l^{|m|})
    """

    th = np.asarray(theta_rad, dtype=float)
    ph = np.asarray(phi_rad, dtype=float)
    if th.shape != ph.shape:
        raise ValueError("theta_rad and phi_rad must have same shape")
    if lmin < 0 or lmax < lmin:
        raise ValueError("invalid l range")

    cols: list[np.ndarray] = []
    names: list[str] = []
    for ell in range(lmin, lmax + 1):
        for m in range(-ell, ell + 1):
            if m == 0:
                y = sph_harm_y(ell, 0, th, ph).real
            elif m > 0:
                y = np.sqrt(2.0) * sph_harm_y(ell, m, th, ph).real
            else:
                y = np.sqrt(2.0) * sph_harm_y(ell, -m, th, ph).imag
            cols.append(y.astype(float))
            names.append(f"{prefix}l{ell}_m{m}")
    if not cols:
        return DesignMatrix(X=np.zeros((th.size, 0)), names=[])
    X = np.stack(cols, axis=1)
    return DesignMatrix(X=X, names=names)


def zscore_columns(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)
