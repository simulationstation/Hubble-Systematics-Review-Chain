from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hubble_systematics.gaussian_linear_model import GaussianPrior


@dataclass(frozen=True)
class Part:
    y: np.ndarray
    y0: np.ndarray
    cov: np.ndarray  # (n,n) or diag var (n,)
    X: np.ndarray  # (n,p_i)
    prior: GaussianPrior


@dataclass(frozen=True)
class Stacked:
    y: np.ndarray
    y0: np.ndarray
    cov: np.ndarray
    X: np.ndarray
    prior: GaussianPrior


def stack_parts(parts: Iterable[Part]) -> Stacked:
    parts = list(parts)
    if not parts:
        raise ValueError("Cannot stack empty parts")

    ys = [np.asarray(p.y, dtype=float).reshape(-1) for p in parts]
    y0s = [np.asarray(p.y0, dtype=float).reshape(-1) for p in parts]
    covs = [np.asarray(p.cov, dtype=float) for p in parts]

    y = np.concatenate(ys, axis=0)
    y0 = np.concatenate(y0s, axis=0)
    cov = _block_diag(covs)

    # Global parameter ordering = insertion order across parts.
    name_to_idx: dict[str, int] = {}
    names: list[str] = []
    for part in parts:
        for n in part.prior.param_names:
            if n not in name_to_idx:
                name_to_idx[n] = len(names)
                names.append(n)
    p = len(names)

    X = np.zeros((y.size, p), dtype=float)
    row0 = 0
    for part in parts:
        n = int(np.asarray(part.y).size)
        local_names = part.prior.param_names
        local_X = np.asarray(part.X, dtype=float)
        if local_X.shape != (n, len(local_names)):
            raise ValueError("Part X shape mismatch")
        for j_local, nname in enumerate(local_names):
            j_global = name_to_idx[nname]
            X[row0 : row0 + n, j_global] = local_X[:, j_local]
        row0 += n

    prior = _combine_priors(names=names, parts=parts, name_to_idx=name_to_idx)
    return Stacked(y=y, y0=y0, cov=cov, X=X, prior=prior)


def _block_diag(mats: list[np.ndarray]) -> np.ndarray:
    if len(mats) == 1:
        return mats[0]
    ndims = {np.asarray(m).ndim for m in mats}
    if ndims == {1}:
        return np.concatenate([np.asarray(m).reshape(-1) for m in mats], axis=0)
    if ndims == {2}:
        sizes = [m.shape[0] for m in mats]
        out = np.zeros((sum(sizes), sum(sizes)))
        i = 0
        for m in mats:
            n = m.shape[0]
            out[i : i + n, i : i + n] = m
            i += n
        return out

    # Mixed: promote 1D diag variances to dense blocks.
    dense = []
    for m in mats:
        m = np.asarray(m)
        if m.ndim == 1:
            dense.append(np.diag(m))
        else:
            dense.append(m)
    sizes = [m.shape[0] for m in dense]
    out = np.zeros((sum(sizes), sum(sizes)))
    i = 0
    for m in dense:
        n = m.shape[0]
        out[i : i + n, i : i + n] = m
        i += n
    return out


def _combine_priors(*, names: list[str], parts: list[Part], name_to_idx: dict[str, int]) -> GaussianPrior:
    p = len(names)
    P = np.zeros((p, p), dtype=float)
    eta = np.zeros(p, dtype=float)

    for part in parts:
        local_names = part.prior.param_names
        if not local_names:
            continue
        P_loc = part.prior.precision_matrix()
        mu_loc = np.asarray(part.prior.mean, dtype=float).reshape(-1)
        if P_loc.shape != (len(local_names), len(local_names)):
            raise ValueError("Local prior precision shape mismatch")
        if mu_loc.shape != (len(local_names),):
            raise ValueError("Local prior mean shape mismatch")
        eta_loc = P_loc @ mu_loc
        for i_loc, ni in enumerate(local_names):
            i_gl = name_to_idx[ni]
            eta[i_gl] += float(eta_loc[i_loc])
            for j_loc, nj in enumerate(local_names):
                j_gl = name_to_idx[nj]
                P[i_gl, j_gl] += float(P_loc[i_loc, j_loc])

    mu = np.zeros(p, dtype=float)
    active = np.any(P != 0.0, axis=1)
    if np.any(active):
        P_sub = P[np.ix_(active, active)]
        eta_sub = eta[active]
        try:
            mu_sub = np.linalg.solve(P_sub, eta_sub)
        except np.linalg.LinAlgError:
            mu_sub = np.linalg.lstsq(P_sub, eta_sub, rcond=None)[0]
        mu[active] = mu_sub

    return GaussianPrior(param_names=names, mean=mu, sigma=None, precision=P)

