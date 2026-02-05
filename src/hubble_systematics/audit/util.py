from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def cholesky_with_jitter(cov: np.ndarray, *, jitter_rel: float = 1e-10, max_tries: int = 6) -> np.ndarray:
    """Cholesky factorization with a tiny diagonal jitter fallback.

    Used only for sampling / logpdf numerics when the covariance is near-singular.
    """
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov must be square")

    n = int(C.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # Try without jitter first.
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        pass

    # Scale jitter by mean diagonal (fallback to 1.0 if invalid).
    diag = np.diag(C)
    finite = np.isfinite(diag)
    scale = float(np.mean(diag[finite])) if np.any(finite) else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    jitter0 = float(jitter_rel) * scale
    jitter = jitter0 if jitter0 > 0.0 else 1e-12

    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(C + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # Give up; let the error propagate with the final jitter attempt for context.
    return np.linalg.cholesky(C + jitter * np.eye(n))


def make_cut_grid(dataset, cut_var: str, scan_cfg: dict[str, Any]) -> np.ndarray:
    cut_min = scan_cfg.get("cut_min")
    cut_max = scan_cfg.get("cut_max")
    n_cuts = int(scan_cfg.get("n_cuts", 12))

    vals = dataset.get_column(cut_var)
    if cut_min is None:
        cut_min = float(np.nanmin(vals))
    if cut_max is None:
        cut_max = float(np.nanmax(vals))
    # Keep cuts inside the available support to avoid empty subsets at endpoints.
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    cut_min = max(float(cut_min), vmin)
    cut_max = min(float(cut_max), vmax)

    if n_cuts < 2:
        raise ValueError("n_cuts must be >=2")
    return np.linspace(float(cut_min), float(cut_max), n_cuts)
