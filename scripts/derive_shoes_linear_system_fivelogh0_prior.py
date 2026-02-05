from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def _resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _load_fits_array(path: Path) -> np.ndarray:
    arr = fits.getdata(str(path), memmap=True)
    if arr is None:
        raise ValueError(f"No data in FITS: {path}")
    return np.asarray(arr, dtype=float)


def solve_shoes_linear_system(*, L: np.ndarray, y: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve q from y = L.T @ q with covariance C (SH0ES linear system convention).

    L: (p, n)
    y: (n,)
    C: (n, n)

    Returns:
      q_hat: (p,)
      cov_q: (p, p)
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float)
    if L.ndim != 2:
        raise ValueError("L must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")
    p, n = L.shape
    if y.size != n:
        raise ValueError("Shape mismatch: expected y.size==L.shape[1]")
    if C.shape != (n, n):
        raise ValueError("Shape mismatch: expected C.shape==(n,n) with n=L.shape[1]")

    # q = (L C^{-1} L^T)^{-1} L C^{-1} y
    # cov(q) = (L C^{-1} L^T)^{-1}
    X = np.linalg.solve(C, L.T)  # (n, p) = C^{-1} L^T
    M = L @ X  # (p, p) = L C^{-1} L^T
    Cy = np.linalg.solve(C, y)  # (n,) = C^{-1} y
    rhs = L @ Cy  # (p,) = L C^{-1} y
    q_hat = np.linalg.solve(M, rhs)
    cov_q = np.linalg.inv(M)
    return q_hat, cov_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--y-path",
        default="data/raw/shoes_linear_system/ally_shoes_ceph_topantheonwt6.0_112221.fits",
        help="FITS path for the SH0ES linear-system y vector.",
    )
    ap.add_argument(
        "--L-path",
        default="data/raw/shoes_linear_system/alll_shoes_ceph_topantheonwt6.0_112221.fits",
        help="FITS path for the SH0ES linear-system L matrix (shape p x n).",
    )
    ap.add_argument(
        "--C-path",
        default="data/raw/shoes_linear_system/allc_shoes_ceph_topantheonwt6.0_112221.fits",
        help="FITS path for the SH0ES linear-system covariance C (shape n x n).",
    )
    ap.add_argument(
        "--param-idx",
        type=int,
        default=46,
        help="Parameter index to summarize (0-based; SH0ES example scripts suggest fivelogH0 is the last index).",
    )
    ap.add_argument(
        "--param-name",
        default="calibrator_offset_mag",
        help="Name to use in the output sigma_overrides mapping for the chosen param index.",
    )
    ap.add_argument(
        "--lstsq-results-path",
        default="data/raw/shoes_linear_system/lstsq_results.txt",
        help="Optional SH0ES lstsq_results.txt path for cross-checking and (by default) choosing the published sigma.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied to the derived sigma.",
    )
    ap.add_argument(
        "--out",
        default="data/processed/external_calibration/shoes_linear_system_sigma_overrides_fivelogh0_v1.json",
        help="Output JSON path for sigma_overrides mapping.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    y_path = _resolve_path(root, str(args.y_path))
    L_path = _resolve_path(root, str(args.L_path))
    C_path = _resolve_path(root, str(args.C_path))
    lstsq_path = _resolve_path(root, str(args.lstsq_results_path)) if args.lstsq_results_path else None

    out_path = _resolve_path(root, str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    y = _load_fits_array(y_path)
    L = _load_fits_array(L_path)
    C = _load_fits_array(C_path)

    q_hat, cov_q = solve_shoes_linear_system(L=L, y=y, C=C)

    j = int(args.param_idx)
    if j < 0 or j >= q_hat.size:
        raise ValueError(f"param-idx out of bounds: {j} for p={q_hat.size}")

    sigma_q_computed = float(np.sqrt(max(float(cov_q[j, j]), 0.0)))
    qj_computed = float(q_hat[j])

    qj_lstsq = None
    sigma_q_lstsq = None
    if lstsq_path is not None and lstsq_path.exists():
        lines = [ln.strip() for ln in lstsq_path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if len(lines) > j:
            parts = lines[j].split()
            if len(parts) >= 2:
                try:
                    qj_lstsq = float(parts[0])
                    sigma_q_lstsq = float(parts[1])
                except Exception:
                    qj_lstsq = None
                    sigma_q_lstsq = None

    # Prefer the published lstsq sigma when available.
    sigma_q = float(sigma_q_lstsq) if (sigma_q_lstsq is not None and np.isfinite(sigma_q_lstsq) and sigma_q_lstsq > 0) else sigma_q_computed
    qj = float(qj_lstsq) if (qj_lstsq is not None and np.isfinite(qj_lstsq)) else qj_computed
    sigma_out = float(args.scale) * float(sigma_q)

    # If this param is fivelog10(H0), convert to (H0_mean, H0_sd) using linear propagation.
    H0_mean = float(10.0 ** (qj / 5.0))
    H0_sd = float(H0_mean * (np.log(10.0) / 5.0) * sigma_q)

    out = {str(args.param_name): sigma_out}
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    meta: dict[str, Any] = {
        "created_by": "scripts/derive_shoes_linear_system_fivelogh0_prior.py",
        "inputs": {
            "y_path": str(args.y_path),
            "L_path": str(args.L_path),
            "C_path": str(args.C_path),
            "param_idx": int(args.param_idx),
            "param_name": str(args.param_name),
            "lstsq_results_path": str(args.lstsq_results_path) if args.lstsq_results_path else None,
            "scale": float(args.scale),
        },
        "results": {
            "q_hat": {"value": qj, "sigma": sigma_q},
            "q_hat_computed": {"value": qj_computed, "sigma": sigma_q_computed},
            "q_hat_lstsq_results": {"value": qj_lstsq, "sigma": sigma_q_lstsq},
            "H0_if_fivelogH0": {"mean": H0_mean, "sigma_approx": H0_sd},
        },
        "notes": [
            "This derives a 1D sigma from the published SH0ES linear-system FITS products (L,y,C), assuming y=L.T@q with covariance C.",
            "The chosen param_idx is treated as fivelog10(H0) for the derived H0 mean/sigma display (per SH0ES example scripts); no parameter-name mapping is bundled in the FITS headers.",
            "Use this as a *calibrator-chain-inspired* prior scale for calibrator-only mechanisms (e.g. calibrator_offset_mag). It is not an independent external calibration dataset.",
        ],
        "shapes": {"y": list(y.shape), "L": list(L.shape), "C": list(C.shape)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"Wrote sigma_overrides to {out_path}")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
