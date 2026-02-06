from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


ApplyTo = Literal["all", "cal", "hf"]


@dataclass(frozen=True)
class ProxySummary:
    proxy: str
    apply_to: ApplyTo
    n: int
    n_nonzero: int
    v_norm2: float
    vCv: float
    sigma_beta_mag: float | None
    notes: list[str]


def _sigma_beta_from_cov(*, cov: np.ndarray, v: np.ndarray) -> float | None:
    cov = np.asarray(cov, dtype=float)
    v = np.asarray(v, dtype=float).reshape(-1)
    if cov.shape != (v.size, v.size):
        raise ValueError("cov/v shape mismatch")
    vv = float(v @ v)
    if not np.isfinite(vv) or vv <= 0.0:
        return None
    vCv = float(v @ (cov @ v))
    if not np.isfinite(vCv):
        return None
    vCv = float(max(vCv, 0.0))
    return float(np.sqrt(vCv)) / vv


def _zscore(x: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    good = np.asarray(good, dtype=bool).reshape(-1)
    if x.shape != good.shape:
        raise ValueError("x/good shape mismatch")
    if not np.any(good):
        raise ValueError("No good entries to z-score")
    mu = float(np.mean(x[good]))
    sd = float(np.std(x[good]) + 1e-12)
    out = np.zeros_like(x, dtype=float)
    out[good] = (x[good] - mu) / sd
    return out, mu, sd


def _apply_mask(v: np.ndarray, *, apply_to: ApplyTo, is_cal: np.ndarray, is_hf: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    is_cal = np.asarray(is_cal, dtype=bool).reshape(-1)
    is_hf = np.asarray(is_hf, dtype=bool).reshape(-1)
    if v.shape != is_cal.shape or v.shape != is_hf.shape:
        raise ValueError("mask shape mismatch")
    if apply_to == "cal":
        return v * is_cal.astype(float)
    if apply_to == "hf":
        return v * is_hf.astype(float)
    if apply_to == "all":
        return v
    raise ValueError(f"Unsupported apply_to: {apply_to}")


def _compute_proxy_sigma(*, ds, cov: np.ndarray, proxy: str, apply_to: ApplyTo) -> tuple[float | None, ProxySummary]:
    proxy = str(proxy)
    apply_to = str(apply_to).lower()  # type: ignore[assignment]
    if apply_to not in {"all", "cal", "hf"}:
        raise ValueError(f"apply_to must be one of all/cal/hf; got {apply_to!r}")

    notes: list[str] = []
    is_cal = np.asarray(ds.is_calibrator, dtype=bool)
    is_hf = np.asarray(ds.is_hubble_flow, dtype=bool)

    if proxy == "c_linear_mag":
        x = np.asarray(ds.c, dtype=float)
        v, mu, sd = _zscore(x, np.isfinite(x))
        notes.append(f"z-score: mu={mu:.4g}, sd={sd:.4g}")
    elif proxy == "x1_linear_mag":
        x = np.asarray(ds.x1, dtype=float)
        v, mu, sd = _zscore(x, np.isfinite(x))
        notes.append(f"z-score: mu={mu:.4g}, sd={sd:.4g}")
    elif proxy == "biascor_m_b_linear_mag":
        x = np.asarray(ds.biascor_m_b, dtype=float)
        v, mu, sd = _zscore(x, np.isfinite(x))
        notes.append(f"z-score: mu={mu:.4g}, sd={sd:.4g}")
    elif proxy == "mwebv_linear_mag":
        x = np.asarray(ds.mwebv, dtype=float)
        good = np.isfinite(x) & (x >= 0.0)
        v, mu, sd = _zscore(x, good)
        notes.append(f"z-score: mu={mu:.4g}, sd={sd:.4g}")
    elif proxy == "pkmjd_err_linear_mag":
        x = np.asarray(ds.pkmjd_err, dtype=float)
        good = np.isfinite(x) & (x > 0.0)
        v, mu, sd = _zscore(x, good)
        notes.append(f"z-score: mu={mu:.4g}, sd={sd:.4g}")
    elif proxy == "host_mass_step_mag":
        thr = 10.0
        x = np.asarray(ds.host_logmass, dtype=float)
        good = np.isfinite(x) & (x > 0.0)
        v = np.zeros_like(x, dtype=float)
        v[good] = (x[good] >= thr).astype(float)
        notes.append(f"binary step: thr={thr:.3f}")
    else:
        raise ValueError(f"Unsupported proxy: {proxy!r}")

    v = _apply_mask(v, apply_to=apply_to, is_cal=is_cal, is_hf=is_hf)
    sigma_beta = _sigma_beta_from_cov(cov=cov, v=v)

    summ = ProxySummary(
        proxy=proxy,
        apply_to=apply_to,
        n=int(v.size),
        n_nonzero=int(np.sum(v != 0.0)),
        v_norm2=float(v @ v),
        vCv=float(v @ (cov @ v)),
        sigma_beta_mag=(None if sigma_beta is None else float(sigma_beta)),
        notes=notes,
    )
    return sigma_beta, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_covproj_v1.json",
        help="Output JSON path for sigma_overrides mapping.",
    )
    ap.add_argument(
        "--raw-dat-path",
        default="data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat",
        help="Path to the Pantheon+SH0ES .dat file (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--raw-cov-path",
        required=True,
        help="Path to the Pantheon+SH0ES .cov file to project onto (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--processed-dir",
        default="data/processed/pantheon_plus_shoes_ladder",
        help="Directory for cached processed Pantheon+SH0ES ladder products (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--tag",
        default="cal+hf_covproj_zHD",
        help="Dataset cache tag (change when swapping raw_cov_path / cuts).",
    )
    ap.add_argument(
        "--proxy",
        action="append",
        default=[],
        help="Proxy parameter name(s) to derive (e.g. c_linear_mag). May be given multiple times. Empty means all supported proxies.",
    )
    ap.add_argument(
        "--apply-to",
        choices=["all", "cal", "hf"],
        default="cal",
        help="Mask for the proxy vector (default: cal).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    raw_dat_path = Path(args.raw_dat_path).expanduser()
    if not raw_dat_path.is_absolute():
        raw_dat_path = (root / raw_dat_path).resolve()
    raw_cov_path = Path(args.raw_cov_path).expanduser()
    if not raw_cov_path.is_absolute():
        raw_cov_path = (root / raw_cov_path).resolve()
    processed_dir = Path(args.processed_dir).expanduser()
    if not processed_dir.is_absolute():
        processed_dir = (root / processed_dir).resolve()

    ds = load_pantheon_plus_shoes_ladder_dataset(
        {
            "raw_dat_path": str(raw_dat_path),
            "raw_cov_path": str(raw_cov_path),
            "include_calibrators": True,
            "include_hubble_flow": True,
            "z_column": "zHD",
            "z_hf_min": 0.023,
            "z_hf_max": 0.15,
            "tag": str(args.tag),
            "processed_dir": str(processed_dir),
        }
    )
    cov = np.asarray(ds.cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Expected dense covariance matrix for cov projection")

    supported = [
        "c_linear_mag",
        "x1_linear_mag",
        "biascor_m_b_linear_mag",
        "mwebv_linear_mag",
        "pkmjd_err_linear_mag",
        "host_mass_step_mag",
    ]
    proxies = list(args.proxy) or supported
    proxies = [str(p) for p in proxies]
    for p in proxies:
        if p not in supported:
            raise ValueError(f"Unsupported proxy {p!r}; supported: {supported}")

    overrides: dict[str, float] = {}
    summaries: list[ProxySummary] = []
    for proxy in proxies:
        sigma_beta, summ = _compute_proxy_sigma(ds=ds, cov=cov, proxy=proxy, apply_to=str(args.apply_to))
        summaries.append(summ)
        if sigma_beta is None:
            continue
        overrides[proxy] = float(sigma_beta)

    out_path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    meta: dict[str, Any] = {
        "created_by": "scripts/derive_proxy_priors_from_cov_projection.py",
        "inputs": {
            "raw_dat_path": str(raw_dat_path),
            "raw_cov_path": str(raw_cov_path),
            "tag": str(args.tag),
            "processed_dir": str(processed_dir),
        },
        "apply_to": str(args.apply_to),
        "supported_proxies": supported,
        "requested_proxies": proxies,
        "summaries": [asdict(s) for s in summaries],
        "notes": [
            "This file encodes per-parameter sigma overrides for selected proxy coefficients by projecting a published covariance matrix C onto the proxy vector v.",
            "We set sigma(beta) so that Var(v^T (beta v)) matches v^T C v, i.e. sigma(beta)=sqrt(v^T C v)/(v^T v).",
            "This is a pragmatic way to translate a covariance budget into a bound on a specific low-dimensional mechanism term.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(overrides)} proxy sigma overrides to {out_path}")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()

