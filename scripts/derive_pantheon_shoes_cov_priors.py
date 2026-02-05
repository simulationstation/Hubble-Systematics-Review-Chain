from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hubble_systematics.probes.pantheon_plus_shoes_ladder import load_pantheon_plus_shoes_ladder_dataset


def _estimate_sigma_from_cov(*, cov: np.ndarray, mask: np.ndarray, method: str = "median") -> float | None:
    cov = np.asarray(cov, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return None
    C = cov[np.ix_(idx, idx)]
    n = C.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    off = off[off > 0.0]
    if off.size == 0:
        return None
    if method == "median":
        v = float(np.median(off))
    elif method == "mean":
        v = float(np.mean(off))
    else:
        raise ValueError(f"Unsupported method: {method!r}")
    return float(np.sqrt(max(v, 0.0)))


def _pkmjd_bin_id(*, t: np.ndarray, edges: list[float]) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    edges = [float(x) for x in edges]
    edges = sorted(edges)
    if len(edges) < 3:
        raise ValueError("edges must have length >=3")
    good = np.isfinite(t) & (t > 0.0)
    inner = np.asarray(edges[1:-1], dtype=float)
    bid = np.zeros_like(t, dtype=int)
    bid[good] = np.digitize(t[good], inner, right=False)
    return bid


def derive_sigma_overrides(
    *,
    cov: np.ndarray,
    idsurvey: np.ndarray,
    is_cal: np.ndarray,
    is_hf: np.ndarray,
    pkmjd: np.ndarray,
    pkmjd_edges: list[float],
    min_cal_per_survey: int,
    min_hf_per_survey: int,
    min_per_time_bin: int,
    include_survey_offsets: bool,
    include_hf_survey_offsets: bool,
    include_cal_survey_offsets: bool,
    include_calibrator_offset: bool,
    include_time_bins: bool,
    include_survey_time_bins: bool,
    method: str,
    scale: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    cov = np.asarray(cov, dtype=float)
    idsurvey = np.asarray(idsurvey, dtype=int)
    is_cal = np.asarray(is_cal, dtype=bool)
    is_hf = np.asarray(is_hf, dtype=bool)
    pkmjd = np.asarray(pkmjd, dtype=float)

    overrides: dict[str, float] = {}
    meta: dict[str, Any] = {
        "method": method,
        "scale": float(scale),
        "min_cal_per_survey": int(min_cal_per_survey),
        "min_hf_per_survey": int(min_hf_per_survey),
        "min_per_time_bin": int(min_per_time_bin),
        "pkmjd_edges": [float(x) for x in pkmjd_edges],
    }

    # Global calibrator offset (all calibrators share the same shift).
    if include_calibrator_offset:
        sig = _estimate_sigma_from_cov(cov=cov, mask=is_cal, method=method)
        if sig is not None:
            overrides["calibrator_offset_mag"] = float(scale) * float(sig)

    surveys = np.unique(idsurvey)
    counts_cal = {int(s): int(np.sum((idsurvey == int(s)) & is_cal)) for s in surveys.tolist()}
    counts_hf = {int(s): int(np.sum((idsurvey == int(s)) & is_hf)) for s in surveys.tolist()}
    meta["counts_cal"] = counts_cal
    meta["counts_hf"] = counts_hf

    if include_survey_offsets:
        for s in surveys.tolist():
            s = int(s)
            sig = _estimate_sigma_from_cov(cov=cov, mask=(idsurvey == s), method=method)
            if sig is None:
                continue
            overrides[f"survey_offset_{s}"] = float(scale) * float(sig)

    if include_hf_survey_offsets:
        for s in surveys.tolist():
            s = int(s)
            if counts_hf.get(s, 0) < min_hf_per_survey:
                continue
            sig = _estimate_sigma_from_cov(cov=cov, mask=is_hf & (idsurvey == s), method=method)
            if sig is None:
                continue
            overrides[f"hf_survey_offset_{s}"] = float(scale) * float(sig)

    if include_cal_survey_offsets:
        for s in surveys.tolist():
            s = int(s)
            if counts_cal.get(s, 0) < min_cal_per_survey:
                continue
            sig = _estimate_sigma_from_cov(cov=cov, mask=is_cal & (idsurvey == s), method=method)
            if sig is None:
                continue
            overrides[f"cal_survey_offset_{s}"] = float(scale) * float(sig)

    bid = _pkmjd_bin_id(t=pkmjd, edges=pkmjd_edges)
    n_bins = int(len(pkmjd_edges) - 1)
    meta["n_pkmjd_bins"] = n_bins

    if include_time_bins:
        for k in range(1, n_bins):
            m = is_cal & (bid == k)
            if int(np.sum(m)) < min_per_time_bin:
                continue
            sig = _estimate_sigma_from_cov(cov=cov, mask=m, method=method)
            if sig is None:
                continue
            overrides[f"pkmjd_bin_offset_{k}"] = float(scale) * float(sig)

    if include_survey_time_bins:
        for s in surveys.tolist():
            s = int(s)
            if counts_cal.get(s, 0) < min_cal_per_survey:
                continue
            in_s = (idsurvey == s)
            for k in range(1, n_bins):
                m = is_cal & in_s & (bid == k)
                if int(np.sum(m)) < min_per_time_bin:
                    continue
                sig = _estimate_sigma_from_cov(cov=cov, mask=m, method=method)
                if sig is None:
                    continue
                overrides[f"survey_pkmjd_bin_offset_{s}_{k}"] = float(scale) * float(sig)

    return overrides, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_cov_v1.json",
        help="Output JSON path for sigma_overrides mapping.",
    )
    ap.add_argument(
        "--raw-dat-path",
        default="data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat",
        help="Path to the Pantheon+SH0ES .dat file (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--raw-cov-path",
        default="data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov",
        help="Path to the Pantheon+SH0ES .cov file to summarize (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--processed-dir",
        default="data/processed/pantheon_plus_shoes_ladder",
        help="Directory for cached processed Pantheon+SH0ES ladder products (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--tag",
        default="cal+hf_stat+sys_zHD",
        help="Dataset cache tag (change when swapping raw_cov_path / cuts).",
    )
    ap.add_argument("--method", choices=["median", "mean"], default="median")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--min-cal-per-survey", type=int, default=4)
    ap.add_argument("--min-hf-per-survey", type=int, default=20)
    ap.add_argument("--min-per-time-bin", type=int, default=10)
    ap.add_argument(
        "--pkmjd-edges",
        default="44672.6,53410.4,54131.12,54885.0,57269.22,59385.6",
        help="Comma-separated pkmjd bin edges (MJD).",
    )
    ap.add_argument("--include-survey-offsets", action="store_true")
    ap.add_argument("--include-hf-survey-offsets", action="store_true")
    ap.add_argument("--include-cal-survey-offsets", action="store_true", default=True)
    ap.add_argument("--include-calibrator-offset", action="store_true", default=True)
    ap.add_argument("--include-time-bins", action="store_true", default=True)
    ap.add_argument("--include-survey-time-bins", action="store_true", default=False)
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

    edges = [float(x) for x in str(args.pkmjd_edges).split(",") if str(x).strip()]

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

    overrides, meta = derive_sigma_overrides(
        cov=ds.cov,
        idsurvey=ds.idsurvey,
        is_cal=ds.is_calibrator,
        is_hf=ds.is_hubble_flow,
        pkmjd=ds.pkmjd,
        pkmjd_edges=edges,
        min_cal_per_survey=int(args.min_cal_per_survey),
        min_hf_per_survey=int(args.min_hf_per_survey),
        min_per_time_bin=int(args.min_per_time_bin),
        include_survey_offsets=bool(args.include_survey_offsets),
        include_hf_survey_offsets=bool(args.include_hf_survey_offsets),
        include_cal_survey_offsets=bool(args.include_cal_survey_offsets),
        include_calibrator_offset=bool(args.include_calibrator_offset),
        include_time_bins=bool(args.include_time_bins),
        include_survey_time_bins=bool(args.include_survey_time_bins),
        method=str(args.method),
        scale=float(args.scale),
    )

    out_path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    meta_out = {
        "created_by": "scripts/derive_pantheon_shoes_cov_priors.py",
        "notes": [
            "This file encodes per-parameter sigma overrides estimated from a Pantheon+SH0ES covariance file.",
            "Method: estimate sigma as sqrt(median positive off-diagonal cov) within each group.",
            "Use as model.priors.sigma_overrides_path in the pantheon_plus_shoes_ladder adapter.",
            "This is not an independent external calibration log; it is a covariance-implied bound from the public data product.",
        ],
        "inputs": {
            "raw_dat_path": str(args.raw_dat_path),
            "raw_cov_path": str(args.raw_cov_path),
            "processed_dir": str(args.processed_dir),
            "tag": str(args.tag),
        },
        "meta": meta,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(overrides)} sigma overrides to {out_path}")
    print(f"Wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
